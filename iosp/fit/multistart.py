"""Batched multistart bilevel IOC: run B*S candidates in parallel, select at the end.

Every candidate is (IK branch b, cost seed s) — a complete, independent bilevel
fit with its branch fixed for all outer steps.  `jax.vmap` runs all candidates
as one batched program; only after convergence does an argmin over TRAINING loss
pick the winner.

Two-phase forward model
-----------------------
  1. Per-segment trajopt (approach, grasp, transport, place), chained through
     IK-seeded endpoints.  Each uses `ioc.inner.solve_implicit`.
  2. Refine trajopt over the full concatenated trajectory.

Hard selection INSIDE the differentiated map makes x*(theta) discontinuous and
breaks the implicit adjoint.  Fixing the branch per candidate keeps the map
smooth; selection on training loss (never held-out) avoids test-set leakage.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ioc import outer as outer_opt
from ioc.inner import make_inner_solver
from iosp.model import fr3, pickplace as pp
from iosp.fit import parametric as s3
from iosp import config
from iosp.model.scenes import scene_a, scene_b, scenes_ab


def build(seed=0, n_iters=60, n_branches=4, scene_b_scale=1.0, z_prior=None):
    """Assemble a forward map over a FLATTENED (candidate x scene) batch.

    Why flattened and not `jax.vmap`-over-candidates: the IK stage is a CUDA
    FFI call, and `jax.vmap` of an `ffi_call` that declares no `vmap_method`
    raises NotImplementedError.  Wrapping a whole bilevel fit in vmap therefore
    fails the moment it reaches `sqp_ik_solve_cuda_batch`.  Folding candidates
    into the solver's EXISTING problem-batch axis avoids the nesting entirely:
    every call stays single-level and the IK is still one kernel launch.

    The outer loop then needs no vmap either.  Each candidate's loss depends
    only on its own row of U, so summing them and taking ONE gradient gives
    each candidate its own gradient exactly -- no interaction, no per-candidate
    loop, and Adam updates the whole (C, K) array elementwise.
    """
    prob = pp.PickPlaceProblem.load(str(config.URDF_PATH), str(config.SRDF_PATH), str(config.MESH_DIR))
    fs = pp.make_composed_forward_solver(n_iters=n_iters)
    scenes = scenes_ab(scene_b_scale)          # M=2: row 0 fit, row 1 held out
    inner, _ = s3._build_inner(prob, scene_a(), config.THETA_IK_STAR, fs, seed)

    # Phase 2: refine solver using pyroffi's dynamics-aware trajopt + implicit
    # adjoint, over the full concatenated trajectory.  Uses fewer L-BFGS
    # iterations than the per-segment solver: the refine starts from stage 1's
    # warm output, so it converges faster.
    refine_fs = pp.make_composed_forward_solver(n_iters=min(n_iters, 20), robot=prob.base.robot)
    full_residual_fn = prob.full_residual_fn()
    full_sc_cal = prob.full_scenes(
        scene_a(),
        prob.grasp_ik(config.THETA_IK_STAR, scene_a()),
        prob.place_ik(config.THETA_IK_STAR, scene_a(),
                      prob.grasp_ik(config.THETA_IK_STAR, scene_a())),
    )
    full_scales = prob.calibrate_full(full_residual_fn, full_sc_cal,
                                      jax.random.PRNGKey(seed + 7))
    refine_inner = make_inner_solver(full_residual_fn, full_scales,
                                     forward_solver=refine_fs)

    refs = prob.branch_refs(scenes, n_branches, jax.random.PRNGKey(seed + 11))
    K = pp.K_IK + pp.K_TRAJOPT + pp.K_FULL
    S = s3.z_scale(pp.K_IK + pp.K_TRAJOPT, pp.K_IK)
    S = jnp.concatenate([S, jnp.ones(pp.K_FULL)])
    # `z = P + S * u`.  P is zero on the synthetic path -- `u = 0` there means
    # zero standoff and flat logits, which is what every recorded result used --
    # and non-zero for teleop demonstrations, where a zero standoff would put
    # the flange inside the object.  See `iosp.fit.teleop`'s module docstring.
    P = jnp.zeros(K, dtype=jnp.float32) if z_prior is None else jnp.asarray(z_prior)
    M = scenes.q_start.shape[0]

    def batched_paths(U, refs_c, space="ee"):
        """(C, M, T, D) for C candidates, each on its own FIXED branch.

        Two-phase pipeline: per-segment trajopt -> full refine, both using
        pyroffi's dynamics-aware solver with implicit adjoint.
        """
        if space not in ("ee", "joint"):
            raise ValueError(f"space must be 'ee' or 'joint', got {space!r}")
        C = U.shape[0]
        Z = P + U * S                                         # (C, K)
        theta_ik = Z[:, : pp.K_IK]                            # (C, 2)
        seg_end = pp.K_IK + pp.K_TRAJOPT
        theta_tr = jax.nn.softmax(Z[:, pp.K_IK : seg_end], axis=-1)  # (C, 7)
        theta_full = jax.nn.softmax(Z[:, seg_end :], axis=-1)         # (C, 4)

        rep = lambda a: jnp.repeat(a, M, axis=0)              # candidate -> C*M
        sc = jax.tree.map(lambda a: jnp.tile(a, (C,) + (1,) * (a.ndim - 1)), scenes)
        tik = rep(theta_ik)                                   # (C*M, K_IK)
        refs_n = rep(refs_c)                                  # (C*M, dof)

        # The in-plane release offsets are applied here too, even though every
        # synthetic scene sets them to zero in `THETA_IK_STAR`: leaving them out
        # would make `place.radial`/`place.tangential` DEAD parameters on this
        # path -- exactly zero gradient, two spurious null directions in a
        # spectrum this study reads rank off.  At the synthetic ground truth the
        # added term is identically zero, so recorded results are unchanged.
        radial, tangential = pp._place_frame(sc.place_pos)
        tgt_pick = sc.pick_pos + tik[:, 0:1] * pp.UP_AXIS
        q_pick = pp._ik_batch(prob, tgt_pick, refs_n, prob._grasp_wxyz(sc))
        tgt_place = (sc.place_pos + tik[:, 1:2] * pp.UP_AXIS
                     + tik[:, 2:3] * radial + tik[:, 3:4] * tangential)
        q_place = pp._ik_batch(prob, tgt_place, q_pick, prob._place_wxyz(sc))

        ps = {
            "approach": pp.Scene(sc.q_start, q_pick, sc.obs_center, sc.obs_radius),
            "grasp": pp.Scene(q_pick, q_pick, sc.obs_center, sc.obs_radius),
            "transport": pp.Scene(q_pick, q_place, sc.obs_center, sc.obs_radius),
            "place": pp.Scene(q_place, q_place, sc.obs_center, sc.obs_radius),
        }
        tt = {p: rep(theta_tr) for p in pp.PHASES}

        # -- phase 1: per-segment solves (pyroffi dynamics-aware + implicit) --
        xs = {}
        for p in pp.PHASES:
            x0 = jax.vmap(prob.seg[p].seed)(ps[p])
            xs[p] = jax.vmap(inner[p].solve_implicit, in_axes=(0, 0, 0))(x0, tt[p], ps[p])

        # -- phase 2: refine over the full trajectory -------------------------
        # Concatenate segment solutions (undetached) and solve a single trajopt
        # over the entire path using pyroffi's dynamics-aware solver +
        # implicit adjoint.
        full_sc = pp.FullScene(sc.q_start, q_place, sc.obs_center,
                               sc.obs_radius, q_pick)
        x0_full = prob.full_seed_from_segments(xs, ps)
        tf = rep(theta_full)                                  # (C*M, 4)
        x_full = jax.vmap(refine_inner.solve_implicit, in_axes=(0, 0, 0))(
            x0_full, tf, full_sc)

        # Read out the refined path.
        q = jax.vmap(prob.seg["full"].unpack)(x_full, full_sc)
        e = q if space == "joint" else jax.vmap(prob.ee_positions)(q)
        path = e.reshape(C, M, e.shape[1], e.shape[-1])
        return path

    u_star = jnp.concatenate([config.THETA_IK_STAR, config.Z_TRAJOPT_STAR,
                              config.Z_FULL_STAR]) / S
    demo = jax.jit(lambda: batched_paths(u_star[None], refs[0][None])[0])()
    demo_q = jax.jit(lambda: batched_paths(u_star[None], refs[0][None], "joint")[0])()
    return dict(prob=prob, batched_paths=batched_paths, refs=refs, demo=demo,
                demo_joint=demo_q, scenes=scenes, scene_b_scale=scene_b_scale,
                u_star=u_star, K=K, S=S, P=P, M=M,
                # The scene axis, split into what the loss sees and what it
                # never does.  Named here rather than hard-coded as rows 0/1 in
                # `run`, because a demonstration set has one row per EPISODE and
                # the split is a property of the dataset, not of the optimizer.
                fit_idx=np.array([0]), gen_idx=np.array([1]), space="ee",
                names=list(pp.THETA_IK_NAMES) + list(pp.THETA_TRAJOPT_NAMES)
                      + list(pp.THETA_FULL_NAMES))


def build_from_demos(demo_dir=None, teleop_root=None, n_fit=None, seed=0,
                     n_iters=60, n_branches=4, anchor_grasp=False,
                     max_episodes=None):
    """`build`, but the demonstration is RECORDED HUMAN TELEOP, not a rollout.

    Same three-stage forward model, same fixed-branch candidate structure, same
    single end-of-run selection.  Four things differ, and each one is forced by
    the demonstration no longer coming from this model:

      * The scene axis is the EPISODE axis.  `build` has M = 2 (scene A, scene
        B); here M is the number of recorded episodes, each with its own cube,
        bucket and start configuration, taken verbatim from that episode's
        randomisation record.  Fitting a rollout on a scene the demonstration
        did not happen in would not be a reconstruction of anything.
      * `demo` is the recorded path, in JOINT space.  `q_d` is what the operator
        commanded, and the 7-DOF arm's self-motion manifold is invisible to an
        EE loss -- a rollout can match the demonstrated EE path to the
        millimetre through a completely different elbow.
      * `u = 0` carries a measured standoff prior (`iosp.fit.teleop.z_prior`),
        because the EE frame is the flange and a zero standoff puts it inside
        the object.
      * `u_star` does not exist.  There is no vector of weights that generated a
        human, so the claim under test is behavioural -- does a cost fitted on
        some episodes reproduce held-out ones -- and every parameter-recovery
        metric is undefined.  It is returned as None so anything reaching for it
        fails loudly instead of scoring against a fiction.

    The robot is the FR3 the episodes were recorded on, NOT the Panda every
    synthetic result used; see `iosp.model.fr3`.
    """
    from iosp.fit import teleop as tl

    kw = {}
    if demo_dir is not None:
        kw["demo_dir"] = demo_dir
    if teleop_root is not None:
        kw["teleop_root"] = teleop_root
    urdf, srdf, mesh_dir, ee_link = fr3.paths()
    prob = pp.PickPlaceProblem.load(urdf, srdf, mesh_dir, ee_link=ee_link)
    names, demo_q, scenes = tl.load_demos(prob=prob, anchor_grasp=anchor_grasp,
                                          max_episodes=max_episodes, **kw)
    M = len(names)
    n_fit = M - max(1, M // 4) if n_fit is None else int(n_fit)
    # `n_fit == M` is allowed: a single-problem reconstruction check has no
    # held-out set, and reporting a held-out RMSE over zero episodes would be a
    # number with no content rather than an error.
    if not 0 < n_fit <= M:
        raise ValueError(f"n_fit must be in (0, {M}]; got {n_fit}")
    fit_idx, gen_idx = np.arange(n_fit), np.arange(n_fit, M)
    fit_scenes = jax.tree.map(lambda a: a[fit_idx], scenes)

    fs = pp.make_composed_forward_solver(n_iters=n_iters, robot=prob.base.robot)

    K = pp.K_IK + pp.K_TRAJOPT + pp.K_FULL
    S = s3.z_scale(pp.K_IK + pp.K_TRAJOPT, pp.K_IK)
    S = jnp.concatenate([S, jnp.ones(pp.K_FULL)])
    standoffs = tl.measure_standoffs(prob, demo_q, scenes, fit_idx)
    P = jnp.concatenate([jnp.asarray(standoffs),
                         jnp.zeros(K - pp.K_IK, dtype=jnp.float32)])
    theta_ik0 = P[: pp.K_IK]
    print(f"[build] {M} episodes ({n_fit} fit / {len(gen_idx)} held out); "
          f"standoff prior grasp {standoffs[0]:.4f} m, place {standoffs[1]:.4f} m",
          flush=True)

    # Feature scales are calibrated on the FIT episodes only, at the prior
    # `theta_ik` -- the same discipline as `iosp.fit.parametric`'s scene-A-only
    # calibration, for the same reason: a scale fitted on the held-out episodes
    # lets them re-normalise the very features being tested.
    inner, _ = s3._build_inner(prob, fit_scenes, theta_ik0, fs, seed)
    refine_fs = pp.make_composed_forward_solver(n_iters=min(n_iters, 20), robot=prob.base.robot)
    full_residual_fn = prob.full_residual_fn()
    q_pick_cal = prob.grasp_ik(theta_ik0, fit_scenes)
    full_sc_cal = prob.full_scenes(fit_scenes, q_pick_cal,
                                   prob.place_ik(theta_ik0, fit_scenes, q_pick_cal))
    full_scales = prob.calibrate_full(full_residual_fn, full_sc_cal,
                                      jax.random.PRNGKey(seed + 7))
    refine_inner = make_inner_solver(full_residual_fn, full_scales,
                                     forward_solver=refine_fs)
    refs = prob.branch_refs(scenes, n_branches, jax.random.PRNGKey(seed + 11))

    def batched_paths(U, refs_c, space="joint"):
        """(C, M, T, D) -- `build.batched_paths` with this problem's closures.

        Kept as a separate body rather than shared with `build`'s: the two
        differ only in which `prob`/`inner`/`scenes`/`P` they close over, and
        the alternative (threading eight objects through a shared function) put
        the forward model's data flow behind an indirection in the one place
        this package most needs it readable.
        """
        if space not in ("ee", "joint"):
            raise ValueError(f"space must be 'ee' or 'joint', got {space!r}")
        C = U.shape[0]
        Z = P + U * S
        theta_ik = Z[:, : pp.K_IK]
        seg_end = pp.K_IK + pp.K_TRAJOPT
        theta_tr = jax.nn.softmax(Z[:, pp.K_IK:seg_end], axis=-1)
        theta_full = jax.nn.softmax(Z[:, seg_end:], axis=-1)

        rep = lambda a: jnp.repeat(a, M, axis=0)
        sc = jax.tree.map(lambda a: jnp.tile(a, (C,) + (1,) * (a.ndim - 1)), scenes)
        tik = rep(theta_ik)
        refs_n = rep(refs_c)

        # Targets and orientations exactly as `grasp_ik`/`place_ik` build them
        # -- this map does NOT call them (it needs the candidate axis folded
        # into the kernel's problem batch), so anything added there has to be
        # mirrored here or it is silently inert inside the fit.
        #
        # SEEDS stay `refs_n` / `q_pick`, deliberately, even when the scene
        # carries `grasp_ref`/`place_ref`: covering several IK branches is what
        # this entry point exists for, and seeding every candidate from the one
        # demonstrated configuration would collapse the branch axis that
        # produced the 4.3x held-out spread.  Anchoring the ORIENTATION is where
        # the gain is anyway (0.530 -> 0.146 rad; the seed adds 0.146 -> 0.084).
        radial, tangential = pp._place_frame(sc.place_pos)
        place_target = (sc.place_pos + tik[:, 1:2] * pp.UP_AXIS
                        + tik[:, 2:3] * radial + tik[:, 3:4] * tangential)
        q_pick = pp._ik_batch(prob, sc.pick_pos + tik[:, 0:1] * pp.UP_AXIS,
                              refs_n, prob._grasp_wxyz(sc))
        q_place = pp._ik_batch(prob, place_target, q_pick, prob._place_wxyz(sc))

        ps = {
            "approach": pp.Scene(sc.q_start, q_pick, sc.obs_center, sc.obs_radius),
            "grasp": pp.Scene(q_pick, q_pick, sc.obs_center, sc.obs_radius),
            "transport": pp.Scene(q_pick, q_place, sc.obs_center, sc.obs_radius),
            "place": pp.Scene(q_place, q_place, sc.obs_center, sc.obs_radius),
        }
        tt = {p: rep(theta_tr) for p in pp.PHASES}

        xs = {}
        for p in pp.PHASES:
            x0 = jax.vmap(prob.seg[p].seed)(ps[p])
            xs[p] = jax.vmap(inner[p].solve_implicit, in_axes=(0, 0, 0))(x0, tt[p], ps[p])

        full_sc = pp.FullScene(sc.q_start, q_place, sc.obs_center,
                               sc.obs_radius, q_pick)
        x_full = jax.vmap(refine_inner.solve_implicit, in_axes=(0, 0, 0))(
            prob.full_seed_from_segments(xs, ps), rep(theta_full), full_sc)

        q = jax.vmap(prob.seg["full"].unpack)(x_full, full_sc)
        e = q if space == "joint" else jax.vmap(prob.ee_positions)(q)
        return e.reshape(C, M, e.shape[1], e.shape[-1])

    return dict(prob=prob, batched_paths=batched_paths, refs=refs,
                demo=demo_q, demo_ee=jax.vmap(prob.ee_positions)(demo_q),
                scenes=scenes, u_star=None, K=K, S=S, P=P, M=M,
                fit_idx=fit_idx, gen_idx=gen_idx, space="joint",
                episodes=names, n_fit=n_fit, standoff_prior=standoffs,
                names=list(pp.THETA_IK_NAMES) + list(pp.THETA_TRAJOPT_NAMES)
                      + list(pp.THETA_FULL_NAMES))


def make_candidates(built, n_starts, seed=0):
    """(C, K) initial cost params and (C, dof) branch refs, C = B*S."""
    refs, K = built["refs"], built["K"]
    rng = np.random.default_rng(seed + 3)
    u0 = [jnp.zeros(K, jnp.float32)] + [
        jnp.asarray(rng.normal(0, 0.5, K), jnp.float32) for _ in range(n_starts - 1)]
    B, S = refs.shape[0], len(u0)
    return (jnp.stack([u for _ in range(B) for u in u0]),
            jnp.concatenate([jnp.repeat(refs[b][None], S, 0) for b in range(B)]), B, S)


def run(seed=0, n_iters=60, n_branches=4, n_starts=3, n_steps=40, lr=config.LR,
        built=None, chunk=None):
    """`built=None` builds the synthetic problem, exactly as before.  Pass a
    `build_from_demos` dict to fit the same candidates against recorded human
    demonstrations instead; everything below is written against `fit_idx` /
    `gen_idx` / `demo`, so the two differ only in what the demonstration is.

    `chunk` splits the CANDIDATE axis into groups of that many, evaluated one
    group at a time.  It changes nothing about the result: each candidate's loss
    depends only on its own row of `U` (that is the property the flattened batch
    is built on), so the gradient of a chunk's summed loss is exactly those
    candidates' gradients, and concatenating the chunks reproduces the whole-
    batch gradient bit for bit.  Every chunk has the same shape, so it also
    costs one compilation, not C/chunk of them.

    It exists because the batch axis is `candidates x scenes` and the refine
    stage's collision Jacobian is dense in it.  MEASURED on the teleop run, on a
    24 GiB A5000, at 12 candidates x 10 episodes:

      chunk=None (120 rows)  died in AUTOTUNING, not execution ("Failed to get
                             configs for 3 out of 5590 instructions") -- the
                             profiler has to hold several configs of a
                             `f32[120, 147, 26, 23, 18, 4, 4]` buffer (rows x
                             refine decision dims x links x waypoints x spheres)
                             at once, and one of them is already 12.2 GiB
      chunk=4    (40 rows)   died in EXECUTION, one 14.19 GiB allocation
      chunk=1    (10 rows)   the configuration these results were run at

    Peak memory is linear in rows, so the knob is predictable: budget ~0.35 GiB
    per row of `chunk * M`.  The synthetic study never hit any of this because
    its scene axis is 2 rather than 10.  `chunk=None` keeps the whole batch in
    one call, which is byte-for-byte the old behaviour.
    """
    import optax
    t0 = time.perf_counter()
    if built is None:
        built = build(seed=seed, n_iters=n_iters, n_branches=n_branches)
    bp, demo = built["batched_paths"], built["demo"]
    fit_idx, gen_idx = built["fit_idx"], built["gen_idx"]
    space = built.get("space", "ee")
    U0, refs_c, B, S = make_candidates(built, n_starts, seed)
    C = U0.shape[0]
    print(f"[build] {time.perf_counter()-t0:.0f}s  B={B} branches x S={S} starts "
          f"= {C} candidates, {n_steps} outer steps each"
          + (f", in chunks of {chunk}" if chunk else ""), flush=True)

    def _per_cand(U, R, idx):
        # An empty scene set (n_fit == M, no held-out) scores as NaN rather than
        # a mean over nothing, which numpy would return as NaN with a warning
        # and which reads as 0.0 after the sqrt in `report`.
        if len(idx) == 0:
            return jnp.full((U.shape[0],), jnp.nan)
        """(C,) mean squared displacement over the scenes in `idx`.

        Mean over scenes as well as waypoints, so an 8-episode fit set and a
        2-episode held-out set are on the same scale and their RMSEs are
        directly comparable -- a sum would make the fit loss look 4x worse for
        free."""
        P_ = bp(U, R, space)[:, idx]                          # (C, |idx|, T, D)
        return jnp.mean(jnp.sum((P_ - demo[idx]) ** 2, -1), axis=(1, 2))

    cs = C if chunk is None else int(chunk)
    if C % cs:
        raise ValueError(f"chunk={cs} must divide the candidate count C={C}")
    bounds = [(i, i + cs) for i in range(0, C, cs)]

    def _chunked(fn, U, out_axis=0):
        """`fn` over each candidate group in turn, results concatenated."""
        return jnp.concatenate([fn(U[a:b], refs_c[a:b]) for a, b in bounds],
                               axis=out_axis)

    per_cand_train = lambda U: _chunked(
        lambda u, r: _per_cand(u, r, fit_idx), U)
    per_cand_held = lambda U: _chunked(
        lambda u, r: _per_cand(u, r, gen_idx), U)

    _gf1 = jax.jit(jax.value_and_grad(
        lambda u, r: jnp.sum(_per_cand(u, r, fit_idx))))

    def gf(U):
        vals, grads = zip(*(_gf1(U[a:b], refs_c[a:b]) for a, b in bounds))
        return sum(float(v) for v in vals), jnp.concatenate(grads, axis=0)
    opt = optax.adamw(lr, weight_decay=0.0)
    U, st = U0, opt.init(U0)
    t0 = time.perf_counter()
    hist = []
    for t in range(n_steps):
        v, g = gf(U)
        hist.append(float(v))
        upd, st = opt.update(g, st, U)
        U = optax.apply_updates(U, upd)
        if t % 5 == 0 or t == n_steps - 1:
            print(f"[fit] step {t:3d}/{n_steps}  sum-loss={float(v):.6f}", flush=True)
    print(f"[fit] {C} candidates x {n_steps} steps in "
          f"{time.perf_counter()-t0:.0f}s (compile incl.)", flush=True)

    tr = np.asarray(per_cand_train(U))
    he = np.asarray(per_cand_held(U))
    return dict(u=np.asarray(U), losses=np.asarray(hist), train=tr, held=he,
                B=B, S=S, refs=np.asarray(built["refs"]), built=built)


def select_winner(out):
    """The ONE selection in this study, on TRAINING loss.

    Asserted, not commented: choosing on held-out loss would leak the test set
    into the number the paper reports."""
    tr, he = out["train"], out["held"]
    w = int(np.argmin(tr))
    leak = int(np.argmin(he))
    assert np.argmin(tr) == w, "winner must be the training-loss argmin"
    out["winner"], out["leak_winner"] = w, leak
    return w, leak


def report(out):
    w, leak = select_winner(out)
    B, S, tr, he = out["B"], out["S"], out["train"], out["held"]
    print(f"\n{'cand':>5} {'branch':>7} {'start':>6} {'train rmse':>11} {'held rmse':>10}")
    for i in range(len(tr)):
        mark = "  <- winner (train argmin)" if i == w else ""
        mark += "  [held argmin: NOT used]" if i == leak and i != w else ""
        print(f"{i:5d} {i//S:7d} {i%S:6d} {np.sqrt(tr[i]):11.5f} "
              f"{np.sqrt(he[i]):10.5f}{mark}")
    print(f"\nselected on training loss: candidate {w} (branch {w//S}, start {w%S})")
    print(f"  held-out RMSE = {np.sqrt(he[w]):.5f}")
    if leak != w:
        print(f"  had we (wrongly) selected on held-out: {np.sqrt(he[leak]):.5f} "
              f"-- {100*(1-np.sqrt(he[leak])/np.sqrt(he[w])):+.1f}%, the size of "
              f"the leakage this guards against")
    spread = np.sqrt(he).max() / np.sqrt(he).min()
    print(f"\nheld-out RMSE spread across candidates: {spread:.1f}x "
          f"({np.sqrt(he).min():.5f} .. {np.sqrt(he).max():.5f})")
    per_branch = [np.sqrt(he[b*S:(b+1)*S]).min() for b in range(B)]
    print("best held-out per branch:", np.round(per_branch, 5),
          "\n-> branch choice matters" if max(per_branch)/min(per_branch) > 1.5
          else "\n-> branches are behaviourally equivalent here")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--n-branches", type=int, default=4)
    ap.add_argument("--n-starts", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=40)
    a = ap.parse_args()
    report(run(seed=a.seed, n_iters=a.n_iters, n_branches=a.n_branches,
               n_starts=a.n_starts, n_steps=a.n_steps))
