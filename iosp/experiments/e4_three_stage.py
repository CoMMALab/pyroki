"""E4 — Inverting spasm's three-stage pick-and-place forward pass.

Three stages: (1) IK seeds endpoints, (2) per-segment trajopt, (3) refine
trajopt over the full concatenated trajectory.  One shared theta over
{smooth, clearance, upright, skeleton}; theta_ik fixed.

Sharing theta is required for differentiability: per-stage weights would be
dead parameters (the implicit adjoint gives zero sensitivity to x0).

`--ablate-stage2` tests whether the segment stage matters by seeding stage 3
from a straight line instead.  Success criterion: held-out EE RMSE.
"""

import argparse
import time

import jax

from iosp.config import enable_compilation_cache
enable_compilation_cache()

import jax.numpy as jnp
import numpy as np

from ioc import outer as outer_opt
from ioc.inner import make_inner_solver
from iosp.model import pickplace as pp
from iosp.config import MESH_DIR, SRDF_PATH, URDF_PATH
from iosp.model.scenes import sample_pickplace_scenes

# Fixed, NOT estimated: IK is a seeding step, not a preference.
THETA_IK = jnp.array([0.105, 0.105, 0.0, 0.0], dtype=jnp.float32)

# Ground truth, pre-softmax.  `skeleton` highest: a demonstrator that mostly
# honours the task skeleton but will trade some fidelity for global smoothness
# -- that exchange rate is what this study recovers.
#
# SELECTED BY MEASUREMENT over 24 candidate vectors, in three rounds, each one
# solved through the full three-stage chain and executed through the actuated
# MuJoCo rollout (`spasm_rollout.rollout_and_score_pickplace`, all 6 fit scenes,
# every run carrying the `mj_rollout.run_events` grasp-timing fix).  Success is
# settled position AND upright; the `err`/`tilt` columns are the first three
# scenes, which are the only ones any vector solves.
#
#   name          vector                                   ok    err mm    tilt
#   base          [1.0, 1.5, 0.5, 0.5, 1.0, 1.5, 2.5]     1/6   39/38/32   0/90/90
#   THIS          [1.0, 1.5, 1.5, 2.0, 1.0, 1.5, 4.5]     3/6   23/29/23   0/0/0
#   smooth_skel   [1.0, 1.5, 1.5, 2.0, 1.0, 1.5, 3.5]     3/6   23/27/26   0/0/0
#   smooth_only   [1.0, 1.5, 1.5, 2.0, 1.0, 1.5, 2.5]     3/6   25/43/21   0/0/0
#   jerk_only     [1.0, 1.5, 0.5, 2.5, 1.0, 1.5, 2.5]     2/6   29/43/37   0/0/90
#   clearance-max [1.0, 1.0, 2.5, 3.0, 0.5, 3.5, 2.5]     2/6   42/46/33   0/90/0
#   smooth_more   [1.0, 1.5, 2.0, 3.0, 1.0, 1.5, 2.5]     0/6   20/37/16   90/90/90
#   effort_max    [1.0, 1.5, 0.5, 0.5, 3.0, 1.5, 2.5]     0/6  405/439/468 90/90/90
#
# `smooth_skel` ties this vector at 3/6 (mean settled error 25.3 mm against
# 25.0); the tie was broken on that mean and the difference is within noise.
#
# What the sweep actually shows, and the reason the winner looks the way it does:
# raising `accel`/`jerk` from 0.5 to 1.5/2.0 is what buys the wins, and every
# variant that ALSO raises `clearance` gives them back.  Note `smooth_more`,
# which has the LOWEST position errors in the whole sweep (16-37 mm) and scores
# 0/6: it lands the cube dead on target and topples it every time.  Position
# accuracy and settled uprightness are close to anti-correlated here, because a
# heavier smoothness or clearance term lifts the release and nothing in
# `STANDARD_FEATURES` prices drop height -- `upright_constraint_fn` holds the
# GRIPPER level but says nothing about how far the object then falls.  Chasing
# error alone selects vectors that fail.
#
# Scenes 3/4/5 fail under ALL 24 vectors (448/~200/407 mm, cube never leaves the
# table), so 3/6 is the ceiling reachable by reweighting and the rest is not a
# weight problem: the synthetic scenes carry `table_box=None` and
# `PickPlaceProblem.full_scenes` does not forward `table_box`/`obs_spheres`
# anyway, so the planner's world is a single obstacle sphere with no table in
# it.  `clearance` at any weight prices distance to that sphere, not to the
# table the arm actually hits.
#
# CHANGES RECORDED NUMBERS: every E4 result and every saved forward extract
# predating this used [1.0, 1.5, 0.5, 0.5, 1.0, 1.5, 2.5].
Z_STAR = jnp.array([1.0, 1.5, 1.5, 2.0, 1.0, 1.5, 4.5], dtype=jnp.float32)  # time,path,accel,jerk,effort,clearance,skeleton
PARAM_NAMES = list(pp.THETA_SHARED_NAMES)


def build(seed=0, n_iters=60, n_scenes=6, constrained=False):
    """Assemble the tied three-stage model and the fit/test scene split.

    `soft_line_search`/`soft_curvature_gate` are OFF, unlike
    `make_composed_forward_solver`'s default: on the single-segment problem
    those were measured to buy landscape smoothness at the cost of FD
    agreement (cos 0.906 -> 0.324), i.e. the adjoint answering a question
    about a slightly different solve than the one that ran.  This study's
    claim is behavioural fidelity, so that is the wrong trade.
    """
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_stock_forward_solver(n_iters=n_iters)

    rng = np.random.default_rng(seed)
    scenes_all = sample_pickplace_scenes(rng, 2 * n_scenes)
    fit = jax.tree.map(lambda a: a[:n_scenes], scenes_all)
    test = jax.tree.map(lambda a: a[n_scenes:], scenes_all)

    # Whitening scales are calibrated ONCE per stage and frozen across every
    # arm and baseline: re-calibrating per arm would let one win by rescaling
    # its features rather than by fitting them.  They stay PER-STAGE even
    # though theta is shared -- sigma carries units, theta carries preference.
    x0_star, phase_scenes_star, q_pick, q_place = prob.seeds(fit, THETA_IK)
    key = jax.random.PRNGKey(seed)
    # Collision as a theta-INDEPENDENT hard constraint (opt-in): each phase's
    # smooth soft-min obstacle clearance (`RobotProblem.collision_constraints_fn`,
    # well-conditioned -- FD-validated cos>0.998, unlike the torque hinge, see
    # `torque-constraint-deferred`) is folded into the constrained implicit
    # adjoint, so the recovered costs are inverted through a collision-feasible
    # forward map rather than one where collision is only a soft cost term.
    def _cfn(seg_problem):
        return seg_problem.collision_constraints_fn() if constrained else None

    inner_by_phase = {}
    for p in pp.PHASES:
        rf = prob.shared_segment_residual_fn(p)
        scales = prob.calibrate_segment(p, rf, phase_scenes_star[p], key)
        inner_by_phase[p] = make_inner_solver(
            rf, scales, forward_solver=forward_solver,
            constraints_fn=_cfn(prob.seg[p]))

    full_rf = prob.shared_full_residual_fn()
    full_scales = prob.calibrate_full(full_rf, prob.full_scenes(fit, q_pick, q_place), key)
    refine = make_inner_solver(full_rf, full_scales, forward_solver=forward_solver,
                               constraints_fn=_cfn(prob.seg["full"]))

    return dict(prob=prob, fit=fit, test=test, inner_by_phase=inner_by_phase,
                refine=refine, seed=seed)


def ee_paths(built, scenes, z, *, stage2=True):
    """(B, T, 3) end-effector paths of the refined trajectory under theta=softmax(z).

    `stage2=False` seeds the refine pass from a straight line instead of the
    segment solutions -- the ablation that measures whether the segment stage
    contributes anything the fit can see.
    """
    prob, refine = built["prob"], built["refine"]
    theta = jax.nn.softmax(z)
    theta_seg, theta_full = prob.split_shared(theta)
    x0, phase_scenes, q_pick, q_place = prob.seeds(scenes, THETA_IK)
    full_sc = prob.full_scenes(scenes, q_pick, q_place)
    if stage2:
        _, _, xs, ps = prob.solve(THETA_IK, {p: theta_seg for p in pp.PHASES},
                                  scenes, built["inner_by_phase"], x0,
                                  refine=refine, theta_full=theta_full)
        x_full, sc_full = xs["full"], ps["full"]
    else:
        x0_full = jax.vmap(prob.seg["full"].seed)(full_sc)
        x_full = jax.vmap(refine.solve_implicit, in_axes=(0, None, 0))(
            x0_full, theta_full, full_sc)
        sc_full = full_sc
    q = jax.vmap(prob.seg["full"].unpack)(x_full, sc_full)
    return jax.vmap(prob.ee_positions)(q)


def make_loss(built, scenes, demos, *, stage2=True):
    def loss(z):
        p = ee_paths(built, scenes, z, stage2=stage2)
        return jnp.mean(jnp.sum((p - demos) ** 2, axis=-1))
    return loss


def fit_z(loss_and_grad, starts, *, lr=0.05, n_steps=40):
    best = None
    for z0 in starts:
        z, trace = outer_opt.adam(loss_and_grad, z0, lr=lr, n_steps=n_steps)
        lN = min(v for _, v in trace)
        if best is None or lN < best["lN"]:
            best = dict(z=z, z0=z0, l0=trace[0][1], lN=lN, trace=trace)
    a0, ah = np.asarray(best["z0"], float), np.asarray(best["z"], float)
    mv = float(np.linalg.norm(ah - a0) / (np.linalg.norm(a0) + 1e-30))
    red = best["l0"] / max(best["lN"], 1e-30)
    best.update(move_rel=mv, loss_reduction=red,
                degenerate=(not np.isfinite(mv)) or mv < 1e-3 or red < 1.05)
    return best


def run(seed=0, n_iters=60, n_scenes=6, n_steps=40, n_starts=8, ablate_stage2=True,
        constrained=False):
    t_wall_start = time.perf_counter()
    t0 = time.perf_counter()
    built = build(seed=seed, n_iters=n_iters, n_scenes=n_scenes, constrained=constrained)
    fit, test = built["fit"], built["test"]
    print(f"[build] {time.perf_counter()-t0:.1f}s  K={pp.K_SHARED} {PARAM_NAMES}  "
          f"N_FULL={pp.N_FULL}  fit/test = {n_scenes}/{n_scenes}  "
          f"theta_ik FIXED at {np.asarray(THETA_IK)}", flush=True)

    t0 = time.perf_counter()
    demos_fit = jax.jit(lambda: ee_paths(built, fit, Z_STAR))()
    demos_test = jax.jit(lambda: ee_paths(built, test, Z_STAR))()
    jax.block_until_ready((demos_fit, demos_test))
    print(f"[demos] {time.perf_counter()-t0:.1f}s (compile incl.), "
          f"shape {tuple(demos_fit.shape)}", flush=True)

    gf = jax.jit(jax.value_and_grad(make_loss(built, fit, demos_fit)))
    jt = jax.jit(make_loss(built, test, demos_test))

    sanity_val, sanity_grad = gf(Z_STAR)
    sanity = float(sanity_val)
    print(f"[sanity] loss(z_star) = {sanity:.3e}", flush=True)
    assert sanity < 1e-4, f"model does not reproduce its own demo: {sanity:.3e}"

    _, g0 = gf(jnp.zeros(pp.K_SHARED, jnp.float32))
    g0 = np.asarray(g0)
    print(f"[grad] dL/dtheta at z=0: {g0}", flush=True)
    assert np.all(np.abs(g0) > 1e-12), f"dead parameter(s): {g0}"

    rng = np.random.default_rng(seed + 1)
    z_zero = jnp.zeros(pp.K_SHARED, dtype=jnp.float32)
    starts = [z_zero] + [jnp.asarray(rng.normal(0, 1, pp.K_SHARED), jnp.float32)
                         for _ in range(n_starts - 1)]

    out = {}
    t_fit = time.perf_counter()
    b = fit_z(gf, starts, n_steps=n_steps)
    t_fit = time.perf_counter() - t_fit
    b["test_rmse"] = float(jnp.sqrt(jt(b["z"])))
    out["fit"] = b
    print(f"[fit] test RMSE {b['test_rmse']:.5f}  move_rel={b['move_rel']:.3e} "
          f"reduction={b['loss_reduction']:.2f}x "
          f"{'DEGENERATE' if b['degenerate'] else 'ok'}  "
          f"fit_time={t_fit:.1f}s", flush=True)

    out["baseline_zero"] = float(jnp.sqrt(jt(z_zero)))
    out["baseline_uniform"] = out["baseline_zero"]  # softmax(0) IS uniform here
    out["oracle"] = float(jnp.sqrt(jt(Z_STAR)))

    if ablate_stage2:
        gf2 = jax.jit(jax.value_and_grad(make_loss(built, fit, demos_fit, stage2=False)))
        jt2 = jax.jit(make_loss(built, test, demos_test, stage2=False))
        b2 = fit_z(gf2, starts, n_steps=n_steps)
        b2["test_rmse"] = float(jnp.sqrt(jt2(b2["z"])))
        out["no_stage2"] = b2
        print(f"[ablate] refine-only (straight-line seed): test RMSE "
              f"{b2['test_rmse']:.5f}", flush=True)

    th, ths = np.asarray(jax.nn.softmax(b["z"])), np.asarray(jax.nn.softmax(Z_STAR))
    out["theta_hat"], out["theta_star"] = th, ths
    out["param_err"] = float(np.linalg.norm(th - ths))

    t_wall = time.perf_counter() - t_wall_start
    out["wall_s"] = float(t_wall)
    out["fit_s"] = float(t_fit)
    print(f"[time] wall={t_wall:.1f}s  fit={t_fit:.1f}s", flush=True)
    return out


def report(out):
    print("\n=== held-out behavioural recovery (EE RMSE, m) ===")
    ref = out["baseline_uniform"]
    rows = [("baseline uniform (theta = 1/K)", ref),
            ("tied three-stage fit", out["fit"]["test_rmse"])]
    if "no_stage2" in out:
        rows.append(("  ablation: refine only, no stage 2", out["no_stage2"]["test_rmse"]))
    rows.append(("oracle theta*", out["oracle"]))
    for n, v in rows:
        print(f"  {n:36s} {v:9.5f}   {100*(1-v/ref):+6.1f}% vs uniform")

    if out["fit"]["degenerate"]:
        print(f"\nNO VERDICT: the fit is degenerate (move_rel="
              f"{out['fit']['move_rel']:.2e}, reduction="
              f"{out['fit']['loss_reduction']:.2f}x) -- its RMSE is its seed's.")
        return
    if "no_stage2" in out:
        a, b = out["fit"]["test_rmse"], out["no_stage2"]["test_rmse"]
        rel = abs(a - b) / max(b, 1e-30)
        verdict = ("stage 2 is load-bearing: seeding the refine pass from the "
                   "segments changes held-out RMSE by "
                   f"{100*(1-a/b):+.1f}%" if rel > 0.05 else
                   "stage 2 washes out: the refine pass reaches the same basin "
                   "from a straight line, so on this problem inverting spasm "
                   "reduces to inverting one global trajopt")
        print(f"\n{verdict}.")
    print(f"\nparam_err (reported, NOT the criterion): {out['param_err']:.4f}")
    for n, h, s in zip(PARAM_NAMES, out["theta_hat"], out["theta_star"]):
        print(f"  {n:12s} {h:8.4f}  {s:8.4f}")


def _serializable(obj):
    """Make nested dicts with numpy arrays JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()
                if k != "trace"}
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return np.asarray(obj).tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


if __name__ == "__main__":
    import json as _json

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--n-scenes", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--no-ablate-stage2", action="store_true")
    ap.add_argument("--constrained", action="store_true",
                    help="Fold per-phase collision as a hard AL constraint into "
                         "the forward solve + constrained implicit adjoint.")
    ap.add_argument("--out", default=None, help="Save results to JSON")
    a = ap.parse_args()
    out = run(seed=a.seed, n_iters=a.n_iters, n_scenes=a.n_scenes,
              n_steps=a.n_steps, n_starts=a.n_starts,
              ablate_stage2=not a.no_ablate_stage2, constrained=a.constrained)
    report(out)
    if a.out:
        import os as _os
        _os.makedirs(_os.path.dirname(a.out) or ".", exist_ok=True)
        out_s = _serializable(out)
        out_s["wall_s"] = out["wall_s"]
        out_s["fit_s"] = out["fit_s"]
        with open(a.out, "w") as f:
            _json.dump(out_s, f, indent=2)
        print(f"\nwrote {a.out}")
