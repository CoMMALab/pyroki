"""Path A on HUMAN demonstrations: the same bilevel forward map as
`iosp.fit.parametric`, fitted against teleoperated pick-and-place instead of
against a rollout of the model itself.

What changes, and why each change is forced
-------------------------------------------
`build_parametric` answers "can the fit recover the weights that generated this
demonstration", and it can only ask that because the demonstration IS a rollout
of the forward model at `THETA_*_STAR`.  There is no such vector here.  A human
at a GELLO leader is not running this planner, so:

  * `theta_star`/`u_star` are None, and every parameter-space metric
    (`param_err`, `captured_frac`, `z_err_*`) is undefined and NOT reported.
    `iosp.fit.procedure.run_procedure` already branches on `theta_star is None`.
    What remains is the behavioural criterion, which is the one that was worth
    reporting anyway: does the fitted cost REPRODUCE held-out demonstrations.
  * The loss no longer has a zero.  Model misspecification -- everything the
    human did that this four-phase planner cannot express -- is a floor under
    `fit_rmse` that no amount of optimization removes, so the number that means
    something is the DROP from `fit_rmse_init`, and the gap between `fit_rmse`
    and `gen_rmse`.
  * Fit and held-out are different EPISODES, not scene A and a jittered scene
    B.  Each episode carries its own randomised cube and bucket, so the demo set
    already spans the "sufficiently different environments" that Cao, Cohen &
    Szpruch make a precondition of identifiability (see `iosp.model.scenes`).

The scenes are the episodes' own
--------------------------------
The recorded scene is used verbatim: `pick_pos` is where THAT episode's cube
actually was, `place_pos` the drop point of THAT episode's bucket, `q_start` the
demo's own first waypoint.  Nothing is re-nominalised onto `iosp.config`'s
canonical task, which is the whole point -- a forward rollout on a scene the
demonstration did not happen in is not a reconstruction of anything.

Two consequences of the recorded scenes to read the results with:

  * `obs_center`/`obs_radius` are CONSTANT across every episode -- the recording
    scene has no obstacle and emits a fixed placeholder held clear of the
    workspace (see `sim_teleop/pickplace/scene.py::iosp_scene_fields`).  The
    `clearance` weight is therefore unidentifiable BY CONSTRUCTION here, and is
    expected to land in the Gram's null space.  If it lands in `U_r` instead,
    something is wrong with the spectrum, not with the demonstrator.
  * The robot is an FR3, not a Panda -- see `iosp.model.fr3`.

The standoff prior
------------------
`u = 0` has to be a sensible planner, because it is both the initialization AND
the value stage 4's refit PINS the null space at -- a `theta_ik` the demos leave
unidentified stays wherever `u = 0` puts it.  `iosp.fit.params`' implicit prior
of `z = 0` is not sensible here: the EE frame is `fr3_hand`, the flange, which
sits `TCP_OFFSET_M` = 0.1034 m behind the fingertips, so a zero standoff asks IK
to put the flange inside the cube.

So the standoff prior is MEASURED, on the fit episodes only, as the median
height of the hand above the target at the skeleton rows the exporter pins the
grasp and the release to (`pickplace.SKELETON_PICK` / `SKELETON_PLACE`).  On the
first recorded session that is 0.105 m at the grasp -- the TCP offset, to 2 mm,
across all ten episodes, which is an independent confirmation that the gripper
channel put the grasp on the right row -- and 0.24 m at the release, because a
human drops the cube into the bucket from above rather than lowering it to the
floor.  Held-out episodes are excluded from the median for the same reason they
are excluded from the feature-scale calibration.

The trajopt logits keep a flat prior (uniform softmax), which is genuinely
uninformative.  `z = z_prior + Z_SCALE * u` leaves `u` dimensionless exactly as
in `iosp.fit.params`.

What the standoff CANNOT absorb: `theta_ik` offsets the IK target along +z only,
and at the release the hand is also ~0.07 m LATERALLY off the bucket centre (the
bucket's inner radius is 0.065 m, so the operator lets go over the rim rather
than the axis).  That lateral term is misspecification and lands in the loss
floor; it is one waypoint of the 23, so it bounds `fit RMSE` from below at
roughly 0.07/sqrt(23) ~ 0.015 m in EE terms before anything else contributes.
"""

import dataclasses
import json
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from ioc import identifiability as ident
from iosp.fit.params import z_scale
from iosp.fit.parametric import _build_inner, screen_stationarity
from iosp.model import fr3, pickplace as pp
from iosp.model.pickplace import split_trajopt as _split_trajopt
from iosp.model.pickplace import split_trajopt_perseg as _split_trajopt_perseg

# `fr3_hand` (the EE frame) to `fr3_hand_tcp`, from the URDF's fixed joint.
TCP_OFFSET_M = 0.1034

DEFAULT_TELEOP_ROOT = pathlib.Path(
    os.environ.get("IOSP_TELEOP_ROOT",
                   pathlib.Path(__file__).resolve().parents[3] / "sim_teleop")
)
DEFAULT_DEMO_DIR = DEFAULT_TELEOP_ROOT / "data" / "demos"

# The in-repo copy of the recorded sessions, so an E10 run does not depend on a
# sibling checkout being present or on `data/demos` still pointing where it did.
#   fit/   23 episodes, session of 2026-09-03 (`sim_teleop/data/train_demos`).
#          FOUR of the recorded 27 are quarantined in `excluded/` -- with the
#          release pinned to the bucket centre their bucket (0.70-0.75 m from
#          the base) is out of reach, the pinned IK target is infeasible, and
#          the rollout runs away to 195-799 rad. See `excluded/EXCLUDED.json`;
#          `find_episodes` never sees them because they are not under `fit/`.
#   test/  10 episodes, session of 2026-09-02 (`sim_teleop/data/test_demos`,
#          which `sim_teleop/data/demos` symlinks to) -- the ORIGINAL ten, which
#          used to be split 8 fit / 2 held and are now held out in full.
# Two SESSIONS, not a shuffle of one: the generalisation question E10 asks is
# whether a cost fitted on one sitting reproduces another, which is strictly
# harder than a within-session split and is why the split is by DIRECTORY
# rather than by index.
LOCAL_DEMO_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "demos"
FIT_DEMO_DIR = LOCAL_DEMO_ROOT / "fit"
TEST_DEMO_DIR = LOCAL_DEMO_ROOT / "test"


def _import_exporter(teleop_root=DEFAULT_TELEOP_ROOT):
    """`sim_teleop.pickplace.iosp_export`, imported from the sibling checkout.

    Imported rather than vendored: it reads `N_FULL`/`PHASE_SPAN` from this
    package, so the episode->waypoint collapse and the forward model's phase
    layout cannot drift apart.  A copy here would be a second thing to keep
    correct, and the first segment-length change would silently break it.
    """
    root = str(pathlib.Path(teleop_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from pickplace import iosp_export
    except ImportError as e:
        raise ImportError(
            f"cannot import the teleop exporter from {root!r}. Point "
            "IOSP_TELEOP_ROOT at the sim_teleop checkout."
        ) from e
    return iosp_export


def _as_dirs(demo_dir):
    """One path or a sequence of them -> a list of paths, order preserved."""
    if isinstance(demo_dir, (str, os.PathLike)):
        return [pathlib.Path(demo_dir)]
    return [pathlib.Path(d) for d in demo_dir]


def find_episodes(demo_dir=DEFAULT_DEMO_DIR):
    """Every episode directory holding both files, sorted by name (= by time).

    `demo_dir` may be a SEQUENCE of directories (or of episode directories
    themselves), in which case each is sorted
    internally and they are CONCATENATED in the order given.  That order is what
    the fit/held split is taken on, so the concatenation must not be re-sorted
    globally -- with a fit directory recorded after the held-out one, a global
    sort would put the held-out episodes first and silently invert the split.
    """
    eps = []
    for d in _as_dirs(demo_dir):
        if (d / "state.jsonl").exists() and (d / "factors.json").exists():
            # `d` is itself an episode.  Accepting these makes an explicit
            # episode LIST a valid `demo_dir`, which is how the demo-count
            # ablation hands over a truncated fit prefix without materialising
            # a directory per n.
            eps.append(d)
            continue
        found = sorted(x for x in d.iterdir()
                       if (x / "state.jsonl").exists() and (x / "factors.json").exists())
        if not found:
            raise FileNotFoundError(
                f"no episodes with state.jsonl + factors.json in {d}")
        eps.extend(found)
    return eps


def load_demos(demo_dir=DEFAULT_DEMO_DIR, teleop_root=DEFAULT_TELEOP_ROOT,
               prob=None, anchor_grasp=False, max_episodes=None,
               return_paths=False):
    """-> (names, (B, N_FULL, dof) waypoints, batched `PickPlaceScene`).

    `return_paths=True` appends the episode DIRECTORIES in batch order.  Names
    alone are not enough to find an episode again once the batch spans two
    sessions: anything that re-resolves a name against a single directory
    (physics rollout, viser playback) silently picks the wrong scene or falls
    off the end.

    `anchor_grasp=True` fills `pick_wxyz`/`grasp_ref` AND `place_wxyz`/
    `place_ref` from each episode's own configuration at the skeleton grasp and
    release rows, which requires `prob` for the FK.
    See `PickPlaceScene.pick_wxyz` for the measured effect and for what it does
    to the claim -- the grasp pose stops being predicted and becomes an input.
    """
    ex = _import_exporter(teleop_root)
    eps = find_episodes(demo_dir)
    if max_episodes is not None:
        eps = eps[: int(max_episodes)]
    paths, fields = ex.build_demo_batch(
        [(d / "state.jsonl", d / "factors.json") for d in eps])
    demo_q = jnp.asarray(paths, dtype=jnp.float32)
    scenes = pp.PickPlaceScene(
        **{k: jnp.asarray(v, dtype=jnp.float32) for k, v in fields.items()})

    if anchor_grasp:
        if prob is None:
            raise ValueError("anchor_grasp=True needs `prob` for the FK")
        # Both events, from the rows the skeleton pins them to.  FK over the
        # whole batch is one batched call, so reading four anchors costs
        # nothing measurable next to a single IK solve.
        q_grasp = demo_q[:, pp.SKELETON_PICK[0]]
        q_place = demo_q[:, pp.SKELETON_PLACE[0]]
        fk = lambda q: prob.base.robot.forward_kinematics(q)[:, prob.ee_index, :4]
        scenes = dataclasses.replace(
            scenes,
            pick_wxyz=fk(q_grasp).astype(jnp.float32),
            grasp_ref=q_grasp.astype(jnp.float32),
            place_wxyz=fk(q_place).astype(jnp.float32),
            place_ref=q_place.astype(jnp.float32))
    if return_paths:
        return [d.name for d in eps], demo_q, scenes, list(eps)
    return [d.name for d in eps], demo_q, scenes


def mixed_split(n_train=None, n_orig_fit=0, fit_dir=FIT_DEMO_DIR,
                orig_dir=TEST_DEMO_DIR):
    """-> (fit episode dirs, held-out episode dirs) for a MIXED split.

    `n_train` episodes from the 2026-09-03 session plus the FIRST `n_orig_fit`
    of the 2026-09-02 session are fitted; the REMAINING 2026-09-02 episodes
    are held out.  `n_orig_fit=0` is the pure cross-session split (fit on one
    sitting, test on the other).

    Why mix at all: the pure split asks the hardest question but confounds two
    things, because the sessions differ in more than identity -- 2026-09-03
    randomised the bucket out to 0.75 m where 2026-09-02 never passed 0.670 m
    (which is what broke the four episodes in `demos/excluded`).  Putting a
    few 2026-09-02 episodes in the fit set means the held-out five are drawn
    from a distribution the fit has actually seen, so a gap is the cost of
    generalising rather than of extrapolating the workspace.

    The 2026-09-02 side is split CHRONOLOGICALLY, first `n_orig_fit` to the
    fit set: a random split of a single sitting leaks late-session technique
    backwards, which is the same reason the within-directory split has always
    been a prefix.
    """
    train = find_episodes(fit_dir)
    orig = find_episodes(orig_dir)
    if n_train is not None:
        if not 0 < int(n_train) <= len(train):
            raise ValueError(f"n_train must be in (0, {len(train)}]; got {n_train}")
        train = train[: int(n_train)]
    if not 0 <= int(n_orig_fit) < len(orig):
        raise ValueError(f"n_orig_fit must be in [0, {len(orig)}); "
                         f"got {n_orig_fit}")
    return train + orig[: int(n_orig_fit)], orig[int(n_orig_fit):]


def z_prior(K, n_ik, standoffs=None):
    """`u = 0` in `z` coordinates: `theta_ik` at `standoffs`, flat logits.

    `standoffs=None` falls back to the TCP offset on the two standoffs and zero
    in-plane offset, which is right for the grasp and wrong for the release --
    prefer `measure_standoffs`, which reads all four off the demonstrations.
    """
    p = np.zeros(K, dtype=np.float32)
    if standoffs is None:
        p[:2] = TCP_OFFSET_M
    else:
        p[:n_ik] = np.asarray(standoffs)
    return jnp.asarray(p)


def measure_standoffs(prob, demo_q, scenes, idx):
    """`theta_ik`'s prior, in metres, from episodes `idx` -- all four entries.

    Each is the median over those episodes of exactly the quantity the matching
    coordinate parameterises, read off the demonstrations instead of guessed:

      grasp.standoff     height of the EE frame above the cube at the pinned
                         grasp row -- comes out at the hand-to-TCP offset, which
                         is an independent check that the gripper channel put
                         the grasp on the right row
      place.standoff     the same above the bucket at the release row
      place.radial       in-plane displacement of the release point along the
                         base->bucket direction (negative = short of centre)
      place.tangential   the same along its left-hand normal

    Median, not mean: one fumbled approach should not move the initialization.
    """
    idx = np.asarray(idx)
    ee = np.asarray(jax.vmap(prob.ee_positions)(demo_q))[idx]
    pick = np.asarray(scenes.pick_pos)[idx]
    place = np.asarray(scenes.place_pos)[idx]
    r_pick, r_place = list(pp.SKELETON_PICK), list(pp.SKELETON_PLACE)

    grasp_z = np.median(ee[:, r_pick, 2] - pick[:, 2:3])
    place_z = np.median(ee[:, r_place, 2] - place[:, 2:3])

    # In-plane, in the scene's own base->bucket frame; see `pp._place_frame`.
    radial, tangential = pp._place_frame(jnp.asarray(place))
    d = ee[:, r_place[0], :2] - place[:, :2]
    rad = np.median((d * np.asarray(radial)[:, :2]).sum(-1))
    tan = np.median((d * np.asarray(tangential)[:, :2]).sum(-1))
    return np.array([grasp_z, place_z, rad, tan], dtype=np.float32)


def release_offset_cap(demo_dir=DEFAULT_DEMO_DIR, idx=None, margin=0.005):
    """Largest in-plane release offset (m) that still drops the cube INSIDE the
    bucket, over episodes `idx` -- read from each episode's own randomisation
    record, not tuned.

    MEASURED, and the reason this exists: `measure_standoffs` returns the
    coordinate-wise MEDIAN of the operators' release offsets, which on this set
    is `|offset| = 0.0719 m`.  Every episode's bucket is smaller than that
    (`bucket_inner_radius` is randomised, ~0.0655 m; the cube is 0.015 m
    half-extent), so the median release point lies OUTSIDE the bucket and
    pinning the event to it fails the task 0/10 (dxy 71 mm, dz +134 mm -- the
    cube lands on the rim) even though it fits the demonstrations best.  The
    median of a correlated distribution need not be a feasible member of it:
    the operators who reached farther released lower, and taking each coordinate's
    median independently combines them into a drop nobody performed.

    So the release offset is PROJECTED into the feasible disc.  `min` over the
    episodes, not mean: a single small bucket makes a larger offset infeasible
    for that scene, and the pinned event is global.
    """
    eps = find_episodes(demo_dir)
    if idx is not None:
        eps = [eps[i] for i in np.asarray(idx)]
    caps = []
    for d in eps:
        with open(d / "factors.json") as fh:
            f = json.load(fh)
        caps.append(float(f["bucket_inner_radius"]) - float(f["cube_half_extent"]))
    return max(0.0, min(caps) - margin)


def project_release_offset(standoffs, cap):
    """`standoffs` with its (radial, tangential) pair shrunk to at most `cap`,
    keeping the demonstrated DIRECTION.  Under the cap it is a no-op."""
    s = np.asarray(standoffs, dtype=np.float32).copy()
    r = float(np.hypot(s[2], s[3]))
    if r > cap and r > 0.0:
        s[2] *= cap / r
        s[3] *= cap / r
    return s


def build_pick_and_place(demo_dir=FIT_DEMO_DIR, teleop_root=DEFAULT_TELEOP_ROOT,
                 held_dir=TEST_DEMO_DIR, n_fit_max=None,
                 n_fit=None, seed=0, n_iters=600, n_restarts=1, space="joint",
                 fast_forward=True, freeze_ik=False, ee_weight=0.0,
                 pin_ik=None, upright_floor=0.0, free_space_only=False,
                 hard_upright=(), duration_weight=1.0, per_segment=False):
    """The `built` dict `iosp.fit.procedure.run_procedure` consumes.

    `space` defaults to "joint", not "ee" as in `build_parametric`: the
    demonstration IS a joint-space path (`q_d`, the commanded configuration),
    and the 7-DOF arm's self-motion manifold is invisible to an EE loss -- a
    fitted rollout can match the demonstrated EE path to the millimetre through
    a completely different elbow.  The EE criterion is still computed and
    reported, from the same rollout, via `ee_*`.

    The first `n_fit` episodes (chronological) are the fit set and the rest are
    held out.  Chronological, not random: the split is then reproducible without
    carrying a seed, and it is the honest one for a human demonstrator whose
    technique drifts over a session -- a random split leaks late-session
    technique into the training set.

    `held_dir` (the default) makes that split CROSS-SESSION instead: `demo_dir`
    supplies the fit episodes, `held_dir` the held-out ones, `n_fit` is then
    determined by the directories and must not be passed.  This is the current
    E10 setup -- 27 fit episodes from the 2026-09-03 session, the original 10
    from 2026-09-02 held out in full.  Pass `held_dir=None` to recover the old
    single-directory prefix split.

    `n_fit_max` keeps only the first n fit episodes (the held-out set is never
    truncated).  This is what the demo-count ablation
    (`iosp.experiments.e10_demo_ablation`) varies, and it TRUNCATES rather than
    down-weighting because the forward map solves every episode in the batch:
    a zero loss weight would still pay for that episode's four segment solves,
    so a weighted mask makes the n=1 fit cost exactly as much as the n=27 one.
    MEASURED on this model at 27 fit + 10 held: one `value_and_grad` is 39 s
    and one loss evaluation 44 s, so the 27-point sweep is ~300 GPU-hours
    masked against ~14x a single fit truncated.  The price of truncating is a
    rebuild (~2 min) and a recompile per n, which is noise against either.
    """
    if space not in ("ee", "joint"):
        raise ValueError(f"space must be 'ee' or 'joint', got {space!r}")

    # `n_iters` DEFAULTS TO 600, not the 60 the old two-feature basis used.
    # MEASURED on the standard basis: worst inner stationarity is 9.1e-2 at 60
    # iterations, 7.7e-3 at 200 and 9.5e-4 at 600, i.e. the 1e-3 tolerance is
    # first met at 600.  That tolerance is not cosmetic -- `ioc.inner`'s implicit
    # adjoint is only valid at a converged solve, so every gradient and every
    # Gram spectrum taken below 600 here is meaningless rather than merely noisy
    # (see `screen_stationarity`).  The basis is simply harder than the old one:
    # RNEA effort residuals are O(1e3-1e4) against O(1e-2) for path, and the
    # duration scalar sits among 42 waypoint variables with very different
    # curvature.  The forward solve costs ~10x accordingly.
    if held_dir is not None:
        # Cross-session split.  `n_fit` is not a free parameter here -- it is
        # however many episodes the fit directory holds -- so a caller passing
        # one is contradicting the directories and gets an error rather than a
        # silently ignored argument.
        if n_fit is not None:
            raise ValueError("n_fit is determined by `held_dir`; pass one or "
                             "the other, not both")
        # Both sides accept a directory, a list of directories, or an explicit
        # list of episode directories -- which is what lets the fit set draw
        # from BOTH sessions (see `mixed_split`).
        fit_eps = find_episodes(demo_dir)
        if n_fit_max is not None:
            if not 0 < int(n_fit_max) <= len(fit_eps):
                raise ValueError(f"n_fit_max must be in (0, {len(fit_eps)}]; "
                                 f"got {n_fit_max}")
            fit_eps = fit_eps[: int(n_fit_max)]
        n_fit = len(fit_eps)
        # Episode directories, not parent directories: `find_episodes` accepts
        # either, and passing the truncated list is what makes the fit set an
        # exact prefix without copying or symlinking anything.
        demo_dir = fit_eps + find_episodes(held_dir)
    elif n_fit_max is not None:
        raise ValueError("n_fit_max needs `held_dir` (it truncates the fit "
                         "directory); use n_fit for a single-directory split")
    names, demo_q, scenes, episode_paths = load_demos(demo_dir, teleop_root,
                                                      return_paths=True)
    B = len(names)
    n_fit = B - max(1, B // 4) if n_fit is None else int(n_fit)
    if not 0 < n_fit < B:
        raise ValueError(f"n_fit must be in (0, {B}); got {n_fit}")
    fit_idx = np.arange(n_fit)
    gen_idx = np.arange(n_fit, B)

    urdf, srdf, mesh_dir, ee_link = fr3.paths()
    prob = pp.PickPlaceProblem.load(urdf, srdf, mesh_dir, ee_link=ee_link)
    # `fast_forward` (default) finds x* with the stock early-stopping solver:
    # the implicit adjoint is analytic and forward-independent (ioc.inner), so
    # this cuts the forward-compile of `loss`/`gf` and every path readout without
    # changing the adjoint's correctness -- only `unrolled`, which differentiates
    # through the solver, needs the soft map and builds its own.
    forward_solver = (pp.make_stock_forward_solver(n_iters=n_iters) if fast_forward
                      else pp.make_composed_forward_solver(n_iters=n_iters))

    K_traj = pp.K_TRAJOPT_PERSEG if per_segment else pp.K_TRAJOPT
    K = pp.K_IK + K_traj
    standoffs = measure_standoffs(prob, demo_q, scenes, fit_idx)
    print(f"  [teleop] standoff prior from the {n_fit} fit episodes: "
          f"grasp {standoffs[0]:.4f} m, place {standoffs[1]:.4f} m "
          f"(hand-to-TCP is {TCP_OFFSET_M:.4f} m)", flush=True)
    S, P = z_scale(K, pp.K_IK), z_prior(K, pp.K_IK, standoffs)
    z_of = lambda u: P + S * u

    # Feature scales are calibrated on the FIT episodes only, for the same
    # reason `build_parametric` calibrates on scene A only: a scale fitted on
    # the held-out scenes lets them re-normalise the very features being tested.
    fit_scenes = jax.tree.map(lambda a: a[fit_idx], scenes)
    # `hard_upright` phases get a HARD (AL) grasp-maintenance constraint instead
    # of relying on the fitted/floored soft `upright` weight -- the gripper axis
    # stays vertical through those phases regardless of the cost weights, so an
    # aggressive fitter that zeroes upright can no longer tip the carried object
    # out.  See `pp.PickPlaceProblem.upright_constraint_fn`.
    constraints_by_phase = ({p: prob.upright_constraint_fn(p) for p in hard_upright}
                            if hard_upright else None)
    if hard_upright:
        print(f"  [teleop] HARD grasp-maintenance (upright AL constraint) on: "
              f"{list(hard_upright)}", flush=True)
    inner, _ = _build_inner(prob, fit_scenes, z_of(jnp.zeros(K))[: pp.K_IK],
                            forward_solver, seed, n_restarts=n_restarts,
                            constraints_by_phase=constraints_by_phase)

    # `freeze_ik` holds theta_ik at the measured standoff prior and fits only
    # theta_trajopt.  theta_ik is measured geometry (median offsets read off the
    # demos by `measure_standoffs`), not a preference; letting the joint-RMSE fit
    # retune it walks the release off the bucket (place.radial ran to -0.15 m,
    # 0/10 rollout success) while init at the prior succeeds 10/10.  Freezing it
    # keeps the events physically right by construction, so the recovered object
    # is purely the free-space cost preference -- which is the IOSP claim.  The
    # first K_IK components of u then have zero effect (dead dims for CMA-ES,
    # zero gradient for the differentiable methods); K is left at 11 so every
    # downstream shape/name is unchanged.
    theta_ik_frozen = P[: pp.K_IK]

    # -- the PRINCIPLED decomposition: task events -> constraints, free-space ->
    # fitted cost.  A pick-and-place demonstration IDENTIFIES its skeleton (grasp
    # the cube, release into the bucket, keep the object seated in transit); only
    # the free-space motion between events is a latent PREFERENCE.  Fitting the
    # skeleton as a cost is fitting known quantities as if unknown, which is what
    # corrupted them (place.radial -> -0.15 m; transport.upright -> 0).  So:
    #
    #   pin_ik="bucket"   release event pinned to the TASK GOAL (bucket CENTRE at
    #                     the measured drop height, radial=tangential=0), grasp at
    #                     the cube -- the symbolic goal, not the demo's own
    #                     marginally-inside release.  This is the release-event
    #                     constraint: q_place = IK(target) is a hard boundary of
    #                     the transport solve, so pinning the target pins the event.
    #   pin_ik="measured" release pinned to the demo's measured offsets (the old
    #                     `freeze_ik`); kept for comparison.
    #   upright_floor>0   grasp-retention constraint: a floor on transport.upright
    #                     so the fit cannot trade away keeping the gripper level
    #                     (upright is the proxy for "don't tip/lose the grasp"
    #                     during the carry).  A soft constraint (fixed floor),
    #                     deliberately not the AL machinery, which has been
    #                     ill-conditioned/inert on this model.
    #   free_space_only   fit the preference on the FREE-SPACE rows only; the
    #                     event rows are determined by the constraints above, so
    #                     scoring them just dilutes the preference signal.
    #
    # ROLLOUT SUCCESS IS NEVER IN THIS LOSS -- it is verification only.
    # `upright` left the cost basis (see `pickplace.STANDARD_FEATURES`): keeping
    # the gripper level during the carry is grasp RETENTION, a task requirement,
    # and it now lives only in `upright_constraint_fn`'s hard constraint. So
    # there is no weight left to floor. `upright_floor` is accepted and ignored,
    # with a warning, rather than removed, so existing command lines and scripts
    # still run instead of dying on an unknown argument.
    if upright_floor > 0.0:
        print(f"  [teleop] NOTE: --upright-floor {upright_floor} ignored; "
              "`upright` is no longer a fitted weight. Use hard_upright=(...) "
              "for grasp retention.", flush=True)
        upright_floor = 0.0
    EVENT_ROWS = sorted(set(pp.SKELETON_PICK) | set(pp.SKELETON_PLACE))
    FREE_ROWS = jnp.asarray([r for r in range(pp.N_FULL) if r not in EVENT_ROWS])
    theta_ik_bucket = jnp.asarray(
        [standoffs[0], standoffs[1], 0.0, 0.0], dtype=jnp.float32)
    if pin_ik == "bucket":
        theta_ik_pinned = theta_ik_bucket
    elif pin_ik == "feasible":
        # The demonstrated release, PROJECTED into the bucket -- the middle
        # ground between "bucket" (task-safe, but discards the measured near-rim
        # release and freezes 0.23 m of place error no weight can touch) and
        # "measured" (fits best, drops the cube on the rim, 0/10).  MEASURED
        # over the offset ray at the fitted weights: |offset| 0.000 -> loss
        # 0.679 / EE 0.155 / 8-8 success; 0.054 -> 0.493 / 0.138 / 8-8;
        # 0.072 (the raw median) -> 0.456 / 0.134 / 0-8.  The success cliff sits
        # exactly at the disc `release_offset_cap` computes, so the projection
        # buys the reconstruction without spending the task.
        cap = release_offset_cap(demo_dir, fit_idx)
        theta_ik_pinned = jnp.asarray(project_release_offset(standoffs, cap),
                                      dtype=jnp.float32)
        print(f"  [teleop] release offset capped at {cap:.4f} m "
              f"(measured |offset| {float(np.hypot(standoffs[2], standoffs[3])):.4f} m) "
              f"-> radial {float(theta_ik_pinned[2]):+.4f}, "
              f"tangential {float(theta_ik_pinned[3]):+.4f}", flush=True)
    elif pin_ik == "measured" or freeze_ik:
        theta_ik_pinned = theta_ik_frozen
    else:
        theta_ik_pinned = None

    _nf = pp.N_FEAT_PER_SEG

    def _weights(z_traj):
        if per_segment:
            return jnp.concatenate([jax.nn.softmax(z_traj[i * _nf:(i + 1) * _nf])
                                    for i in range(len(pp.PHASES))])
        return jax.nn.softmax(z_traj)

    _do_split = _split_trajopt_perseg if per_segment else _split_trajopt

    def _rollout(u):
        z = z_of(u)
        theta_ik = theta_ik_pinned if theta_ik_pinned is not None else z[: pp.K_IK]
        z_traj = z[pp.K_IK:]
        x0, _, _, _ = prob.seeds(scenes, theta_ik)
        _, _, xs, ps = prob.solve(theta_ik, _do_split(_weights(z_traj)),
                                  scenes, inner, x0)
        return xs, ps

    def _durations(xs):
        """(B, 4) fitted seconds per phase, read out of each segment's own
        decision vector.  See `RobotProblem.duration`."""
        return jnp.stack([jax.vmap(prob.seg[p].duration)(xs[p]) for p in pp.PHASES],
                         axis=-1)

    def ee_paths(u):
        xs, ps = _rollout(u)
        return prob.full_ee_paths(scenes, xs, ps)

    def joint_paths(u):
        xs, ps = _rollout(u)
        return prob.full_joint_paths(scenes, xs, ps)

    paths = joint_paths if space == "joint" else ee_paths
    paths_j = jax.jit(paths)
    ee_paths_j = jax.jit(ee_paths)

    demo_dur = getattr(scenes, "phase_durations", None)
    if demo_dur is None:
        print("  [teleop] NOTE: episodes carry no `phase_durations`; the `time` "
              "feature has no data to fit against and will be gauge.", flush=True)
    else:
        _d = np.asarray(demo_dur)
        print(f"  [teleop] demo phase durations, s (mean over {B} episodes): "
              + ", ".join(f"{p} {v:.2f}" for p, v in zip(pp.PHASES, _d.mean(0))),
              flush=True)

    demo = demo_q if space == "joint" else jax.vmap(prob.ee_positions)(demo_q)
    ee_demo = jax.vmap(prob.ee_positions)(demo_q)

    screen_stationarity(prob, fit_scenes, inner, z_of(jnp.zeros(K))[: pp.K_IK],
                        _do_split(_weights(jnp.zeros(K_traj))),
                        "teleop (path A, human demos)")

    def loss_a(u):
        # Base term in the loss `space` (joint by default: preserves the
        # redundant arm's homotopy/elbow branch, which an EE loss is blind to).
        # `ee_weight > 0` adds a Cartesian alignment term from the SAME rollout,
        # so the object's placement is pulled toward the (successful) demo EE
        # path even as the joint term pins the branch -- the hypothesis being
        # that joint keeps the arm on the right route while EE keeps the hand
        # (and thus the cube/bucket) where the task needs it.  Units differ
        # (rad^2 vs m^2 per waypoint), so `ee_weight` is a relative scale, not 1.
        xs, ps = _rollout(u)
        base_paths = (prob.full_joint_paths(scenes, xs, ps) if space == "joint"
                      else prob.full_ee_paths(scenes, xs, ps))
        rows = FREE_ROWS if free_space_only else slice(None)
        loss = jnp.mean(jnp.sum(
            (base_paths[fit_idx][:, rows] - demo[fit_idx][:, rows]) ** 2, axis=-1))
        # DURATION term.  The waypoint export resamples each phase to a fixed row
        # count, so `base_paths` contains no timing at all: two episodes, one
        # twice as fast, give identical matrices.  Without this term the `time`
        # feature is pure gauge -- it can only reach the loss through how the
        # dt-scaled accel/jerk/effort terms rebalance the SHAPE, which is a very
        # weak channel.  Scored as a RELATIVE error so seconds^2 does not have to
        # be commensurate with rad^2, and only where the demo recorded it.
        if duration_weight and demo_dur is not None:
            # `grasp` excluded: its recorded duration is one sample by
            # construction of the skeleton cuts, not a measurement.
            ti = jnp.asarray([pp.PHASES.index(p) for p in pp.TIMED_PHASES])
            T_fit = _durations(xs)[fit_idx][:, ti]
            T_dem = demo_dur[fit_idx][:, ti]
            loss = loss + duration_weight * jnp.mean(((T_fit - T_dem) / T_dem) ** 2)
        if ee_weight and space == "joint":
            ee = prob.ee_positions(base_paths)
            loss = loss + ee_weight * jnp.mean(jnp.sum(
                (ee[fit_idx][:, rows] - ee_demo[fit_idx][:, rows]) ** 2, axis=-1))
        return loss

    def _rmse(P_, D, idx):
        return float(jnp.sqrt(jnp.mean(jnp.sum((P_[idx] - D[idx]) ** 2, axis=-1))))

    def theta_of(u):
        z = np.asarray(z_of(u))
        ik = np.asarray(theta_ik_pinned) if theta_ik_pinned is not None else z[: pp.K_IK]
        return np.concatenate([ik, np.asarray(_weights(jnp.asarray(z[pp.K_IK:])))])

    _traj_names = (list(pp.THETA_TRAJOPT_PERSEG_NAMES) if per_segment
                   else list(pp.THETA_TRAJOPT_NAMES))

    return dict(
        gf=jax.jit(jax.value_and_grad(loss_a)),
        # Value-only loss: FD/CMA-ES need the loss VALUE, never its gradient.
        # Routing them through `gf(u)[0]` made every probe also build the
        # implicit adjoint's dense Hessian (see ioc.inner), which is what drove
        # the FD stage to >108 GB host RAM and OOM.  `loss` gives byte-identical
        # values with none of that curvature work.
        loss=jax.jit(loss_a),
        paths_fn=paths_j, demo_paths=demo, space=space,
        ee_paths_fn=ee_paths_j, ee_demo_paths=ee_demo,
        jac_fn=ident.make_jac_fn(lambda u: paths(u)[fit_idx]),
        rmse_a=lambda u: _rmse(paths_j(u), demo, fit_idx),
        rmse_b=lambda u: _rmse(paths_j(u), demo, gen_idx),
        ee_rmse_a=lambda u: _rmse(ee_paths_j(u), ee_demo, fit_idx),
        ee_rmse_b=lambda u: _rmse(ee_paths_j(u), ee_demo, gen_idx),
        durations_fn=jax.jit(lambda u: _durations(_rollout(u)[0])),
        demo_durations=demo_dur,
        K=K, n_ik=pp.K_IK, theta_of=theta_of, standoff_prior=standoffs,
        # No ground-truth cost exists for a human demonstrator: `run_procedure`
        # skips every parameter-space metric on `theta_star is None`.
        theta_star=None, u_star=None,
        names=list(pp.THETA_IK_NAMES) + _traj_names,
        episodes=names, episode_paths=[str(d) for d in episode_paths],
        n_fit=n_fit, fit_idx=fit_idx, gen_idx=gen_idx,
        scenes=scenes, demo_q=demo_q, prob=prob,
    )


build_teleop = build_pick_and_place
