"""Physics validation of a SPaSM rollout on one teleop pick-and-place episode.

The question this answers is upstream of the bilevel fit: BEFORE trusting IOSP to
tune the weights, can a hand-picked SPaSM construction (theta_ik standoffs + the
per-segment / refine cost weights) actually roll out a trajectory that PICKS the
cube and DROPS it in the bucket?  `iosp.fit.multistart.build_from_demos` gives the
exact three-stage forward map (IK -> per-segment trajopt -> global refine); this
script drives its joint-space output through the SAME MuJoCo scene
`iosp.viz.e10_teleop_viser` plays the recorded demo in, and checks task success by
CONTACT-DRIVEN physics -- the cube is grasped by the fingers closing on it and
lands in the bucket or it does not.  No kinematic "attach to the hand" hack.

The SPaSM rollout emits only the 7 arm joints.  The gripper is scheduled off the
task skeleton (`pickplace.SKELETON_PICK`/`SKELETON_PLACE`): open on approach,
close once the pick waypoint is reached, hold closed through transport, open at
the release waypoint.

"Manually tune a SPaSM construction" == choose `--u` (the outer parameter in
u-coordinates; u=0 is the measured standoff prior + flat softmax weights, i.e.
the honest default construction) or override the four standoffs directly.

Usage:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.viz.e10_spasm_sim --episode-index 0
"""
import argparse
import json

import numpy as np


def _install_fast_forward_solver():
    """Swap IOSP's differentiable forward solver for the STOCK trajopt config.

    IOSP's `make_composed_forward_solver` deliberately runs a fixed-length scan
    (`early_stop=False`) with `soft_line_search=True`+`soft_curvature_gate=True`.
    Those three exist ONLY to make `q*(theta)` smooth for the bilevel adjoint,
    and the two soft surrogates are what blow up the cold compile (the argmax
    over trial alphas and the L-BFGS curvature gate become dense temperature-
    weighted combinations over every trial point).  A forward-only feasibility
    rollout differentiates nothing, so the stock hard config -- early-stopping
    `while_loop`, hard line search, hard curvature gate -- is both far faster to
    compile AND converges at least as well (it runs to `grad_tol` instead of a
    fixed schedule).  Not bit-identical to the fitted map, but the difference is
    line-search tie-breaking, invisible to task success.

    Monkeypatched (not threaded through `build_from_demos`) because both the
    segment and the refine solver are built by the same factory inside it; one
    swap covers both and leaves every other closure identical.
    """
    from iosp.model import pickplace as pp

    def fast_forward(n_iters=60, **_ignored):
        return pp.make_stock_forward_solver(n_iters=n_iters)

    pp.make_composed_forward_solver = fast_forward


def spasm_joint_rollout(episode_index, u=None, branch=0, n_iters=60, faithful=False):
    """(N_FULL, 7) SPaSM joint path for one episode, at outer parameter `u`.

    Builds the three-stage forward map over the first `episode_index + 1`
    episodes (so the target episode's own scene drives its own rollout) and
    reads out that episode's row.  `u=None` -> u=0 -> the measured standoff
    prior with flat cost weights.

    `faithful=False` (default) uses the fast-compiling stock solver; set it True
    to roll out the exact differentiable map IOSP inverts (slow cold compile).
    """
    if not faithful:
        _install_fast_forward_solver()
    import jax
    import jax.numpy as jnp
    from iosp.fit import multistart as ms
    from iosp.model import pickplace as pp

    built = ms.build_from_demos(n_fit=episode_index + 1,
                                max_episodes=episode_index + 1,
                                n_iters=n_iters, n_branches=max(branch + 1, 4))
    K = built["K"]
    u_vec = jnp.zeros(K, jnp.float32) if u is None else jnp.asarray(u, jnp.float32)
    refs = built["refs"]
    fn = jax.jit(lambda U, r: built["batched_paths"](U, r, "joint"))
    path = fn(u_vec[None], refs[branch][None])       # (1, M, N_FULL, 7)
    q = np.asarray(path[0, episode_index])           # (N_FULL, 7)

    # Report the construction actually rolled out, in interpretable units.
    Z = np.asarray(built["P"]) + np.asarray(u_vec) * np.asarray(built["S"])
    theta_ik = Z[: pp.K_IK]
    seg_end = pp.K_IK + pp.K_TRAJOPT
    w_seg = np.asarray(jax.nn.softmax(jnp.asarray(Z[pp.K_IK:seg_end])))
    w_full = np.asarray(jax.nn.softmax(jnp.asarray(Z[seg_end:])))
    print("\nSPaSM construction rolled out:")
    print(f"  episode           {built['episodes'][episode_index]}")
    print(f"  theta_ik (m)      grasp {theta_ik[0]:.4f}  place {theta_ik[1]:.4f}  "
          f"radial {theta_ik[2]:+.4f}  tangential {theta_ik[3]:+.4f}")
    for n, v in zip(pp.THETA_TRAJOPT_NAMES, w_seg):
        print(f"  seg   {n:20s} {v:.3f}")
    for n, v in zip(pp.THETA_FULL_NAMES, w_full):
        print(f"  refine {n:19s} {v:.3f}")
    return q, built["episodes"][episode_index]


# -- physics execution -------------------------------------------------------

GRIPPER_FULL_OPEN_M = 0.08


def _set_arm(data, model, q):
    import sandbox as sb
    data.ctrl[:7] = sb.clamp_to_limits(model, q)


def _set_gripper(data, open_frac):
    data.ctrl[7] = 255.0 * float(np.clip(open_frac, 0.0, 1.0))


def _mj_steps(model, data, n):
    import mujoco
    for _ in range(n):
        mujoco.mj_step(model, data)


def _cube_xyz(ctx):
    a = ctx["cube_adr"]
    return np.asarray(ctx["data"].qpos[a:a + 3]).copy()


def _in_bucket(scene, cube_xyz, margin=0.0):
    """(is_in, horizontal_dist, height_above_floor)."""
    cx, cy = scene.bucket_center_xy
    dxy = float(np.hypot(cube_xyz[0] - cx, cube_xyz[1] - cy))
    floor_top = scene.table_top_z + scene.bucket_floor_thickness
    rim_top = floor_top + scene.bucket_wall_height
    z = float(cube_xyz[2])
    inside = (dxy <= scene.bucket_inner_radius + margin
              and floor_top - scene.cube_half_extent <= z <= rim_top + scene.cube_half_extent)
    return inside, dxy, z - floor_top


def physics_success(episode_index, q_path, *, episode_path=None, verbose=False,
                    **kw):
    """Drive `q_path` (N_FULL, 7) through one episode's physics scene and report
    whether the cube ends in the bucket.  The reusable entry point for scoring a
    reconstruction's ROLLOUT SUCCESS (as opposed to its EE RMSE).

    PASS `episode_path` -- the episode directory the caller actually built --
    whenever the batch is not exactly `DEFAULT_DEMO_DIR`'s episodes in order.
    `episode_index` alone resolves against that one directory, which is wrong
    in two ways once the fit and held-out sets come from different sessions:
    an index past its length raises IndexError, and an index inside it
    SILENTLY scores the path in some other episode's scene -- a different cube
    and bucket -- which reads as a catastrophic rollout failure (dxy of
    200-460 mm) rather than as the lookup bug it is.  `episode_index` is kept
    only as the fallback for the single-session callers in this module's CLI.
    """
    import pathlib as _pl
    from iosp.viz import e10_teleop_viser as tv
    if episode_path is not None:
        ep = _pl.Path(episode_path)
        ctx = tv.build_ctx(str(ep.parent), ep.name)
    else:
        episodes = tv.find_episodes(tv.DEFAULT_DEMO_DIR)
        ctx = tv.build_ctx(str(tv.DEFAULT_DEMO_DIR), episodes[episode_index])
    landed = execute(ctx, np.asarray(q_path), verbose=verbose, **kw)
    cube = _cube_xyz(ctx)
    inside, dxy, dz = _in_bucket(ctx["scene"], cube)
    return dict(success=bool(inside), landed=landed, dxy=float(dxy),
                dz=float(dz), cube=cube.tolist())


def execute(ctx, q_path, *, settle_steps=60, grasp_dwell=400, release_dwell=200,
            interp=6, verbose=True):
    """Drive the arm through `q_path` in the sim, with a skeleton-keyed gripper.

    Each waypoint is approached over `interp` interpolated sub-targets, each held
    `settle_steps` physics steps so the position actuators track it.  The gripper
    closes when the pick waypoint is reached (extra `grasp_dwell` steps to load
    the fingers) and opens at the release waypoint (`release_dwell` steps to let
    the cube fall).
    """
    from iosp.model import pickplace as pp
    model, data = ctx["model"], ctx["data"]
    scene = ctx["scene"]
    grasp_row = pp.SKELETON_PICK[0]        # 7: cube reached, close here
    release_row = pp.SKELETON_PLACE[0]     # 19: over the bucket, open here

    _set_gripper(data, 1.0)                # start open
    landed_at = None

    def check_landed(tag):
        nonlocal landed_at
        if landed_at is None:
            inside, dxy, dz = _in_bucket(scene, _cube_xyz(ctx))
            if inside:
                landed_at = tag
                if verbose:
                    print(f"  >>> BOX LANDED IN THE BUCKET at {tag} "
                          f"(horiz {dxy*1000:.0f} mm from centre, "
                          f"{dz*1000:+.0f} mm above floor) <<<", flush=True)

    # Gripper state HELD across a waypoint's motion, and toggled only AFTER the
    # arm has arrived: closing on approach to the pick pose would shut the hand
    # in mid-air before it reaches the cube; opening on approach to the release
    # pose would drop the cube before it is over the bucket.  So the hand stays
    # open through the whole descent to the pick row, then closes; stays closed
    # through the whole transport to the release row, then opens.
    prev = q_path[0]
    for i, q in enumerate(q_path):
        for a in np.linspace(0.0, 1.0, interp + 1)[1:]:
            _set_arm(data, model, (1 - a) * prev + a * q)
            _mj_steps(model, data, settle_steps)
        prev = q

        if i == grasp_row:
            _set_gripper(data, 0.0)                   # close ON the cube
            _mj_steps(model, data, grasp_dwell)      # let the fingers load
            if verbose:
                print(f"  row {i} (pick): cube {_cube_xyz(ctx).round(3).tolist()} "
                      f"after closing", flush=True)
        elif i == release_row:
            if verbose:
                print(f"  row {i} (place): cube {_cube_xyz(ctx).round(3).tolist()} "
                      f"before release (lifted {_cube_xyz(ctx)[2]-ctx['cube_pos0'][2]:+.3f} m)",
                      flush=True)
            _set_gripper(data, 1.0)                   # open OVER the bucket
            _mj_steps(model, data, release_dwell)    # let the cube fall
            check_landed(f"release (row {i})")
        check_landed(f"row {i}")

    _mj_steps(model, data, 400)                      # final settle
    check_landed("final settle")
    return landed_at


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--branch", type=int, default=0,
                    help="IK branch reference (0 = q_start)")
    ap.add_argument("--u", type=str, default=None,
                    help="outer parameter as a JSON list of length K; "
                         "default u=0 (measured standoff prior + flat weights)")
    ap.add_argument("--standoffs", type=str, default=None,
                    help="override theta_ik directly, JSON [grasp,place,radial,tangential] in m")
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--faithful", action="store_true",
                    help="roll out the exact differentiable map IOSP fits "
                         "(soft line search + curvature gate; slow cold compile)")
    ap.add_argument("--settle-steps", type=int, default=60)
    ap.add_argument("--interp", type=int, default=6)
    ap.add_argument("--grasp-dwell", type=int, default=400)
    ap.add_argument("--release-dwell", type=int, default=200)
    args = ap.parse_args()

    u = json.loads(args.u) if args.u else None

    q_path, episode = spasm_joint_rollout(
        args.episode_index, u=u, branch=args.branch, n_iters=args.n_iters,
        faithful=args.faithful)

    # `--standoffs` overrides theta_ik post-hoc for quick reach tuning without
    # re-deriving u.  It re-runs the rollout with the override folded into u.
    if args.standoffs is not None:
        import jax.numpy as jnp
        from iosp.fit import multistart as ms
        from iosp.model import pickplace as pp
        built = ms.build_from_demos(n_fit=args.episode_index + 1,
                                    max_episodes=args.episode_index + 1,
                                    n_iters=args.n_iters,
                                    n_branches=max(args.branch + 1, 4))
        S, P = np.asarray(built["S"]), np.asarray(built["P"])
        u_vec = np.zeros(built["K"], np.float32) if u is None else np.asarray(u, np.float32)
        so = np.asarray(json.loads(args.standoffs), np.float32)
        u_vec[: pp.K_IK] = (so - P[: pp.K_IK]) / S[: pp.K_IK]
        import jax
        fn = jax.jit(lambda U, r: built["batched_paths"](U, r, "joint"))
        q_path = np.asarray(fn(jnp.asarray(u_vec)[None],
                               built["refs"][args.branch][None])[0, args.episode_index])
        print(f"  theta_ik overridden to {so.tolist()} m")

    # Build the physics scene exactly as the teleop playback does.
    from iosp.viz import e10_teleop_viser as tv
    episodes = tv.find_episodes(tv.DEFAULT_DEMO_DIR)
    ctx = tv.build_ctx(str(tv.DEFAULT_DEMO_DIR), episodes[args.episode_index])
    scene = ctx["scene"]
    print(f"\nbucket centre {scene.bucket_center_xy}, inner radius "
          f"{scene.bucket_inner_radius:.3f} m; cube spawn {scene.cube_spawn_pos()}")
    print(f"executing {len(q_path)} SPaSM waypoints in physics...", flush=True)

    landed = execute(ctx, q_path, settle_steps=args.settle_steps,
                     grasp_dwell=args.grasp_dwell, release_dwell=args.release_dwell,
                     interp=args.interp)

    cube = _cube_xyz(ctx)
    inside, dxy, dz = _in_bucket(scene, cube)
    print(f"\nfinal cube {cube.round(4).tolist()}  "
          f"(horiz {dxy*1000:.0f} mm from bucket centre, {dz*1000:+.0f} mm above floor)")
    print("RESULT:", "SUCCESS -- box in bucket" if inside
          else "FAILURE -- box not in bucket",
          f"(landed at: {landed})" if landed else "")


if __name__ == "__main__":
    main()
