"""Run SPaSM's OWN tetris trajectory optimizer, then test it under dynamics.

`iosp.checks.dynamic_report` scores IOSP's composed planner.  This scores
SPaSM's, on the same footing: it calls `spasm.tetris_traj.opt` -- SPaSM's
trajectory optimizer, with SPaSM's `TrajOptParams`, against SPaSM's own
`Simulation` -- and then rolls the result out through the actuated MuJoCo
model and re-scores the poses the ARM ACTUALLY ACHIEVED with SPaSM's packing
cost.  Nothing about the plan is ours; only the feasibility test is.

SPaSM's plan is `2 * num_blocks - 1` segments of `T + 2` rows:

    even index 2k   pick-and-place of block k -- the block is attached for the
                    WHOLE segment (`tetris_traj.__main__` sets
                    `block_poses_matrix[k] = ee` at every step), so the grasp is
                    the segment's FIRST row and the release its LAST;
    odd index 2k+1  the empty-handed return from block k's goal to block k+1's
                    start.

So the concatenated path has `num_blocks` separate grasp/release events rather
than the single one `iosp.viz.mj_rollout` was written for.

    python -m iosp.checks.spasm_trajopt --num-blocks 3
"""
from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import sys
import types

import numpy as np

# The ONE place iosp reaches into the SPaSM checkout.  Everything else --
# geometry, costs, skeletons, success tests -- lives in `iosp.model.spasm_costs`
# and `iosp.model.spasm_tasks`, so only this module needs SPaSM installed.
SPASM_ROOT = pathlib.Path(os.environ.get(
    "SPASM_ROOT", pathlib.Path(__file__).resolve().parents[2].parent / "spasm"))


class _StubVisualizer:
    """SPaSM's `Simulation.__init__` opens a meshcat zmq client; we don't render."""

    def __init__(self, *a, **k):
        pass

    def delete(self):
        pass

    def __getitem__(self, k):
        return self

    def __setitem__(self, k, v):
        pass

    def set_object(self, *a, **k):
        pass

    def set_transform(self, *a, **k):
        pass

    def set_property(self, *a, **k):
        pass


def _install_meshcat_stub():
    """Make `import meshcat` succeed without it, and never open a socket."""
    try:
        import meshcat  # noqa: F401
    except ModuleNotFoundError:
        meshcat = types.ModuleType("meshcat")
        geometry = types.ModuleType("meshcat.geometry")
        transformations = types.ModuleType("meshcat.transformations")

        class _Any:
            def __init__(self, *a, **k):
                pass

        for _n in ("Box", "Sphere", "Cylinder", "MeshPhongMaterial",
                   "MeshLambertMaterial", "MeshBasicMaterial", "Line",
                   "LineSegments", "PointsGeometry", "LineBasicMaterial",
                   "PointsMaterial", "Points"):
            setattr(geometry, _n, _Any)
        transformations.translation_matrix = (
            lambda t: np.block([[np.eye(3), np.asarray(t, float).reshape(3, 1)],
                                [np.zeros((1, 3)), np.ones((1, 1))]]))
        transformations.quaternion_matrix = lambda q: np.eye(4)
        meshcat.geometry, meshcat.transformations = geometry, transformations
        sys.modules["meshcat"] = meshcat
        sys.modules["meshcat.geometry"] = geometry
        sys.modules["meshcat.transformations"] = transformations
    import meshcat
    meshcat.Visualizer = _StubVisualizer


def _import_spasm():
    if not (SPASM_ROOT / "spasm").is_dir():
        raise SystemExit(
            f"SPaSM checkout not found at {SPASM_ROOT}; set SPASM_ROOT. Only "
            "this module needs it -- the tasks themselves are in "
            "`iosp.model.spasm_costs`.")
    root = str(SPASM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _install_meshcat_stub()
    return root


@contextlib.contextmanager
def _spasm_cwd():
    """SPaSM resolves `kinematics/urdf/...` relative to its own root."""
    prev = os.getcwd()
    os.chdir(_import_spasm())
    try:
        yield
    finally:
        os.chdir(prev)


def _spasm_tetris_sim(num_blocks=3, extra_cells=None):
    """SPaSM's own `Simulation`, with the goal widened to match ours."""
    import jax.numpy as jnp
    from iosp.model import spasm_costs as SC
    from iosp.model import spasm_tasks as ST
    if extra_cells is None:
        extra_cells = ST.EXTRA_GOAL_CELLS
    with _spasm_cwd():
        from spasm.tetris_env import Simulation, create_walls
        sim = Simulation(num_blocks=num_blocks)
        gd = SC.goal_dims(num_blocks, extra_cells)
        sim.goal_dims = jnp.asarray(gd, jnp.float32)
        sim.goal_walls = create_walls(sim.goal_position, sim.goal_dims,
                                      SC.WALL_HEIGHT, SC.WALL_THICKNESS)
        return sim


def solve_packing(num_blocks=3, extra_cells=None, seed=None, tries=8):
    """Run SPaSM's own packing solver -> (num_blocks, 4) skeleton.

    This is what produced `resources/tetris_packing_*.npy`; the docstring in
    `spasm_tasks.tetris_skeleton` has always pointed at a `--solve-packing`
    flag, which did not exist until now.

    The solve runs against OUR widened goal (`_spasm_tetris_sim`), because
    SPaSM's cell-exact sizing has no valid tiling -- see `EXTRA_GOAL_CELLS`.
    `solve.solve` loops until it beats `cost_thresh`, so a threshold that is out
    of reach for a given goal width would hang; each attempt is therefore run
    with its own key and the best result across `tries` is kept.
    """
    import time as _time
    import jax
    import numpy as _np
    from iosp.model import spasm_costs as SC
    from iosp.model import spasm_tasks as ST

    if extra_cells is None:
        extra_cells = ST.EXTRA_GOAL_CELLS
    sim = _spasm_tetris_sim(num_blocks, extra_cells)

    with _spasm_cwd():
        from spasm.solve import SpasmParams, solve

        params = SpasmParams()
        # `solve.py.__main__`'s per-count tuning, which is not importable.
        # num_blocks=1 is deliberately absent: SPaSM's own sampler dies on it
        # (a float32 indexer inside `sample_particles`), and its `__main__`
        # rejects it too.
        if num_blocks == 3:
            params.sampling_batch, params.opt_batch = 512, 64
            params.opt_steps = 25
        elif num_blocks == 5:
            params.sampling_batch, params.opt_batch = 4096, 256
            params.opt_steps, params.cost_thresh = 25, 0.42
        elif num_blocks == 8:
            params.sampling_batch, params.opt_batch = 2048 * 128, 256
            params.opt_steps, params.cost_thresh = 50, 0.66
        else:
            raise SystemExit(f"no SPaSM tuning for num_blocks={num_blocks} "
                             "(it supports 3, 5, 8)")

        best, best_cost = None, float("inf")
        for t in range(tries):
            key = jax.random.key(int(_time.time()) + t if seed is None
                                 else seed + t)
            poses = _np.asarray(solve(params, sim, key), float)
            # Score with OUR ported cost, not SPaSM's, so the saved skeleton is
            # judged by the same function every downstream check uses.
            c = float(SC.tetris_cost(poses, num_blocks, extra_cells))
            print(f"  [solve_packing] try {t}: cost {c:.4f}"
                  f"{'  <- best' if c < best_cost else ''}", flush=True)
            if c < best_cost:
                best, best_cost = poses, c
        return best.astype(_np.float32), best_cost


def _spasm_tower_sim(num_blocks=10, num_obs=10):
    with _spasm_cwd():
        from spasm.tower_env import TowerSimulation
        return TowerSimulation(num_blocks=num_blocks, num_obs=num_obs)


def _tower_opt_pinned(params, sim, initial_state, final_state, TT):
    """`tower_traj.opt` with the placement row pinned, as tetris already does.

    The two SPaSM trajopts differ by one slice:

        tetris_traj:  q_trajs.at[:, 1:-1, :].add(-lr * grad[:, :-1, :])
        tower_traj:   q_trajs.at[:, 1:,   :].add(grad * -lr)

    The LAST row of a pick-and-place segment is the block's placement pose --
    the IK solution that puts the gripper level, at the right height, over the
    stack.  Tetris holds it fixed and lets the optimizer shape only the interior;
    tower lets the optimizer move it too, and it drifts: the released wrist ends
    up tilted by 5-40 degrees, which `matrix_to_xyzyaw` then discards (it keeps
    only x, y, z and yaw), so no kinematic check can see it.  Under dynamics the
    block is dropped tilted and the tower falls over.
    """
    import jax
    import jax.numpy as jnp

    T = 10
    num_trajs = 2 * sim.num_blocks - 1
    q_trajs = TT.q_traj_init(initial_state, final_state, T)

    def opt_step(i, q_trajs):
        lr = (1.0 - i / params.trajopt_steps) * params.trajopt_lr
        grad = jax.grad(TT.cost, argnums=3)(
            params, sim, q_trajs[:, 0, :], q_trajs[:, 1:, :], i)
        q_trajs = q_trajs.at[:, 1:-1, :].add(-lr * grad[:, :-1, :])
        return_starts = q_trajs[::2, -1, :][:-1]
        q_trajs = q_trajs.at[1::2, 0, :].set(return_starts)
        return_ends = q_trajs[::2, 0, :][1:]
        q_trajs = q_trajs.at[1::2, -1, :].set(return_ends)
        return q_trajs

    return jax.lax.fori_loop(0, params.trajopt_steps, opt_step, q_trajs)


def solve_tower_plan(num_blocks=10, skeleton=None, steps=None,
                     orientation_weight=None, pin_place_row=False):
    """-> (segments, sim, skeleton) via SPaSM's `tower_traj.opt`.

    `pin_place_row=True` swaps in `_tower_opt_pinned`, which is SPaSM's own
    optimizer and cost with the placement row held fixed the way tetris holds
    it.
    """
    import jax.numpy as jnp
    from iosp.model import spasm_tasks as ST

    sim = _spasm_tower_sim(num_blocks)
    sim.z_error_mul = 5.0            # what `tower_traj.__main__` sets
    if skeleton is None:
        skeleton = ST.tower_skeleton(num_blocks)
    init = ST.tower_init_state(num_blocks)

    with _spasm_cwd():
        from spasm.tower_traj import TrajOptParams, opt
        params = TrajOptParams()
        if steps is not None:
            params.trajopt_steps = int(steps)
        if orientation_weight is not None:
            params.orientation_weight = float(orientation_weight)
        from spasm import tower_traj as TT
        fn = ((lambda p, s_, a, b: _tower_opt_pinned(p, s_, a, b, TT))
              if pin_place_row else opt)
        segs = np.asarray(fn(params, sim, jnp.asarray(init, jnp.float32),
                             jnp.asarray(skeleton, jnp.float32)), np.float32)
    return segs, sim, skeleton


def solve_plan(num_blocks=3, extra_cells=None, skeleton=None, steps=None):
    """-> (segments, sim, skeleton) with segments (2n-1, T+2, 7), via SPaSM."""
    import jax.numpy as jnp
    from iosp.model import spasm_tasks as ST

    sim = _spasm_tetris_sim(num_blocks, extra_cells)
    if skeleton is None:
        skeleton = ST.tetris_skeleton(num_blocks, extra_cells)

    with _spasm_cwd():
        from spasm import tetris_traj as TT
        from spasm.tetris_traj import TrajOptParams, opt

        # `tetris_traj.cost.arm_collision_cost_fn.single_q_cost` references a
        # bare name `initial_state` that is defined nowhere -- the enclosing
        # function's parameter is `initial_state_q` (joint configs, not block
        # poses), so the lookup falls through to module globals and raises
        # NameError.  The sibling `held_block_collision_cost_fn` uses
        # `initial_poses` for the identical "blocks j > block_idx are still
        # where they started" term, and that value is
        # `q_to_block(q_trajs[::2, 0, :])` -- the pick rows, which `opt` never
        # moves -- so it is constant and equal to the blocks' spawn poses.
        # Binding that as the module global resolves the name to exactly the
        # value the code means, without editing the SPaSM checkout.
        TT.initial_state = jnp.asarray(np.stack(
            [np.asarray(p) for p in sim.block_poses_original]), jnp.float32)

        params = TrajOptParams()
        if steps is not None:
            params.trajopt_steps = int(steps)
        initial_state = jnp.asarray(np.stack(
            [np.asarray(p) for p in sim.block_poses_original]), jnp.float32)
        segs = np.asarray(opt(params, sim, initial_state,
                              jnp.asarray(skeleton, jnp.float32)), np.float32)
    return segs, sim, skeleton


def concat_segments(segs):
    """Flatten SPaSM's segments into one path plus its grasp/release events.

    Consecutive segments share a row (a return starts where the previous place
    ended), so the joins are de-duplicated; the events are returned as row
    indices into the concatenated path.
    """
    rows, events = [], []
    for i, seg in enumerate(segs):
        start = sum(len(r) for r in rows)
        s = seg if i == 0 else seg[1:]      # drop the shared row
        if i > 0:
            start -= 1
        rows.append(s)
        if i % 2 == 0:                      # a carry segment
            events.append((start, start + len(seg) - 1, i // 2))
    return np.concatenate(rows, axis=0), events


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", choices=("tetris", "tower"), default="tetris")
    ap.add_argument("--num-blocks", type=int, default=None)
    ap.add_argument("--extra-cells", type=int, default=None)
    ap.add_argument("--pin-place-row", action="store_true",
                    help="hold each segment's placement row fixed, as "
                         "tetris_traj does but tower_traj does not")
    ap.add_argument("--orientation-weight", type=float, default=None,
                    help="override SPaSM's tower orientation_weight (0.60)")
    ap.add_argument("--steps", type=int, default=None,
                    help="override SPaSM's trajopt_steps")
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--safety", type=float, default=None,
                    help="fraction of the joint velocity limits to command "
                         "(default 0.5); lower = slower execution")
    ap.add_argument("--dwell", type=float, default=0.6)
    ap.add_argument("--save", default=None, help="write the plan to an .npz")
    ap.add_argument("--solve-packing", action="store_true",
                    help="re-solve the PACKING SKELETON for --num-blocks and "
                         "write it to resources/, instead of planning a "
                         "trajectory")
    ap.add_argument("--packing-tries", type=int, default=8,
                    help="restarts to keep the best packing from")
    ap.add_argument("--separate-packing", action="store_true",
                    help="refine the SAVED skeleton for --num-blocks until no "
                         "two blocks overlap, and rewrite it; use when "
                         "--solve-packing cannot reach positive clearance")
    ap.add_argument("--separate-target-mm", type=float, default=8.0,
                    help="clearance the separation pass aims for")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    from iosp.checks import spasm_rollout as SR

    n = args.num_blocks or (3 if args.domain == "tetris" else 10)

    if args.separate_packing:
        from iosp.model import spasm_tasks as ST
        extra = ST.EXTRA_GOAL_CELLS if args.extra_cells is None else args.extra_cells
        old = ST.tetris_skeleton(n, extra)
        rep0 = ST.packing_report(old, n, extra)
        poses, rep = ST.separate_packing(
            old, n, extra, target_mm=args.separate_target_mm, verbose=True)
        out = ST._packing_path(n, extra)
        np.save(out, poses)
        print(f"wrote {out}: clearance "
              f"{rep0['min_block_clearance_mm']:.2f} -> "
              f"{rep['min_block_clearance_mm']:.2f} mm, wall margin "
              f"{rep0['min_wall_margin_mm']:.2f} -> "
              f"{rep['min_wall_margin_mm']:.2f} mm")
        return poses

    if args.solve_packing:
        from iosp.model import spasm_tasks as ST
        extra = ST.EXTRA_GOAL_CELLS if args.extra_cells is None else args.extra_cells
        poses, cost = solve_packing(n, extra, seed=args.seed,
                                    tries=args.packing_tries)
        out = ST._packing_path(n, extra)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, poses)
        print(f"wrote {out}  cost {cost:.4f} "
              f"(SPaSM threshold {ST.TETRIS_COST_THRESH.get(n)})")
        return poses
    if args.domain == "tower":
        segs, sim, skeleton = solve_tower_plan(
            n, steps=args.steps, orientation_weight=args.orientation_weight,
            pin_place_row=args.pin_place_row)
    else:
        segs, sim, skeleton = solve_plan(n, args.extra_cells, steps=args.steps)
    q, events = concat_segments(segs)
    print(f"SPaSM {args.domain} trajopt: {segs.shape} -> path {q.shape}, "
          f"{len(events)} pick-place events at rows "
          f"{[(g, r) for g, r, _ in events]}")

    if args.domain == "tower":
        return SR.rollout_and_score_tower(q, events, n, skeleton,
                                          settle=args.settle, dwell=args.dwell,
                                          save=args.save, safety=args.safety)
    return SR.rollout_and_score(q, events, n, args.extra_cells, skeleton,
                                settle=args.settle, dwell=args.dwell,
                                save=args.save)


if __name__ == "__main__":
    main()
