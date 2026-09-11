"""Actuated MuJoCo rollout of the tetris-packing task via mjviser.

Plans the trajectory with SPaSM's own trajectory optimizer for a dynamically
feasible path, then drives it through a real MuJoCo scene under physics: the
Panda's position servos track the plan under gravity, and the tetromino is held
purely by contact friction (the gripper closes on it and lifts it -- no
kinematic carry).  Where the block ends up is determined by the physics, not
asserted.

Everything is a geom, so `mj_step` resolves contacts: the block can slip, tip,
collide with walls, or be nudged by a later placement.  See `iosp.viz.mj_scene`
for the `--robot` choice (Menagerie Panda vs. the planner's own spherized
collision model).

    # solve with SPaSM and roll out (needs SPaSM checkout + GPU)
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.viz.tetris_viser --num-blocks 3

    # replay a saved forward-pass extract -- no solve, no GPU work
        python -m iosp.viz.tetris_viser --from-npz scratch/feas/tetris_aligned.npz

    # look at the curated environment alone
        python -m iosp.viz.tetris_viser --no-solve --num-blocks 3
"""
from __future__ import annotations

import argparse

import numpy as np

from iosp.viz import mj_rollout as R
from iosp.viz import mj_scene as M

# ---------------------------------------------------------------------------
# SPaSM-based planning
# ---------------------------------------------------------------------------

def _plan_with_spasm(num_blocks=3, steps=None):
    """-> (q, events) via SPaSM's trajectory optimizer."""
    from iosp.checks.spasm_trajopt import solve_plan, concat_segments
    segs, _sim, _skeleton = solve_plan(num_blocks, steps=steps)
    q, events = concat_segments(segs)
    print(f"[tetris_viser] SPaSM trajopt: {segs.shape} -> path {q.shape}, "
          f"{len(events)} pick-place events")
    return q, events


# ---------------------------------------------------------------------------
# SPaSM tetris environment geometry (faithfully matches Simulation.render())
# ---------------------------------------------------------------------------

from iosp.model.spasm_costs import SPH_RADIUS  # noqa: E402

def _create_tetris_spheres(shape: str, sph_radius: float = SPH_RADIUS):
    from iosp.model.spasm_costs import create_tetris_spheres
    return create_tetris_spheres(shape, sph_radius)


def _transform_spheres(spheres, pose_xyzyaw):
    from iosp.model.spasm_costs import block_pose_to_spheres
    return block_pose_to_spheres(spheres, pose_xyzyaw)


_BLOCK_SHAPES = ["O", "L", "O", "O", "O", "L", "L", "L"]
_BLOCK_POSES = [
    [0.50, 0.35, None, 0],
    [0.15, -0.6, None, 0],
    [0.00, 0.6, None, 0],
    [0.15, 0.6, None, 0],
    [0.00, -0.6, None, 0],
    [0.50, -0.3, None, 0],
    [0.50, -0.1, None, 0],
    [0.50, 0.1, None, 0],
]

_RAW_COLORS = [0xe81416, 0xffa500, 0xfaeb36, 0x79c314, 0x487de7, 0x87369d, 0x5eb40d, 0xffa500]
_PASTEL_FACTOR = 0.6

def _pastel_rgb(hex_color):
    r = (hex_color >> 16) & 0xFF
    g = (hex_color >> 8) & 0xFF
    b = hex_color & 0xFF
    pf = _PASTEL_FACTOR
    return (int(r * pf + 255 * (1 - pf)),
            int(g * pf + 255 * (1 - pf)),
            int(b * pf + 255 * (1 - pf)))

BLOCK_COLORS = [_pastel_rgb(c) for c in _RAW_COLORS]

WALL_COLOR = (220, 218, 210)
TABLE_COLOR = (255, 255, 255)
GOAL_FLOOR_COLOR = (255, 255, 255)


def _spasm_spawn(num_blocks):
    from iosp.model import spasm_tasks as ST
    g = ST.tetris_geometry(num_blocks)
    return g["block_spheres"], [np.asarray(p, float) for p in g["block_poses"]]


def _get_block_spheres_and_poses(num_blocks):
    spheres, poses = _spasm_spawn(num_blocks)
    colors = [BLOCK_COLORS[k % len(BLOCK_COLORS)] for k in range(num_blocks)]
    return list(spheres), [p.astype(np.float32) for p in poses], colors


def _goal_dims(num_blocks):
    from iosp.model import spasm_tasks as ST
    g = ST.tetris_geometry(num_blocks)
    return (np.asarray(g["goal_dims"], float),
            np.asarray(g["goal_position"], float))


def _create_walls(goal_pos, goal_dims, wall_height=0.045, wall_thickness=0.015):
    cx, cy, cz = goal_pos
    cdx, cdy, _ = goal_dims
    walls = []
    walls.append((cx - cdx/2, cy + cdy/2, cz,
                  cx + cdx/2, cy + cdy/2 + wall_thickness, cz + wall_height))
    walls.append((cx - cdx/2, cy - cdy/2 - wall_thickness, cz,
                  cx + cdx/2, cy - cdy/2, cz + wall_height))
    walls.append((cx - cdx/2 - wall_thickness, cy - cdy/2, cz,
                  cx - cdx/2, cy + cdy/2, cz + wall_height))
    walls.append((cx + cdx/2, cy - cdy/2, cz,
                  cx + cdx/2 + wall_thickness, cy + cdy/2, cz + wall_height))
    return walls


# ---------------------------------------------------------------------------
# MuJoCo scene
# ---------------------------------------------------------------------------

SOLVED_GROUP = 3

def _load_solved_poses(num_blocks, path=None):
    from iosp.model import spasm_tasks as ST
    try:
        return (np.load(path).astype(np.float32) if path
                else ST.tetris_skeleton(num_blocks))
    except SystemExit:
        return None


def _rgba(rgb, alpha=1.0):
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, alpha)


def _add_tetris_env(b, i, num_blocks, pick=None, place=None, solved_poses=None):
    """Add scene `i`'s tetris environment to the world builder."""
    off = b.offset(i)
    p = lambda v: off + np.asarray(v, float)

    b.box(f"s{i}_table", p([0.30, 0.0, -0.011]), (0.8, 1.5, 0.02), M.TABLE_RGBA)

    gd, gp = _goal_dims(num_blocks)
    b.box(f"s{i}_goal_floor", p(gp),
          (float(gd[0]), float(gd[1]), float(gd[2])), _rgba(GOAL_FLOOR_COLOR))

    for j, (x1, y1, z1, x2, y2, z2) in enumerate(_create_walls(gp, gd)):
        b.box(f"s{i}_wall{j}", p([(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]),
              (x2 - x1, y2 - y1, z2 - z1), _rgba(WALL_COLOR))

    spheres_list, poses, colors = _get_block_spheres_and_poses(num_blocks)
    for k, (sphs, pose, color) in enumerate(zip(spheres_list, poses, colors)):
        for j, (x, y, z, r) in enumerate(_transform_spheres(sphs, pose)):
            b.sphere(f"s{i}_blk{k}_{j}", p([x, y, z]), r, _rgba(color))
    if solved_poses is not None:
        for k, (sphs, pose, color) in enumerate(zip(spheres_list, solved_poses,
                                                    colors)):
            for j, (x, y, z, r) in enumerate(_transform_spheres(sphs, pose)):
                b.sphere(f"s{i}_sol{k}_{j}", p([x, y, z]), r, _rgba(color),
                         collide=False, group=SOLVED_GROUP)

    if pick is not None:
        b.marker(f"s{i}_pick_marker", p(pick), 0.02, M.MARKER_PICK_RGBA)
        b.marker(f"s{i}_place_marker", p(place), 0.02, M.MARKER_PLACE_RGBA)


def _standoff():
    from iosp.model.tetris import GRASP_OFFSET
    return GRASP_OFFSET


def __getattr__(name):
    if name == "STANDOFF":
        return _standoff()
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def _load_npz(path):
    """(q, pick, place, grasp_row, release_row) from a `forward_extract` dump."""
    d = np.load(path, allow_pickle=True)
    dom = str(d["domain"])
    if dom != "tetris":
        raise SystemExit(f"{path} is a {dom!r} extract, not tetris")
    return (np.asarray(d["q"]), np.asarray(d["pick_pos"]),
            np.asarray(d["place_pos"]), int(d["idx_pick"]), int(d["idx_place"]),
            np.asarray(d["spawn_poses"]) if "spawn_poses" in d else None,
            int(d["num_blocks"]) if "num_blocks" in d else None)


def _scene_spawns(num_blocks, sampled, i):
    """All `num_blocks` spawn poses for npz scene `i`.

    Only the block this scene picks was randomised by the sampler; the rest sit
    at SPaSM's nominal spawns (or are pre-placed in the goal by `preplace`).
    """
    from iosp.model import spasm_tasks as ST
    nominal = np.asarray(ST.tetris_geometry(num_blocks)["block_poses"], float)
    if sampled is None:
        return None
    spawns = nominal.copy()
    spawns[i % num_blocks] = np.asarray(sampled[i], float)
    return spawns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=None,
                    help="override SPaSM's trajopt_steps")
    ap.add_argument("--robot", choices=M.ROBOTS, default="menagerie",
                    help="'spherized' is the planner's own collision model")
    ap.add_argument("--spread", type=float, default=1.8,
                    help="lateral offset between scenes, metres")
    ap.add_argument("--fps", type=float, default=60.0,
                    help="playback frames per second")
    ap.add_argument("--scene", type=int, default=None,
                    help="execute only this scene; the default chains one "
                         "demonstration per packing slot into the full sequence")
    ap.add_argument("--all-scenes", action="store_true",
                    help="score every npz scene separately and serve them in "
                         "one dropdown, as pickplace_viser does")
    ap.add_argument("--n-scenes", type=int, default=None,
                    help="with --all-scenes, cap how many scenes to roll out")
    ap.add_argument("--walls", choices=("solid", "outline"), default="solid",
                    help="'outline' paints the goal footprint on the floor "
                         "instead of building physical walls")
    ap.add_argument("--dwell", type=float, default=0.6)
    ap.add_argument("--settle", type=float, default=R.DEFAULT_SETTLE,
                    help="[s] simulated after release, to let the block come "
                         "to rest")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-solve", action="store_true",
                    help="environment geometry only -- no forward solve")
    ap.add_argument("--from-npz", type=str, default=None,
                    help="play a saved forward-pass extract "
                         "(scratch/feas/tetris_aligned.npz) instead of re-solving")
    ap.add_argument("--solved-npy", type=str, default=None,
                    help="override the packing skeleton shown in geom group 3")
    args = ap.parse_args()

    if args.all_scenes and not args.from_npz:
        ap.error("--all-scenes needs --from-npz PATH (it replays that "
                 "extract's scenes)")
    if args.all_scenes and args.scene is not None:
        ap.error("--all-scenes and --scene are mutually exclusive")

    solved_poses = _load_solved_poses(args.num_blocks, args.solved_npy)
    if solved_poses is None:
        print(f"[tetris_viser] no {args.num_blocks}-block solution at "
              f"{args.solved_npy}; showing spawn poses only")

    if args.no_solve:
        b = M.WorldBuilder(n_scenes=1, spread=args.spread, robot=args.robot)
        _add_tetris_env(b, 0, args.num_blocks, solved_poses=solved_poses)
        world = b.compile()
        print(f"[tetris_viser] environment-only mode, num_blocks={args.num_blocks}, "
              f"robot={args.robot}")
        M.play(world, 1, lambda _t: world.set_neutral(), port=args.port,
               fps=args.fps, banner="[tetris_viser] no trajectory (--no-solve)")
        return

    if args.from_npz:
        q, pick, place, grasp_row, release_row, sampled_spawns, npz_nb = \
            _load_npz(args.from_npz)
        # The extract records which packing skeleton it planned for; adopt it so
        # the goal box and block count match the plan without a second flag.
        if npz_nb is not None and npz_nb != args.num_blocks:
            print(f"[tetris_viser] {args.from_npz} is a {npz_nb}-block extract; "
                  f"using --num-blocks {npz_nb}")
            args.num_blocks = npz_nb
        # NPZ contains per-scene single-block trajectories; chain them all.
        n_scenes = q.shape[0]
        T = q.shape[1]
        if args.all_scenes:
            print(f"[tetris_viser] {args.from_npz}: q{q.shape}")
            _run_all_scenes(args, q, grasp_row, release_row, sampled_spawns)
            return
        if args.scene is not None:
            i = min(max(args.scene, 0), n_scenes - 1)
            rows = q[i]
            events = [(grasp_row, release_row, i % args.num_blocks)]
            preplace = list(range(i % args.num_blocks))
            label = f"scene {i} from npz"
        else:
            slots = list(range(min(args.num_blocks, n_scenes)))
            rows = np.concatenate([q[k] for k in slots], axis=0)
            events = [(k * T + grasp_row, k * T + release_row, k) for k in slots]
            preplace = None
            label = f"npz, {len(slots)} blocks chained"
        print(f"[tetris_viser] {args.from_npz}: q{q.shape}")
    else:
        print("Planning with SPaSM...", flush=True)
        rows, events = _plan_with_spasm(args.num_blocks, steps=args.steps)
        preplace = None
        label = f"SPaSM plan, {args.num_blocks} blocks"

    _run_dynamic(args, rows, events, preplace, label, solved_poses)


def _cost(v):
    return "unavailable (SPaSM not importable)" if v is None else f"{v:.4f}"


def _mm(v):
    return "n/a" if v is None else f"{v:.1f} mm"


def _scene_entry(args, rows, events, preplace, label, spawn_poses=None):
    """Roll out and score ONE scene -> a `mj_scene.play_multi` entry dict.

    Delegates the scene and the verdict to `iosp.checks.spasm_rollout`, so the
    viewer and the check report the same answer about the same trajectory.
    """
    from iosp.checks import spasm_rollout as SR
    from iosp.model import spasm_tasks as ST

    nb = args.num_blocks
    world, sim, names = SR.build_scene(nb, robot=args.robot, preplace=preplace,
                                       walls=args.walls,
                                       spawn_poses=spawn_poses)
    skeleton = ST.tetris_skeleton(nb)

    print(f"[tetris_viser] actuated rollout: {label}")
    ro = R.run_events(world, rows, [(g, r, names[k]) for g, r, k in events],
                      settle=args.settle, dwell=args.dwell)

    world.data.qpos[:] = ro.qpos[-1]
    world.forward()
    achieved = np.stack([world.body_xyzyaw(nm) for nm in names])
    tilt = np.array([world.body_tilt_deg(nm) for nm in names])
    scored = [k for _, _, k in events]
    ok, v = ST.packing_success(achieved, nb, tilt_deg=tilt, slots=scored)
    worst, mean = ro.tracking_mm_deg(0)

    print(f"\n===== tetris {label}: EXECUTED =====")
    print(f"  TASK {'SUCCESS' if ok else 'FAIL'}: "
          f"in the walls {'yes' if v['inside_walls'] else 'NO'} "
          f"(margin {v['min_wall_margin_mm']:+.1f} mm), "
          f"no overlap {'yes' if v['no_overlap'] else 'NO'} "
          f"({_mm(v['min_block_clearance_mm'])}), "
          f"upright {'yes' if v['upright'] else 'NO'} "
          f"({v['max_tilt_deg']:.1f} deg)")
    for k in scored:
        d = np.linalg.norm(achieved[k, :2] - skeleton[k, :2]) * 100
        dy = abs(((achieved[k, 3] - skeleton[k, 3] + np.pi) % (2 * np.pi)) - np.pi)
        print(f"    block {k}: {d:5.2f} cm and {np.degrees(dy):5.1f} deg from "
              f"its skeleton pose   tilt {tilt[k]:4.1f} deg")
    print(f"  arm tracking max {worst:.1f} deg (mean {mean:.1f})")
    print(f"  [diagnostic] SPaSM packing cost {_cost(v['spasm_cost'])} "
          f"(threshold {ST.TETRIS_COST_THRESH.get(nb)}) -- a planning "
          f"objective, not the task test")

    verdict_md = (("<span style='color:#2ab05e'>**TASK SUCCESS**</span>" if ok
                   else "<span style='color:#d43d3d'>**TASK FAIL**</span>")
                  + f" &nbsp; {label}"
                  + f"\n\nin the walls {v['inside_walls']} "
                    f"({v['min_wall_margin_mm']:+.1f} mm) &nbsp;|&nbsp; "
                    f"no overlap {v['no_overlap']} "
                    f"({_mm(v['min_block_clearance_mm'])}) &nbsp;|&nbsp; "
                    f"upright {v['upright']} ({v['max_tilt_deg']:.1f}&deg;)"
                  + f"\n\n<sub>SPaSM cost {_cost(v['spasm_cost'])} is a "
                    f"planning objective, not the task test.</sub>")

    set_row = R.replayer(world, ro)

    # Determine which block each timestep corresponds to.
    T_per_block = len(rows) // max(len(events), 1)

    def _info(t):
        u = ro.frame_row[int(t)]
        k = events[min(int(u) // max(T_per_block, 1), len(events) - 1)][2]
        return (f"block **{k}** &nbsp;|&nbsp; t={t * ro.dt:.2f}s")

    return dict(world=world, n_frames=ro.n_frames, set_row=set_row,
                label=f"{label} [{'OK' if ok else 'FAIL'}]",
                verdict_md=verdict_md, info_fn=_info, ok=bool(ok))


def _run_dynamic(args, rows, events, preplace, label, solved_poses,
                 spawn_poses=None):
    """One rollout, served on its own (the chained or single-scene modes)."""
    e = _scene_entry(args, rows, events, preplace, label, spawn_poses)
    print(f"[tetris_viser] {e['n_frames']} recorded frames, robot={args.robot}")
    M.play(e["world"], e["n_frames"], e["set_row"], port=args.port,
           fps=args.fps, info_fn=e["info_fn"], verdict_md=e["verdict_md"])


def _run_all_scenes(args, q, grasp_row, release_row, sampled_spawns=None):
    """Every npz scene as its own rollout, in one dropdown -- the same view
    `pickplace_viser` gives.  Scene `i` packs slot `i % num_blocks`, with the
    slots below it already placed, so each is scored independently."""
    n = q.shape[0] if args.n_scenes is None else min(args.n_scenes, q.shape[0])
    entries = []
    for i in range(n):
        k = i % args.num_blocks
        entries.append(_scene_entry(
            args, q[i], [(grasp_row, release_row, k)],
            list(range(k)), f"scene {i} (slot {k})",
            _scene_spawns(args.num_blocks, sampled_spawns, i)))
    n_ok = sum(e["ok"] for e in entries)
    print(f"\n  ---> {n_ok}/{len(entries)} succeeded")
    print(f"[tetris_viser] {len(entries)} scenes, robot={args.robot}")
    M.play_multi(entries, port=args.port, fps=args.fps)


if __name__ == "__main__":
    main()
