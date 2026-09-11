"""Actuated MuJoCo rollout of the tower-stacking task via mjviser.

Plans the trajectory with SPaSM's own tower trajectory optimizer for a
dynamically feasible path, then drives it through a real MuJoCo scene under
physics: the Panda's position servos track the plan under gravity, and the cube
is held purely by contact friction (the gripper closes on it and lifts it -- no
kinematic carry).  Where the cube ends up is determined by the physics, not
asserted.

Everything runs through `mj_step` -- there is no kinematic replay path.  See
`iosp.viz.mj_scene` for the `--robot` choice.

    # solve with SPaSM and roll out (needs SPaSM checkout + GPU)
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.viz.tower_viser --num-blocks 10

    # replay a saved forward-pass extract -- no solve, no GPU work
        python -m iosp.viz.tower_viser --from-npz scratch/feas/tower_aligned.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from iosp.viz import mj_scene as M

# ---------------------------------------------------------------------------
# SPaSM-based planning
# ---------------------------------------------------------------------------

def _plan_with_spasm(num_blocks=10, steps=None, pin_place_row=True):
    """-> (q, events) via SPaSM's tower trajectory optimizer."""
    from iosp.checks.spasm_trajopt import solve_tower_plan, concat_segments
    segs, _sim, _skeleton = solve_tower_plan(num_blocks, steps=steps,
                                              pin_place_row=pin_place_row)
    q, events = concat_segments(segs)
    print(f"[tower_viser] SPaSM trajopt: {segs.shape} -> path {q.shape}, "
          f"{len(events)} pick-place events")
    return q, events


# ---------------------------------------------------------------------------
# SPaSM tower environment geometry (matches TowerSimulation.render())
# ---------------------------------------------------------------------------

BLOCK_DIM = 0.06
BLOCK_HALF = BLOCK_DIM / 2.0
BASE_XY = (0.45, 0.0)

TABLE_DIMS = (1.1, 1.5, 0.02)
TABLE_POS = (0.15, 0.0, -0.011)

BASE_MARKER_COLOR = (230, 204, 50)
STACKED_COLOR = (165, 140, 100)

import random as _random
_TOWER_BLOCK_HEX = [0xfd3f52, 0xff6b6b, 0xfd7e03, 0xffbc16, 0xa9e507,
                     0x65d73d, 0x38c188, 0x0cd4ae, 0x02ccd0, 0x31b5e7]
_random.seed(42)
_random.shuffle(_TOWER_BLOCK_HEX)


def _hex_rgb(h):
    return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)


def _rgba(rgb, alpha=1.0):
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, alpha)


def _spawn_poses(num_blocks=10):
    return ([[0.4 - (i - num_blocks // 2) * 0.12, 0.30, BLOCK_DIM / 2, 0.0]
             for i in range(num_blocks // 2, num_blocks)] +
            [[0.4 - i * 0.12, 0.5, BLOCK_DIM / 2, 0.0]
             for i in range(num_blocks // 2)])


def _standoff():
    from iosp.model.tower import GRASP_OFFSET
    return GRASP_OFFSET


def __getattr__(name):
    if name == "STANDOFF":
        return _standoff()
    raise AttributeError(name)

PHASE_NAMES = ("approach", "place_traj", "return_traj")


def _phase_of(t, grasp_row, release_row):
    if t <= grasp_row:
        return PHASE_NAMES[0]
    return PHASE_NAMES[1] if t <= release_row else PHASE_NAMES[2]


def _load_npz(path):
    from iosp.checks.forward_extract import clip_to_limits
    d = np.load(path, allow_pickle=True)
    dom = str(d["domain"])
    if dom != "tower":
        raise SystemExit(f"{path} is a {dom!r} extract, not tower")
    # An extract saved before `forward_extract` enforced the limits can still
    # hold rows the arm cannot reach; clip on the way in so a stale npz plays
    # the same way a fresh one does, and says so.
    q = clip_to_limits(np.asarray(d["q"]),
                       {k: d[k] for k in ("q_lo", "q_hi") if k in d})
    return dict(q=q, pick=np.asarray(d["pick_pos"]),
                place=np.asarray(d["place_pos"]),
                obs_center=np.asarray(d["obs_center"]),
                obs_radius=np.asarray(d["obs_radius"]),
                grasp_row=int(d["idx_pick"]), release_row=int(d["idx_place"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-blocks", type=int, default=10)
    ap.add_argument("--steps", type=int, default=None,
                    help="override SPaSM's trajopt_steps")
    ap.add_argument("--no-pin-place-row", action="store_true",
                    help="let SPaSM move the placement row (tower drifts)")
    ap.add_argument("--n-scenes", type=int, default=4,
                    help="number of scenes for --from-npz")
    ap.add_argument("--all-scenes", action="store_true",
                    help="use every scene in the npz (overrides --n-scenes)")
    ap.add_argument("--robot", choices=M.ROBOTS, default="menagerie",
                    help="'spherized' is the planner's own collision model")
    ap.add_argument("--spread", type=float, default=1.8)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--scene", type=int, default=None,
                    help="execute only this scene; the default chains all")
    ap.add_argument("--dwell", type=float, default=0.6)
    ap.add_argument("--settle", type=float, default=2.0,
                    help="[s] simulated after release, to let the cube come "
                         "to rest")
    ap.add_argument("--from-npz", type=str, default=None,
                    help="play a saved forward-pass extract "
                         "(scratch/feas/tower_aligned.npz) instead of re-solving")
    args = ap.parse_args()

    if args.from_npz:
        d = _load_npz(args.from_npz)
        n = d["q"].shape[0] if args.all_scenes else min(args.n_scenes, d["q"].shape[0])
        q, pick, place = d["q"][:n], d["pick"][:n], d["place"][:n]
        grasp_row, release_row = d["grasp_row"], d["release_row"]
        T = q.shape[1]
        print(f"[tower_viser] {args.from_npz}: q{q.shape}, no solve")

        if args.scene is not None:
            i = min(max(args.scene, 0), n - 1)
            rows = q[i]
            events = [(grasp_row, release_row, i)]
            preplace = list(range(i))
            label = f"npz scene {i}, level {i} ({i} pre-placed)"
        else:
            slots = list(range(n))
            rows = np.concatenate([q[k] for k in slots], axis=0)
            events = [(k * T + grasp_row, k * T + release_row, k) for k in slots]
            preplace = None
            label = f"npz, {len(slots)} levels chained"
    else:
        print("Planning with SPaSM...", flush=True)
        rows, events = _plan_with_spasm(
            args.num_blocks, steps=args.steps,
            pin_place_row=not args.no_pin_place_row)
        preplace = None
        label = f"SPaSM plan, {args.num_blocks} blocks"

    _run_dynamic(args, rows, events, preplace, label)


def _run_dynamic(args, rows, events, preplace, label):
    """Actuated rollout scored by the SPaSM tower task test."""
    from iosp.checks import spasm_rollout as SR
    from iosp.model import spasm_tasks as ST
    from iosp.viz import mj_rollout as R

    nb = args.num_blocks
    world, sim, names = SR.build_tower_scene(nb, robot=args.robot,
                                              preplace=preplace)
    skeleton = ST.tower_skeleton(nb)

    print(f"[tower_viser] actuated rollout: {label}")
    ro = R.run_events(world, rows, [(g, r, names[k]) for g, r, k in events],
                      settle=args.settle, dwell=args.dwell, gravcomp=True,
                      grasp_half_width=BLOCK_HALF - 0.01)

    world.data.qpos[:] = ro.qpos[-1]
    world.forward()
    # Score only the levels this rollout actually BUILT.  `tower_success` marks
    # every block it is given against that block's own level height, so handing
    # it all ten while the plan places six reports a 541 mm error for the four
    # still sitting in their spawn cells -- a FAIL that says nothing about the
    # six that were placed perfectly.  The tower that was attempted is levels
    # 0..max(event level), and that is what gets graded.
    n_built = max(k for _, _, k in events) + 1
    achieved = np.stack([world.body_xyzyaw(nm) for nm in names[:n_built]])
    tilt = np.array([world.body_tilt_deg(nm) for nm in names[:n_built]])
    ok, v = ST.tower_success(achieved, tilt, n_built)
    worst, mean = ro.tracking_mm_deg(0)
    h = float(sim.block_height)

    print(f"\n===== tower {label}: EXECUTED =====")
    print(f"  TASK {'SUCCESS' if ok else 'FAIL'}: "
          f"at height {'yes' if v['at_height'] else 'NO'} "
          f"(worst {v['max_z_err_mm']:.0f} mm), "
          f"supported {'yes' if v['supported'] else 'NO'} "
          f"(worst offset {v['max_stack_offset_mm']:.0f} mm), "
          f"upright {'yes' if v['upright'] else 'NO'} "
          f"(max tilt {v['max_tilt_deg']:.1f} deg)")
    print(f"  arm tracking max {worst:.1f} deg (mean {mean:.1f})")
    for k in [ev[2] for ev in events]:
        print(f"    level {k}: z {achieved[k,2]:.3f} (target {k*h+h/2:.3f}), "
              f"xy {np.linalg.norm(achieved[k,:2]-skeleton[k,:2])*100:5.2f} cm "
              f"from plan, tilt {tilt[k]:5.1f} deg")

    verdict_md = (("<span style='color:#2ab05e'>**TASK SUCCESS**</span>" if ok
                   else "<span style='color:#d43d3d'>**TASK FAIL**</span>")
                  + f" &nbsp; {label}"
                  + f"\n\nat height {v['at_height']} "
                    f"(worst {v['max_z_err_mm']:.0f} mm) &nbsp;|&nbsp; "
                    f"supported {v['supported']} "
                    f"(worst {v['max_stack_offset_mm']:.0f} mm) &nbsp;|&nbsp; "
                    f"upright {v['upright']} ({v['max_tilt_deg']:.1f}&deg;)")

    set_row = R.replayer(world, ro)
    T_per_block = len(rows) // max(len(events), 1)

    def _info(t):
        u = ro.frame_row[int(t)]
        k = events[min(int(u) // max(T_per_block, 1), len(events) - 1)][2]
        return (f"block **{k}** &nbsp;|&nbsp; t={t * ro.dt:.2f}s")

    print(f"[tower_viser] {ro.n_frames} recorded frames, robot={args.robot}")
    M.play(world, ro.n_frames, set_row, port=args.port, fps=args.fps,
           info_fn=_info, verdict_md=verdict_md)


if __name__ == "__main__":
    main()
