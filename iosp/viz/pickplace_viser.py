"""Dynamic (actuated) rollout of the pick-and-place model in MuJoCo via mjviser.

Drives the planned joint paths through the Panda's position servos under gravity
and watches where the cube actually ends up in the bucket.  Contact physics
determines whether the grasp holds and the release lands.

The scene matches the E10 teleop layout: a table, a cube spawned on the table,
and an n-gon bucket at the place target.

    # replay a saved forward-pass extract through physics — no solve, no GPU
    MUJOCO_GL=egl python -m iosp.viz.pickplace_viser \\
        --from-npz scratch/feas/pickplace.npz

    # solve and roll out (slow; needs a GPU)
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
        python -m iosp.viz.pickplace_viser --solve --n-scenes 4

    # environment geometry only
    MUJOCO_GL=egl python -m iosp.viz.pickplace_viser --no-solve
"""
from __future__ import annotations

import argparse

import numpy as np

from iosp.checks import spasm_rollout as SR
from iosp.viz import mj_scene as M

# approach -> grasp -> transport -> place
PHASE_NAMES = ("approach", "grasp", "transport", "place")


def _phase_of(t, grasp_row, release_row):
    if t < grasp_row:
        return PHASE_NAMES[0]
    if t == grasp_row:
        return PHASE_NAMES[1]
    return PHASE_NAMES[2] if t <= release_row else PHASE_NAMES[3]


GRIPPER_BODIES = ("hand", "left_finger", "right_finger")


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def _load_npz(path):
    """A `iosp.checks.forward_extract` pick-place dump."""
    d = np.load(path, allow_pickle=True)
    dom = str(d["domain"])
    if dom != "pickplace":
        raise SystemExit(f"{path} is a {dom!r} extract, not pickplace")
    out = dict(q=np.asarray(d["q"]), pick=np.asarray(d["pick_pos"]),
               place=np.asarray(d["place_pos"]),
               grasp_row=int(d["idx_pick"]), release_row=int(d["idx_place"]))
    for k in ("bucket_center", "bucket_inner_radius",
              "bucket_wall_height", "bucket_wall_thickness",
              "bucket_floor_thickness", "table_top_z"):
        if k in d:
            out[k] = np.asarray(d[k])
    return out


def _solve(seed=0, n_iters=60, n_scenes=4):
    """Roll the composed chain out at e4's ground-truth Z_STAR."""
    from iosp.checks import forward_extract as FE
    q, meta = FE._pickplace()
    n = min(n_scenes, q.shape[0])
    out = dict(q=np.asarray(q)[:n],
               pick=np.asarray(meta["pick_pos"])[:n],
               place=np.asarray(meta["place_pos"])[:n],
               grasp_row=int(meta["idx_pick"]),
               release_row=int(meta["idx_place"]))
    for k in ("bucket_center", "bucket_inner_radius",
              "bucket_wall_height", "bucket_wall_thickness",
              "bucket_floor_thickness", "table_top_z"):
        if k in meta:
            out[k] = np.asarray(meta[k])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--robot", choices=M.ROBOTS, default="menagerie",
                    help="'spherized' is the planner's own collision model")
    ap.add_argument("--spread", type=float, default=1.4)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dwell", type=float, default=0.6)
    ap.add_argument("--settle", type=float, default=1.5,
                    help="[s] simulated after release")
    ap.add_argument("--no-solve", action="store_true",
                    help="environment geometry only -- no forward solve")
    ap.add_argument("--from-npz", type=str, default=None,
                    help="play a saved forward-pass extract "
                         "(scratch/feas/pickplace.npz)")
    ap.add_argument("--solve", action="store_true",
                    help="solve a fresh rollout instead (slow: the fit's own chain)")
    args = ap.parse_args()

    if not (args.no_solve or args.from_npz or args.solve):
        ap.error("pick one of --from-npz PATH, --solve, or --no-solve")

    if args.no_solve:
        pick = np.array([SR.PP_TABLE_CENTER_XY[0] - 0.1,
                         SR.PP_TABLE_CENTER_XY[1] - 0.18,
                         SR.PP_TABLE_HEIGHT + SR.PP_OBJ_SIZE / 2 + 1e-3])
        b = M.WorldBuilder(n_scenes=1, spread=args.spread, robot=args.robot)
        table_pos = (SR.PP_TABLE_CENTER_XY[0], SR.PP_TABLE_CENTER_XY[1],
                     SR.PP_TABLE_HEIGHT - 0.5 * SR.PP_TABLE_THICKNESS)
        table_dims = (2 * SR.PP_TABLE_HALF_XY[0], 2 * SR.PP_TABLE_HALF_XY[1],
                      SR.PP_TABLE_THICKNESS)
        b.box("table", table_pos, table_dims, M.TABLE_RGBA)
        SR._add_bucket(b, SR.PP_BUCKET_CENTER_XY, SR.PP_TABLE_HEIGHT)
        nm = "s0_cube"
        b.free_body(nm, [("box", (SR.PP_OBJ_SIZE,) * 3, (0, 0, 0),
                          SR.PP_OBJ_RGBA)],
                    pos=tuple(pick), collide=True)
        for g in GRIPPER_BODIES:
            b.exclude(f"s0_{g}", nm)
        world = b.compile()
        print(f"[pickplace_viser] environment-only mode, robot={args.robot}")
        M.play(world, 1, lambda _t: world.set_neutral(), port=args.port,
               fps=args.fps, banner="[pickplace_viser] no trajectory (--no-solve)")
        return

    if args.from_npz:
        d = _load_npz(args.from_npz)
        n = min(args.n_scenes, d["q"].shape[0])
        data = {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim > 0 else v)
                for k, v in d.items()}
        print(f"[pickplace_viser] {args.from_npz}: q{data['q'].shape}")
    else:
        print("Building forward model and rolling out (slow)...", flush=True)
        data = _solve(seed=args.seed, n_iters=args.n_iters,
                      n_scenes=args.n_scenes)

    # -- dynamic (actuated) rollout ----------------------------------------
    from iosp.viz import mj_rollout as R

    q = data["q"]
    pick, place = data["pick"], data["place"]
    grasp_row, release_row = data["grasp_row"], data["release_row"]
    n_scenes = q.shape[0]

    bucket_kw = {}
    if "bucket_center" in data:
        bucket_kw["bucket_center_xy"] = data["bucket_center"][..., :2]
    for k in ("bucket_inner_radius", "bucket_wall_height",
              "bucket_wall_thickness", "bucket_floor_thickness",
              "table_top_z"):
        if k in data:
            bucket_kw[k] = data[k]

    results = []
    for i in range(n_scenes):
        per_scene_kw = {}
        for k, v in bucket_kw.items():
            if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n_scenes:
                per_scene_kw[k] = v[i]
            else:
                per_scene_kw[k] = v
        r = SR.rollout_and_score_pickplace(
            q[i], grasp_row, release_row, pick[i], place[i],
            settle=args.settle, dwell=args.dwell, robot=args.robot,
            progress=(i == 0),
            **per_scene_kw)
        results.append(r)

    n_ok = sum(r["ok"] for r in results)
    print(f"\n  ---> {n_ok}/{n_scenes} succeeded")

    entries = []
    for i, r in enumerate(results):
        ro_i = r["rollout"]
        v = r["verdict"]
        tag = "OK" if r["ok"] else "FAIL"
        verdict_md = (
            ("<span style='color:#2ab05e'>**TASK SUCCESS**</span>" if r["ok"]
             else "<span style='color:#d43d3d'>**TASK FAIL**</span>")
            + f"\n\nerror {v['err_mm']:.1f} mm"
              f" (xy {v['err_xy_mm']:.1f} mm)"
              f" &nbsp;|&nbsp; upright {v['upright']}"
              f" ({v['tilt_deg']:.1f}&deg;)")
        set_row_i = R.replayer(r["world"], ro_i)

        def _make_info(ro_inner):
            def _info(t):
                u = ro_inner.frame_row[int(t)]
                ph = _phase_of(int(u), grasp_row, release_row)
                return f"phase **{ph}** &nbsp;|&nbsp; t={t * ro_inner.dt:.2f}s"
            return _info

        entries.append(dict(
            world=r["world"], n_frames=ro_i.n_frames,
            set_row=set_row_i, label=f"scene {i} [{tag}]",
            verdict_md=verdict_md, info_fn=_make_info(ro_i)))

    print(f"[pickplace_viser] {n_scenes} scenes, robot={args.robot}")
    M.play_multi(entries, port=args.port, fps=args.fps)


if __name__ == "__main__":
    main()
