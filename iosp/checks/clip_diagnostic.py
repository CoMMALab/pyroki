"""Diagnose obstacle clipping during dynamic rollouts of forward-solved plans.

For each domain (tetris, tower, pickplace), loads the forward-extracted .npz,
runs a dynamic rollout, and records:
  - Contact penetration at each recorded frame (arm-obstacle, object-obstacle)
  - EE path and carried-object path
  - Where and when clipping occurs

Produces a per-scene JSON report for visualization, and a text summary.

    python -m iosp.checks.clip_diagnostic scratch/feas/tetris_fresh.npz
    python -m iosp.checks.clip_diagnostic scratch/feas/tower_fresh.npz
    python -m iosp.checks.clip_diagnostic scratch/feas/pickplace_fresh.npz
"""
from __future__ import annotations

import argparse
import json
import pathlib

import mujoco
import numpy as np

from iosp.viz import mj_scene as M, mj_rollout as R


# ---------------------------------------------------------------------------
# Per-domain scene builders (same as dynamic_report but returns more info)
# ---------------------------------------------------------------------------

def _build_tetris_scene(b, d, i, num_blocks=1):
    from iosp.viz import tetris_viser as V
    so = np.array([0.0, 0.0, V.STANDOFF])
    V._add_tetris_env(b, 0, num_blocks, pick=d["pick"][i], place=d["place"][i])
    V._carried_tetromino(b, 0, d["pick"][i] - so, collide=True)
    return dict(grasp_off=so, place_off=so, score_on="settled",
                domain_info=f"tetris {num_blocks}-block, scene {i}")


def _build_tower_scene(b, d, i, stack_level=None):
    from iosp.viz import tower_viser as V
    so = np.array([0.0, 0.0, V.STANDOFF])
    if stack_level is None:
        rest_z = float(d["place"][i, 2]) - V.STANDOFF
        stack_level = int(round(rest_z / V.BLOCK_DIM - 0.5))
    base_xy = (float(d["place"][i, 0]), float(d["place"][i, 1]))
    V._add_tower_env(b, 0, stack_level, pick=d["pick"][i], place=d["place"][i],
                     obs_center=None, skip_spawn_near=d["pick"][i] - so,
                     base_xy=base_xy)
    V._carried_cube(b, 0, d["pick"][i] - so, collide=True)
    return dict(grasp_off=so, place_off=so, score_on="settled",
                domain_info=f"tower level {stack_level}, scene {i}")


def _build_pickplace_scene(b, d, i):
    from iosp.checks import spasm_rollout as SR
    zero = np.zeros(3)
    off = b.offset(0)
    table_pos = tuple(off + np.array([SR.PP_TABLE_CENTER_XY[0],
                                       SR.PP_TABLE_CENTER_XY[1],
                                       SR.PP_TABLE_HEIGHT - 0.5 * SR.PP_TABLE_THICKNESS]))
    table_dims = (2 * SR.PP_TABLE_HALF_XY[0], 2 * SR.PP_TABLE_HALF_XY[1],
                  SR.PP_TABLE_THICKNESS)
    b.box("s0_table", table_pos, table_dims, M.TABLE_RGBA)
    SR._add_bucket(b, SR.PP_BUCKET_CENTER_XY, SR.PP_TABLE_HEIGHT)
    pick = np.asarray(d["pick"][i], float)
    nm = "s0_carried"
    b.free_body(nm, [("box", (SR.PP_OBJ_SIZE,) * 3, (0, 0, 0), SR.PP_OBJ_RGBA)],
                pos=tuple(off + pick), collide=True)
    return dict(grasp_off=zero, place_off=zero, score_on="settled",
                domain_info=f"pickplace, scene {i}")


BUILDERS = {"tetris": _build_tetris_scene, "pickplace": _build_pickplace_scene,
            "tower": _build_tower_scene}


def _load(path):
    d = np.load(path, allow_pickle=True)
    out = {k: np.asarray(d[k]) for k in d.files if d[k].ndim > 0}
    return (str(d["domain"]),
            dict(q=out["q"], pick=out["pick_pos"], place=out["place_pos"],
                 **{k: out[k] for k in ("obs_center", "obs_radius") if k in out}),
            int(d["idx_pick"]), int(d["idx_place"]))


# ---------------------------------------------------------------------------
# Contact analysis
# ---------------------------------------------------------------------------

def _geom_name(model, gid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return n if n else f"g{gid}"


def _body_of_geom(model, gid):
    bid = model.geom_bodyid[gid]
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"b{bid}"


def analyze_contacts(world, model, data):
    """Return list of penetrating contacts with metadata."""
    contacts = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        if c.dist >= 0:
            continue
        g1 = _geom_name(model, c.geom1)
        g2 = _geom_name(model, c.geom2)
        b1 = _body_of_geom(model, c.geom1)
        b2 = _body_of_geom(model, c.geom2)
        contacts.append(dict(
            geom1=g1, geom2=g2, body1=b1, body2=b2,
            depth_mm=float(-c.dist * 1000),
            pos=c.pos.tolist(),
        ))
    return contacts


def classify_contact(c, scene_idx):
    """Classify a contact as arm-obstacle, object-obstacle, arm-table, etc."""
    prefix = f"s{scene_idx}_"
    names = [c["geom1"], c["geom2"]]
    bodies = [c["body1"], c["body2"]]

    is_arm = any("link" in n or "hand" in n or "finger" in n for n in bodies)
    is_carried = any("carried" in n for n in names)
    is_table = any("table" in n for n in names)
    is_wall = any("wall" in n for n in names)
    is_floor = any("floor" in n or "goal_floor" in n for n in names)
    is_obstacle = any("obs" in n for n in names)
    is_stack = any("stack" in n or "block" in n for n in names)

    if is_carried and is_wall:
        return "object-wall"
    if is_carried and is_table:
        return "object-table"
    if is_carried and is_obstacle:
        return "object-obstacle"
    if is_carried and is_stack:
        return "object-stack"
    if is_arm and is_table:
        return "arm-table"
    if is_arm and is_wall:
        return "arm-wall"
    if is_arm and is_obstacle:
        return "arm-obstacle"
    if is_arm and is_arm:
        return "arm-self"
    return "other"


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def diagnose_scene(domain, d, scene_idx, grasp_row, release_row, **kw):
    """Full diagnostic for one scene: kinematic + dynamic analysis."""
    b = M.WorldBuilder(n_scenes=1, spread=1.8, robot="menagerie")
    cfg = BUILDERS[domain](b, d, scene_idx, **kw)
    world = b.compile()
    model, data = world.model, world.data
    q = d["q"][scene_idx]

    # --- Kinematic pass: set each row, check contacts ---
    kin_report = []
    ee_path = []
    for t in range(q.shape[0]):
        world.set_arm(0, q[t])
        world.forward()
        ee = np.asarray(world.ee_position(0))
        ee_path.append(ee.tolist())
        contacts = analyze_contacts(world, model, data)
        classified = [(classify_contact(c, 0), c) for c in contacts]
        kin_report.append(dict(
            row=t,
            ee=ee.tolist(),
            contacts=[dict(type=cls, **c) for cls, c in classified],
            worst_mm=max((c["depth_mm"] for c in contacts), default=0),
        ))

    # --- Dynamic pass ---
    b2 = M.WorldBuilder(n_scenes=1, spread=1.8, robot="menagerie")
    cfg2 = BUILDERS[domain](b2, d, scene_idx, **kw)
    world2 = b2.compile()
    q_batch = q[None]  # (1, T, 7)
    ro = R.run(world2, q_batch, grasp_row, release_row,
               carried_names=[f"s0_carried"],
               settle=1.5, progress=False)

    # Analyze every recorded frame for contacts
    dyn_clips = []
    carried_path = []
    for fi in range(ro.n_frames):
        world2.data.qpos[:] = ro.qpos[fi]
        mujoco.mj_forward(world2.model, world2.data)
        contacts = analyze_contacts(world2, world2.model, world2.data)
        classified = [(classify_contact(c, 0), c) for c in contacts]
        obstacle_clips = [dict(type=cls, **c) for cls, c in classified
                          if "wall" in cls or "obstacle" in cls or "stack" in cls]
        if obstacle_clips:
            dyn_clips.append(dict(
                frame=fi, row=float(ro.frame_row[fi]),
                clips=obstacle_clips,
            ))
        try:
            cp = np.asarray(world2.body_pos(f"s0_carried"))
            carried_path.append(cp.tolist())
        except Exception:
            carried_path.append(None)

    # Final object position
    world2.data.qpos[:] = ro.qpos[-1]
    mujoco.mj_forward(world2.model, world2.data)
    try:
        final_obj = np.asarray(world2.body_pos(f"s0_carried"))
    except Exception:
        final_obj = np.zeros(3)
    target = d["place"][scene_idx] - cfg2["place_off"]
    err_mm = float(np.linalg.norm(final_obj - target) * 1000)

    # Release position
    rf = min(ro.release_frame, ro.n_frames - 1)
    world2.data.qpos[:] = ro.qpos[rf]
    mujoco.mj_forward(world2.model, world2.data)
    try:
        release_obj = np.asarray(world2.body_pos(f"s0_carried"))
    except Exception:
        release_obj = np.zeros(3)
    release_err_mm = float(np.linalg.norm(release_obj - target) * 1000)

    worst, mean = ro.tracking_mm_deg(0)

    # Joint limit check
    lo, hi = world2.joint_limits()
    jl_over = np.degrees(np.maximum(0.0, np.maximum(lo - q, q - hi))).max(axis=1)

    return dict(
        domain=domain,
        scene=scene_idx,
        info=cfg.get("domain_info", ""),
        pick=d["pick"][scene_idx].tolist(),
        place=d["place"][scene_idx].tolist(),
        target=target.tolist(),
        grasp_row=grasp_row,
        release_row=release_row,
        n_rows=q.shape[0],
        ee_path=ee_path,
        carried_path=carried_path,
        kin_contacts=kin_report,
        dyn_obstacle_clips=dyn_clips,
        n_clip_frames=len(dyn_clips),
        worst_clip_mm=max((max(c["depth_mm"] for c in cf["clips"])
                           for cf in dyn_clips), default=0),
        final_obj=final_obj.tolist(),
        release_obj=release_obj.tolist(),
        settled_err_mm=err_mm,
        release_err_mm=release_err_mm,
        tracking_max_deg=worst,
        tracking_mean_deg=mean,
        joint_limit_over_deg=jl_over.tolist(),
        max_joint_limit_over_deg=float(jl_over.max()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--n-scenes", type=int, default=None)
    ap.add_argument("--num-blocks", type=int, default=1, help="tetris only")
    ap.add_argument("--out", default=None,
                    help="write JSON report to this path")
    args = ap.parse_args()

    all_reports = []
    for path in args.npz:
        domain, d, grasp_row, release_row = _load(path)
        n = d["q"].shape[0] if args.n_scenes is None else min(args.n_scenes, d["q"].shape[0])

        print(f"\n{'='*60}")
        print(f"  {domain.upper()}: {path}  ({n} scenes)")
        print(f"  grasp_row={grasp_row}, release_row={release_row}")
        print(f"{'='*60}")

        for i in range(n):
            kw = dict(num_blocks=args.num_blocks) if domain == "tetris" else {}
            report = diagnose_scene(domain, d, i, grasp_row, release_row, **kw)
            all_reports.append(report)

            clip_types = {}
            for cf in report["dyn_obstacle_clips"]:
                for c in cf["clips"]:
                    t = c["type"]
                    clip_types[t] = max(clip_types.get(t, 0), c["depth_mm"])

            kin_types = {}
            for kr in report["kin_contacts"]:
                for c in kr["contacts"]:
                    t = c["type"]
                    kin_types[t] = max(kin_types.get(t, 0), c["depth_mm"])

            status = "OK" if report["n_clip_frames"] == 0 else "CLIPS"
            print(f"\n  Scene {i}: {status}")
            print(f"    Release err: {report['release_err_mm']:.0f} mm, "
                  f"Settled err: {report['settled_err_mm']:.0f} mm")
            print(f"    Tracking: max {report['tracking_max_deg']:.1f} deg "
                  f"(mean {report['tracking_mean_deg']:.1f})")
            print(f"    Joint limit over: max {report['max_joint_limit_over_deg']:.1f} deg")
            if kin_types:
                print(f"    Kinematic penetration (by type):")
                for t, mm in sorted(kin_types.items(), key=lambda x: -x[1]):
                    print(f"      {t}: {mm:.1f} mm")
            if clip_types:
                print(f"    Dynamic obstacle clips ({report['n_clip_frames']} frames):")
                for t, mm in sorted(clip_types.items(), key=lambda x: -x[1]):
                    print(f"      {t}: {mm:.1f} mm")
            else:
                print(f"    No obstacle clips in dynamic rollout")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
