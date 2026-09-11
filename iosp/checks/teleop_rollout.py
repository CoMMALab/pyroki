"""Execute pick-and-place plans in the teleop scene and score the task.

The counterpart of `iosp.checks.spasm_trajopt` for the third domain.  Scenes
come from `iosp.model.teleop_task` (sim_teleop's own geometry and
randomisation), the demonstration is synthetic rather than recorded, and success
is the geometric task test: the cube ends up in the bucket.

The plan here is the SKELETON -- IK at the grasp and release poses with a lift
between them, the same shape as SPaSM's `q_traj_init` -- so the harness can be
exercised and the scene validated before the IOSP composed planner is wired in
behind it.  Swapping the planner in changes `plan_skeleton` and nothing else.

    python -m iosp.checks.teleop_rollout --n-scenes 6
"""
from __future__ import annotations

import argparse

import numpy as np

from iosp.model import teleop_task as T
from iosp.viz import mj_rollout as R


def _dls_ik(world, target_pos, target_yaw, q0, iters=300, damping=0.08,
            pos_w=1.0, rot_w=0.35):
    """Damped least squares IK to a position plus a down-and-yaw orientation.

    Small and self-contained on purpose: this generates the skeleton the
    rollout executes, so it should depend on the MuJoCo model being simulated
    rather than on a second kinematics stack that could disagree with it.
    """
    import mujoco

    m, d = world.model, world.data
    bid = world.ee_body[0]
    off = np.asarray(world._ee_offset, float)
    adr = world.arm_adr[0]
    dofs = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in world._arm_joint_names])
    lo, hi = world.joint_limits()

    # Desired rotation: gripper pointing down, rotated by the object's yaw.
    cz, sz = np.cos(target_yaw), np.sin(target_yaw)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    R_des = Rz @ np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

    q = np.array(q0, float)
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        world.set_arm(0, q)
        world.forward()
        Rc = d.xmat[bid].reshape(3, 3)
        cur = d.xpos[bid] + Rc @ off
        e_pos = np.asarray(target_pos, float) - cur
        # Orientation error as a rotation vector.
        Re = R_des @ Rc.T
        w = np.empty(4)
        mujoco.mju_mat2Quat(w, Re.ravel())
        ang = 2.0 * np.arctan2(np.linalg.norm(w[1:]), w[0])
        axis = w[1:] / (np.linalg.norm(w[1:]) + 1e-12)
        e_rot = axis * ang

        mujoco.mj_jac(m, d, jacp, jacr, cur, bid)
        J = np.vstack([pos_w * jacp[:, dofs], rot_w * jacr[:, dofs]])
        e = np.concatenate([pos_w * e_pos, rot_w * e_rot])
        if np.linalg.norm(e_pos) < 5e-4 and abs(ang) < 5e-3:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        q = np.clip(q + dq, lo, hi)
    world.set_arm(0, q)
    world.forward()
    Rc = d.xmat[bid].reshape(3, 3)
    err = np.linalg.norm(np.asarray(target_pos, float)
                         - (d.xpos[bid] + Rc @ off))
    return q, float(err)


def plan_skeleton(world, scene, n_interp=6):
    """-> (q rows, [(grasp_row, release_row, body)], diagnostics).

    home -> above the cube -> grasp -> lift -> above the bucket -> release ->
    home.  Approach and retreat are VERTICAL over the cube so the gripper comes
    straight down onto it rather than sweeping in from the side.
    """
    sk = T.skeleton(scene)
    q_home = T._home_q()
    lift = 0.12

    up = np.array([0.0, 0.0, lift])
    waypoints = [
        ("pre_pick", sk["pick_pos"] + up, sk["yaw"]),
        ("pick", sk["pick_pos"], sk["yaw"]),
        ("lift", sk["pick_pos"] + up, sk["yaw"]),
        ("pre_place", sk["place_pos"] + up, sk["yaw"]),
        ("place", sk["place_pos"], sk["yaw"]),
        ("retreat", sk["place_pos"] + up, sk["yaw"]),
    ]

    qs, errs, q_ref = [q_home], {}, q_home
    for name, pos, yaw in waypoints:
        q_ref, e = _dls_ik(world, pos, float(yaw), q_ref)
        errs[name] = e * 1000
        qs.append(q_ref.copy())
    qs.append(q_home)

    key = np.stack(qs)                       # (8, 7)
    rows, key_row = [], []
    for i in range(len(key) - 1):
        seg = np.linspace(key[i], key[i + 1], n_interp + 1)
        key_row.append(len(rows))
        rows.extend(seg[:-1])
    key_row.append(len(rows))
    rows.append(key[-1])
    q = np.stack(rows)

    grasp_row = key_row[2]      # the "pick" waypoint
    release_row = key_row[5]    # the "place" waypoint
    return q, [(grasp_row, release_row, T.CUBE_BODY)], errs


def run_scene(scene, settle=1.5, dwell=0.6, progress=False):
    world = T.build_world(scene)
    q, events, ik_err = plan_skeleton(world, scene)

    cube_before = world.body_xyzyaw(T.CUBE_BODY)[:3].copy()
    ro = R.run_events(world, q, events, settle=settle, dwell=dwell,
                      progress=progress, hand_body=T.EE_BODY,
                      grasp_half_width=scene.cube_half_extent)

    # How far the cube moved before the gripper closed -- the check that the
    # approach did not shove it (see `teleop_task.build_world`).
    gf = min(ro.grasp_frame, ro.n_frames - 1)
    world.data.qpos[:] = ro.qpos[gf]
    world.forward()
    shove = float(np.linalg.norm(world.body_xyzyaw(T.CUBE_BODY)[:3] - cube_before))

    world.data.qpos[:] = ro.qpos[-1]
    world.forward()
    cube = world.body_xyzyaw(T.CUBE_BODY)
    tilt = world.body_tilt_deg(T.CUBE_BODY)
    ok, rep = T.place_success(cube[:3], scene, tilt)
    worst, mean = ro.tracking_mm_deg(0)
    return dict(ok=ok, rep=rep, cube=cube, tilt=tilt, shove_mm=shove * 1000,
                ik_err_mm=ik_err, tracking=(worst, mean), rollout=ro,
                world=world, q=q, events=events)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-scenes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--dwell", type=float, default=0.6)
    ap.add_argument("--save", default=None)
    ap.add_argument("--viser", action="store_true",
                    help="serve scene 0's rollout in the browser (mjviser)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scenes = T.sample_scenes(rng, args.n_scenes)
    n_ok, rows = 0, []
    print(f"\n===== teleop pick-and-place, EXECUTED ({args.n_scenes} scenes) =====")
    for i, sc in enumerate(scenes):
        r = run_scene(sc, args.settle, args.dwell)
        n_ok += r["ok"]
        rep = r["rep"]
        why = []
        if not rep["inside"]:
            why.append("outside bucket")
        if not rep["seated"]:
            why.append("not seated")
        if not rep["upright"]:
            why.append("tipped")
        print(f"  scene {i}: {'SUCCESS' if r['ok'] else '   FAIL'}  "
              f"radial {rep['radial_mm']:5.1f} mm "
              f"(clearance {rep['clearance_mm']:+6.1f}), "
              f"dz {rep['dz_mm']:+6.1f} mm, tilt {r['tilt']:4.1f} deg"
              + (f"   [{', '.join(why)}]" if why else ""))
        print(f"            IK err mm {({k: round(v, 1) for k, v in r['ik_err_mm'].items()})}"
              f"  pre-grasp shove {r['shove_mm']:.1f} mm"
              f"  tracking max {r['tracking'][0]:.1f} deg")
        rows.append(r)
    print(f"  ---> {n_ok}/{len(scenes)} succeeded\n")

    if args.viser:
        serve(rows[0], scenes[0], port=args.port, fps=args.fps)
    if args.save:
        r0 = rows[0]
        np.savez(args.save, q=r0["q"], qpos=r0["rollout"].qpos,
                 frame_row=r0["rollout"].frame_row, dt=r0["rollout"].dt)
        print(f"  wrote {args.save}")
    return n_ok, len(scenes)




def serve(result, scene, port=8080, fps=60.0):
    """Play a recorded rollout in the browser, like `tetris_viser --dynamic`.

    The rollout is replayed from its stored `qpos`, not re-integrated, so
    scrubbing the slider is instant and shows exactly the execution that was
    scored.
    """
    from iosp.viz import mj_rollout as R
    from iosp.viz import mj_scene as M

    world, ro = result["world"], result["rollout"]
    rep, ok = result["rep"], result["ok"]
    set_row = R.replayer(world, ro)

    verdict = (("<span style='color:#2ab05e'>**SUCCESS**</span>" if ok
                else "<span style='color:#d43d3d'>**FAIL**</span>")
               + f" &nbsp; cube {rep['radial_mm']:.1f} mm from the bucket axis "
                 f"(clearance {rep['clearance_mm']:+.1f} mm), "
                 f"dz {rep['dz_mm']:+.1f} mm, tilt {result['tilt']:.1f}&deg;"
               + f"\n\n<sub>in bucket {rep['inside']}, seated {rep['seated']}, "
                 f"upright {rep['upright']} &nbsp;|&nbsp; pre-grasp shove "
                 f"{result['shove_mm']:.1f} mm</sub>")

    gf, rf = ro.grasp_frame, ro.release_frame

    def info(t):
        phase = ("approach" if t < gf else
                 "carry" if t <= rf else "release / settle")
        return f"phase **{phase}** &nbsp;|&nbsp; t={t * ro.dt:.2f}s"

    print(f"[teleop_rollout] serving {ro.n_frames} frames on :{port}")
    M.play(world, ro.n_frames, set_row, port=port, fps=fps,
           info_fn=info, verdict_md=verdict)


if __name__ == "__main__":
    main()
