"""Browser playback of a recorded E10 teleop episode, in the real scene.

Recreates `sim_teleop/sandbox.py`'s setup (FR3 + Franka Hand + the episode's
own table/cube/bucket, via `remu.sim.scene.build_scene_xml` +
`pickplace.scene.write_mjcf`, viewed through `remu.viewer.viser_viewer.
ViserViewer` / mjviser) instead of hand-rolling geometry -- that module is
already the proven correspondence between the recorded `state.jsonl` and a
compiled MuJoCo model, including the exact mesh set, and a parallel
implementation here would drift from it the first time either changes.

Playback drives the FR3's own position actuators toward the recorded `q_d`
(and the gripper tendon toward `gripper_target`) and STEPS PHYSICS, exactly
like `sandbox.py`'s `tick()` -- mjviser then just renders the resulting
rollout.  The cube is picked up by real contact when the fingers close on it,
not by a hand-written "attach to the hand frame while the gripper is shut"
kinematic hack: that was tried first and is strictly worse than the thing
`sandbox.py` already solved (friction tuning, gain retuning, contact).

Usage:
    python iosp/viz/e10_teleop_viser.py [--episode ep_20260902_093953]
                                        [--demo-dir sim_teleop/data/demos]
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

TELEOP_ROOT = pathlib.Path(
    os.environ.get("IOSP_TELEOP_ROOT",
                    pathlib.Path(__file__).resolve().parents[2].parent / "sim_teleop")
)
DEFAULT_DEMO_DIR = TELEOP_ROOT / "data" / "demos"

# A local menagerie checkout, so `remu.sim.scene.default_fr3_mjcf()` doesn't
# need to `git clone` (which fails offline / behind this box's proxy). Only
# set if the caller hasn't already pointed FR3_MJCF elsewhere.
_LOCAL_MENAGERIE_FR3 = pathlib.Path(
    "/home/sadmin/Work/mujoco/mujoco_menagerie/franka_fr3/fr3.xml")
if "FR3_MJCF" not in os.environ and _LOCAL_MENAGERIE_FR3.exists():
    os.environ["FR3_MJCF"] = str(_LOCAL_MENAGERIE_FR3)

for p in (TELEOP_ROOT, TELEOP_ROOT / "remu" / "src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

GRIPPER_FULL_OPEN_M = 0.08  # Franka Hand max width; sandbox.py's ctrl convention


def find_episodes(demo_dir):
    return sorted(d.name for d in pathlib.Path(demo_dir).iterdir()
                  if (d / "state.jsonl").exists() and (d / "factors.json").exists())


def load_episode(demo_dir, name):
    """-> (PickPlaceScene, q_d (N,7), gripper_target (N,), stamp_ns (N,))."""
    from pickplace.scene import PickPlaceScene

    ep_dir = pathlib.Path(demo_dir) / name
    factors = json.loads((ep_dir / "factors.json").read_text())
    scene = PickPlaceScene.from_factors(factors)

    rows = [json.loads(line) for line in (ep_dir / "state.jsonl").read_text().splitlines()
            if line.strip()]
    q_d = np.array([r["q_d"] for r in rows], dtype=np.float64)
    # The COMMAND channel, not the measured one.  `gripper_width` is
    # `qpos[7]+qpos[8]` sampled after the step, so once the fingers are on the
    # cube it stalls at the cube's width (~0.03 m) while the operator was
    # commanding 0.  Replaying the measurement therefore re-commands a
    # PARTIALLY OPEN hand exactly during the grasp (ctrl ~93 instead of 0),
    # the fingers never load the cube, and the lift comes up empty -- which
    # reads as a friction/gain mismatch but is a channel mix-up.
    # `gripper_target` is what record.py's tick() sent: ctrl = 255 * target/0.08.
    gripper_target = np.array([r["gripper_target"] for r in rows], dtype=np.float64)
    stamp_ns = np.array([r["stamp_ns"] for r in rows], dtype=np.int64)
    return scene, q_d, gripper_target, stamp_ns


def build_ctx(demo_dir, episode):
    """Everything the playback loop needs for one episode, bundled so
    switching episodes is one dict swap instead of a pile of `nonlocal`s.

    Builds and configures the model exactly as `sandbox.py.main()` does:
    grasp-friction retuned (the sandbox's measured fix for the cube slipping
    31mm through the fingers at MuJoCo's default torsional friction) and the
    position-actuator gains retuned against the episode's own start pose (the
    model's stock gains are 2-8x overdamped, which is what makes tracking
    look like it is moving through oil).
    """
    import mujoco
    import sandbox as sb  # sim_teleop/sandbox.py

    scene, q_d, gripper_target, stamp_ns = load_episode(demo_dir, episode)
    model, data = sb.build_model(scene)
    sb.set_grasp_friction(model)

    # `write_mjcf` bakes the cube's spawn *position* into the compiled model
    # but not its recorded yaw -- sandbox.py's own `randomize()` writes that
    # separately (qpos[adr+3:adr+7] from `scene.cube_yaw`). Skipping it leaves
    # the cube unrotated while the recorded grasp targeted its true ~18-25 deg
    # yaw, so the fingers close on a corner instead of two parallel faces and
    # it slips out -- reads exactly like a friction bug but isn't one.
    from pickplace.scene import CUBE_FREEJOINT

    cube_adr = model.joint(CUBE_FREEJOINT).qposadr[0]
    yaw = scene.cube_yaw
    data.qpos[cube_adr : cube_adr + 3] = scene.cube_spawn_pos()
    data.qpos[cube_adr + 3 : cube_adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))

    data.qpos[:7] = q_d[0]
    data.ctrl[:7] = q_d[0]
    data.ctrl[7] = 255.0 * np.clip(gripper_target[0] / GRIPPER_FULL_OPEN_M, 0.0, 1.0)
    mujoco.mj_forward(model, data)
    # Reference pose is Q_HOME, matching record.py's own retune call -- kd is
    # derived from diag(M) at that pose, so tuning against the episode's start
    # pose gives different damping on every episode and none of them the
    # damping the demonstration was recorded under.
    sb.retune_position_gains(model, data, sb.Q_HOME,
                             damping_ratio=1.5, stiffness_scale=1.0)

    return {
        "episode": episode, "scene": scene, "q_d": q_d, "gripper_target": gripper_target,
        "t_s": (stamp_ns - stamp_ns[0]) / 1e9, "n": len(q_d),
        "model": model, "data": data, "sim_carry": 0.0, "cube_adr": cube_adr,
        "cube_pos0": np.asarray(scene.cube_spawn_pos()),
        "cube_quat0": np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
    }


def drive_to(ctx, i):
    """Set the actuator targets for recorded frame `i` (does not step)."""
    import sandbox as sb

    data = ctx["data"]
    data.ctrl[:7] = sb.clamp_to_limits(ctx["model"], ctx["q_d"][i])
    frac_open = np.clip(ctx["gripper_target"][i] / GRIPPER_FULL_OPEN_M, 0.0, 1.0)
    data.ctrl[7] = 255.0 * frac_open


def step(ctx, dt):
    """Advance physics by `dt` seconds of sim time, catching up like
    `sandbox.py`'s `sim_carry` (bounded, so a stall doesn't burst-step)."""
    import mujoco

    model, data = ctx["model"], ctx["data"]
    max_catchup = max(4.0 * model.opt.timestep, 0.05)
    ctx["sim_carry"] = min(ctx["sim_carry"] + dt, max_catchup)
    while ctx["sim_carry"] >= model.opt.timestep:
        mujoco.mj_step(model, data)
        ctx["sim_carry"] -= model.opt.timestep


def reset(ctx):
    import mujoco

    model, data = ctx["model"], ctx["data"]
    mujoco.mj_resetData(model, data)
    cube_adr = ctx["cube_adr"]
    data.qpos[cube_adr : cube_adr + 3] = ctx["cube_pos0"]
    data.qpos[cube_adr + 3 : cube_adr + 7] = ctx["cube_quat0"]
    data.qpos[:7] = ctx["q_d"][0]
    data.ctrl[:7] = ctx["q_d"][0]
    # mj_resetData zeros ctrl, and ctrl[7]=0 is a CLOSED hand; leaving it there
    # slams the fingers shut for the frames before the first drive_to().
    data.ctrl[7] = 255.0 * np.clip(ctx["gripper_target"][0] / GRIPPER_FULL_OPEN_M, 0.0, 1.0)
    mujoco.mj_forward(model, data)
    ctx["sim_carry"] = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-dir", type=str, default=str(DEFAULT_DEMO_DIR))
    parser.add_argument("--episode", type=str, default=None,
                        help="episode name; defaults to the first one found")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    episodes = find_episodes(args.demo_dir)
    if not episodes:
        raise FileNotFoundError(f"no episodes with state.jsonl + factors.json in {args.demo_dir}")
    episode = args.episode or episodes[0]
    if episode not in episodes:
        raise ValueError(f"{episode!r} not found in {args.demo_dir}; have {episodes}")

    print(f"Loading episode {episode}...", flush=True)
    print("Building scene (FR3 + hand + table/cube/bucket)...", flush=True)
    ctx = build_ctx(args.demo_dir, episode)

    from remu.viewer.viser_viewer import ViserViewer

    viewer = ViserViewer(ctx["model"], ctx["data"], port=args.port)
    print(f"Viser server at http://localhost:{args.port}", flush=True)

    server = viewer.server
    ep_dropdown = server.gui.add_dropdown("Episode", episodes, initial_value=episode)
    progress = server.gui.add_slider("Frame (progress)", min=0, max=ctx["n"] - 1, step=1,
                                     initial_value=0, disabled=True)
    play_toggle = server.gui.add_checkbox("Play", initial_value=True)
    loop_toggle = server.gui.add_checkbox("Loop", initial_value=True)
    speed_slider = server.gui.add_slider("Speed", min=0.1, max=4.0, step=0.1,
                                         initial_value=args.speed)
    restart_btn = server.gui.add_button("Restart episode")

    play_t = 0.0

    def switch_episode(_=None):
        nonlocal ctx, play_t
        ctx = build_ctx(args.demo_dir, ep_dropdown.value)
        play_t = 0.0
        progress.max = ctx["n"] - 1
        progress.value = 0
        import mjviser
        viewer.scene = mjviser.ViserMujocoScene(server, ctx["model"], num_envs=1)
        viewer.model, viewer.data = ctx["model"], ctx["data"]
        viewer.sync(ctx["model"], ctx["data"])

    def do_restart(_=None):
        nonlocal play_t
        reset(ctx)
        play_t = 0.0
        viewer.sync(ctx["model"], ctx["data"])

    ep_dropdown.on_update(switch_episode)
    restart_btn.on_click(do_restart)

    print(f"{ctx['n']} frames, {ctx['t_s'][-1]:.1f}s recorded", flush=True)

    # Playback loop: advance by wall-clock time against the recording's own
    # timestamps (not a fixed frame step), same reasoning as sandbox.py's
    # sim_carry -- otherwise per-tick jitter reads as speed drift.
    prev_wall = time.time()
    try:
        while True:
            now = time.time()
            dt = now - prev_wall
            prev_wall = now
            if play_toggle.value:
                play_t += dt * speed_slider.value
                t_end = ctx["t_s"][-1]
                if play_t > t_end:
                    if loop_toggle.value:
                        reset(ctx)
                        play_t = 0.0
                    else:
                        play_t = t_end
                        play_toggle.value = False
                i = int(np.searchsorted(ctx["t_s"], play_t))
                i = min(max(i, 0), ctx["n"] - 1)
                progress.value = i
                drive_to(ctx, i)
                step(ctx, dt * speed_slider.value)
                viewer.sync(ctx["model"], ctx["data"])
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
