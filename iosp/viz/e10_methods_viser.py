"""Side-by-side physics playback of every E10 method's reconstruction, in one scene.

E10 fits five outer-loop methods (the u=0 random-initialisation baseline,
implicit differentiation, finite differences, CMA-ES and unrolled
differentiation) to eight teleoperated pick-and-place demonstrations, and scores
each fit by ROLLOUT SUCCESS: driving its joint path through contact physics and
asking whether the cube ends up in the bucket.  `summary.json` reports that as a
count.  This script shows it happening.

One MuJoCo model is composed from N copies of the episode's own scene (FR3 +
Franka Hand + that episode's table / cube / bucket), laid out along +y and each
tinted its own hue -- robot and cube share a hue so a failure reads as "the
GREEN arm dropped the GREEN cube on the table".  All copies are driven from the
SAME waypoint schedule (`e10_spasm_sim.execute`'s: `interp` sub-targets per
waypoint, gripper closed at `SKELETON_PICK[0]` and opened at
`SKELETON_PLACE[0]`), so at every instant the arms differ only by the
construction being rolled out and the failures are directly comparable.

The copies are attached with `MjSpec.attach` rather than by string-splicing five
scene XMLs: attach renames joints, actuators, meshes and defaults consistently,
which hand-written prefixing gets wrong the first time the scene changes.  They
sit `--spacing` apart (> the 0.9 m table width) so no copy can ever touch
another -- the physics of each rollout stays exactly the physics
`rollout_success` scored.

Nothing is re-fitted here: joint paths come from the saved
`joint_paths.npz` of a prior `e10_method_comparison` run.

Usage:
    PYTHONPATH=. python -m iosp.viz.e10_methods_viser --episode-index 0
    PYTHONPATH=. python -m iosp.viz.e10_methods_viser \
        --results-dir iosp/data/results/e10_methods_principled --episode-index 9
"""
import argparse
import json
import pathlib
import time

import numpy as np

from iosp.viz import e10_teleop_viser as tv   # also sets FR3_MJCF + sys.path

DEFAULT_RESULTS = (pathlib.Path(__file__).resolve().parents[1]
                   / "data" / "results" / "e10_methods_principled")

# Hue per method: (robot rgb, cube rgb).  The cube is the saturated end of the
# same hue as its arm, so the pairing is readable without a legend while the
# arms stay light enough to see joint geometry against.
METHOD_COLORS = {
    "init":     ((0.62, 0.62, 0.66), (0.35, 0.35, 0.40)),   # grey  -- baseline
    "implicit": ((0.45, 0.80, 0.45), (0.10, 0.60, 0.15)),   # green -- the win
    "fd":       ((0.95, 0.70, 0.35), (0.85, 0.45, 0.05)),   # orange
    "cmaes":    ((0.55, 0.65, 0.95), (0.15, 0.30, 0.85)),   # blue
    "unrolled": ((0.90, 0.55, 0.75), (0.75, 0.10, 0.40)),   # magenta
    "demo":     ((1.00, 0.92, 0.60), (0.95, 0.80, 0.10)),   # gold  -- the human
}
DEMO = "demo"
DEFAULT_ORDER = ("demo", "init", "fd", "unrolled", "cmaes", "implicit")

GRIPPER_FULL_OPEN_M = 0.08
Q_HOME = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8])


# -- loading the saved fits --------------------------------------------------

def load_method_paths(results_dir, wanted):
    """-> ({method: (B, N_FULL, 7)}, episodes, summary).

    `init` (the u=0 baseline) has no row of its own in `joint_paths.npz` -- the
    experiment scores it via `init_rollout` but only saves paths for fitted
    methods.  When some method's `u_hat` is exactly zero its saved path IS the
    u=0 path (this is the usual case: unrolled NaNs out of its first step and
    never leaves the initialisation), so alias it.  Otherwise the caller is told
    to drop `init` rather than shown a silently wrong grey robot.
    """
    results_dir = pathlib.Path(results_dir)
    jp = np.load(results_dir / "joint_paths.npz")
    summary = json.loads((results_dir / "summary.json").read_text())
    u_hats = np.load(results_dir / "u_hats.npz")

    paths, missing = {}, []
    for m in wanted:
        if m in jp.files:
            paths[m] = np.asarray(jp[m], np.float64)
        elif m == "init":
            zero = [k for k in u_hats.files if not np.any(u_hats[k])]
            if zero:
                print(f"  init: aliased to '{zero[0]}' (its u_hat is exactly 0, "
                      f"so its saved path is the u=0 path)")
                paths["init"] = np.asarray(jp[zero[0]], np.float64)
            else:
                missing.append("init (no method sits at u=0; rerun with "
                               "--recompute-init to roll it out here)")
        else:
            missing.append(m)
    for m in missing:
        print(f"  WARNING: no saved path for {m}; dropping it")
    return paths, summary["episodes"], summary


def recompute_init_path(episode_index, n_iters=60):
    """Roll out the u=0 construction for one episode through the forward map.

    The honest fallback when no fitted method landed on u=0.  Costs a JAX build
    of the three-stage map (`e10_spasm_sim` installs the fast stock solver).
    """
    from iosp.viz.e10_spasm_sim import spasm_joint_rollout
    q, _ep = spasm_joint_rollout(episode_index, u=None, n_iters=n_iters)
    return np.asarray(q, np.float64)


# -- composing the N-copy scene ---------------------------------------------

def _single_scene_xml(scene):
    """The one-copy FR3 + hand + this episode's table/cube/bucket, as a path.

    Ground-less: the floor and light belong to the parent, or five coincident
    planes z-fight.
    """
    import tempfile
    from pickplace.scene import write_mjcf
    from remu.sim.scene import build_scene_xml

    frag = write_mjcf(pathlib.Path(tempfile.mkdtemp()) / "pickplace.xml", scene)
    return str(build_scene_xml(extra_object_mjcfs=[frag], add_ground=False))


def build_multi_model(scene, methods, spacing):
    """Compile one model holding `len(methods)` copies of `scene`, offset in y."""
    import mujoco

    child_src = _single_scene_xml(scene)
    ref = mujoco.MjSpec.from_file(child_src)

    parent = mujoco.MjSpec()
    # Solver/compiler settings come from the child, not MjSpec's defaults --
    # attach keeps the PARENT's options, so an empty parent would silently
    # roll these out under a different integrator than the fit was scored with.
    parent.option = ref.option
    parent.compiler = ref.compiler
    parent.visual = ref.visual
    parent.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1],
                               type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    y0 = -0.5 * spacing * (len(methods) - 1)
    offsets = {}
    for i, name in enumerate(methods):
        child = mujoco.MjSpec.from_file(child_src)
        for k in list(child.keys):
            child.delete(k)      # the FR3's 'home' keyframe is nq=9, not nq=N*16
        off = (0.0, y0 + i * spacing, 0.0)
        frame = parent.worldbody.add_frame(pos=list(off))
        parent.attach(child, prefix=f"{name}_", frame=frame)
        offsets[name] = np.array(off)

    parent.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
        rgba=[0.30, 0.32, 0.36, 1.0],
    )
    model = parent.compile()
    return model, mujoco.MjData(model), offsets


def copy_indices(model, name):
    """The per-copy index bundle: arm joints/dofs/qpos, actuators, cube qpos."""
    arm = [model.joint(f"{name}_fr3_joint{k}") for k in range(1, 8)]
    return dict(
        qadr=np.array([j.qposadr[0] for j in arm]),
        dofadr=np.array([j.dofadr[0] for j in arm]),
        act=np.array([model.actuator(f"{name}_fr3_joint{k}").id
                      for k in range(1, 8)]),
        grip_act=model.actuator(f"{name}_fr3_actuator8").id,
        cube_qadr=model.joint(f"{name}_pp_cube_free").qposadr[0],
    )


def tint_copy(model, name, robot_rgb, cube_rgb):
    """Recolour one copy's visible robot and cube geoms.

    Matched on the geom's BODY name, not its own: the menagerie FR3's visual
    mesh geoms are unnamed, so a geom-name match colours the cube and leaves
    every arm the stock white.  Body names are always present and always carry
    the attach prefix, which is the only thing distinguishing the copies.

    Only groups 0-2 (what mjviser shows by default) are touched, so toggling
    the collision groups on still reads as collision geometry.  `matid` is
    cleared because mjviser resolves material colour ahead of geom rgba --
    tinting rgba alone leaves the menagerie's materials in charge.

    The material's own luminance is folded back in as a brightness scale rather
    than flattening the arm to one colour: the FR3's black joint housings and
    white shells stay distinguishable, so a tinted arm still reads as an arm
    instead of a silhouette.
    """
    import mujoco

    for g in range(model.ngeom):
        if model.geom_group[g] > 2:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 model.geom_bodyid[g]) or ""
        if not body.startswith(f"{name}_"):
            continue
        stem = body[len(name) + 1:]
        if stem.startswith("pp_cube"):
            rgb = np.asarray(cube_rgb, float)
        elif stem.startswith("fr3"):
            matid = model.geom_matid[g]
            src = (model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[g])[:3]
            lum = float(np.dot(src, (0.2126, 0.7152, 0.0722)))
            rgb = np.clip(np.asarray(robot_rgb, float) * (0.45 + 1.0 * lum),
                          0.0, 1.0)
        else:
            continue                      # table / bucket keep their own colour
        model.geom_rgba[g, :3] = rgb
        model.geom_matid[g] = -1


def set_grasp_friction(model, torsional=0.1):
    """sandbox.py's measured grasp fix, made prefix-agnostic.

    `sandbox.set_grasp_friction` matches the cube geom by exact name, which no
    longer holds once every copy's geoms carry a method prefix.
    """
    import mujoco

    n = 0
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 model.geom_bodyid[g]) or ""
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "finger" in body or gname.endswith("pp_cube_geom"):
            model.geom_friction[g, 1] = torsional
            n += 1
    return n


def retune_position_gains(model, data, idx, q_ref, damping_ratio=1.5):
    """sandbox.py's gain retune for ONE copy, indexed instead of hardcoded [:7].

    Same derivation (kd = 2*zeta*sqrt(kp*M_ii) at `q_ref`); `sandbox`'s version
    assumes the arm owns qpos[:7] and actuators 0..6, which is true of the
    single-arm scene and of no copy but the first here.
    """
    import mujoco

    saved = data.qpos.copy()
    data.qpos[idx["qadr"]] = q_ref
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)

    for a, d in zip(idx["act"], idx["dofadr"]):
        kp = model.actuator_gainprm[a][0]
        kd = 2.0 * damping_ratio * np.sqrt(kp * max(M[d, d], 1e-6))
        model.actuator_biasprm[a][1] = -kp
        model.actuator_biasprm[a][2] = -kd


# -- the shared waypoint schedule -------------------------------------------

def build_schedule(paths, interp, settle_steps, grasp_dwell, release_dwell,
                   final_settle=400):
    """-> (entries, q_of) for the timeline every copy is driven along.

    `entries` is a list of (kind, row, alpha, hold_steps); `q_of(entry, q_path)`
    turns one into that copy's arm target.  The structure is shared because
    every method's path has the same N_FULL rows and the same skeleton events,
    so a single global step counter indexes all copies -- which is the whole
    point: at any frozen frame the arms differ only by their construction.
    """
    from iosp.model import pickplace as pp

    grasp_row = pp.SKELETON_PICK[0]
    release_row = pp.SKELETON_PLACE[0]
    if not paths:                      # only the recorded demo is being shown
        return [], np.array([0]), grasp_row, release_row
    n_rows = next(iter(paths.values())).shape[0]

    entries, grip = [], 1.0
    for i in range(n_rows):
        for a in np.linspace(0.0, 1.0, interp + 1)[1:]:
            entries.append((i, float(a), grip, settle_steps))
        if i == grasp_row:
            grip = 0.0                                   # close ON the cube
            entries.append((i, 1.0, grip, grasp_dwell))
        elif i == release_row:
            grip = 1.0                                   # open OVER the bucket
            entries.append((i, 1.0, grip, release_dwell))
    entries.append((n_rows - 1, 1.0, grip, final_settle))

    holds = np.array([e[3] for e in entries])
    return entries, np.cumsum(holds), grasp_row, release_row


def entry_q(entry, q_path):
    row, alpha, _grip, _hold = entry
    prev = q_path[max(row - 1, 0)]
    return (1.0 - alpha) * prev + alpha * q_path[row]


# -- runtime ----------------------------------------------------------------

def build_ctx(demo_dir, episode, episode_index, methods, paths, spacing,
              sched_kw, demo_source="recorded"):
    """Everything one episode's playback needs, bundled for a one-dict swap.

    `demo` is the one copy that is not a fit, and `demo_source` picks which
    demonstration it shows:

    "recorded" replays `state.jsonl` at its own recorded rate -- the human's
    actual 15-25 Hz command stream, all ~500 frames of it, driven exactly as
    `e10_teleop_viser` drives it (position targets toward `q_d`, gripper toward
    `gripper_target`, NOT the measured `gripper_width`, which stalls at the
    cube's width during the grasp and replays as a half-open hand).  It is NOT
    resampled onto the methods' waypoint schedule: stretching a recording to
    another clock changes the physics it was recorded under.  The two clocks
    happen to be close anyway (18.6 s of schedule against 15.8-21.6 s of
    recording), so the copies stay roughly in step without any warping, and
    the timeline runs until the slower of the two finishes.

    "skeleton" instead drives the demo copy from `joint_paths.npz["demo"]` --
    the demonstration sampled down to the same N_FULL skeleton rows the fits
    are scored against -- on the identical schedule as every other copy.  That
    is the strictly comparable reference (same clock, same gripper timing, so a
    difference is the construction and nothing else), and it is also the target
    the loss actually sees, but it is not what the human did.
    """
    import mujoco

    scene, q_d, gripper_target, stamp_ns = tv.load_episode(demo_dir, episode)
    model, data, offsets = build_multi_model(scene, methods, spacing)
    set_grasp_friction(model)

    idx = {m: copy_indices(model, m) for m in methods}
    for m in methods:
        tint_copy(model, m, *METHOD_COLORS[m])
        retune_position_gains(model, data, idx[m], Q_HOME)

    replay = DEMO in methods and demo_source == "recorded"
    sched_methods = [m for m in methods if m != DEMO or not replay]
    q_paths = {m: np.asarray(paths[m][episode_index], np.float64)
               for m in sched_methods}
    entries, ends, grasp_row, release_row = build_schedule(q_paths, **sched_kw)

    demo_t = (stamp_ns - stamp_ns[0]) / 1e9
    demo_steps = int(demo_t[-1] / model.opt.timestep) if replay else 0

    yaw = scene.cube_yaw
    ctx = dict(episode=episode, episode_index=episode_index, scene=scene,
               model=model, data=data, offsets=offsets, idx=idx,
               q_paths=q_paths, entries=entries, ends=ends,
               grasp_row=grasp_row, release_row=release_row,
               methods=list(methods), sched_methods=sched_methods,
               replay=replay, demo_source=demo_source,
               demo_q=q_d, demo_grip=gripper_target, demo_t=demo_t,
               demo_steps=demo_steps,
               total_steps=max(int(ends[-1]), demo_steps),
               cube_pos0=np.asarray(scene.cube_spawn_pos()),
               cube_quat0=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
               step_i=0, entry_k=-1, landed={m: None for m in methods})
    reset(ctx)
    return ctx


def reset(ctx):
    import mujoco

    model, data = ctx["model"], ctx["data"]
    mujoco.mj_resetData(model, data)
    for m in ctx["methods"]:
        i = ctx["idx"][m]
        a = i["cube_qadr"]
        data.qpos[a:a + 3] = ctx["cube_pos0"] + ctx["offsets"][m]
        data.qpos[a + 3:a + 7] = ctx["cube_quat0"]
        q0 = (ctx["demo_q"][0] if (m == DEMO and ctx["replay"])
              else ctx["q_paths"][m][0])
        data.qpos[i["qadr"]] = q0
        data.ctrl[i["act"]] = q0
        data.ctrl[i["grip_act"]] = 255.0     # start OPEN (ctrl 0 is a shut hand)
    mujoco.mj_forward(model, data)
    ctx["step_i"] = 0
    ctx["entry_k"] = 0
    if ctx["entries"]:
        apply_entry(ctx, 0)
    if ctx["replay"]:
        drive_demo(ctx)
    ctx["landed"] = {m: None for m in ctx["methods"]}


def apply_entry(ctx, k):
    """Set every schedule-driven copy's actuator targets from entry `k`.

    Skips the demo copy when it is replaying the recording -- that one runs on
    its own clock, from `drive_demo`.
    """
    model, data = ctx["model"], ctx["data"]
    entry = ctx["entries"][k]
    for m in ctx["sched_methods"]:
        i = ctx["idx"][m]
        q = entry_q(entry, ctx["q_paths"][m])
        lo = model.actuator_ctrlrange[i["act"], 0]
        hi = model.actuator_ctrlrange[i["act"], 1]
        data.ctrl[i["act"]] = np.clip(q, lo, hi)
        data.ctrl[i["grip_act"]] = 255.0 * entry[2]


def drive_demo(ctx):
    """Point the demo copy at the recorded frame for the current physics step.

    Frame lookup is against the recording's own timestamps rather than a fixed
    frame stride: the teleop loop ran at a jittery ~27 Hz, and indexing by a
    nominal rate drifts against the real motion by the end of an episode.
    """
    model, data = ctx["model"], ctx["data"]
    i = ctx["idx"][DEMO]
    t = ctx["step_i"] * model.opt.timestep
    f = min(int(np.searchsorted(ctx["demo_t"], t)), len(ctx["demo_q"]) - 1)
    lo = model.actuator_ctrlrange[i["act"], 0]
    hi = model.actuator_ctrlrange[i["act"], 1]
    data.ctrl[i["act"]] = np.clip(ctx["demo_q"][f], lo, hi)
    data.ctrl[i["grip_act"]] = 255.0 * np.clip(
        ctx["demo_grip"][f] / GRIPPER_FULL_OPEN_M, 0.0, 1.0)
    ctx["demo_frame"] = f


def cube_xyz(ctx, m):
    """That copy's cube, back in the single-scene frame the bucket is defined in."""
    a = ctx["idx"][m]["cube_qadr"]
    return np.asarray(ctx["data"].qpos[a:a + 3]) - ctx["offsets"][m]


def in_bucket(scene, xyz):
    cx, cy = scene.bucket_center_xy
    dxy = float(np.hypot(xyz[0] - cx, xyz[1] - cy))
    floor_top = scene.table_top_z + scene.bucket_floor_thickness
    rim_top = floor_top + scene.bucket_wall_height
    z = float(xyz[2])
    inside = (dxy <= scene.bucket_inner_radius
              and floor_top - scene.cube_half_extent <= z <= rim_top + scene.cube_half_extent)
    return inside, dxy, z - floor_top


def advance(ctx, n_steps):
    """Step physics `n_steps`, re-applying targets as the schedule crosses entries."""
    import mujoco

    model, data = ctx["model"], ctx["data"]
    ends, total = ctx["ends"], ctx["total_steps"]
    for _ in range(int(n_steps)):
        if ctx["step_i"] >= total:
            return True
        k = min(int(np.searchsorted(ends, ctx["step_i"], side="right")),
                len(ctx["entries"]) - 1)
        if ctx["entries"] and k != ctx["entry_k"]:
            ctx["entry_k"] = k
            apply_entry(ctx, k)
        if ctx["replay"]:
            drive_demo(ctx)
        mujoco.mj_step(model, data)
        ctx["step_i"] += 1
        for m in ctx["methods"]:
            if ctx["landed"][m] is None:
                inside, _dxy, _dz = in_bucket(ctx["scene"], cube_xyz(ctx, m))
                if inside:
                    ctx["landed"][m] = ctx["step_i"]
    return ctx["step_i"] >= total


def status_markdown(ctx, summary):
    """Live per-method readout: colour, cube state, and the run's own verdict."""
    ep_i = ctx["episode_index"]
    n_fit = summary["n_fit"]
    tag = "fit" if ep_i < n_fit else "HELD OUT"
    lines = [f"**{ctx['episode']}**  (episode {ep_i}, {tag})",
             "",
             "| | method | cube | run verdict |",
             "|---|---|---|---|"]
    for m in ctx["methods"]:
        _robot, cube = METHOD_COLORS[m]
        swatch = "#%02x%02x%02x" % tuple(int(255 * c) for c in cube)
        inside, dxy, dz = in_bucket(ctx["scene"], cube_xyz(ctx, m))
        live = ("IN BUCKET" if inside
                else f"{dxy * 1000:.0f} mm out, {dz * 1000:+.0f} mm")
        if m == DEMO:
            label = ("demo (recorded)" if ctx["replay"]
                     else "demo (skeleton)")
            rec = "human"
        else:
            label = m
            roll = (summary["methods"].get(m, {}) or {}).get("rollout")
            if m == "init":
                roll = summary.get("init_rollout")
            rec = "success" if roll and roll["per_episode"][ep_i] else (
                "fail" if roll else "--")
        lines.append(
            f'| <span style="color:{swatch}">&#9632;</span> | **{label}** | {live} | {rec} |')
    lines += ["", f"step {ctx['step_i']}/{ctx['total_steps']}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo-dir", type=str, default=str(tv.DEFAULT_DEMO_DIR))
    ap.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS))
    ap.add_argument("--episode-index", type=int, default=0,
                    help="0-9; indices >= summary['n_fit'] are the held-out demos")
    ap.add_argument("--methods", type=str, default=",".join(DEFAULT_ORDER),
                    help="comma-separated, left to right along +y")
    ap.add_argument("--demo-source", choices=("recorded", "skeleton"),
                    default="recorded",
                    help="how the 'demo' copy is driven: 'recorded' replays "
                         "state.jsonl at its own rate (what the human did); "
                         "'skeleton' drives the N_FULL-row demo path on the "
                         "same schedule as the fits (what the loss sees)")
    ap.add_argument("--recompute-init", action="store_true",
                    help="roll out the u=0 baseline through the forward map "
                         "instead of aliasing a method whose u_hat is 0")
    ap.add_argument("--spacing", type=float, default=1.2,
                    help="metres between copies (table is 0.9 m across)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--settle-steps", type=int, default=60)
    ap.add_argument("--interp", type=int, default=6)
    ap.add_argument("--grasp-dwell", type=int, default=400)
    ap.add_argument("--release-dwell", type=int, default=200)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_COLORS]
    if unknown:
        raise ValueError(f"no colour assigned for {unknown}; "
                         f"known: {sorted(METHOD_COLORS)}")

    print(f"Loading fits from {args.results_dir}", flush=True)
    paths, episodes, summary = load_method_paths(args.results_dir, methods)
    methods = [m for m in methods if m in paths or (m == "init" and args.recompute_init)]
    if not methods:
        raise SystemExit("no method paths to show")

    ep_names = tv.find_episodes(args.demo_dir)
    if len(ep_names) < len(episodes):
        raise SystemExit(f"{args.demo_dir} has {len(ep_names)} episodes, the run "
                         f"fitted {len(episodes)}")
    if not 0 <= args.episode_index < len(episodes):
        raise SystemExit(f"--episode-index must be in [0, {len(episodes) - 1}]")

    def ensure_init(ep_i):
        """Fill in `init` for episode `ep_i` when it is being recomputed.

        Done per episode rather than once: `spasm_joint_rollout` builds the
        forward map over episodes 0..ep_i and reads out that episode's own row,
        so one episode's u=0 path is not another's.
        """
        if not args.recompute_init or "init" not in methods:
            return
        cache = ensure_init.cache
        if ep_i not in cache:
            print(f"Rolling out the u=0 baseline for episode {ep_i}...", flush=True)
            cache[ep_i] = recompute_init_path(ep_i)
        arr = paths.get("init")
        if arr is None:
            arr = np.zeros((len(episodes),) + cache[ep_i].shape)
            paths["init"] = arr
        arr[ep_i] = cache[ep_i]
    ensure_init.cache = {}

    sched_kw = dict(interp=args.interp, settle_steps=args.settle_steps,
                    grasp_dwell=args.grasp_dwell, release_dwell=args.release_dwell)

    ensure_init(args.episode_index)
    print(f"Composing {len(methods)} copies of the scene "
          f"({', '.join(methods)}) at {args.spacing} m spacing...", flush=True)
    ctx = build_ctx(args.demo_dir, episodes[args.episode_index],
                    args.episode_index, methods, paths, args.spacing, sched_kw,
                    demo_source=args.demo_source)
    print(f"  nq={ctx['model'].nq}  nu={ctx['model'].nu}  "
          f"{ctx['total_steps']} physics steps per rollout", flush=True)
    if ctx["replay"]:
        print(f"  demo: replaying {len(ctx['demo_q'])} recorded frames "
              f"({ctx['demo_t'][-1]:.1f}s) against "
              f"{ctx['ends'][-1] * ctx['model'].opt.timestep:.1f}s of schedule",
              flush=True)

    from remu.viewer.viser_viewer import ViserViewer

    viewer = ViserViewer(ctx["model"], ctx["data"], port=args.port)
    print(f"Viser server at http://localhost:{args.port}", flush=True)
    server = viewer.server

    labels = [f"{i}: {n}" + ("" if i < summary["n_fit"] else "  (held out)")
              for i, n in enumerate(episodes)]
    ep_dropdown = server.gui.add_dropdown("Demonstration", labels,
                                          initial_value=labels[args.episode_index])
    play_toggle = server.gui.add_checkbox("Play", initial_value=True)
    loop_toggle = server.gui.add_checkbox("Loop", initial_value=True)
    speed_slider = server.gui.add_slider("Speed", min=0.1, max=8.0, step=0.1,
                                         initial_value=args.speed)
    restart_btn = server.gui.add_button("Restart rollout")
    status = server.gui.add_markdown(status_markdown(ctx, summary))

    def switch_episode(_=None):
        nonlocal ctx
        import mjviser
        i = labels.index(ep_dropdown.value)
        ensure_init(i)
        ctx = build_ctx(args.demo_dir, episodes[i], i, methods, paths,
                        args.spacing, sched_kw, demo_source=args.demo_source)
        viewer.scene = mjviser.ViserMujocoScene(server, ctx["model"], num_envs=1)
        viewer.model, viewer.data = ctx["model"], ctx["data"]
        viewer.sync(ctx["model"], ctx["data"])

    ep_dropdown.on_update(switch_episode)
    restart_btn.on_click(lambda _=None: (reset(ctx),
                                         viewer.sync(ctx["model"], ctx["data"])))

    # Wall-clock paced, like e10_teleop_viser: `speed` x realtime worth of
    # physics per frame, bounded so a stall does not burst-step the whole
    # remaining rollout into one frame.
    dt_model = ctx["model"].opt.timestep
    prev_wall = time.time()
    last_status = 0.0
    try:
        while True:
            now = time.time()
            dt = min(now - prev_wall, 0.1)
            prev_wall = now
            if play_toggle.value:
                n = int(round(dt * speed_slider.value / dt_model))
                done = advance(ctx, n) if n else False
                viewer.sync(ctx["model"], ctx["data"])
                if done:
                    if loop_toggle.value:
                        reset(ctx)
                    else:
                        play_toggle.value = False
            if now - last_status > 0.25:
                status.content = status_markdown(ctx, summary)
                last_status = now
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
