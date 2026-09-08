"""Support utilities for bench_ik.py: GPU monitoring, CSV/table row builders,
and the self-contained cuRobo baseline child.

Kept JAX-free at module level (only numpy + stdlib) so bench_ik.py can import
it unconditionally, even under ``_IS_CUROBO_CHILD`` where JAX/pyroffi are not
installed. The cuRobo functions defer their torch/curobo imports inside the
functions themselves, same discipline as bench_ik.py's cuRobo section.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import datetime
import json
import os
import pathlib
import shutil
import sys
import threading
import xml.etree.ElementTree as ET

import numpy as np

# Success thresholds shared by every method's scoring (see bench_ik.py's
# module docstring for the rationale).
POS_THR_M   = 1e-3
ROT_THR_RAD = 0.05

# ---------------------------------------------------------------------------
# GPU monitoring (NVML)
# ---------------------------------------------------------------------------

# NVML indices are PHYSICAL, so the handle must track CUDA_VISIBLE_DEVICES —
# otherwise a child pinned to GPU k would report GPU 0's util/VRAM.
try:
    if "--cpu-only" in sys.argv[1:]:
        raise RuntimeError("CPU-only mode")
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _physical_gpu_idx = int(_cuda_visible.split(",")[0].strip()) if _cuda_visible.strip() else 0
    _NVML_HANDLE: object | None = _pynvml.nvmlDeviceGetHandleByIndex(_physical_gpu_idx)
    _NVML_OK = True
except Exception:
    _NVML_HANDLE = None
    _NVML_OK = False


@contextlib.contextmanager
def _gpu_monitor(interval_s: float = 0.02):
    """Sample GPU utilisation and VRAM in a background thread.

    Yields a dict; on exit it contains ``gpu_util`` (util % samples) and
    ``vram_mb`` (used VRAM MiB samples).
    """
    samples: dict[str, list[float]] = {"gpu_util": [], "vram_mb": []}
    stop_evt = threading.Event()

    def _sample() -> None:
        while not stop_evt.is_set():
            if _NVML_OK and _NVML_HANDLE is not None:
                util = _pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                mem  = _pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                samples["gpu_util"].append(float(util.gpu))
                samples["vram_mb"].append(float(mem.used) / 1024 ** 2)
            stop_evt.wait(interval_s)

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    try:
        yield samples
    finally:
        stop_evt.set()
        t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

_COL_W = 20  # method name column width
_NUM_W = 10  # numeric column width

def _table_header(cols: list[str]) -> str:
    row = f"  {'Method':<{_COL_W}}"
    for c in cols:
        row += f"  {c:>{_NUM_W}}"
    return row

def _table_sep(n_cols: int) -> str:
    return "  " + "-" * (_COL_W + n_cols * (_NUM_W + 2))

def _table_row(label: str, vals: list[str]) -> str:
    row = f"  {label:<{_COL_W}}"
    for v in vals:
        row += f"  {v:>{_NUM_W}}"
    return row


def _seq_row(label: str, results: list["SolveResult"]) -> tuple[str, dict]:
    pos    = np.array([r.pos_err * 1e3 for r in results])
    rot    = np.array([r.rot_err       for r in results])
    t      = np.array([r.time_ms       for r in results])
    solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
    n      = len(results)
    vals = [
        f"{np.median(t):.3f}",
        f"{np.percentile(t, 95):.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
    ]
    return _table_row(label, vals), {"t_med": float(np.median(t))}


def _seq_row_coll(
    label: str, results: list["SolveResult"], coll_free: int,
) -> tuple[str, dict]:
    """Like _seq_row but with an extra coll_free column."""
    pos    = np.array([r.pos_err * 1e3 for r in results])
    rot    = np.array([r.rot_err       for r in results])
    t      = np.array([r.time_ms       for r in results])
    solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
    n      = len(results)
    vals = [
        f"{np.median(t):.3f}",
        f"{np.percentile(t, 95):.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
        f"{coll_free}/{n}",
    ]
    return _table_row(label, vals), {"t_med": float(np.median(t))}


def _batch_row(label: str, result: "BatchResult") -> tuple[str, dict]:
    pos    = result.pos_errs * 1e3
    rot    = result.rot_errs
    solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))
    n      = len(pos)

    def _fmt_pct(v: float) -> str:
        return f"{v:.0f}%" if not np.isnan(v) else "n/a"

    def _fmt_mb(v: float) -> str:
        return f"{v:.0f}" if not np.isnan(v) else "n/a"

    vals = [
        f"{result.time_ms:.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
        _fmt_pct(result.peak_gpu_util),
        _fmt_pct(result.avg_gpu_util),
        _fmt_mb(result.peak_vram_mb),
    ]
    return _table_row(label, vals), {}


def _batch_row_coll(
    label: str, result: "BatchResult", coll_free: int,
) -> tuple[str, dict]:
    """Like _batch_row but with an extra coll_free column."""
    pos    = result.pos_errs * 1e3
    rot    = result.rot_errs
    solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))
    n      = len(pos)

    def _fmt_pct(v: float) -> str:
        return f"{v:.0f}%" if not np.isnan(v) else "n/a"

    def _fmt_mb(v: float) -> str:
        return f"{v:.0f}" if not np.isnan(v) else "n/a"

    vals = [
        f"{result.time_ms:.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
        f"{coll_free}/{n}",
        _fmt_pct(result.peak_gpu_util),
        _fmt_pct(result.avg_gpu_util),
        _fmt_mb(result.peak_vram_mb),
    ]
    return _table_row(label, vals), {}


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "timestamp", "robot", "mode", "solver", "collision_free",
    "n_problems", "n_timed",
    "t_med_ms", "t_p95_ms",
    "pos_med_mm", "pos_p95_mm",
    "rot_med_rad", "rot_p95_rad",
    "success_n", "success_total",
    "coll_free_n",
    "peak_gpu_pct", "avg_gpu_pct", "peak_vram_mb",
]


def _write_csv(
    path: pathlib.Path,
    timestamp: str,
    robot_name: str,
    seq_results: dict[str, list["SolveResult"]],
    batch_results: dict[str, "BatchResult"],
    seq_coll_results: dict[str, list["SolveResult"]],
    batch_coll_results: dict[str, "BatchResult"],
    seq_coll_free: dict[str, int],
    batch_coll_free: dict[str, int],
    n_timed: int,
) -> None:
    """Append all benchmark results to *path* as CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    rows: list[dict] = []

    # -- Sequential (no collision) ------------------------------------------
    for solver, results in seq_results.items():
        pos = np.array([r.pos_err * 1e3 for r in results])
        rot = np.array([r.rot_err       for r in results])
        t   = np.array([r.time_ms       for r in results])
        solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
        rows.append({
            "timestamp":      timestamp,
            "robot":          robot_name,
            "mode":           "sequential",
            "solver":         solver,
            "collision_free": False,
            "n_problems":     len(results),
            "n_timed":        n_timed,
            "t_med_ms":       round(float(np.median(t)),        6),
            "t_p95_ms":       round(float(np.percentile(t, 95)), 6),
            "pos_med_mm":     round(float(np.median(pos)),       6),
            "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
            "rot_med_rad":    round(float(np.median(rot)),       6),
            "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
            "success_n":      solved,
            "success_total":  len(results),
            "coll_free_n":    "",
            "peak_gpu_pct":   "",
            "avg_gpu_pct":    "",
            "peak_vram_mb":   "",
        })

    # -- Batch (no collision) ------------------------------------------------
    for solver, result in batch_results.items():
        pos    = result.pos_errs * 1e3
        rot    = result.rot_errs
        solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))

        def _fmtf(v): return round(float(v), 6) if not np.isnan(v) else ""

        rows.append({
            "timestamp":      timestamp,
            "robot":          robot_name,
            "mode":           "batch",
            "solver":         solver,
            "collision_free": False,
            "n_problems":     len(pos),
            "n_timed":        n_timed,
            "t_med_ms":       round(result.time_ms, 6),
            "t_p95_ms":       "",
            "pos_med_mm":     round(float(np.median(pos)),        6),
            "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
            "rot_med_rad":    round(float(np.median(rot)),         6),
            "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
            "success_n":      solved,
            "success_total":  len(pos),
            "coll_free_n":    "",
            "peak_gpu_pct":   _fmtf(result.peak_gpu_util),
            "avg_gpu_pct":    _fmtf(result.avg_gpu_util),
            "peak_vram_mb":   _fmtf(result.peak_vram_mb),
        })

    # -- Sequential (collision-free) -----------------------------------------
    for solver, results in seq_coll_results.items():
        pos = np.array([r.pos_err * 1e3 for r in results])
        rot = np.array([r.rot_err       for r in results])
        t   = np.array([r.time_ms       for r in results])
        solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
        rows.append({
            "timestamp":      timestamp,
            "robot":          robot_name,
            "mode":           "sequential",
            "solver":         solver,
            "collision_free": True,
            "n_problems":     len(results),
            "n_timed":        n_timed,
            "t_med_ms":       round(float(np.median(t)),          6),
            "t_p95_ms":       round(float(np.percentile(t, 95)),  6),
            "pos_med_mm":     round(float(np.median(pos)),         6),
            "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
            "rot_med_rad":    round(float(np.median(rot)),         6),
            "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
            "success_n":      solved,
            "success_total":  len(results),
            "coll_free_n":    seq_coll_free.get(solver, ""),
            "peak_gpu_pct":   "",
            "avg_gpu_pct":    "",
            "peak_vram_mb":   "",
        })

    # -- Batch (collision-free) ----------------------------------------------
    for solver, result in batch_coll_results.items():
        pos    = result.pos_errs * 1e3
        rot    = result.rot_errs
        solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))

        def _fmtf(v): return round(float(v), 6) if not np.isnan(v) else ""  # noqa: F811

        rows.append({
            "timestamp":      timestamp,
            "robot":          robot_name,
            "mode":           "batch",
            "solver":         solver,
            "collision_free": True,
            "n_problems":     len(pos),
            "n_timed":        n_timed,
            "t_med_ms":       round(result.time_ms, 6),
            "t_p95_ms":       "",
            "pos_med_mm":     round(float(np.median(pos)),         6),
            "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
            "rot_med_rad":    round(float(np.median(rot)),         6),
            "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
            "success_n":      solved,
            "success_total":  len(pos),
            "coll_free_n":    batch_coll_free.get(solver, ""),
            "peak_gpu_pct":   _fmtf(result.peak_gpu_util),
            "avg_gpu_pct":    _fmtf(result.avg_gpu_util),
            "peak_vram_mb":   _fmtf(result.peak_vram_mb),
        })

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# cuRobo baseline child (runs in the ``curobo`` conda env)
# ---------------------------------------------------------------------------
#
# Reached only when bench_ik.py's _IS_CUROBO_CHILD is True (bench_ik.py
# re-invoked with ``--solver cuRobo`` under a cuRobo-capable interpreter —
# see _curobo_python_cmd / bench_ik._run_solver_subprocess). All curobo/torch
# imports are deferred into the functions themselves so importing this module
# never requires torch or curobo, and vice versa.
#
# Protocol (mirrors pyroffi's bench and cuRobo's own ik_benchmark.py):
#
#   * Targets come from the ``.npz`` sidecar bench_ik.py's _run_robot_benchmark
#     writes during target generation (``bench_ik_targets_<robot>.npz``): key
#     ``seq`` is (32, 7) and ``batch`` is (256, 7), both wxyz_xyz float32.
#     cuRobo is the LAST candidate solver per robot, so the file is guaranteed
#     to exist by the time this child starts.
#   * Four rows, built and timed SEQUENTIALLY (one solver at a time, freed
#     between blocks to bound VRAM): sequential/batch x no-scene/with-scene.
#   * Sequential: one correctness solve then N_TIMED timed reps per problem,
#     each preceded by ``reset_seed()`` (fresh multistarts, matching pyroffi's
#     fresh-seed timed reps). Per-problem time = median of the reps.
#   * Batch: one correctness solve over all targets, then N_TIMED timed reps
#     with fresh seeds; effective per-problem time = median(rep)/n_targets,
#     timed loop wrapped in _gpu_monitor (same as pyroffi).
#   * Warmup: N_WARMUP solves with ``exit_early=False`` (full optimizer path)
#     so the worst-case CUDA graph is captured before any timed solve;
#     correctness/timed solves use ``exit_early=True`` (cuRobo's protocol).
#   * cuRobo's native tolerances are 5 mm / 0.05 rad; success is still SCORED
#     at pyroffi's POS_THR_M / ROT_THR_RAD thresholds, applied to
#     ``result.position_error`` / ``result.rotation_error`` (max over tool
#     links).
#   * ``coll_free_n`` is cuRobo's own ``result.feasible`` (self + world
#     collision and joint-limit check), counted over the correctness solve.
#
# Robot frames: cuRobo ships configs only for panda (franka.yml) and g1
# (unitree_g1.yml), i.e. _CUROBO_ROBOT_FILES. The per-robot tool frame is
# narrowed in-memory BEFORE ``IKSolverCfg.create`` so goals and comparison
# happen in the SAME frame as pyroffi's targets: panda -> ``panda_hand``
# (franka.yml's default), g1 -> ``right_hand_palm_link`` (pyroffi's EE link;
# present in unitree_g1.yml's kinematics even though the shipped config lists
# the 4 fingertip + ankle frames). No static offset is needed.

_CUROBO_ROBOT_FILES = {"panda": "franka.yml", "g1": "unitree_g1.yml"}
_CUROBO_TOOL_FRAMES = {"panda": ["panda_hand"], "g1": ["right_hand_palm_link"]}
_CUROBO_N_SEEDS = 32  # cuRobo LM/seed multistarts (uniform across robots)

# cspace joints outside the narrowed tool-frame tree, which this cuRobo's
# KinematicsLoader requires to be explicitly locked. panda's finger locks
# ship in franka.yml itself; unitree_g1.yml ships none, so with the single
# right_hand_palm_link goal the tree is base->waist->right arm->palm: both
# legs, the whole left arm+hand, and the right fingers all need locking. 0.0
# is within every joint's URDF limits and neutral (locked joints don't affect
# the palm pose, only the collision variant's feasibility checks).
_CUROBO_LOCK_JOINTS = {
    "g1": {
        # left leg
        "left_hip_pitch_joint": 0.0, "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0, "left_knee_joint": 0.0,
        "left_ankle_pitch_joint": 0.0, "left_ankle_roll_joint": 0.0,
        # right leg
        "right_hip_pitch_joint": 0.0, "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0, "right_knee_joint": 0.0,
        "right_ankle_pitch_joint": 0.0, "right_ankle_roll_joint": 0.0,
        # left arm
        "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        # left hand
        "left_hand_thumb_0_joint": 0.0, "left_hand_thumb_1_joint": 0.0,
        "left_hand_thumb_2_joint": 0.0, "left_hand_middle_0_joint": 0.0,
        "left_hand_middle_1_joint": 0.0, "left_hand_index_0_joint": 0.0,
        "left_hand_index_1_joint": 0.0,
        # right fingers (palm is the tool frame, so these are outside the tree)
        "right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": 0.0,
        "right_hand_thumb_2_joint": 0.0, "right_hand_middle_0_joint": 0.0,
        "right_hand_middle_1_joint": 0.0, "right_hand_index_0_joint": 0.0,
        "right_hand_index_1_joint": 0.0,
    },
}


def _curobo_env_to_scene_dict(env: dict) -> dict | None:
    """Convert the env JSON to the SceneCfg dict schema
    ``IKSolverCfg.create(scene_model=...)`` consumes:
    ``{"sphere": {name: {radius, pose}}, "cuboid": {name: {dims, pose}}}``
    with ``pose = [x, y, z, qw, qx, qy, qz]``.

    Prefers the pre-built ``curobo_world_model`` key when present (written by
    bench_ik._build_and_save_env), else falls back to the raw
    ``spheres``/``cuboids`` lists — the SAME source pyroffi's env-to-geoms
    conversion consumes, so both sides collide against the same obstacles.
    The floor plane has no SceneCfg primitive and is omitted (matching the
    pre-built key). Returns None if there are no obstacles at all.
    """
    scene = env.get("curobo_world_model")
    if scene is None:
        scene = {}
        spheres = {
            s["name"]: {"radius": s["radius"], "pose": [*s["center"], 1, 0, 0, 0]}
            for s in env.get("spheres", [])
        }
        if spheres:
            scene["sphere"] = spheres
        cuboids = {}
        for b in env.get("cuboids", []):
            wxyz = b.get("wxyz", [1.0, 0.0, 0.0, 0.0])
            cuboids[b["name"]] = {"dims": b["dims"], "pose": [*b["center"], *wxyz]}
        if cuboids:
            scene["cuboid"] = cuboids
    if not scene:
        return None
    return scene


def _curobo_make_goal(pose7, tool_frames: list[str], device):
    """Build a GoalToolPose from (B, 7) wxyz_xyz numpy poses."""
    import torch
    from curobo._src.types.pose import Pose
    from curobo._src.types.tool_pose import GoalToolPose

    pos = torch.from_numpy(np.ascontiguousarray(pose7[:, 4:7])).to(
        device, dtype=torch.float32
    )
    quat = torch.from_numpy(np.ascontiguousarray(pose7[:, 0:4])).to(
        device, dtype=torch.float32
    )
    return GoalToolPose.from_poses(
        {tool_frames[0]: Pose(position=pos, quaternion=quat)},
        ordered_tool_frames=tool_frames,
    )


def _curobo_build_solver(robot_data: dict, world_dict: dict | None,
                          max_batch_size: int, collision_free: bool):
    """Build one IKSolver. *robot_data* is deep-copied because the no-scene
    variant mutates the collision model (matching cuRobo's own benchmark).
    *world_dict* is the raw SceneCfg dict (see _curobo_env_to_scene_dict);
    passing a dict routes through create_solver_core_cfg -> SceneCfg.create
    exactly like a YAML path would."""
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.solver.solver_ik_cfg import IKSolverCfg
    from curobo._src.types.device_cfg import DeviceCfg

    robot_data_copy = copy.deepcopy(robot_data)
    if not collision_free:
        # cuRobo's "collision free" benchmark configuration. NOTE: unlike
        # cuRobo's benchmark we must NOT null lock_joints here — the finger
        # joints are in cspace but outside the panda_hand tree, so this
        # cuRobo version's KinematicsLoader validation requires them locked.
        robot_data_copy["kinematics"]["collision_link_names"] = None
    cfg = IKSolverCfg.create(
        robot=robot_data_copy,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model=world_dict if collision_free else None,
        self_collision_check=collision_free,
        device_cfg=DeviceCfg(),
        num_seeds=_CUROBO_N_SEEDS,
        # cuRobo's native tolerances (its published protocol scores at 5 mm);
        # our rows still score success at POS_THR_M / ROT_THR_RAD.
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        use_cuda_graph=True,
        optimizer_collision_activation_distance=0.0025,
        # Stays at the uniform 32: the g1-specific 128-seed / 240-iter
        # overrides in cuRobo's benchmark compensate its multi-link (4
        # fingertip) goal, which we do not use.
        seed_solver_num_seeds=_CUROBO_N_SEEDS,
        max_batch_size=max_batch_size,
    )
    return IKSolver(cfg)


def _curobo_run_sequential(
    solver, poses: np.ndarray, tool_frames: list[str], device,
    n_warmup: int, n_timed: int,
):
    """One correctness solve + n_timed timed reps per problem.

    Returns ([(pos_err_m, rot_err_rad, time_ms), ...], n_feasible).
    """
    from curobo._src.util.cuda_event_timer import CudaEventTimer

    solver.config.exit_early = False
    goal0 = _curobo_make_goal(poses[:1], tool_frames, device)
    for _ in range(n_warmup):
        solver.reset_seed()
        solver.solve_pose(goal_tool_poses=goal0, seed_config=None)
    import torch
    torch.cuda.empty_cache()

    solver.config.exit_early = True
    results: list[tuple[float, float, float]] = []
    n_feasible = 0
    for i in range(len(poses)):
        goal = _curobo_make_goal(poses[i : i + 1], tool_frames, device)
        solver.reset_seed()
        res = solver.solve_pose(goal_tool_poses=goal, seed_config=None)
        pos_err = float(res.position_error.view(-1)[0])
        rot_err = float(res.rotation_error.view(-1)[0])
        n_feasible += int(bool(res.feasible.view(-1)[0]))

        times: list[float] = []
        for _ in range(n_timed):
            solver.reset_seed()
            timer = CudaEventTimer().start()
            solver.solve_pose(goal_tool_poses=goal, seed_config=None)
            times.append(timer.stop() * 1e3)
        results.append((pos_err, rot_err, float(np.median(times))))
    return results, n_feasible


def _curobo_run_batch(
    solver, poses: np.ndarray, tool_frames: list[str], device,
    n_warmup: int, n_timed: int,
):
    """One correctness solve over the whole batch + n_timed timed reps.

    Returns (pos_errs_m, rot_errs_rad, effective_ms_per_problem, n_feasible,
    (peak_gpu_pct, avg_gpu_pct, peak_vram_mb)).
    """
    import torch
    from curobo._src.util.cuda_event_timer import CudaEventTimer

    solver.config.exit_early = False
    goal = _curobo_make_goal(poses, tool_frames, device)
    for _ in range(n_warmup):
        solver.reset_seed()
        solver.solve_pose(goal_tool_poses=goal, seed_config=None)
    torch.cuda.empty_cache()

    solver.config.exit_early = True
    solver.reset_seed()
    res = solver.solve_pose(goal_tool_poses=goal, seed_config=None)
    pos_errs = res.position_error.view(-1).cpu().numpy().astype(np.float64)
    rot_errs = res.rotation_error.view(-1).cpu().numpy().astype(np.float64)
    n_feasible = int(res.feasible.view(-1).sum())

    times: list[float] = []
    with _gpu_monitor() as samples:
        for _ in range(n_timed):
            solver.reset_seed()
            timer = CudaEventTimer().start()
            solver.solve_pose(goal_tool_poses=goal, seed_config=None)
            times.append(timer.stop() * 1e3)
    effective_ms = float(np.median(times)) / len(poses)
    gpu = (
        float(np.max(samples["gpu_util"])) if samples["gpu_util"] else float("nan"),
        float(np.mean(samples["gpu_util"])) if samples["gpu_util"] else float("nan"),
        float(np.max(samples["vram_mb"])) if samples["vram_mb"] else float("nan"),
    )
    return pos_errs, rot_errs, effective_ms, n_feasible, gpu


def _curobo_seq_row(ts: str, robot: str, collision_free: bool,
                     results: list[tuple[float, float, float]],
                     coll_free: int | None, n_timed: int) -> dict:
    pos = np.array([r[0] * 1e3 for r in results])
    rot = np.array([r[1] for r in results])
    t = np.array([r[2] for r in results])
    solved = sum(r[0] < POS_THR_M and r[1] < ROT_THR_RAD for r in results)
    return {
        "timestamp":      ts,
        "robot":          robot,
        "mode":           "sequential",
        "solver":         "cuRobo",
        "collision_free": collision_free,
        "n_problems":     len(results),
        "n_timed":        n_timed,
        "t_med_ms":       round(float(np.median(t)),        6),
        "t_p95_ms":       round(float(np.percentile(t, 95)), 6),
        "pos_med_mm":     round(float(np.median(pos)),        6),
        "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
        "rot_med_rad":    round(float(np.median(rot)),        6),
        "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
        "success_n":      solved,
        "success_total":  len(results),
        "coll_free_n":    coll_free if coll_free is not None else "",
        "peak_gpu_pct":   "",
        "avg_gpu_pct":    "",
        "peak_vram_mb":   "",
    }


def _curobo_batch_row(ts: str, robot: str, collision_free: bool,
                       pos_errs: np.ndarray, rot_errs: np.ndarray, time_ms: float,
                       coll_free: int | str, gpu: tuple[float, float, float],
                       n_timed: int) -> dict:
    pos = pos_errs * 1e3
    rot = rot_errs
    solved = int(np.sum((pos_errs < POS_THR_M) & (rot_errs < ROT_THR_RAD)))

    def _fmtf(v):
        return round(float(v), 6) if not np.isnan(v) else ""

    return {
        "timestamp":      ts,
        "robot":          robot,
        "mode":           "batch",
        "solver":         "cuRobo",
        "collision_free": collision_free,
        "n_problems":     len(pos),
        "n_timed":        n_timed,
        "t_med_ms":       round(time_ms, 6),
        "t_p95_ms":       "",
        "pos_med_mm":     round(float(np.median(pos)),         6),
        "pos_p95_mm":     round(float(np.percentile(pos, 95)), 6),
        "rot_med_rad":    round(float(np.median(rot)),         6),
        "rot_p95_rad":    round(float(np.percentile(rot, 95)), 6),
        "success_n":      solved,
        "success_total":  len(pos),
        "coll_free_n":    coll_free,
        "peak_gpu_pct":   _fmtf(gpu[0]),
        "avg_gpu_pct":    _fmtf(gpu[1]),
        "peak_vram_mb":   _fmtf(gpu[2]),
    }


def _run_curobo_child(
    robot_name: str,
    csv_file: pathlib.Path,
    *,
    env_file: pathlib.Path,
    n_targets: int,
    n_targets_batch: int,
    n_warmup: int,
    n_timed: int,
) -> None:
    """Entry point for the cuRobo child (``--robot X --solver cuRobo``).

    All curobo/torch imports are local to this function (see the section
    docstring above): this keeps the module importable in the plain
    JAX/pyroffi env, since this function is simply never called there.
    """
    import curobo.runtime as runtime

    # Must be set BEFORE importing the solver stack (same as cuRobo's own
    # benchmark): disable torch.compile/jit so timings reflect eager kernels.
    runtime.enable_torch_compile = False
    runtime.enable_torch_jit = False

    import torch
    from curobo._src.geom.types import SceneCfg
    from curobo._src.util.logging import setup_curobo_logger
    from curobo._src.util_file import (
        get_assets_path,
        get_robot_configs_path,
        join_path,
        load_yaml,
    )

    # Enable CUDA event timing for accurate GPU measurements (must come after
    # the curobo imports, exactly like cuRobo's own ik_benchmark.py).
    runtime.enable_cuda_event_timer = True

    # Seeds / precision flags — identical to cuRobo's own ik_benchmark.py.
    torch.manual_seed(2)
    np.random.seed(2)
    torch._dynamo.config.compiled_autograd = True
    torch._dynamo.config.cache_size_limit = 64
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    setup_curobo_logger("error")

    targets_npz = csv_file.parent / f"bench_ik_targets_{robot_name}.npz"
    if not targets_npz.exists():
        raise SystemExit(
            f"targets file not found: {targets_npz}\n"
            f"  It is written by _run_robot_benchmark during target generation; "
            f"run the dispatcher (or a non-cuRobo child for this robot first)."
        )

    device = torch.device("cuda")
    data = np.load(targets_npz)
    seq_poses = data["seq"].astype(np.float32)
    batch_poses = data["batch"].astype(np.float32)
    if seq_poses.shape != (n_targets, 7):
        raise SystemExit(f"targets 'seq' shape is {seq_poses.shape}, expected ({n_targets}, 7)")
    if batch_poses.shape != (n_targets_batch, 7):
        raise SystemExit(
            f"targets 'batch' shape is {batch_poses.shape}, expected ({n_targets_batch}, 7)"
        )

    env = json.loads(env_file.read_text())
    world_dict = _curobo_env_to_scene_dict(env)
    if world_dict is not None:
        # Fail fast on a schema mismatch (validates field names against the
        # Sphere/Cuboid dataclasses) before any solver is built.
        SceneCfg.create(world_dict)

    robot_data = load_yaml(
        join_path(get_robot_configs_path(), _CUROBO_ROBOT_FILES[robot_name])
    )
    if "kinematics" not in robot_data:
        # Newer robot YAMLs wrap the config under "robot_cfg" (franka.yml);
        # g1's ships kinematics at the top level (same unwrap as cuRobo's own
        # benchmark).
        robot_data = robot_data["robot_cfg"]
    tool_frames = _CUROBO_TOOL_FRAMES[robot_name]
    # Links live in the URDF, not the YAML — validate the tool frame is a
    # real link before building any solver (fails fast on a frame-name typo).
    kin = robot_data["kinematics"]
    urdf_path = join_path(get_assets_path(), kin["urdf_path"])
    urdf_links = {el.get("name") for _, el in ET.iterparse(urdf_path) if el.tag == "link"}
    urdf_links |= set(kin.get("extra_links") or {})
    if tool_frames[0] not in urdf_links:
        raise SystemExit(
            f"tool frame '{tool_frames[0]}' not found in {robot_name}'s URDF links "
            f"({pathlib.Path(urdf_path).name})"
        )
    # Narrow to pyroffi's EE frame (see section docstring) BEFORE create().
    kin["tool_frames"] = tool_frames
    # Lock the cspace joints the narrowed tree drops (see _CUROBO_LOCK_JOINTS).
    if robot_name in _CUROBO_LOCK_JOINTS:
        kin["lock_joints"] = _CUROBO_LOCK_JOINTS[robot_name]

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"cuRobo baseline: robot={robot_name} tool_frames={tool_frames}")
    print(f"  targets: seq={seq_poses.shape} batch={batch_poses.shape} "
          f"({targets_npz.name})")
    n_obs = len(SceneCfg.create(world_dict).objects) if world_dict is not None else 0
    print(f"  scene: {env_file.name} ({n_obs} obstacles), num_seeds={_CUROBO_N_SEEDS}")

    rows: list[dict] = []

    # 1) sequential, no scene
    print("\n[cuRobo] sequential, no scene ...")
    solver = _curobo_build_solver(robot_data, world_dict, 1, collision_free=False)
    seq_results, _ = _curobo_run_sequential(
        solver, seq_poses, tool_frames, device, n_warmup, n_timed
    )
    del solver
    torch.cuda.empty_cache()
    rows.append(_curobo_seq_row(ts, robot_name, False, seq_results, None, n_timed))

    # 2) sequential, with scene
    print("[cuRobo] sequential, with scene ...")
    solver = _curobo_build_solver(robot_data, world_dict, 1, collision_free=True)
    seq_coll_results, seq_coll_free = _curobo_run_sequential(
        solver, seq_poses, tool_frames, device, n_warmup, n_timed
    )
    del solver
    torch.cuda.empty_cache()
    rows.append(
        _curobo_seq_row(ts, robot_name, True, seq_coll_results, seq_coll_free, n_timed)
    )

    # 3) batch, no scene
    print("[cuRobo] batch, no scene ...")
    solver = _curobo_build_solver(robot_data, world_dict, n_targets_batch, collision_free=False)
    b_pos, b_rot, b_ms, _, b_gpu = _curobo_run_batch(
        solver, batch_poses, tool_frames, device, n_warmup, n_timed
    )
    del solver
    torch.cuda.empty_cache()
    # "" (not 0): no collision check ran, so coll_free_n is not measured —
    # matches the sequential no-scene row and the pyroffi dispatcher rows.
    rows.append(_curobo_batch_row(ts, robot_name, False, b_pos, b_rot, b_ms, "", b_gpu, n_timed))

    # 4) batch, with scene
    print("[cuRobo] batch, with scene ...")
    solver = _curobo_build_solver(robot_data, world_dict, n_targets_batch, collision_free=True)
    bc_pos, bc_rot, bc_ms, bc_feasible, bc_gpu = _curobo_run_batch(
        solver, batch_poses, tool_frames, device, n_warmup, n_timed
    )
    del solver
    torch.cuda.empty_cache()
    rows.append(
        _curobo_batch_row(
            ts, robot_name, True, bc_pos, bc_rot, bc_ms, bc_feasible, bc_gpu, n_timed
        )
    )

    # -- summary table --------------------------------------------------------
    print(f"\ncuRobo baseline results (scored at {POS_THR_M * 1e3:g} mm / {ROT_THR_RAD} rad):")
    hdr = (f"  {'mode':<12} {'scene':<7} {'t_med(ms)':>10} {'t_p95(ms)':>10} "
           f"{'pos_med(mm)':>11} {'rot_med(rad)':>12} {'success':>9} {'coll_free':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {r['mode']:<12} {'yes' if r['collision_free'] else 'no':<7} "
            f"{r['t_med_ms']:>10} {str(r['t_p95_ms']):>10} "
            f"{r['pos_med_mm']:>11} {r['rot_med_rad']:>12} "
            f"{str(r['success_n']) + '/' + str(r['success_total']):>9} "
            f"{str(r['coll_free_n']):>9}"
        )

    # -- append CSV rows -------------------------------------------------------
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_file.exists()
    with csv_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nappended {len(rows)} cuRobo rows to {csv_file}")


def _curobo_python_cmd() -> list[str] | None:
    """Locate a python interpreter with cuRobo installed.

    Precedence: the CUROBO_PYTHON env var, a sibling ``curobo`` conda env next
    to the active one (i.e. ``<envs>/curobo/bin/python``), then
    ``conda run -n curobo``. Returns None if no candidate is found.
    """
    override = os.environ.get("CUROBO_PYTHON")
    if override:
        if pathlib.Path(override).is_file():
            return [override]
        print(f"  warning: CUROBO_PYTHON={override} not found; trying other candidates")
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        envs_dir = pathlib.Path(prefix).resolve().parent
    else:
        # Not inside a conda env: infer <envs>/ from sys.executable
        # (<envs>/<env>/bin/python).
        envs_dir = pathlib.Path(sys.executable).resolve().parent.parent.parent
    sibling = envs_dir / "curobo" / "bin" / "python"
    if sibling.is_file():
        return [str(sibling)]
    if shutil.which("conda") is not None:
        return ["conda", "run", "--no-capture-output", "-n", "curobo"]
    return None
