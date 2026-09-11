"""The teleop pick-and-place task, driven synthetically instead of by a human.

`iosp.fit.teleop` fits the same forward model against RECORDED demonstrations of
this scene.  This module keeps the scene and throws the recordings away: scenes
are sampled from `sim_teleop.pickplace.randomize`'s own ranges and the
demonstration is a rollout of the planner, exactly as `tetris` and `tower` do.
So the pick-and-place domain finally has a task that is physically realizable,
which `iosp.config`'s abstract one never was -- its `PICK_POS`/`PLACE_POS` sit
0.3 m up with the table top at z=0 and no bucket anywhere, so the object starts
in mid-air and the release target is a point in free space.

Everything geometric is imported, not restated:

    scene         `sim_teleop.pickplace.scene.PickPlaceScene`
    randomisation `sim_teleop.pickplace.randomize.RandomizationRanges`
    MuJoCo world  `sim_teleop.pickplace.model.build_model` (FR3 + Franka Hand +
                  table + bucket + cube)
    IOSP context  the scene's own `iosp_scene_fields`

The task skeleton, mirroring tetris's packing skeleton and tower's stack:

    pick    the cube where it spawned, at the cube's own yaw
    place   `scene.drop_target_xyz()` -- the cube centre resting in the bucket

Success is geometric and task-level, the same standard the other two domains
use: the cube ends up INSIDE THE BUCKET, on its floor, and nothing was knocked
over on the way.
"""
from __future__ import annotations

import contextlib
import functools
import os
import pathlib
import sys

import numpy as np

TELEOP_ROOT = pathlib.Path(os.environ.get(
    "IOSP_TELEOP_ROOT",
    pathlib.Path(__file__).resolve().parents[3] / "sim_teleop"))

# Franka flange -> fingertip.  `iosp.fit.teleop` MEASURED this on the recorded
# episodes as the median hand height above the cube at the grasp row and got
# 0.105 m, matching the URDF's 0.1034 m to 2 mm -- an independent confirmation
# that the same offset the tetris and tower models needed is a property of the
# gripper, not of any one domain.
GRASP_OFFSET = 0.1034

# A human drops the cube into the bucket from above rather than lowering it to
# the floor; `iosp.fit.teleop` measures 0.24 m at the release on the recorded
# session.  Kept as the planned release height so the synthetic demonstrations
# have the same shape as the human ones.
RELEASE_OFFSET = 0.24


def _import_teleop():
    if not (TELEOP_ROOT / "pickplace").is_dir():
        raise SystemExit(
            f"sim_teleop checkout not found at {TELEOP_ROOT}; set "
            "IOSP_TELEOP_ROOT. This task is defined by that repo.")
    root = str(TELEOP_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


@contextlib.contextmanager
def _teleop_cwd():
    prev = os.getcwd()
    os.chdir(_import_teleop())
    try:
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def sample_scenes(rng, n, ranges=None):
    """-> list of `PickPlaceScene`, sampled from sim_teleop's own ranges.

    Only the factors that change the TASK are varied -- the cube's pose, size
    and the bucket's pose and radius.  Friction, colour and lighting are left
    nominal: they are there for vision policies and only add variance here.
    """
    _import_teleop()
    from pickplace.randomize import RandomizationRanges
    from pickplace.scene import PickPlaceScene

    r = ranges or RandomizationRanges()
    nom = PickPlaceScene()
    out = []
    for _ in range(n):
        cx, cy = nom.cube_spawn_xy
        if r.cube_xy_offset is not None:
            cx += rng.uniform(*r.cube_xy_offset)
            cy += rng.uniform(*r.cube_xy_offset)
        bx, by = nom.bucket_center_xy
        if r.bucket_xy_offset is not None:
            bx += rng.uniform(*r.bucket_xy_offset)
            by += rng.uniform(*r.bucket_xy_offset)
        out.append(PickPlaceScene(
            cube_spawn_xy=(cx, cy),
            cube_yaw=(rng.uniform(*r.cube_yaw) if r.cube_yaw else 0.0),
            cube_half_extent=(rng.uniform(*r.cube_half_extent)
                              if r.cube_half_extent else nom.cube_half_extent),
            bucket_center_xy=(bx, by),
            bucket_inner_radius=(rng.uniform(*r.bucket_inner_radius)
                                 if r.bucket_inner_radius
                                 else nom.bucket_inner_radius),
        ))
    return out


def skeleton(scene):
    """The task skeleton for one scene: the grasp and release EE targets.

    -> dict with `pick_pos`/`place_pos` (EE targets, offset along the approach
    axis from the object poses) plus the object poses themselves, which is what
    a success test has to compare against.
    """
    cube = np.asarray(scene.cube_spawn_pos(), np.float32)
    drop = np.asarray(scene.drop_target_xyz(), np.float32)
    up = np.array([0.0, 0.0, 1.0], np.float32)
    return dict(
        cube_pos=cube, drop_pos=drop, yaw=np.float32(scene.cube_yaw),
        pick_pos=cube + up * GRASP_OFFSET,
        place_pos=drop + up * RELEASE_OFFSET,
    )


def iosp_fields(scene, q_start):
    """The scene as iosp's `PickPlaceScene` context, via sim_teleop's own map."""
    _import_teleop()
    return scene.iosp_scene_fields(q_start)


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]
GRIP_ACTUATOR = "fr3_actuator8"
EE_BODY = "fr3_link7"
CUBE_BODY = "pp_cube"
GRIPPER_BODIES = ("fr3_hand", "fr3_left_finger", "fr3_right_finger")


@functools.lru_cache(maxsize=8)
def _home_q():
    """A neutral FR3 pose to start from and return to."""
    return np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8])


def build_world(scene):
    """-> `MjWorld` wrapping sim_teleop's compiled model for `scene`.

    The model is taken as compiled, with no contact exclusions bolted on.  The
    tetris and tower scenes needed the gripper excluded from the object because
    the planner's arm has no hand and its approach swept the fingers through the
    block; here the grasp offset is the MEASURED flange-to-fingertip distance,
    so the fingers straddle the cube instead of driving into it.  Whether that
    holds is a measurable claim, and `iosp.checks.teleop_rollout` reports the
    pre-grasp disturbance so it stays checked rather than assumed.
    """
    from iosp.viz import mj_scene as M

    _import_teleop()
    from pickplace.model import build_model
    from pickplace.scene import CUBE_BODY as _CB

    world = M.MjWorld.wrap(
        build_model(scene), ARM_JOINTS, EE_BODY, [0.0, 0.0, GRASP_OFFSET],
        free_bodies=[_CB], finger_joints=FINGER_JOINTS,
        grip_actuator=GRIP_ACTUATOR, arm_actuators=ARM_JOINTS)
    world.set_arm(0, _home_q())
    world.set_fingers(0, M.FINGER_OPEN)
    world.forward()
    return world


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------

TILT_TOL_DEG = 25.0


def place_success(cube_xyz, scene, tilt_deg=None):
    """Is the cube IN THE BUCKET?  -> (ok, dict).

    Three conditions, all geometric: the cube's centre is inside the bucket's
    inner radius with its own half-width to spare, it is resting at the bucket
    floor rather than perched on the rim or still in the air, and it has not
    been tipped over.
    """
    cube_xyz = np.asarray(cube_xyz, float)
    cx, cy = scene.bucket_center_xy
    r_in = scene.bucket_inner_radius
    half = scene.cube_half_extent

    radial = float(np.hypot(cube_xyz[0] - cx, cube_xyz[1] - cy))
    inside = radial <= max(r_in - half, 0.0)

    floor_z = scene.table_top_z + scene.bucket_floor_thickness + half
    dz = float(cube_xyz[2] - floor_z)
    seated = abs(dz) <= 0.015

    upright = True if tilt_deg is None else bool(float(tilt_deg) <= TILT_TOL_DEG)
    ok = inside and seated and upright
    return ok, dict(ok=ok, inside=inside, seated=seated, upright=upright,
                    radial_mm=radial * 1000,
                    clearance_mm=(r_in - half - radial) * 1000,
                    dz_mm=dz * 1000,
                    tilt_deg=None if tilt_deg is None else float(tilt_deg))
