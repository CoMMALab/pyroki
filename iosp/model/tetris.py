"""Tetris-packing IOSP model: inverting a differentiable planner for
block-packing on a Panda, adapting SPaSM's tetris problem structure.

The forward model
-----------------
A Panda picks blocks from scattered initial positions and places them inside
a walled goal region.  Each block is a spherized L- or O-shaped tetromino
(matching SPaSM's `spasm.tetris.env`), and the trajectory for each block is a
two-phase composed plan:

  1. **Pick-to-place**: IK for the pick pose (block's current position +
     standoff), trajopt from q_start to q_pick, then from q_pick to q_place
     (target inside the goal region + standoff).
  2. **Return**: trajopt from q_place back to q_home.

For the IOSP paper, we use N=1 block as the base case (single pick-place with
wall + obstacle geometry) and N=3 for scaling.

Cost features (tied across all segments)
-----------------------------------------
  ``effort``     velocity norm ||q[t+1] - q[t]||^2
  ``smooth``     acceleration norm ||q[t+2] - 2q[t+1] + q[t]||^2
  ``clearance``  obstacle avoidance (walls + static blocks)
  ``orient``     EE tilt away from pointing downward
  ``skeleton``   deviation from the task skeleton (pick/place poses)

theta = softmax(z) over these 5 features, shared across all segments and the
refine pass -- same tied-model argument as `pickplace.py`.
"""

import dataclasses
import os

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("IOSP_JAX_CACHE_DIR",
                   os.path.expanduser("~/.cache/jax_pyroffi_iosp")),
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from ioc.robot.problem import RobotProblem, Scene

from pyroffi.optimization_engines import _implicit_diff
_implicit_diff.CANONICAL_BY_DEFAULT = True
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda_batch

# ---------------------------------------------------------------------------
# Geometry: block and wall definitions
# ---------------------------------------------------------------------------

# NOT a standoff: the offset between the frame the IOSP/pyroffi FK reports as
# the end effector (`panda_link7` + 0.107 m, i.e. the HAND ORIGIN) and the frame
# SPaSM plans in (`panda_grasptarget`, another 0.105 m along the approach axis,
# between the fingertips).  SPaSM sets the grasp pose EQUAL to the block pose
# (`tetris_traj.q_to_block = matrix_to_xyzyaw(get_ee_pose(q))`), so reproducing
# its demonstrations means putting our EE exactly this far above the block.
# The old value of 0.06 drove `panda_grasptarget` 4.5 cm BELOW the block origin
# -- the gripper closed through the block rather than around its peg.
#
# 0.105 is the nominal `panda_grasptarget` offset, and it is right for a WELDED
# grasp, where the block only has to hang below the hand.  It is wrong for the
# contact-physics rollout (`mj_rollout.run_events`, no weld): measured on this
# menagerie hand, the finger pads span dz = -0.016 .. -0.0905 from the hand
# origin, so a hand 0.105 above the block puts the pad BOTTOM at the very top of
# the 1.5 cm handle -- zero overlap, and the fingers close on air.  The block
# was never touched (0 deg tilt, and the "slip" was just the hand completing its
# carry alone), and no friction value changes that.
# 0.053 is the pad CENTRE, which straddles the handle at the block origin.
GRASP_OFFSET = 0.053
STANDOFF = GRASP_OFFSET          # back-compat alias

# [m] how far ABOVE its skeleton slot the block is released, so it drops in
# vertically rather than being carried down alongside the goal wall.  The wall
# is 0.045 tall and is NOT in the trajopt problem, so a block driven down into
# the slot has nothing pricing its clearance against the rim.
RELEASE_DROP = 0.085

CLEARANCE_MARGIN = 0.05
SOFTMIN_TAU = 0.02
SOFTNESS = 60.0

SPH_RADIUS = 0.03

DOWN_WXYZ = jnp.array([0.0, 1.0, 0.0, 0.0])
UP_AXIS = jnp.array([0.0, 0.0, 1.0])
IK_RNG_KEY = jax.random.PRNGKey(0)
IK_CONTINUITY_WEIGHT = 1.0

TORQUE_DT = 0.1     # [s] waypoint spacing for the finite-difference qd/qdd
GRAVITY = -9.81


SELF_MARGIN = 0.01   # [m] self-collision clearance margin (hinge point)


def _self_collision_residual(robot_coll, robot, q):
    """Smooth arm SELF-collision residual (arm-vs-arm), the term SPaSM prices in
    `arm_collision_cost` but iosp's obstacle-only `clearance` lacks.

    `compute_self_collision_distance` returns per-pair distances over the active
    (SRDF-filtered, non-adjacent) self-collision pairs, so a soft-min over them
    -- not the built-in hard min -- keeps this smooth for the implicit-diff
    adjoint (cf. `clearance_residual`'s soft-min fix).  Returns (T,)."""
    d = jax.vmap(lambda qi: robot_coll.compute_self_collision_distance(robot, qi))(q)
    d_min = -SOFTMIN_TAU * jax.scipy.special.logsumexp(-d / SOFTMIN_TAU, axis=-1)
    return jax.nn.softplus(SOFTNESS * (SELF_MARGIN - d_min)) / SOFTNESS


def _torque_residual(robot, q, dt=TORQUE_DT, gravity=GRAVITY):
    """RNEA joint torques at the interior knots (GRiD inverse dynamics).

    The `torque` cost feature: prices dynamic effort (mass/gravity/Coriolis),
    not just kinematic velocity, so the demonstrations are dynamically -- not
    only geometrically -- meaningful.  Central differences for qd/qdd, matching
    `ioc.robot.bases.dynamic`; routed through GRiD's CUDA FFI (`use_cuda=True`),
    whose analytic `idsva_so` custom_jvp keeps `jax.hessian` (the implicit
    adjoint) working through it."""
    qd = (q[2:] - q[:-2]) / (2.0 * dt)
    qdd = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt ** 2)
    qm = q[1:-1]
    tau = robot.inverse_dynamics(qm, qd, qdd, gravity=gravity, use_cuda=True)
    return tau.reshape(-1)


def create_tetris_spheres(shape="L", sph_radius=SPH_RADIUS):
    """Spherized tetromino, matching SPaSM's `create_tetris_spheres`."""
    coords = {
        "L": jnp.array([(0, 0, 0), (0, 1, 0), (0, -1, 0), (1, -1, 0)], dtype=jnp.float32),
        "O": jnp.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)], dtype=jnp.float32),
    }[shape]
    n = coords.shape[0]
    spheres = jnp.zeros((n + 2, 4), dtype=jnp.float32)
    spheres = spheres.at[:n, :3].set(coords * sph_radius * 2)
    spheres = spheres.at[:n, 3].set(sph_radius)
    stick = jnp.array([
        [0.0, 0.0, -sph_radius * 1.25, sph_radius / 2],
        [0.0, 0.0, -sph_radius * 2.0, sph_radius / 2],
    ], dtype=jnp.float32)
    spheres = spheres.at[n:, :].set(stick)
    z_offset = -spheres[-1, 2]
    spheres = spheres.at[:, 2].add(z_offset)
    return spheres


def block_pose_to_spheres(block_spheres, pose_xyzyaw):
    """Transform local block spheres by a pose (x, y, z, yaw) -> (K, 4)."""
    pos = pose_xyzyaw[:3]
    yaw = pose_xyzyaw[3]
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    R = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=jnp.float32)
    centers = block_spheres[:, :3] @ R.T + pos
    return jnp.concatenate([centers, block_spheres[:, 3:]], axis=-1)


def create_goal_walls(goal_center, goal_dims, wall_height=0.045,
                      wall_thickness=0.015):
    """AABB walls around the goal region, as (N_walls, 6) [x1,y1,z1,x2,y2,z2]."""
    cx, cy, cz = goal_center
    dx, dy, _ = goal_dims
    walls = jnp.array([
        [cx - dx/2, cy + dy/2, cz,
         cx + dx/2, cy + dy/2 + wall_thickness, cz + wall_height],
        [cx - dx/2, cy - dy/2 - wall_thickness, cz,
         cx + dx/2, cy - dy/2, cz + wall_height],
        [cx - dx/2 - wall_thickness, cy - dy/2, cz,
         cx - dx/2, cy + dy/2, cz + wall_height],
        [cx + dx/2, cy - dy/2, cz,
         cx + dx/2 + wall_thickness, cy + dy/2, cz + wall_height],
    ], dtype=jnp.float32)
    return walls


# ---------------------------------------------------------------------------
# Segment layout
# ---------------------------------------------------------------------------

PHASES = ("approach", "place_traj", "return_traj")
N_APPROACH = 8
N_PLACE_TRAJ = 10
N_RETURN = 6
SEGMENT_LEN = {"approach": N_APPROACH, "place_traj": N_PLACE_TRAJ,
                "return_traj": N_RETURN}

N_FULL = N_APPROACH + (N_PLACE_TRAJ - 1) + (N_RETURN - 1)

PHASE_SPAN, _s = {}, 0
for _p in PHASES:
    PHASE_SPAN[_p] = (_s, _s + SEGMENT_LEN[_p])
    _s += SEGMENT_LEN[_p] - 1
del _s, _p
assert PHASE_SPAN["return_traj"][1] == N_FULL

IDX_PICK = PHASE_SPAN["approach"][1] - 1
IDX_PLACE = PHASE_SPAN["place_traj"][1] - 1

FEATURE_NAMES = ("effort", "smooth", "clearance", "held", "orient", "torque",
                 "skeleton")
K = len(FEATURE_NAMES)
SEGMENT_FEATURES = ("effort", "smooth", "clearance", "held", "orient", "torque")
K_SEG = len(SEGMENT_FEATURES)


# ---------------------------------------------------------------------------
# Scene dataclass
# ---------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TetrisScene:
    """Context for one tetris pick-place demonstration."""
    q_start: jnp.ndarray       # (dof,) home configuration
    pick_pos: jnp.ndarray      # (3,) block's current EE target (with standoff)
    place_pos: jnp.ndarray     # (3,) target inside goal (with standoff)
    obs_center: jnp.ndarray    # (N_obs, 3) obstacle sphere centers
    obs_radius: jnp.ndarray    # (N_obs,) obstacle sphere radii
    block_spheres: jnp.ndarray # (6, 4) the carried block's own sphere layout
    pick_yaw: jnp.ndarray      # (,) block yaw at the pick, SPaSM convention
    place_yaw: jnp.ndarray     # (,) block yaw in the packing skeleton
    slot: jnp.ndarray          # (,) which skeleton slot this scene fills


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TetrisFullScene:
    """Context for the stage-3 refine solve."""
    q_start: jnp.ndarray       # (dof,)
    q_goal: jnp.ndarray        # (dof,) = q_home for return
    obs_center: jnp.ndarray    # (N_obs, 3)
    obs_radius: jnp.ndarray    # (N_obs,)
    q_pick: jnp.ndarray        # (dof,)
    q_place: jnp.ndarray       # (dof,)
    block_spheres: jnp.ndarray # (6, 4) the carried block's sphere layout


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SegmentScene:
    """Minimal segment scene, compatible with RobotProblem.unpack/seed."""
    q_start: jnp.ndarray       # (dof,)
    q_goal: jnp.ndarray        # (dof,)
    obs_center: jnp.ndarray    # (N_obs, 3) or (3,) if single obs
    obs_radius: jnp.ndarray    # (N_obs,) or (1,) if single obs
    block_spheres: jnp.ndarray # (6, 4) the carried block's sphere layout
    holding: jnp.ndarray       # (,) 1.0 while this segment carries the block


# ---------------------------------------------------------------------------
# Collision helpers (smooth, jit-friendly)
# ---------------------------------------------------------------------------

def _multi_sphere_clearance(robot_coll, robot, q, obs_centers, obs_radii):
    """Smooth clearance residual against multiple obstacle spheres.

    Returns a (T,) vector of clearance violations, soft-min over all
    robot spheres × all obstacles.
    """
    coll = robot_coll.at_config(robot, q)
    coll_pos = coll.pose.translation()   # (..., T, S, 3)
    coll_rad = coll.radius               # (..., T, S)

    # (..., T, S, N_obs) pairwise distances — vmap-safe via expand_dims
    # coll_pos: (..., T, S, 3) -> (..., T, S, 1, 3)
    # obs_centers: (..., N_obs, 3) -> (..., 1, 1, N_obs, 3)
    cp = jnp.expand_dims(coll_pos, -2)
    oc = jnp.expand_dims(jnp.expand_dims(obs_centers, -3), -3)
    d = (jnp.linalg.norm(cp - oc, axis=-1)
        - jnp.expand_dims(coll_rad, -1)
        - obs_radii)
    d_flat = d.reshape(*d.shape[:-2], -1)   # (..., T, S*N_obs)
    d_min = -SOFTMIN_TAU * jax.scipy.special.logsumexp(
        -d_flat / SOFTMIN_TAU, axis=-1)
    return jax.nn.softplus(SOFTNESS * (CLEARANCE_MARGIN - d_min)) / SOFTNESS


def _held_block_residual(robot, ee_index, q, block_spheres, obs_center,
                         obs_radius):
    """Clearance of the CARRIED BLOCK against the world, (T,).

    The clearance feature above prices the ARM's spheres against the obstacles
    and the arm against itself.  It says nothing about the block in the
    gripper, because the planner's robot model has nothing attached to it -- so
    a plan that sweeps the block straight through a goal wall costs exactly
    zero, and no amount of re-weighting `clearance` can see it.  SPaSM's own
    trajectory optimizer has this term and weights it at 1.20 against an
    arm-collision weight of 0.005, i.e. 240x; the port of this domain dropped
    it.

    The block rides the EE: its pose is the EE pose less `GRASP_OFFSET` along
    the approach axis, with the EE's yaw, so its spheres follow from forward
    kinematics alone -- no extra decision variables.  `obs_center`/`obs_radius`
    already carry the goal walls and the blocks packed in earlier slots, which
    are exactly what it has to miss.
    """
    fk = robot.forward_kinematics(q)[..., ee_index, :]
    quat, pos = fk[..., 0:4], fk[..., 4:7]

    # Block frame: 180 deg about x, then the EE's yaw about z -- SPaSM's
    # `yaw_to_quat_xyz`.  Yaw is read off the EE quaternion.
    yaw = jnp.arctan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
    c, s_ = jnp.cos(yaw), jnp.sin(yaw)

    loc = block_spheres[:, :3] * jnp.array([1.0, -1.0, -1.0])   # the x flip
    bx = c[:, None] * loc[None, :, 0] - s_[:, None] * loc[None, :, 1]
    by = s_[:, None] * loc[None, :, 0] + c[:, None] * loc[None, :, 1]
    bz = jnp.broadcast_to(loc[None, :, 2], bx.shape)
    origin = pos - UP_AXIS * GRASP_OFFSET                       # (T, 3)
    cen = jnp.stack([bx, by, bz], axis=-1) + origin[:, None, :]  # (T, S, 3)
    rad = block_spheres[:, 3]                                    # (S,)

    d = (jnp.linalg.norm(cen[:, :, None, :] - obs_center[None, None, :, :],
                         axis=-1)
         - rad[None, :, None] - obs_radius[None, None, :])
    d_min = -SOFTMIN_TAU * jax.scipy.special.logsumexp(
        -d.reshape(d.shape[0], -1) / SOFTMIN_TAU, axis=-1)
    return jax.nn.softplus(SOFTNESS * (CLEARANCE_MARGIN - d_min)) / SOFTNESS


def _orient_residual(robot, ee_index, q):
    """Tilt of the EE away from pointing straight down: (T, 2) -> flat."""
    quat = robot.forward_kinematics(q)[..., ee_index, 0:4]
    return quat[:, 1:3].reshape(-1)


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class TetrisProblem:
    """Tetris-packing IOSP problem: pick-place with wall and block obstacles."""

    def __init__(self, base: RobotProblem, seg: dict):
        self.base = base
        self.seg = seg

    @property
    def dof(self):
        return self.base.dof

    @property
    def ee_index(self):
        return self.base.ee_index

    @staticmethod
    def load(urdf_path, srdf_path, mesh_dir):
        base = RobotProblem.load(urdf_path, srdf_path, mesh_dir, n_timesteps=2)
        seg = {p: dataclasses.replace(base, n_timesteps=SEGMENT_LEN[p])
               for p in PHASES}
        seg["full"] = dataclasses.replace(
            base, n_timesteps=N_FULL,
            pinned_rows=((IDX_PICK, "q_pick"), (IDX_PLACE, "q_place")))
        return TetrisProblem(base=base, seg=seg)

    # -- IK ------------------------------------------------------------------

    def pick_ik(self, pick_pos, refs, pick_yaw=None):
        return _ik_batch(self, pick_pos, refs, pick_yaw)

    def place_ik(self, place_pos, q_pick, place_yaw=None):
        return _ik_batch(self, place_pos, q_pick, place_yaw)

    # -- residuals -----------------------------------------------------------

    def segment_residual_fn(self, phase):
        problem = self.seg[phase]

        def residual_fn(x_flat, scene: SegmentScene):
            q = problem.unpack(x_flat, scene)
            v = q[1:] - q[:-1]
            a = q[2:] - 2.0 * q[1:-1] + q[:-2]
            clearance = jnp.concatenate([
                _multi_sphere_clearance(
                    self.base.robot_coll, self.base.robot, q,
                    scene.obs_center.reshape(-1, 3),
                    scene.obs_radius.reshape(-1)),
                _self_collision_residual(
                    self.base.robot_coll, self.base.robot, q)[..., None]], axis=-1)
            # The block is in the gripper only on the transport segment; on
            # the approach it is still on the table and on the return the
            # gripper is empty, so `holding` gates the term rather than the
            # phase name (it keeps the residual shape identical across phases,
            # which the tied-theta model needs).
            held = _held_block_residual(
                self.base.robot, self.ee_index, q, scene.block_spheres,
                scene.obs_center.reshape(-1, 3),
                scene.obs_radius.reshape(-1)) * scene.holding
            orient = _orient_residual(self.base.robot, self.ee_index, q)
            torque = _torque_residual(self.base.robot, q)
            return (v.reshape(-1), a.reshape(-1), clearance, held, orient,
                    torque)

        return residual_fn

    def full_residual_fn(self):
        problem = self.seg["full"]

        def residual_fn(x_flat, scene: TetrisFullScene):
            q = problem.unpack(x_flat, scene)
            v = q[1:] - q[:-1]
            a = q[2:] - 2.0 * q[1:-1] + q[:-2]
            clearance = jnp.concatenate([
                _multi_sphere_clearance(
                    self.base.robot_coll, self.base.robot, q,
                    scene.obs_center.reshape(-1, 3),
                    scene.obs_radius.reshape(-1)),
                _self_collision_residual(
                    self.base.robot_coll, self.base.robot, q)[..., None]], axis=-1)
            # Held only between the grasp and release rows.
            hold_mask = ((jnp.arange(q.shape[0]) >= IDX_PICK)
                         & (jnp.arange(q.shape[0]) <= IDX_PLACE)).astype(q.dtype)
            held = _held_block_residual(
                self.base.robot, self.ee_index, q, scene.block_spheres,
                scene.obs_center.reshape(-1, 3),
                scene.obs_radius.reshape(-1)) * hold_mask
            orient = _orient_residual(self.base.robot, self.ee_index, q)
            torque = _torque_residual(self.base.robot, q)
            skel = jnp.concatenate([
                q[IDX_PICK] - scene.q_pick,
                q[IDX_PLACE] - scene.q_place])
            return (v.reshape(-1), a.reshape(-1), clearance, held, orient,
                    torque, skel)

        return residual_fn

    def calibrate_segment(self, phase, residual_fn, scenes, key,
                          n_probe=16, jitter=0.15):
        problem = self.seg[phase]

        def raw(scene, k):
            x0 = problem.seed(scene)
            x = x0 + jitter * jax.random.normal(k, x0.shape)
            rs = residual_fn(x, scene)
            return jnp.stack([jnp.sum(r**2) for r in rs])

        keys = jax.random.split(key, n_probe)
        vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)),
                        in_axes=(0, None))(scenes, keys)
        scales = jnp.mean(jnp.abs(vals.reshape(-1, vals.shape[-1])), axis=0)
        # A feature can be STRUCTURALLY absent on a phase: `held` is gated off
        # on the approach and the return, where the gripper is empty, so its
        # residual is identically zero there and has no scale to estimate.
        # That is not a degenerate cost -- the term contributes nothing either
        # way -- so give it a neutral 1.0 rather than dividing by zero.  A
        # feature that is zero on EVERY phase would still be a real problem,
        # and `calibrate_full` (where nothing is gated) still asserts.
        absent = scales <= 1e-8
        scales = jnp.where(absent, 1.0, scales)
        n_absent = int(jnp.sum(absent))
        if n_absent:
            names = [n for n, a in zip(SEGMENT_FEATURES, np.asarray(absent)) if a]
            print(f"  [calibrate] {phase}: {names} identically zero on this "
                  f"phase; neutral scale 1.0", flush=True)
        return scales

    def calibrate_full(self, residual_fn, scenes, key,
                       n_probe=16, jitter=0.15):
        problem = self.seg["full"]

        def raw(scene, k):
            x0 = problem.seed(scene)
            x = x0 + jitter * jax.random.normal(k, x0.shape)
            rs = residual_fn(x, scene)
            return jnp.stack([jnp.sum(r**2) for r in rs])

        keys = jax.random.split(key, n_probe)
        vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)),
                        in_axes=(0, None))(scenes, keys)
        scales = jnp.mean(jnp.abs(vals.reshape(-1, vals.shape[-1])), axis=0)
        scales = jnp.where(scales > 1e-8, scales, 1.0)  # pinned feature -> benign 0 scale
        return scales

    def ee_positions(self, q):
        return self.base.ee_positions(q)

    # -- composed forward solve ----------------------------------------------

    def seeds(self, scenes: TetrisScene, standoff=GRASP_OFFSET):
        """Compute IK and per-segment seeds."""
        q_pick = self.pick_ik(scenes.pick_pos, scenes.q_start,
                              getattr(scenes, "pick_yaw", None))
        q_place = self.place_ik(scenes.place_pos, q_pick,
                                getattr(scenes, "place_yaw", None))

        one = jnp.ones_like(scenes.pick_yaw)
        zero = jnp.zeros_like(scenes.pick_yaw)
        seg_scenes = {
            "approach": SegmentScene(scenes.q_start, q_pick,
                                     scenes.obs_center, scenes.obs_radius,
                                     scenes.block_spheres, zero),
            "place_traj": SegmentScene(q_pick, q_place,
                                       scenes.obs_center, scenes.obs_radius,
                                       scenes.block_spheres, one),
            "return_traj": SegmentScene(q_place, scenes.q_start,
                                        scenes.obs_center, scenes.obs_radius,
                                        scenes.block_spheres, zero),
        }
        x0 = {p: jax.vmap(self.seg[p].seed)(seg_scenes[p]) for p in PHASES}
        return x0, seg_scenes, q_pick, q_place

    def solve(self, scenes, inner_by_phase, theta_seg, theta_full,
              refine_inner, *, stage2=True):
        """Full composed forward solve: IK -> per-segment -> refine."""
        x0, seg_scenes, q_pick, q_place = self.seeds(scenes)

        xs = {}
        for phase in PHASES:
            xs[phase] = jax.vmap(
                inner_by_phase[phase].solve_implicit,
                in_axes=(0, None, 0))(x0[phase], theta_seg, seg_scenes[phase])

        # Stage 3: refine
        full_sc = TetrisFullScene(
            scenes.q_start, scenes.q_start,
            scenes.obs_center, scenes.obs_radius,
            q_pick, q_place, scenes.block_spheres)

        if stage2:
            rows = []
            for i, ph in enumerate(PHASES):
                q = jax.vmap(self.seg[ph].unpack)(xs[ph], seg_scenes[ph])
                rows.append(q[:, 1:] if i > 0 else q)
            q_cat = jnp.concatenate(rows, axis=1)
            x0_full = q_cat[:, 1:-1, :].reshape(q_cat.shape[0], -1)
        else:
            x0_full = jax.vmap(self.seg["full"].seed)(full_sc)

        xs["full"] = jax.vmap(
            refine_inner.solve_implicit,
            in_axes=(0, None, 0))(x0_full, theta_full, full_sc)

        return xs, seg_scenes, full_sc, q_pick, q_place


def _grasp_rotation(yaw):
    """SPaSM's `yaw_to_quat_xyz`: 180 deg about x, THEN the yaw about z.

    `DOWN_WXYZ` is exactly the x-flip, so yaw=0 reproduces the old fixed target.
    The packing skeleton in `saved/tetris.npy` carries yaws of up to +-3 rad, so
    a position-only IK target cannot express the demonstrated grasp at all.
    """
    return jaxlie.SO3.from_z_radians(yaw) @ jaxlie.SO3(wxyz=DOWN_WXYZ)


def _target_pose_batch(pos_batch, yaw_batch=None):
    n = pos_batch.shape[0]
    if yaw_batch is None:
        rot = jaxlie.SO3(wxyz=jnp.broadcast_to(DOWN_WXYZ, (n, 4)).astype(jnp.float32))
    else:
        rot = jax.vmap(_grasp_rotation)(jnp.asarray(yaw_batch, jnp.float32))
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=rot, translation=pos_batch.astype(jnp.float32))


# [rad] kept inside the URDF's joint limits by every IK solve here.  5 deg,
# which covers the 4 deg by which this URDF is looser than the executing Panda.
DEFAULT_JOINT_MARGIN = float(np.radians(5.0))


# Seeds per IK problem.  The kernel derives each seed from `IK_RNG_KEY` AND the
# problem's position in the batch, so the branch it finds is not monotone in
# this number -- 256 was measured WORSE than 128 on one tower place pose.  128
# is where the tower's place poses stopped depending on it (one went 50.3 mm ->
# 16.9 mm from 32 -> 128, and no further at 256).
IK_NUM_SEEDS = 128


def _ik_batch(problem, target_pos, refs, target_yaw=None, continuity_weight=None):
    q = sqp_ik_solve_cuda_batch(
        problem.base.robot, problem.ee_index,
        _target_pose_batch(target_pos, target_yaw),
        IK_RNG_KEY, refs.astype(jnp.float32),
        continuity_weight=(IK_CONTINUITY_WEIGHT if continuity_weight is None
                           else continuity_weight),
        num_seeds=IK_NUM_SEEDS,
    ).astype(refs.dtype)
    # The joint limits are ALWAYS enforced; `IOSP_JOINT_MARGIN` only buys extra
    # room inside them.  It used to gate the clip entirely (default "0" = no
    # clip at all), and an unclipped IK solution is not a harmless seed: the
    # pick and place rows are PINNED to it by the skeleton residual, so a
    # branch a few degrees outside a limit becomes a hard waypoint the plan is
    # built around.  Measured on the tower stack, whose place rows asked for
    # joint 3 at -4 deg past its +-limit on level 2 and joint 6 at 26 deg past
    # on level 3; under physics the servo simply saturates at the limit, and
    # the residual pose error there is exactly that violation -- it does not
    # shrink when the trajectory is slowed, because it is not a tracking lag.
    #
    # The default margin is NOT zero, and that matters as much as the clip: the
    # URDF here is 4 deg looser than the Panda that executes the plan, so an IK
    # solution resting on the URDF limit is still outside the real one.  Solving
    # INSIDE the margin is not the same as solving to the limit and clipping
    # afterwards -- a clipped pinned row no longer reaches the pose it was
    # solved for, and measured that moved the tower's level-1 and level-2
    # placements 43 mm and 36 mm off their targets.  See
    # `forward_extract.EXEC_JOINT_MARGIN`, which this must cover.
    margin = float(os.environ.get("IOSP_JOINT_MARGIN", str(DEFAULT_JOINT_MARGIN)))
    lo = jnp.asarray(problem.base.robot.joints.lower_limits) + margin
    hi = jnp.asarray(problem.base.robot.joints.upper_limits) - margin
    return jnp.clip(q, lo, hi)


def make_tetris_forward_solver(n_iters=60, robot=None, method=None, gd_lr=0.1,
                               *, stock=False):
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt
    method = method or os.environ.get("IOSP_TRAJOPT", "lbfgs")
    if method == "projected_gd":
        lo = tuple(float(v) for v in np.asarray(robot.joints.lower_limits))
        hi = tuple(float(v) for v in np.asarray(robot.joints.upper_limits))
        cfg = DynamicsTrajOptConfig(n_iters=n_iters, method="projected_gd",
                                    gd_lr=gd_lr, q_lo=lo, q_hi=hi, dof=len(lo))
    elif stock:
        cfg = DynamicsTrajOptConfig(n_iters=n_iters)
    else:
        cfg = DynamicsTrajOptConfig(n_iters=n_iters, early_stop=False, unroll_tail=0,
                                    soft_line_search=False, soft_curvature_gate=False)
    return lambda x0, cost_fn: dynamics_trajopt(x0, cost_fn, cfg)


# ---------------------------------------------------------------------------
# Scene sampling
# ---------------------------------------------------------------------------

GOAL_CENTER = jnp.array([0.3, 0.0, -0.005], dtype=jnp.float32)

# Goal footprint, matching SPaSM's `Simulation.__init__`: the region is sized in
# CELLS of one sphere diameter, plus a one-radius buffer.  It was previously a
# fixed [0.18, 0.30], which is neither -- for the num_blocks=1 case the real
# walled region is 0.15 x 0.15, so `sample_tetris_scenes` was drawing place
# targets up to y = +-0.10 against a wall at +-0.075 and asking the planner to
# put blocks through it.
SPH_RADIUS = 0.03
# (wide, tall) in cells, exactly `tetris_env.Simulation.__init__`.
_GOAL_CELLS = {1: (2, 2), 3: (6, 2), 5: (10, 2), 8: (16, 2)}


def goal_dims(num_blocks=1):
    """(dx, dy, dz) of the walled goal region for `num_blocks`."""
    gw, gt = _GOAL_CELLS.get(num_blocks, (10, 2))
    buf = SPH_RADIUS
    return jnp.array([gt * SPH_RADIUS * 2 + buf, gw * SPH_RADIUS * 2 + buf,
                      0.01], dtype=jnp.float32)


GOAL_DIMS = goal_dims(1)

# Resting height of a tetromino's grasp frame on the table.  This is the block
# pose SPaSM's `create_tetris_spheres` produces (the four body spheres at local
# +2r under the 180-degree x flip), NOT an approximation: the old 0.06 put the
# planner's grasp target 3.5 cm below the peg the gripper actually has to take,
# which no amount of weight tuning can recover.
BLOCK_Z = 0.095

# Half-footprint of a tetromino: an O covers 2x2 cells, so its centre has to
# stay this far inside the goal walls for the block itself to fit.
BLOCK_FOOTPRINT_HALF = SPH_RADIUS * 2

Q_HOME = jnp.array([0.0, -0.785, 0.0, -1.571, 0.0, 1.571, 0.785],
                    dtype=jnp.float32)



def _jitter_spawn(rng, nominal, k, goal_pos, goal_dims, jitter_xy, jitter_yaw,
                  block_reach=0.09, tries=32):
    """A randomised spawn pose for block `k`, or its nominal pose.

    The spawn is where the block STARTS, which the packing skeleton does not
    depend on -- so this can be randomised freely, unlike the block's shape.
    Rejection sampling keeps the draw physically sane: the block must not land
    on the goal region (it is supposed to be carried there) and must not overlap
    another block's spawn.  After `tries` failures the nominal pose is returned,
    so the sampler cannot hang or silently produce an invalid scene.
    """
    if jitter_xy <= 0.0 and jitter_yaw <= 0.0:
        return nominal[k].copy()

    gp = np.asarray(goal_pos, float)[:2]
    gd = np.asarray(goal_dims, float)[:2]
    keep_out = gd / 2.0 + block_reach          # goal footprint + block extent

    for _ in range(tries):
        cand = nominal[k].copy()
        if jitter_xy > 0.0:
            # uniform in a disc, so the offset has no corner bias
            th = rng.uniform(0.0, 2.0 * np.pi)
            rad = jitter_xy * np.sqrt(rng.uniform(0.0, 1.0))
            cand[0] += rad * np.cos(th)
            cand[1] += rad * np.sin(th)
        if jitter_yaw > 0.0:
            cand[3] += rng.uniform(-jitter_yaw, jitter_yaw)

        if np.all(np.abs(cand[:2] - gp) < keep_out):
            continue                            # would sit on the goal box
        others = [nominal[j][:2] for j in range(len(nominal)) if j != k]
        if any(np.linalg.norm(cand[:2] - o) < 2 * block_reach for o in others):
            continue                            # would overlap another block
        return cand
    return nominal[k].copy()


def sample_tetris_scenes(rng, n, num_blocks=3, jitter_q=0.05,
                         jitter_spawn_xy=0.04, jitter_spawn_yaw=0.25):
    """Scenes that walk SPaSM's packing, with geometry taken from SPaSM itself.

    Scene i fills SLOT ``i % num_blocks`` of `spasm_tasks.tetris_skeleton()`: it
    picks that block from its SPaSM spawn pose and places it at the skeleton
    pose -- including the skeleton's YAW, which two of the three slots need and
    which the old position-only sampler could not express -- with every earlier
    slot already occupied and therefore an obstacle.  So the batch walks the
    real packing sequence, and the later slots are genuinely harder because the
    box is filling up.

    Nothing geometric is defined here: the goal walls, the resting height, the
    spawn poses and the shapes all come from `spasm_tasks`, which imports
    SPaSM's `Simulation`.  Both EE targets sit `GRASP_OFFSET` above their block
    along the approach axis, which is what puts `panda_grasptarget` -- the frame
    SPaSM plans in -- exactly on the block pose.

    Randomisation.  `jitter_q` perturbs the arm's start pose; `jitter_spawn_xy`
    and `jitter_spawn_yaw` perturb where each block STARTS.  Only the spawn is
    safe to randomise: the packing skeleton is a solved arrangement for these
    shapes in this goal box, so it does not depend on where a block was picked
    up from, but it WOULD be invalidated by changing a block's size or shape.
    Set both spawn jitters to 0.0 to recover the old deterministic sampler.

    NOTE: a randomised spawn has to reach the MuJoCo rollout too, or the arm
    will fly to a pose no block occupies -- pass the spawns to
    `spasm_rollout.build_scene(spawn_poses=...)`, which `forward_extract` records
    in its npz for exactly this purpose.
    """
    from iosp.model import spasm_tasks as ST

    g = ST.tetris_geometry(num_blocks)
    skeleton = ST.tetris_skeleton(num_blocks)
    spawn = np.asarray(g["block_poses"], np.float32)     # (num_blocks, 4)

    walls = create_goal_walls(jnp.asarray(g["goal_position"]),
                              jnp.asarray(g["goal_dims"]))
    wall_spheres = np.asarray(_walls_to_spheres(walls))
    n_placed_max = num_blocks - 1
    n_obs = wall_spheres.shape[0] + n_placed_max * SPHERES_PER_BLOCK

    up = np.array([0.0, 0.0, GRASP_OFFSET], dtype=np.float32)
    # The place target is lifted by RELEASE_DROP so the block is let go ABOVE
    # its slot and falls in vertically, instead of being carried down into the
    # slot and scraping the goal wall on the way.
    drop = np.array([0.0, 0.0, RELEASE_DROP], dtype=np.float32)

    qs, picks, places, p_yaw, q_yaw, slots, obs_c, obs_r, b_sph = (
        [] for _ in range(9))
    for i in range(n):
        k = i % num_blocks
        q0 = np.asarray(Q_HOME) + rng.normal(scale=jitter_q, size=7).astype(np.float32)

        placed = ([ST.tetris_block_spheres(num_blocks, skeleton)[j]
                   for j in range(k)] if k else [])
        # Pad to a fixed count so the batch is one rectangular array and the
        # residual stays jit-shape-stable; padding rows are zero-radius spheres
        # parked far below the table, which the clearance term ignores.
        pad = np.tile(np.array([[0.0, 0.0, -10.0, 0.0]], np.float32),
                      (n_obs - wall_spheres.shape[0]
                       - len(placed) * SPHERES_PER_BLOCK, 1))
        allobs = np.concatenate([wall_spheres] + placed + [pad], axis=0)

        sp = _jitter_spawn(rng, spawn, k, g["goal_position"], g["goal_dims"],
                           jitter_spawn_xy, jitter_spawn_yaw)

        qs.append(q0)
        picks.append(sp[:3] + up)
        places.append(skeleton[k, :3] + up + drop)
        p_yaw.append(np.float32(sp[3]))
        q_yaw.append(np.float32(skeleton[k, 3]))
        slots.append(np.int32(k))
        b_sph.append(np.asarray(g["block_spheres"][k], np.float32))
        obs_c.append(allobs[:, :3])
        obs_r.append(allobs[:, 3])

    f32 = lambda a: jnp.asarray(np.stack(a), dtype=jnp.float32)
    return TetrisScene(
        q_start=f32(qs), pick_pos=f32(picks), place_pos=f32(places),
        obs_center=f32(obs_c), obs_radius=f32(obs_r),
        block_spheres=f32(b_sph),
        pick_yaw=f32(p_yaw), place_yaw=f32(q_yaw),
        slot=jnp.asarray(np.stack(slots), dtype=jnp.int32))


SPHERES_PER_BLOCK = 6


def block_pose_spheres(shape, pose_xyzyaw):
    """A tetromino's spheres in world frame, (6, 4), at an [x,y,z,yaw] pose."""
    from iosp.model.spasm_costs import (block_pose_to_spheres,
                                        create_tetris_spheres)
    return block_pose_to_spheres(create_tetris_spheres(shape),
                                 pose_xyzyaw).astype(np.float32)


SPHERES_PER_BLOCK = 6


def block_pose_spheres(shape, pose_xyzyaw):
    """A tetromino's spheres in world frame, (6, 4), at an [x,y,z,yaw] pose.

    Mirrors SPaSM's `block_pose_to_spheres`: the block frame is flipped 180 deg
    about x and then yawed, so its +z points down and the grasp peg stands up.
    """
    from scipy.spatial.transform import Rotation
    sph = np.asarray(_create_tetris_spheres_np(shape))
    x, y, z, yaw = [float(v) for v in pose_xyzyaw]
    R = (Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)).as_matrix()
    pos = sph[:, :3] @ R.T + np.array([x, y, z])
    return np.hstack([pos, sph[:, 3:4]]).astype(np.float32)


def _create_tetris_spheres_np(shape, sph_radius=SPH_RADIUS):
    """`tetris_env.create_tetris_spheres`, in numpy."""
    coords = {"L": np.array([(0, 0, 0), (0, 1, 0), (0, -1, 0), (1, -1, 0)], float),
              "O": np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)], float)}[shape]
    nc = coords.shape[0]
    sph = np.zeros((nc + 2, 4))
    sph[:nc, :3] = coords * sph_radius * 2
    sph[:nc, 3] = sph_radius
    sph[nc] = [0.0, 0.0, -sph_radius * 1.25, sph_radius / 2]
    sph[nc + 1] = [0.0, 0.0, -sph_radius * 2, sph_radius / 2]
    sph[:, 2] += -sph[-1, 2]
    return sph


def _walls_to_spheres(walls, radius=0.02, overlap=0.75):
    """Approximate AABB walls as spheres for the clearance residual.

    The count is derived from each wall's LENGTH so consecutive spheres always
    overlap.  It used to be a fixed `n_per_edge=5` per wall, which made the
    approximation depend on how long the wall happened to be: on the widened
    goal the two 0.51 m walls got 5 spheres of radius 0.02 spaced 128 mm apart,
    leaving 88 mm GAPS.  The cost then saw a dotted line rather than a wall,
    and a 6 cm block passed between the dots for free -- which is what let a
    plan carry a block in through the side of the box at zero clearance cost.

    `overlap` is the fraction of a sphere diameter to step by; below 1.0 the
    spheres overlap and the wall is sealed.
    """
    step = 2.0 * radius * overlap
    spheres = []
    for w in walls:
        x1, y1, z1 = float(w[0]), float(w[1]), float(w[2])
        x2, y2, z2 = float(w[3]), float(w[4]), float(w[5])
        length = max(abs(x2 - x1), abs(y2 - y1))
        height = abs(z2 - z1)
        n_len = max(int(np.ceil(length / step)) + 1, 2)
        n_h = max(int(np.ceil(height / step)) + 1, 1)
        for i in range(n_len):
            t1 = i / max(n_len - 1, 1)
            for j in range(n_h):
                t2 = j / max(n_h - 1, 1) if n_h > 1 else 0.5
                spheres.append([x1 + t1 * (x2 - x1), y1 + t1 * (y2 - y1),
                                z1 + t2 * (z2 - z1), radius])
    return jnp.array(spheres, dtype=jnp.float32)
