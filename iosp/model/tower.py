"""Block-stacking IOSP model: inverting a differentiable planner for
vertical tower construction on a Panda, adapting SPaSM's tower problem.

The forward model
-----------------
A Panda picks blocks from scattered initial positions and stacks them into a
vertical tower.  Each block is a 6cm cube (matching SPaSM's
`spasm.tower.env.TowerSimulation`).  The trajectory for each block is a
two-phase composed plan identical in structure to `tetris.py`:

  1. **Pick-to-place**: IK for the pick pose and place pose (atop the
     previous block or at the base), then per-segment trajopt.
  2. **Return**: trajopt back to home.

The stacking constraint distinguishes this from tetris: each block's place
target is at z = block_height * (stack_level + 0.5), and the cost includes a
z-alignment term that penalizes deviation from the target height.

Cost features (tied across all segments)
-----------------------------------------
  ``effort``     velocity norm
  ``smooth``     acceleration norm
  ``clearance``  obstacle avoidance (static obstacles + already-stacked blocks)
  ``orient``     EE tilt away from pointing downward
  ``z_align``    z-height deviation from the stacking target
  ``skeleton``   deviation from the task skeleton (pick/place poses)

theta = softmax(z) over these 6 features.
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
from iosp.model.tetris import (
    _multi_sphere_clearance, _orient_residual, _target_pose_batch,
    _ik_batch, _torque_residual, _self_collision_residual, CLEARANCE_MARGIN, SOFTMIN_TAU, SOFTNESS,
    DOWN_WXYZ, UP_AXIS, IK_RNG_KEY, IK_CONTINUITY_WEIGHT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BLOCK_SPHERES = 8
BLOCK_DIM = 0.06
BLOCK_HALF = BLOCK_DIM / 2.0

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

# Vertical STANDOFF above the grasp and the placement.
#
# Nothing else in this cost makes the last approach row sit over the block: the
# segment is free to reach the pick pose from whatever direction is cheapest,
# and measured it comes in 49-61 deg off vertical, sweeping up to 140 mm
# sideways in the final row.  Under physics that is not a near-miss but a
# collision -- the open finger pads strike the cube on the way down and plow it
# 64 mm out of the grip, so the fingers close on nothing (tower levels 3 and 4
# never left the table).  Pinning the row BEFORE each grasp row to the same
# pose lifted straight up makes that last row a vertical descent, which is what
# every pick-and-place stack does and what `pickplace` already gets from its
# fitted `theta_ik` standoff.
#
# These pin rows that already exist -- `N_FULL` and the phase lengths are
# unchanged -- so this costs the approach one of its six free travel rows and
# the place-transport one of its eight.
# The RETREAT after the release needs the same treatment, and for the same
# reason: measured, the row after the release moved 91 mm sideways while rising
# only 59 mm, so the fingers -- open by then, and 40 mm off the hand axis --
# swept across the cube they had just set down and dragged it 60 mm off the
# stack (tower level 3 placed to 5 mm, then lost it on the way out).  Sending
# the hand back to the SAME standoff it descended from makes the departure
# vertical.  There is no matching pin after the grasp: the block is held at
# that point, so a diagonal lift carries it rather than knocking it.
# [m] straight up from the grasp pose.  A TALLER standoff is not a safer one:
# the pose above a placement is reached by folding the wrist, and past about
# 6 cm the arm starts running out of joint 5 and the "vertical" standoff drifts
# sideways instead -- measured lateral error at the place standoff, per scene:
#   4 cm  [ 0.1 17.3 30.6  0.2  0.1  0.0]
#   6 cm  [ 0.0  8.6 39.6  0.1  0.1  0.2]   <- flattest overall
#   8 cm  [ 6.3  4.0 48.5  0.1  0.1  0.2]
#  10 cm  [26.7 13.1 57.1  0.1  0.1  0.1]
# and a standoff that is 27 mm off to the side is not a vertical descent at all,
# which is the whole reason the pin exists.
PRE_GRASP_STANDOFF = 0.06
STANDOFF_SELECT_WEIGHT = 0.25    # standoff's share of the yaw-branch score
IDX_PRE_PICK = IDX_PICK - 1
IDX_PRE_PLACE = IDX_PLACE - 1
IDX_POST_PLACE = IDX_PLACE + 1

FEATURE_NAMES = ("effort", "smooth", "clearance", "orient", "z_align",
                 "torque", "skeleton")
K = len(FEATURE_NAMES)
SEGMENT_FEATURES = ("effort", "smooth", "clearance", "orient", "z_align",
                    "torque")
K_SEG = len(SEGMENT_FEATURES)

Q_HOME = jnp.array([0.0, -0.785, 0.0, -1.571, 0.0, 1.571, 0.785],
                    dtype=jnp.float32)

# NOT a standoff: the gap between the frame pyroffi's FK calls the end effector
# (`panda_link7` + 0.107 m, the HAND ORIGIN) and the frame SPaSM plans in
# (`panda_grasptarget`, a further 0.105 m between the fingertips).  SPaSM sets
# the grasp pose equal to the block pose, so matching it means putting our EE
# exactly this far above the block.  The old 0.08 drove `panda_grasptarget`
# 2.5 cm below the block centre -- through the cube rather than around it.
GRASP_OFFSET = 0.105
STANDOFF = GRASP_OFFSET          # back-compat alias


# ---------------------------------------------------------------------------
# Scene dataclasses
# ---------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TowerScene:
    """Context for one block-stacking pick-place demonstration."""
    q_start: jnp.ndarray        # (dof,)
    pick_pos: jnp.ndarray       # (3,) EE target at block's current position
    place_pos: jnp.ndarray      # (3,) EE target at stack position
    target_z: jnp.ndarray       # (1,) target z height for the placed block
    obs_center: jnp.ndarray     # (N_obs, 3) obstacle sphere centers
    obs_radius: jnp.ndarray     # (N_obs,) obstacle sphere radii
    pick_yaw: jnp.ndarray       # (,) block yaw at its spawn cell
    place_yaw: jnp.ndarray      # (,) block yaw in the solved stack
    level: jnp.ndarray          # (,) which stack level this scene builds


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TowerFullScene:
    """Context for the stage-3 refine solve."""
    q_start: jnp.ndarray
    q_goal: jnp.ndarray         # = q_home
    obs_center: jnp.ndarray
    obs_radius: jnp.ndarray
    q_pick: jnp.ndarray
    q_place: jnp.ndarray
    target_z: jnp.ndarray       # (1,)
    q_prepick: jnp.ndarray      # PRE_GRASP_STANDOFF above the pick
    q_preplace: jnp.ndarray     # PRE_GRASP_STANDOFF above the place


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TowerSegScene:
    """Segment scene with z-alignment target."""
    q_start: jnp.ndarray
    q_goal: jnp.ndarray
    obs_center: jnp.ndarray
    obs_radius: jnp.ndarray
    target_z: jnp.ndarray       # (1,)


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class TowerProblem:
    """Block-stacking IOSP problem."""

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
            pinned_rows=((IDX_PRE_PICK, "q_prepick"), (IDX_PICK, "q_pick"),
                         (IDX_PRE_PLACE, "q_preplace"), (IDX_PLACE, "q_place"),
                         (IDX_POST_PLACE, "q_preplace")))
        return TowerProblem(base=base, seg=seg)

    def yaw_symmetric_pair_ik(self, pos, refs, yaw):
        """IK for a grasp row AND its standoff -> (q_at, q_standoff, yaw).

        The blocks are CUBES: gripping one at yaw, yaw+90, yaw+180 or yaw+270
        is the same physical grasp, and `spasm_tasks.tower_success` scores
        height, support offset and tilt -- never yaw.  Pinning one particular
        yaw is therefore a constraint the task never asked for, and it is not
        free: it is joint 5, the wrist flex, that runs out of range on this
        stack (the level-2 place wanted 221.0 deg against a 219 deg URDF limit
        and the executing Panda's 215), and which multiple of 90 deg the wrist
        takes decides whether it does.  Solving all four and keeping whichever
        actually reaches the target took the level-1 place from 55 mm to 0.1.

        The four are solved as one batched call, and the winner is chosen AFTER
        the joint-limit clip, so a branch that only reaches by saturating a
        joint loses to one that genuinely reaches.

        The row and its standoff are scored TOGETHER, by the worse of the two
        residuals, and share the winning yaw.  Choosing them independently is
        wrong twice over: the wrist would have to spin 90 deg during what is
        supposed to be a straight vertical descent, sweeping the held cube's
        corners out to r*sqrt(2) right above the stack; and a yaw that reaches
        the placement beautifully is not necessarily one that reaches the pose
        100 mm above it (measured: 0.11 mm at the place, 56.6 mm at its
        standoff, same yaw).

        This is applied to the PLACE side only.  The pick already solves to
        0.03-0.3 mm at the skeleton's own yaw, and searching there is actively
        harmful: `q_pick` seeds the place IK, so a pick branch chosen without
        reference to the place can strand it (measured, place went to 37-54 mm).
        """
        if yaw is None:
            q = _ik_batch(self, pos, refs, None)
            return q, self.standoff_ik(pos, q, None), None
        n_yaw = 4
        up = jnp.array([0.0, 0.0, PRE_GRASP_STANDOFF], pos.dtype)
        offs = jnp.arange(n_yaw, dtype=jnp.float32) * (jnp.pi / 2.0)
        yaws = yaw[None, :] + offs[:, None]                       # (4, n)

        # UNROLLED, not `jax.vmap`ed over the yaw axis, and that is load-bearing.
        # `sqp_ik_solve_cuda_batch` seeds each problem from `IK_RNG_KEY` AND its
        # POSITION in the batch (see its docstring), so folding the four yaws
        # into one 4*n batch seeds them differently from four separate n-row
        # calls and lands on different IK branches -- measured, it cost one
        # scene's place row 0.2 mm -> 37.6 mm.  Four is static, so this unrolls
        # under jit and each yaw keeps the batch shape it would have alone.
        q_at, q_off = [], []
        for k in range(n_yaw):
            a = _ik_batch(self, pos, refs, yaws[k])
            q_at.append(a)
            q_off.append(_ik_batch(self, pos + up, a, yaws[k]))
        q_at, q_off = jnp.stack(q_at), jnp.stack(q_off)           # (4, n, dof)

        def err_of(q, target):
            ee = self.base.ee_positions(q.reshape(-1, q.shape[-1]))
            return jnp.linalg.norm(ee.reshape(n_yaw, -1, 3) - target[None], axis=-1)

        # The two residuals are NOT equally important, and scoring them by the
        # worse of the pair was measured to be actively wrong: it traded a
        # 0.2 mm placement for a better standoff and left the block 37 mm off.
        # The grasp row decides where the block ends up; the standoff only
        # shapes the approach, and a standoff 25 mm off still descends within
        # 14 deg of vertical -- against the 50-60 deg the plan had with no
        # standoff at all.  So the standoff is a tie-breaker, not a veto.
        err = err_of(q_at, pos) + STANDOFF_SELECT_WEIGHT * err_of(q_off, pos + up)
        best = jnp.argmin(err, axis=0)                            # (n,)
        take = lambda a: jnp.take_along_axis(a, best[None, :, None], axis=0)[0]
        return (take(q_at), take(q_off),
                jnp.take_along_axis(yaws, best[None, :], axis=0)[0])

    def pick_ik(self, pick_pos, refs, pick_yaw=None):
        return _ik_batch(self, pick_pos, refs, pick_yaw)

    def place_ik(self, place_pos, q_pick, place_yaw=None):
        return _ik_batch(self, place_pos, q_pick, place_yaw)

    def standoff_ik(self, pos, refs, yaw=None):
        """IK for the pose `PRE_GRASP_STANDOFF` straight above `pos`.

        Seeded from the grasp configuration it stands off from, so continuity
        picks the IK branch that reaches the grasp by descending rather than by
        reconfiguring the arm between the two rows.
        """
        up = jnp.array([0.0, 0.0, PRE_GRASP_STANDOFF], pos.dtype)
        return _ik_batch(self, pos + up, refs, yaw)

    # -- z-alignment residual -----------------------------------------------

    def _z_align_residual(self, q, target_z):
        """Penalty for EE z deviating from the target stacking height.
        Returns (T,) residual."""
        ee = self.base.ee_positions(q)  # (T, 3)
        return ee[:, 2] - target_z[0]

    # -- residuals -----------------------------------------------------------

    def segment_residual_fn(self, phase):
        problem = self.seg[phase]

        def residual_fn(x_flat, scene: TowerSegScene):
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
            orient = _orient_residual(self.base.robot, self.ee_index, q)
            z_align = self._z_align_residual(q, scene.target_z)
            torque = _torque_residual(self.base.robot, q)
            return (v.reshape(-1), a.reshape(-1), clearance, orient, z_align,
                    torque)

        return residual_fn

    def full_residual_fn(self):
        problem = self.seg["full"]

        def residual_fn(x_flat, scene: TowerFullScene):
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
            orient = _orient_residual(self.base.robot, self.ee_index, q)
            z_align = self._z_align_residual(q, scene.target_z)
            torque = _torque_residual(self.base.robot, q)
            skel = jnp.concatenate([
                q[IDX_PICK] - scene.q_pick,
                q[IDX_PLACE] - scene.q_place])
            return (v.reshape(-1), a.reshape(-1), clearance, orient,
                    z_align, torque, skel)

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
        assert bool(jnp.all(scales > 1e-8)), \
            f"{phase}: degenerate feature scale {scales}"
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

    def seeds(self, scenes: TowerScene):
        # The standoffs take the SAME yaw the grasp row settled on, not the
        # skeleton's: the standoff exists so the last 100 mm is a straight
        # vertical descent, and a wrist that spins 90 deg on the way down is not
        # that.
        pick_yaw = getattr(scenes, "pick_yaw", None)
        q_pick = self.pick_ik(scenes.pick_pos, scenes.q_start, pick_yaw)
        q_prepick = self.standoff_ik(scenes.pick_pos, q_pick, pick_yaw)
        q_place, q_preplace, _ = self.yaw_symmetric_pair_ik(
            scenes.place_pos, q_pick, getattr(scenes, "place_yaw", None))

        seg_scenes = {
            "approach": TowerSegScene(scenes.q_start, q_pick,
                                      scenes.obs_center, scenes.obs_radius,
                                      scenes.target_z),
            "place_traj": TowerSegScene(q_pick, q_place,
                                         scenes.obs_center, scenes.obs_radius,
                                         scenes.target_z),
            "return_traj": TowerSegScene(q_place, scenes.q_start,
                                          scenes.obs_center, scenes.obs_radius,
                                          scenes.target_z),
        }
        x0 = {p: jax.vmap(self.seg[p].seed)(seg_scenes[p]) for p in PHASES}
        return x0, seg_scenes, q_pick, q_place, q_prepick, q_preplace

    def solve(self, scenes, inner_by_phase, theta_seg, theta_full,
              refine_inner, *, stage2=True):
        x0, seg_scenes, q_pick, q_place, q_prepick, q_preplace = self.seeds(scenes)

        xs = {}
        for phase in PHASES:
            xs[phase] = jax.vmap(
                inner_by_phase[phase].solve_implicit,
                in_axes=(0, None, 0))(x0[phase], theta_seg, seg_scenes[phase])

        full_sc = TowerFullScene(
            scenes.q_start, scenes.q_start,
            scenes.obs_center, scenes.obs_radius,
            q_pick, q_place, scenes.target_z, q_prepick, q_preplace)

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


def make_tower_forward_solver(n_iters=60, robot=None, method=None, gd_lr=0.1,
                              *, stock=False):
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt
    method = method or os.environ.get("IOSP_TRAJOPT", "lbfgs")
    if method == "projected_gd":
        margin = float(os.environ.get("IOSP_JOINT_MARGIN", "0.07"))
        lo = tuple(float(v) + margin for v in np.asarray(robot.joints.lower_limits))
        hi = tuple(float(v) - margin for v in np.asarray(robot.joints.upper_limits))
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

BASE_XY = jnp.array([0.45, 0.0], dtype=jnp.float32)


def _block_obstacle_spheres(block_pos, block_half=BLOCK_HALF, n_spheres=8):
    """Approximate a cube at `block_pos` (x,y,z center) as collision spheres."""
    h = block_half
    offsets = np.array([
        [-h, -h, -h], [-h, -h, h], [-h, h, -h], [-h, h, h],
        [h, -h, -h], [h, -h, h], [h, h, -h], [h, h, h],
    ], dtype=np.float32)
    r = h * 0.5
    centers = np.asarray(block_pos) + offsets
    return centers, np.full(n_spheres, r, dtype=np.float32)


def sample_tower_scenes(rng, n, stack_level=None, num_blocks=10, jitter_q=0.05):
    """Scenes that walk SPaSM's stack, with geometry taken from SPaSM itself.

    Scene i places LEVEL ``i % num_blocks`` of `spasm_tasks.tower_skeleton()`:
    it picks that block from its SPaSM spawn cell and puts it at the solved
    pose for that level -- xy, height and yaw -- with every lower level already
    stacked and therefore an obstacle.  Pass `stack_level` to pin every scene to
    one level instead.

    Nothing geometric is defined here.  In particular the obstacles are SPaSM's
    ten spheres, not the three this module used to invent -- one of which had
    radius 0.2 at the origin, with the robot's own base inside it.
    """
    from iosp.model import spasm_tasks as ST

    from iosp.model import spasm_costs as SC
    skeleton = ST.tower_skeleton(num_blocks)
    spawn = ST.tower_init_state(num_blocks)
    obs_c_all = SC.TOWER_OBSTACLE_POSES.astype(np.float32)
    obs_r_all = SC.TOWER_OBSTACLE_RADII.astype(np.float32)
    bdim = float(SC.BLOCK_DIMS[2])

    up = np.array([0.0, 0.0, GRASP_OFFSET], dtype=np.float32)
    n_obs = obs_c_all.shape[0] + (num_blocks - 1) * N_BLOCK_SPHERES

    qs, picks, places, p_yaw, q_yaw, tgts, lvls, obs_c, obs_r = ([] for _ in range(9))
    for i in range(n):
        k = (i % num_blocks) if stack_level is None else int(stack_level)
        q0 = np.asarray(Q_HOME) + rng.normal(scale=jitter_q, size=7).astype(np.float32)

        oc, orr = [obs_c_all], [obs_r_all]
        for lvl in range(k):
            bc, br = _block_obstacle_spheres(skeleton[lvl, :3])
            oc.append(bc)
            orr.append(br)
        pad = n_obs - sum(a.shape[0] for a in oc)
        if pad > 0:
            oc.append(np.full((pad, 3), 100.0, np.float32))
            orr.append(np.full(pad, 0.01, np.float32))

        qs.append(q0)
        picks.append(spawn[k, :3] + up)
        places.append(skeleton[k, :3] + up)
        p_yaw.append(np.float32(spawn[k, 3]))
        q_yaw.append(np.float32(skeleton[k, 3]))
        tgts.append(np.array([skeleton[k, 2] + GRASP_OFFSET], np.float32))
        lvls.append(np.int32(k))
        obs_c.append(np.concatenate(oc, axis=0))
        obs_r.append(np.concatenate(orr, axis=0))

    f32 = lambda a: jnp.asarray(np.stack(a), dtype=jnp.float32)
    return TowerScene(
        q_start=f32(qs), pick_pos=f32(picks), place_pos=f32(places),
        target_z=f32(tgts), obs_center=f32(obs_c), obs_radius=f32(obs_r),
        pick_yaw=f32(p_yaw), place_yaw=f32(q_yaw),
        level=jnp.asarray(np.stack(lvls), dtype=jnp.int32))
