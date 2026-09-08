"""Shared primitives for IK solvers.

Provides the core residual function and constants shared by all IK solvers
in this package (_hjcd_ik, _ls_ik, etc.).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from .._robot import Robot

# Step-size candidates for the vectorised LM line search.
# All five are evaluated in parallel via vmap; the best is kept.
_LS_ALPHAS = jnp.array([1.0, 0.5, 0.25, 0.1, 0.025])


def split_cuda_and_post_constraints(
    constraints: Sequence | None,
    constraint_args: Sequence | None,
    constraint_weights: Sequence[float] | None,
    collision_constraint_indices: Sequence[int] | None,
    collision_free: bool,
) -> tuple[
    tuple,
    tuple,
    Float[Array, "n_constraints"] | None,
    tuple,
    tuple,
    Float[Array, "n_constraints"] | None,
]:
    """Split constraints into CUDA-scored and post-refinement groups.

    Collision constraints are selected by index in ``collision_constraint_indices``.
    When ``collision_free`` is False those constraints are dropped entirely.
    Non-collision constraints are always retained for post-refinement.
    """
    all_fns = tuple(constraints) if constraints else ()
    all_args = tuple(constraint_args) if constraint_args is not None else ()
    if len(all_fns) == 0:
        return (), (), None, (), (), None

    if len(all_args) != len(all_fns):
        if len(all_args) == 0:
            all_args = tuple(() for _ in range(len(all_fns)))
        else:
            raise ValueError(
                "constraint_args must be None/empty or match constraints length"
            )

    if constraint_weights is None:
        all_w = jnp.ones((len(all_fns),), dtype=jnp.float32)
    else:
        all_w = jnp.array(constraint_weights, dtype=jnp.float32)
        if all_w.shape[0] != len(all_fns):
            raise ValueError("constraint_weights length must match constraints length")

    collision_idx = set(int(i) for i in (collision_constraint_indices or ()))
    for idx in collision_idx:
        if idx < 0 or idx >= len(all_fns):
            raise ValueError("collision_constraint_indices contains an out-of-range index")

    non_collision_idx = [i for i in range(len(all_fns)) if i not in collision_idx]
    if collision_free:
        cuda_idx = list(range(len(all_fns)))
    else:
        cuda_idx = non_collision_idx

    cuda_fns = tuple(all_fns[i] for i in cuda_idx)
    cuda_args = tuple(all_args[i] for i in cuda_idx)
    cuda_w = all_w[jnp.array(cuda_idx, dtype=jnp.int32)] if len(cuda_idx) > 0 else None

    post_fns = tuple(all_fns[i] for i in non_collision_idx)
    post_args = tuple(all_args[i] for i in non_collision_idx)
    post_w = (
        all_w[jnp.array(non_collision_idx, dtype=jnp.int32)]
        if len(non_collision_idx) > 0
        else None
    )

    return cuda_fns, cuda_args, cuda_w, post_fns, post_args, post_w


@functools.partial(jax.jit, static_argnames=("target_link_index",))
def _ik_residual(
    cfg: Float[Array, "n_act"],
    robot: Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
) -> Float[Array, "6"]:
    """SE(3) log-map residual.  Layout: [pos(3), ori(3)]."""
    Ts_world_link = robot.forward_kinematics(cfg)
    T_actual = jaxlie.SE3(Ts_world_link[target_link_index])
    return (T_actual.inverse() @ target_pose).log()


def _ik_residual_kernel_convention(
    cfg: Float[Array, "n_act"],
    robot: Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
) -> Float[Array, "6"]:
    """The CUDA kernels' residual, in JAX. Layout: [pos(3), ori(3)].

    World-frame position difference stacked with a world-frame rotation vector::

        pos = p_ee - p_tgt
        ori = log(R_ee @ R_tgt^-1)

    This is NOT :func:`_ik_residual`, which is the SE(3) LOCAL log-map
    ``(T_actual^-1 @ T_target).log()``. The two differ by an invertible 6x6 A
    whose position block carries a rotation, so their Jacobians differ
    elementwise.

    Both are valid optimality conditions -- r = 0 at the same configurations --
    and the implicit-diff tangent is invariant to the choice, because for
    full-row-rank J_q::

        pinv(A J_q) (A J_t) = J_q^+ J_t

    But that invariance holds ONLY when J_q, J_t and J_theta all come from the
    SAME residual. The CUDA task-Jacobian kernel returns J_q in THIS convention,
    so the implicit rule uses this function for the remaining blocks. Pairing
    the kernel's J_q with `_ik_residual`'s J_t would produce a confidently wrong
    gradient with no error raised anywhere -- the reason the two live side by
    side with this note rather than one being quietly swapped for the other.
    """
    Ts_world_link = robot.forward_kinematics(cfg)
    T_actual = jaxlie.SE3(Ts_world_link[target_link_index])
    pos = T_actual.translation() - target_pose.translation()
    ori = (T_actual.rotation() @ target_pose.rotation().inverse()).log()
    return jnp.concatenate([pos, ori])


@jax.jit
def _adaptive_weights(f: Float[Array, "6"]) -> Float[Array, "6"]:
    """Adaptive position / orientation balance weights.

    When position error dominates, orientation residuals are down-weighted
    so the solver focuses on closing the translational gap first.  The scale
    is clipped to [0.05, 1.0] so orientation never loses all influence.
    """
    pos_err = jnp.linalg.norm(f[:3]) + 1e-8
    ori_err = jnp.linalg.norm(f[3:]) + 1e-8
    ori_scale = jnp.clip(pos_err / ori_err, 0.05, 1.0)
    return jnp.concatenate([jnp.ones(3), jnp.full(3, ori_scale)])


# ---------------------------------------------------------------------------
# Self-collision tables for the CUDA IK kernels
# ---------------------------------------------------------------------------

# Empty tables: the kernels read n_self_pairs == 0 as "self-collision disabled",
# so these are also the default for callers that pass no checker.
_EMPTY_SELF_TABLES = (
    jnp.zeros((0, 4), jnp.float32),   # sph_local
    jnp.zeros((1,), jnp.int32),       # link_start
    jnp.zeros((1,), jnp.int32),       # link_joint
    jnp.zeros((0,), jnp.int32),       # pair_i
    jnp.zeros((0,), jnp.int32),       # pair_j
)


def self_collision_table_arrays(robot, collision_checker):
    """Self-collision buffers for the CUDA IK kernels, empties when unavailable.

    TODO(task 6): activation is implicit in the checker's TYPE, so passing a
    collision_checker changes IK behaviour for every existing caller with no
    opt-in and no way to disable it short of dropping the checker. Add an
    explicit parameter (defaulting to current behaviour) and surface the SRDF
    requirement at the API level, not just in this docstring.

    Deliberately *not* a new user-facing argument. A caller that already passes
    ``collision_checker`` has expressed intent to avoid collisions, and a
    spherized model carries its own SRDF-filtered active-pair table -- so the
    self-collision check comes along automatically rather than needing to be
    requested separately. Anything else (capsule models, custom checkers) gets
    empties, which the kernels read as "disabled".

    ``sph_local`` travels with the tables because ``link_start`` indexes THIS
    buffer. It is not interchangeable with a solver's ``robot_spheres_local``,
    which drops links with no parent joint and so has different offsets.

    Note the SRDF requirement: a spherized model built WITHOUT one reports
    adjacent links as permanently overlapping, and every configuration would be
    rejected. That is a property of the model, not of this code path.
    """
    from ..collision._robot_collision import RobotCollisionSpherized
    from ..cuda_kernels.collision._fused_self_collision_ffi import static_arrays

    if robot is None or not isinstance(collision_checker, RobotCollisionSpherized):
        return _EMPTY_SELF_TABLES

    # Deliberately unguarded: a checker that IS spherized but whose tables fail
    # to build is a bug, and swallowing it here yields a silent no-op that looks
    # exactly like "self-collision had no effect".
    #
    # static_arrays() does plain-numpy conversion (np.asarray) on the checker's
    # geometry, which is only valid outside an active JAX trace. robot/
    # collision_checker are always concrete Python objects here (never a scan
    # carry or other dynamic value) -- but a caller of THIS function may itself
    # be running inside a jax.jit/lax.scan trace (e.g. bench_ik.py's timing
    # harness), where any jnp op -- even one applied to a closed-over constant
    # -- returns a Tracer, and np.asarray() on a Tracer always raises
    # TracerArrayConversionError. ensure_compile_time_eval forces this block to
    # run eagerly regardless of the surrounding trace, which is exactly the
    # constant-folding case it exists for.
    with jax.ensure_compile_time_eval():
        sph_local, link_start, link_joint, pair_i, pair_j = static_arrays(
            robot, collision_checker)
        return (jnp.asarray(sph_local, jnp.float32),
                jnp.asarray(link_start, jnp.int32),
                jnp.asarray(link_joint, jnp.int32),
                jnp.asarray(pair_i, jnp.int32),
                jnp.asarray(pair_j, jnp.int32))
