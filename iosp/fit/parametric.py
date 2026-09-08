"""Path A: build the bilevel forward map for the KNOWN cost library.

This is the piece every path-A experiment shares, and until this refactor it
lived inside `study3_identifiable_refit` -- so recording an animation, running
the multistart, or sweeping the loss space all had to import an experiment
script and inherit its `__main__`, its RKHS code and its argument parsing.

`build_parametric` returns everything a caller needs to fit and to score:
the jitted value-and-grad, the rollout in the loss space AND in EE space, the
demonstrations, the Jacobian for the sensitivity Gram, and the ground truth.

Always screen before trusting a spectrum.  `screen_stationarity` checks that
the inner solves actually converged: `G = J^T J` is built from `dx*/dtheta`,
which only exists where `x*(theta)` is differentiable, so a non-converged inner
solve makes the eigenvalues meaningless rather than merely noisy.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ioc import identifiability as ident
from ioc.inner import make_inner_solver
from iosp import config
from iosp.config import THETA_IK_STAR, Z_TRAJOPT_STAR, URDF_PATH, SRDF_PATH, MESH_DIR
from iosp.fit.params import z_scale
from iosp.model import pickplace as pp
from iosp.model.scenes import scene_a, scene_b, scenes_ab
from iosp.model.pickplace import split_trajopt as _split_trajopt

# ---------------------------------------------------------------------------
# Path A -- known cost library (theory doc §2)
# ---------------------------------------------------------------------------

def screen_stationarity(prob, scenes, inner, theta_ik, by_phase, label,
                        tol=1e-3):
    """CONFOUND 4.  `ioc/inner.py` is explicit that the implicit adjoint is only
    valid at a CONVERGED inner solve -- on contexts that plateau, adjoint-vs-FD
    agreement collapses from cos 0.9999 to 0.59, and the Jacobian this study
    eigendecomposes is exactly that adjoint.  The forward solver runs a fixed
    60 iterations, a budget calibrated on the NAMED cost and never re-checked
    for the held-out scene or for the RKHS cost, which is a different surface
    entirely.  So: measure `||grad_x C||` per phase per scene and report it.

    Reported, not asserted: a plateaued context invalidates the spectrum, and
    the right response is to see that in the log next to the numbers rather
    than to crash a 25-minute run.
    """
    x0, phase_scenes, _, _ = prob.seeds(scenes, theta_ik)
    worst = 0.0
    print(f"  [{label}] inner stationarity ||grad_x C|| per phase:", flush=True)
    for p in pp.PHASES:
        s = jax.vmap(inner[p].stationarity, in_axes=(0, None, 0))(
            x0[p], by_phase[p], phase_scenes[p])
        s = np.asarray(s)
        m = float(s.max())
        # `max(x, nan)` returns x, so a NaN phase would silently pass as OK
        worst = float("nan") if not np.isfinite(m) else (
            max(worst, m) if np.isfinite(worst) else worst)
        flag = ("  <-- NaN" if not np.isfinite(m)
                else "  <-- NOT CONVERGED" if m > tol else "")
        print(f"    {p:10s} worst={m:.3e} over {s.size} scenes "
              f"[{' '.join(f'{v:.1e}' for v in s.ravel())}]{flag}", flush=True)
    if worst > tol:
        print(f"  [{label}] WARNING: worst stationarity {worst:.3e} > {tol:.0e}; "
              "the sensitivity spectrum below is NOT trustworthy", flush=True)
    return worst


def _build_inner(prob, scenes, theta_ik, forward_solver, seed=0,
                 residual_override=None, n_restarts=1, restart_jitter=0.35,
                 constraints_by_phase=None):
    """`constraints_by_phase` (optional) maps a phase name to a
    `constraints_fn(ctx) -> tuple[AugmentedLagrangianTerm]`; that phase's forward
    solve then becomes AL-constrained and its implicit adjoint linearizes the
    augmented stationarity (see `ioc.inner`).  Constraints must be
    theta-INDEPENDENT.  `None` (default) is byte-identical to the unconstrained
    build every existing caller relies on."""
    x0, phase_scenes, _, _ = prob.seeds(scenes, theta_ik)
    inner = {}
    for p in pp.PHASES:
        residual_fn, _ = prob.make_segment_inner(p, forward_solver)
        if residual_override is not None and p in residual_override:
            residual_fn = residual_override[p]
        scales = prob.calibrate_segment(p, residual_fn, phase_scenes[p],
                                        jax.random.PRNGKey(seed))
        cfn = None if constraints_by_phase is None else constraints_by_phase.get(p)
        # `n_restarts=1` (the default) is byte-for-byte the previous behaviour
        # for every existing caller.  Above 1, each segment solve becomes the
        # best of `n_restarts` local solves, which is what makes x*(theta)
        # single-valued enough for the implicit adjoint when the segment is
        # multimodal -- measured on this model: the transport segment flips
        # sides of the obstacle for isolated outer steps, spiking held-out EE
        # RMSE 20-40x before snapping back.  See `ioc.inner.make_inner_solver`.
        inner[p] = make_inner_solver(residual_fn, scales,
                                     forward_solver=forward_solver,
                                     n_restarts=n_restarts,
                                     restart_jitter=restart_jitter,
                                     constraints_fn=cfn)
    return inner, x0


def build_parametric(seed=0, n_iters=60, n_restarts=1, space="ee",
                     scene_b_scale=1.0):
    """`space` selects the OUTER LOSS coordinates, and nothing else.

    "ee" (default) is byte-for-byte every existing caller: the loss is mean
    squared EE position error, (T, 3).  "joint" scores the same waypoints in
    configuration space, (T, dof), which makes the redundant arm's self-motion
    manifold observable to the outer loss -- `full_ee_path` cannot see it, which
    is why an IK winner flip costs ~0 there and 2.5-4.4 rad here.

    Both modes always expose `ee_paths_fn`/`ee_demo`, so a joint-space fit is
    still scored on the EE criterion the paper reports and the two runs are
    directly comparable.
    """
    if space not in ("ee", "joint"):
        raise ValueError(f"space must be 'ee' or 'joint', got {space!r}")
    prob = pp.PickPlaceProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    forward_solver = pp.make_composed_forward_solver(n_iters=n_iters)
    scenes = scenes_ab(scene_b_scale)
    theta_trajopt_star = jax.nn.softmax(Z_TRAJOPT_STAR)

    # Feature scales are CALIBRATED ON SCENE A ONLY and then applied to both
    # scenes.  Calibrating per-scene (the previous version) would silently let
    # the held-out scene re-normalize the very features being tested, which
    # makes the generalization number optimistic; scales belong to the fitted
    # model, not to the evaluation scene.
    inner, _ = _build_inner(prob, scene_a(), THETA_IK_STAR, forward_solver, seed,
                            n_restarts=n_restarts)

    K = pp.K_IK + pp.K_TRAJOPT
    S = z_scale(K, pp.K_IK)  # confound 1: u is dimensionless, z = S * u

    def _rollout(u):
        z = S * u
        theta_ik, z_traj = z[: pp.K_IK], z[pp.K_IK :]
        x0, _, _, _ = prob.seeds(scenes, theta_ik)
        _, _, xs, ps = prob.solve(theta_ik, _split_trajopt(jax.nn.softmax(z_traj)),
                                  scenes, inner, x0)
        return xs, ps

    def ee_paths(u):
        """-> (2, T, 3): row 0 = scene A (fit), row 1 = scene B (held out)."""
        xs, ps = _rollout(u)
        return prob.full_ee_paths(scenes, xs, ps)

    def joint_paths(u):
        """-> (2, T, dof), the same waypoints in configuration space."""
        xs, ps = _rollout(u)
        return prob.full_joint_paths(scenes, xs, ps)

    paths = ee_paths if space == "ee" else joint_paths

    z_star = jnp.concatenate([THETA_IK_STAR, Z_TRAJOPT_STAR])
    u_star = z_star / S
    paths_j = jax.jit(paths)
    demo = jax.block_until_ready(paths_j(u_star))  # (2, T, .), theta_star rollouts
    # Always available, whatever the loss space: the EE criterion is what the
    # paper reports, so a joint-space fit still has to be scored on it.
    ee_paths_j = paths_j if space == "ee" else jax.jit(ee_paths)
    ee_demo = demo if space == "ee" else jax.block_until_ready(ee_paths_j(u_star))

    screen_stationarity(prob, scenes, inner, THETA_IK_STAR,
                        _split_trajopt(theta_trajopt_star), "path A (parametric)")

    def loss_a(u):
        return jnp.mean(jnp.sum((paths(u)[0] - demo[0]) ** 2, axis=-1))

    def rmse(u, i):
        P = paths_j(u)
        return float(jnp.sqrt(jnp.mean(jnp.sum((P[i] - demo[i]) ** 2, axis=-1))))

    def theta_of(u):
        z = np.asarray(S) * np.asarray(u)
        return np.concatenate([z[: pp.K_IK], np.asarray(jax.nn.softmax(z[pp.K_IK :]))])

    return dict(
        gf=jax.jit(jax.value_and_grad(loss_a)),
        # exposed so a recorder can plot the SAME rollout the gradient is
        # computed from; without it a caller rebuilding `paths` itself can
        # silently pair a restart-enabled rollout with a no-restart gradient.
        paths_fn=paths_j, demo_paths=demo,
        ee_paths_fn=ee_paths_j, ee_demo_paths=ee_demo, space=space,
        jac_fn=ident.make_jac_fn(lambda u: paths(u)[0]),
        rmse_a=lambda u: rmse(u, 0),
        rmse_b=lambda u: rmse(u, 1),
        u_star=u_star, K=K, n_ik=pp.K_IK,
        theta_of=theta_of,
        theta_star=np.concatenate([np.asarray(THETA_IK_STAR), np.asarray(theta_trajopt_star)]),
        names=list(pp.THETA_IK_NAMES) + list(pp.THETA_TRAJOPT_NAMES),
    )


