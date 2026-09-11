"""Experiment 9: IOSP on block stacking — can the composed implicit adjoint
recover cost preferences for a vertical tower construction task?

Same structure as `e8_tetris.py` but uses `iosp.model.tower` with the
z-alignment feature.  The N=1 case (place one block on the table) is the
base; N=3 tests scaling where each successive block adds obstacles.

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    XLA_FLAGS="--xla_disable_hlo_passes=fusion" \\
        python -m iosp.experiments.e9_tower --seed 0 --stack-level 0
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from iosp import config
config.enable_compilation_cache()

from ioc import outer as outer_opt
from ioc.inner import make_inner_solver
from iosp.model import tower as tw
from iosp.config import URDF_PATH, SRDF_PATH, MESH_DIR

# effort/smooth raised from (0.5, 1.5); see `e8_tetris.Z_STAR` for the
# reasoning.  Tower was the worse of the two -- joint6 commanded to 298 deg
# against a 215 deg limit, 84 deg of overshoot -- and unlike tetris the
# violation genuinely broke the task: 0/6 executed rollouts at the old weights
# against 6/6 here.
# clearance raised 1.0 -> 3.0 (4.3% -> 24.8% of the softmax).  Measured over
# logits 1/2/3/4: the tallest passing stack is 8 levels at ALL of them, so the
# weight is free, and it buys real margin on the obstacles the arm CAN avoid --
# closest-approach on the binding one goes 62.6 -> 91.5 mm (102.3 at logit 4,
# which was not taken: at 47% of the basis clearance starts to swamp the other
# six features).
#
# It cannot fix every clip.  The worst remaining approach, -9.7 mm into the
# sphere at [0, -0.1, 0.9], is at ROW 0 -- `q_start`, a PINNED row -- and comes
# from `sample_tower_scenes`' 0.05 rad jitter on Q_HOME, which itself clears by
# +19.7 mm.  No cost weight moves a pinned row.
Z_STAR = jnp.array([3.5, 2.5, 3.0, 0.8, 1.2, 1.0, 2.0], dtype=jnp.float32)  # ...,z_align,torque,skeleton
PARAM_NAMES = list(tw.FEATURE_NAMES)


def build(seed=0, n_iters=60, n_scenes=6, stack_level=0, jitter_q=0.05,
          fit_scenes=None, test_scenes=None):
    """`fit_scenes`/`test_scenes` override the sampler.

    Pass them when two conditions have to be graded against the SAME held-out
    set -- `e12_demo_diversity` compares a narrow demo set against a randomised
    one, and that comparison is only meaningful if the thing they are both
    tested on is held fixed.  The inner solvers are calibrated on whichever
    scenes end up as `fit`, which is the existing behaviour.
    """
    prob = tw.TowerProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    # See `e8_tetris.build`: IOSP_STOCK_SOLVER=1 restores the early-stopping
    # `while_loop`, which is much cheaper to compile when not differentiating.
    stock = os.environ.get("IOSP_STOCK_SOLVER", "0") == "1"
    fs = tw.make_tower_forward_solver(n_iters=n_iters, robot=prob.base.robot,
                                      stock=stock)

    if fit_scenes is None or test_scenes is None:
        rng = np.random.default_rng(seed)
        scenes_all = tw.sample_tower_scenes(rng, 2 * n_scenes,
                                            stack_level=stack_level,
                                            jitter_q=jitter_q)
        fit = jax.tree.map(lambda a: a[:n_scenes], scenes_all)
        test = jax.tree.map(lambda a: a[n_scenes:], scenes_all)
    else:
        fit, test = fit_scenes, test_scenes

    key = jax.random.PRNGKey(seed)
    x0, seg_scenes, q_pick, q_place, q_prepick, q_preplace = prob.seeds(fit)

    inner_by_phase = {}
    for p in tw.PHASES:
        rf = prob.segment_residual_fn(p)
        scales = prob.calibrate_segment(p, rf, seg_scenes[p], key)
        inner_by_phase[p] = make_inner_solver(rf, scales, forward_solver=fs)

    full_rf = prob.full_residual_fn()
    full_sc = tw.TowerFullScene(
        fit.q_start, fit.q_start,
        fit.obs_center, fit.obs_radius,
        q_pick, q_place, fit.target_z, q_prepick, q_preplace)
    full_scales = prob.calibrate_full(full_rf, full_sc, key)
    refine = make_inner_solver(full_rf, full_scales, forward_solver=fs)

    return dict(prob=prob, fit=fit, test=test, inner_by_phase=inner_by_phase,
                refine=refine, fs=fs, seed=seed, stack_level=stack_level,
                full_scales=full_scales)


def ee_paths(built, scenes, z, *, stage2=True):
    prob, refine = built["prob"], built["refine"]
    theta = jax.nn.softmax(z)
    theta_seg = theta[:tw.K_SEG]
    theta_full = theta

    xs, seg_scenes, full_sc, _, _ = prob.solve(
        scenes, built["inner_by_phase"], theta_seg, theta_full,
        refine, stage2=stage2)

    q = jax.vmap(prob.seg["full"].unpack)(xs["full"], full_sc)
    return jax.vmap(prob.ee_positions)(q)


def make_loss(built, scenes, demos, *, stage2=True):
    def loss(z):
        p = ee_paths(built, scenes, z, stage2=stage2)
        return jnp.mean(jnp.sum((p - demos) ** 2, axis=-1))
    return loss


def fit_z(loss_and_grad, starts, *, lr=0.05, n_steps=40):
    best = None
    for z0 in starts:
        z, trace = outer_opt.adam(loss_and_grad, z0, lr=lr, n_steps=n_steps)
        lN = min(v for _, v in trace)
        if best is None or lN < best["lN"]:
            best = dict(z=z, z0=z0, l0=trace[0][1], lN=lN, trace=trace)
    a0, ah = np.asarray(best["z0"], float), np.asarray(best["z"], float)
    mv = float(np.linalg.norm(ah - a0) / (np.linalg.norm(a0) + 1e-30))
    red = best["l0"] / max(best["lN"], 1e-30)
    best.update(move_rel=mv, loss_reduction=red,
                degenerate=(not np.isfinite(mv)) or mv < 1e-3 or red < 1.05)
    return best


def run(seed=0, n_iters=60, n_scenes=6, n_steps=40, n_starts=8,
        stack_level=0, out=None):
    t_wall_start = time.perf_counter()
    t0 = time.perf_counter()
    built = build(seed=seed, n_iters=n_iters, n_scenes=n_scenes,
                  stack_level=stack_level)
    fit, test = built["fit"], built["test"]
    print(f"[build] {time.perf_counter()-t0:.1f}s  K={tw.K} {PARAM_NAMES}  "
          f"stack_level={stack_level}  fit/test={n_scenes}/{n_scenes}", flush=True)

    t0 = time.perf_counter()
    demos_fit = jax.jit(lambda: ee_paths(built, fit, Z_STAR))()
    demos_test = jax.jit(lambda: ee_paths(built, test, Z_STAR))()
    jax.block_until_ready((demos_fit, demos_test))
    print(f"[demos] {time.perf_counter()-t0:.1f}s, shape {tuple(demos_fit.shape)}",
          flush=True)

    gf = jax.jit(jax.value_and_grad(make_loss(built, fit, demos_fit)))
    jt = jax.jit(make_loss(built, test, demos_test))

    sanity_val, sanity_grad = gf(Z_STAR)
    sanity = float(sanity_val)
    print(f"[sanity] loss(z_star) = {sanity:.3e}", flush=True)
    assert sanity < 1e-4, f"model does not reproduce its own demo: {sanity:.3e}"

    _, g0 = gf(jnp.zeros(tw.K, jnp.float32))
    g0 = np.asarray(g0)
    print(f"[grad] dL/dtheta at z=0: {g0}", flush=True)

    rng = np.random.default_rng(seed + 1)
    z_zero = jnp.zeros(tw.K, dtype=jnp.float32)
    starts = [z_zero] + [jnp.asarray(rng.normal(0, 1, tw.K), jnp.float32)
                         for _ in range(n_starts - 1)]

    result = {}
    t_fit = time.perf_counter()
    b = fit_z(gf, starts, n_steps=n_steps)
    t_fit = time.perf_counter() - t_fit
    b["test_rmse"] = float(jnp.sqrt(jt(b["z"])))
    result["fit"] = b
    print(f"[fit] test RMSE {b['test_rmse']:.5f}  move_rel={b['move_rel']:.3e} "
          f"reduction={b['loss_reduction']:.2f}x "
          f"{'DEGENERATE' if b['degenerate'] else 'ok'}  "
          f"fit_time={t_fit:.1f}s", flush=True)

    result["baseline"] = float(jnp.sqrt(jt(z_zero)))
    result["oracle"] = float(jnp.sqrt(jt(Z_STAR)))

    th = np.asarray(jax.nn.softmax(b["z"]))
    ths = np.asarray(jax.nn.softmax(Z_STAR))
    result["theta_hat"], result["theta_star"] = th, ths
    result["param_err"] = float(np.linalg.norm(th - ths))

    t_wall = time.perf_counter() - t_wall_start
    result["wall_s"] = t_wall
    result["fit_s"] = t_fit
    print(f"[time] wall={t_wall:.1f}s  fit={t_fit:.1f}s", flush=True)

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez_compressed(out, **{k: v for k, v in result.items()
                                    if not isinstance(v, dict)},
                            test_rmse=b["test_rmse"],
                            seed=seed, stack_level=stack_level)
        print(f"[out] wrote {out}", flush=True)

    return result


def report(result):
    ref = result["baseline"]
    print("\n=== held-out behavioural recovery (EE RMSE, m) ===")
    for n, v in (("baseline uniform (theta = 1/K)", ref),
                 ("tower IOSP fit", result["fit"]["test_rmse"]),
                 ("oracle theta*", result["oracle"])):
        print(f"  {n:36s} {v:9.5f}   {100*(1-v/ref):+6.1f}% vs uniform")

    if result["fit"]["degenerate"]:
        print(f"\nNO VERDICT: fit degenerate")
        return

    print(f"\nparam_err (reported, NOT criterion): {result['param_err']:.4f}")
    for n, h, s in zip(PARAM_NAMES, result["theta_hat"], result["theta_star"]):
        print(f"  {n:12s} {h:8.4f}  {s:8.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--n-scenes", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--stack-level", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report(run(seed=a.seed, n_iters=a.n_iters, n_scenes=a.n_scenes,
               n_steps=a.n_steps, n_starts=a.n_starts,
               stack_level=a.stack_level, out=a.out))
