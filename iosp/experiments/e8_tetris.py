"""Experiment 8: IOSP on tetris-packing — can the composed implicit adjoint
recover cost preferences for a block-packing task?

Structure matches `e4_three_stage.py`:
  1. Build the tetris forward model (IK -> segment trajopt -> refine)
  2. Roll out synthetic demos at ground-truth Z_STAR
  3. Recover Z_STAR via multistart implicit-adjoint Adam
  4. Score held-out EE RMSE (the paper's criterion), report param_err as context

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    XLA_FLAGS="--xla_disable_hlo_passes=fusion" \\
        python -m iosp.experiments.e8_tetris --seed 0 --num-blocks 1
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
from iosp.model import tetris as tt
from iosp.config import URDF_PATH, SRDF_PATH, MESH_DIR

# effort/smooth raised from (0.5, 1.5) after scoring EXECUTED rollouts with
# `iosp.checks.dynamic_report`: at the old weights effort carried only 0.078 of
# the softmax against skeleton's 0.349, which left the free interior of the
# carry phase unpriced.  The plan then bulged the EE to 1.15 m -- past joint2's
# and joint6's hard stops by up to 45 deg -- while still hitting the pinned
# rows exactly, so every purely kinematic check passed.  At (3.5, 2.5) the
# overshoot is 0.0 deg and reach is unchanged.
# `held` sits between clearance and orient.  It is the CARRIED BLOCK's
# clearance, which this model had no term for at all: the arm's own clearance
# says nothing about the block in its gripper, so a plan sweeping the block
# through a goal wall cost zero.  SPaSM's trajopt weights the equivalent term
# 240x its arm-collision weight (1.20 vs 0.005), so it starts high here too.
Z_STAR = jnp.array([3.5, 2.5, 1.0, 3.0, 0.8, 1.0, 2.0], dtype=jnp.float32)  # ...,held,orient,torque,skeleton
PARAM_NAMES = list(tt.FEATURE_NAMES)


def build(seed=0, n_iters=60, n_scenes=6, num_blocks=3):
    prob = tt.TetrisProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    # IOSP_STOCK_SOLVER=1 swaps the fixed-length unroll for the stock
    # early-stopping `while_loop`.  The unroll exists only to make q*(theta)
    # differentiable for the bilevel outer loop; for forward-only iteration it
    # is pure compile-time cost.
    stock = os.environ.get("IOSP_STOCK_SOLVER", "0") == "1"
    fs = tt.make_tetris_forward_solver(n_iters=n_iters, robot=prob.base.robot,
                                       stock=stock)

    rng = np.random.default_rng(seed)
    scenes_all = tt.sample_tetris_scenes(rng, 2 * n_scenes, num_blocks=num_blocks)
    fit = jax.tree.map(lambda a: a[:n_scenes], scenes_all)
    test = jax.tree.map(lambda a: a[n_scenes:], scenes_all)

    key = jax.random.PRNGKey(seed)
    x0, seg_scenes, _, _ = prob.seeds(fit)

    inner_by_phase = {}
    for p in tt.PHASES:
        rf = prob.segment_residual_fn(p)
        scales = prob.calibrate_segment(p, rf, seg_scenes[p], key)
        inner_by_phase[p] = make_inner_solver(rf, scales, forward_solver=fs)

    full_rf = prob.full_residual_fn()
    full_sc = tt.TetrisFullScene(
        fit.q_start, fit.q_start,
        fit.obs_center, fit.obs_radius,
        prob.pick_ik(fit.pick_pos, fit.q_start),
        prob.place_ik(fit.place_pos,
                      prob.pick_ik(fit.pick_pos, fit.q_start)),
        fit.block_spheres)
    full_scales = prob.calibrate_full(full_rf, full_sc, key)
    refine = make_inner_solver(full_rf, full_scales, forward_solver=fs)

    return dict(prob=prob, fit=fit, test=test, inner_by_phase=inner_by_phase,
                refine=refine, fs=fs, seed=seed, full_scales=full_scales)


def ee_paths(built, scenes, z, *, stage2=True):
    prob, refine = built["prob"], built["refine"]
    theta = jax.nn.softmax(z)
    theta_seg = theta[:tt.K_SEG]
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
        num_blocks=3, out=None):
    t_wall_start = time.perf_counter()
    t0 = time.perf_counter()
    built = build(seed=seed, n_iters=n_iters, n_scenes=n_scenes,
                  num_blocks=num_blocks)
    fit, test = built["fit"], built["test"]
    print(f"[build] {time.perf_counter()-t0:.1f}s  K={tt.K} {PARAM_NAMES}  "
          f"num_blocks={num_blocks}  fit/test={n_scenes}/{n_scenes}", flush=True)

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

    _, g0 = gf(jnp.zeros(tt.K, jnp.float32))
    g0 = np.asarray(g0)
    print(f"[grad] dL/dtheta at z=0: {g0}", flush=True)

    rng = np.random.default_rng(seed + 1)
    z_zero = jnp.zeros(tt.K, dtype=jnp.float32)
    starts = [z_zero] + [jnp.asarray(rng.normal(0, 1, tt.K), jnp.float32)
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
                            seed=seed, num_blocks=num_blocks)
        print(f"[out] wrote {out}", flush=True)

    return result


def report(result):
    ref = result["baseline"]
    print("\n=== held-out behavioural recovery (EE RMSE, m) ===")
    for n, v in (("baseline uniform (theta = 1/K)", ref),
                 ("tetris IOSP fit", result["fit"]["test_rmse"]),
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
    ap.add_argument("--num-blocks", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report(run(seed=a.seed, n_iters=a.n_iters, n_scenes=a.n_scenes,
               n_steps=a.n_steps, n_starts=a.n_starts,
               num_blocks=a.num_blocks, out=a.out))
