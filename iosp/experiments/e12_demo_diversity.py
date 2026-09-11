"""E12 - does demonstration DIVERSITY buy cost identifiability?

Two demonstration sets of the same size are fit against the same known
`Z_STAR`, and graded on the same held-out scenes:

  NARROW   every demo builds the SAME stack level, differing only by the start-
           pose jitter -- five recordings of one task.
  DIVERSE  the levels walk (domain randomisation over the sampler's own axes:
           which block, which spawn cell, how tall the stack already is, and so
           which blocks are obstacles) -- five recordings of five situations.

The claim being tested is not "more data is better" -- the sets are the same
size.  It is that a cost is identified by the demonstrations SPANNING the
directions it can vary in, so what matters is the rank of the feature-gradient
Gram matrix the set induces, not how many rows it has.  If that is right,
DIVERSE should show more eigendirections above the floor, a smaller maximum
collinearity between features, and -- the part that matters -- less of its
recovery error living in the identifiable subspace, plus a lower held-out
error.  Both conditions are scored on the SAME held-out scenes, drawn with
full diversity, so the comparison is about what was learned and not about what
it was tested on.

Replaces `fig2_multistart` on the paper slate: that figure showed an optimiser
converging, which is a claim about the solver; this one is a claim about the
data, which is the subject of the paper.

    CUDA_VISIBLE_DEVICES=<i> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.experiments.e12_demo_diversity \\
            --out iosp/data/results/diversity/demo_diversity.npz
"""
from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from iosp import config
from iosp.fit import gram as GR

# The narrow set still needs SOME spread or every demo is the same trajectory
# and the fit has nothing at all to chew on; this is the start-pose jitter a
# single repeated task would realistically show.
NARROW_JITTER = 0.02
DIVERSE_JITTER = 0.05


def _scenes(rng, n, stack_level, jitter_q, num_blocks=10):
    from iosp.model import tower as tw
    return tw.sample_tower_scenes(rng, n, stack_level=stack_level,
                                  num_blocks=num_blocks, jitter_q=jitter_q)


def _one(name, fit_scenes, test_scenes, seed, n_iters, n_steps, n_starts):
    """Fit one condition and analyse its Gram. -> dict"""
    from iosp.experiments import e9_tower as E
    from iosp.model import tower as M

    print(f"\n===== condition: {name} =====", flush=True)
    t0 = time.perf_counter()
    built = E.build(seed=seed, n_iters=n_iters,
                    fit_scenes=fit_scenes, test_scenes=test_scenes)
    print(f"[build] {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    demos_fit = jax.jit(lambda: E.ee_paths(built, fit_scenes, E.Z_STAR))()
    demos_test = jax.jit(lambda: E.ee_paths(built, test_scenes, E.Z_STAR))()
    jax.block_until_ready((demos_fit, demos_test))
    print(f"[demos] {time.perf_counter()-t0:.1f}s {tuple(demos_fit.shape)}",
          flush=True)

    gf = jax.jit(jax.value_and_grad(E.make_loss(built, fit_scenes, demos_fit)))
    jt = jax.jit(E.make_loss(built, test_scenes, demos_test))
    sanity = float(gf(E.Z_STAR)[0])
    assert sanity < 1e-4, f"{name}: model does not reproduce its own demo: {sanity:.3e}"

    rng = np.random.default_rng(seed + 1)
    z_zero = jnp.zeros(M.K, jnp.float32)
    starts = [z_zero] + [jnp.asarray(rng.normal(0, 1, M.K), jnp.float32)
                         for _ in range(n_starts - 1)]
    t0 = time.perf_counter()
    best = E.fit_z(gf, starts, n_steps=n_steps)
    print(f"[fit] {time.perf_counter()-t0:.1f}s  {best['l0']:.3e} -> "
          f"{best['lN']:.3e}", flush=True)

    theta_hat = np.asarray(jax.nn.softmax(best["z"]), float)
    theta_star = np.asarray(jax.nn.softmax(E.Z_STAR), float)
    delta = theta_hat - theta_star

    # Gram at the GROUND TRUTH, on the scenes this condition was fit on: it
    # describes what these demonstrations could have revealed, independently of
    # where the optimiser happened to land.
    theta = jax.nn.softmax(E.Z_STAR)
    xs_all, _, full_sc, _, _ = built["prob"].solve(
        fit_scenes, built["inner_by_phase"], theta[:M.K_SEG], theta,
        built["refine"])
    cols = GR.feature_columns(built["prob"].full_residual_fn(), xs_all["full"],
                              full_sc, M.K, built["full_scales"])
    G = GR.gram(cols)
    eigvals, eigvecs, top_idx, null_idx, k = GR.spectrum(G)
    coll = GR.collinearity(G)

    out = dict(
        name=name, G=G, eigvals=eigvals, eigvecs=eigvecs, k=k,
        top_idx=np.asarray(top_idx), null_idx=np.asarray(null_idx),
        theta_hat=theta_hat, theta_star=theta_star, delta=delta,
        raw_err=float(np.linalg.norm(delta)),
        top_err=GR.project(delta, eigvecs, top_idx),
        null_err=GR.project(delta, eigvecs, null_idx),
        held_out_rmse=float(jnp.sqrt(jt(best["z"]))),
        baseline_rmse=float(jnp.sqrt(jt(z_zero))),
        oracle_rmse=float(jnp.sqrt(jt(E.Z_STAR))),
        fit_loss0=float(best["l0"]), fit_lossN=float(best["lN"]),
        **coll)
    ev = eigvals[np.argsort(eigvals)[::-1]]
    print(f"[gram] k={k}/{M.K}  eff_rank={coll['eff_rank']:.2f}  "
          f"max|cos|={coll['max_cos']:.4f}  spectrum={np.round(ev, 6)}")
    print(f"[err ] raw={out['raw_err']:.4f} top={out['top_err']:.4f} "
          f"null={out['null_err']:.4f}  held-out RMSE={out['held_out_rmse']:.5f}")
    return out


def run(seed=0, n_demos=5, n_test=6, n_iters=60, n_steps=40, n_starts=8,
        narrow_level=0, out=None):
    from iosp.experiments import e9_tower as E

    # The held-out set is drawn ONCE, with full diversity, and both conditions
    # are graded on it.  Drawing it per-condition would let the narrow fit be
    # tested on narrow scenes, which is the comparison quietly answering itself.
    test_scenes = _scenes(np.random.default_rng(seed + 777), n_test,
                          stack_level=None, jitter_q=DIVERSE_JITTER)
    narrow = _scenes(np.random.default_rng(seed + 1), n_demos,
                     stack_level=narrow_level, jitter_q=NARROW_JITTER)
    diverse = _scenes(np.random.default_rng(seed + 2), n_demos,
                      stack_level=None, jitter_q=DIVERSE_JITTER)

    conds = [_one("narrow", narrow, test_scenes, seed, n_iters, n_steps, n_starts),
             _one("diverse", diverse, test_scenes, seed, n_iters, n_steps, n_starts)]

    print("\n===== summary =====")
    hdr = f"{'condition':10s} {'k':>3s} {'eff_rank':>9s} {'max|cos|':>9s} " \
          f"{'raw':>8s} {'top':>8s} {'null':>8s} {'heldout':>9s}"
    print(hdr)
    for c in conds:
        print(f"{c['name']:10s} {c['k']:3d} {c['eff_rank']:9.2f} "
              f"{c['max_cos']:9.4f} {c['raw_err']:8.4f} {c['top_err']:8.4f} "
              f"{c['null_err']:8.4f} {c['held_out_rmse']:9.5f}")
    print(f"{'baseline':10s} {'':3s} {'':9s} {'':9s} {'':8s} {'':8s} {'':8s} "
          f"{conds[0]['baseline_rmse']:9.5f}")
    print(f"{'oracle':10s} {'':3s} {'':9s} {'':9s} {'':8s} {'':8s} {'':8s} "
          f"{conds[0]['oracle_rmse']:9.5f}")

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        payload = dict(order=np.array(E.PARAM_NAMES, dtype=object),
                       n_demos=n_demos, n_test=n_test, seed=seed,
                       narrow_level=narrow_level)
        for c in conds:
            for key, val in c.items():
                if key == "name":
                    continue
                payload[f"{c['name']}_{key}"] = val
        np.savez(out, allow_pickle=True, **payload)
        print(f"\nwrote {out}")
    return conds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-demos", type=int, default=5,
                    help="demonstrations PER CONDITION (both get the same count)")
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--narrow-level", type=int, default=0,
                    help="the single stack level the NARROW condition repeats")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    config.setup()
    run(**vars(args))


if __name__ == "__main__":
    main()
