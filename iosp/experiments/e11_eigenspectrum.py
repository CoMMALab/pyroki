"""Eigenspectrum data collection on the CURATED synthetic domains.

Fits the cost weights back from self-demonstrations generated at a KNOWN
`Z_STAR`, then builds the feature-gradient Gram matrix at that ground truth and
decomposes the recovery error onto its identifiable and near-null subspaces.
The fit comes first and the Gram second, deliberately: the Gram is a property
of the demonstrations, and it is used to EXPLAIN a recovery error that was
measured without reference to it, not to define one.

This supersedes `e0d_eigen_projection`, which ran on one hardcoded pick-place
scene whose `theta*` was itself a fit.  Tower and tetris now have curated
geometry and a known ground-truth weight vector (see `e9_tower.Z_STAR`), so
"how far is theta_hat from theta*" is a real number here rather than a
self-comparison, and `raw vs identifiable vs null` says something checkable.

    CUDA_VISIBLE_DEVICES=<i> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m iosp.experiments.e11_eigenspectrum \\
            --domain tower --out iosp/data/results/identifiability/tower_eigen.npz
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

DOMAINS = ("tower", "tetris")


def _domain(name):
    """-> (experiment module, model module).  Both expose the same surface."""
    if name == "tower":
        from iosp.experiments import e9_tower as E
        from iosp.model import tower as M
    elif name == "tetris":
        from iosp.experiments import e8_tetris as E
        from iosp.model import tetris as M
    else:
        raise SystemExit(f"unknown domain {name!r}, want one of {DOMAINS}")
    return E, M


def solve_at(E, M, built, scenes, z):
    """-> (xs_full, full_scenes) at the weights `z`."""
    theta = jax.nn.softmax(z)
    xs, _, full_sc, _, _ = built["prob"].solve(
        scenes, built["inner_by_phase"], theta[:M.K_SEG], theta, built["refine"])
    return xs["full"], full_sc


def collect(domain="tower", seed=0, n_scenes=6, n_iters=60, n_steps=40,
            n_starts=8, stack_level=None):
    E, M = _domain(domain)
    t0 = time.perf_counter()
    kw = dict(seed=seed, n_iters=n_iters, n_scenes=n_scenes)
    if domain == "tower":
        kw["stack_level"] = stack_level
    built = E.build(**kw)
    fit, test = built["fit"], built["test"]
    print(f"[build] {time.perf_counter()-t0:.1f}s  domain={domain}  K={M.K}  "
          f"{E.PARAM_NAMES}", flush=True)

    # --- demonstrations at the known ground truth -------------------------
    t0 = time.perf_counter()
    demos_fit = jax.jit(lambda: E.ee_paths(built, fit, E.Z_STAR))()
    demos_test = jax.jit(lambda: E.ee_paths(built, test, E.Z_STAR))()
    jax.block_until_ready((demos_fit, demos_test))
    print(f"[demos] {time.perf_counter()-t0:.1f}s {tuple(demos_fit.shape)}",
          flush=True)

    gf = jax.jit(jax.value_and_grad(E.make_loss(built, fit, demos_fit)))
    jt = jax.jit(E.make_loss(built, test, demos_test))
    sanity = float(gf(E.Z_STAR)[0])
    print(f"[sanity] loss(z_star) = {sanity:.3e}", flush=True)
    assert sanity < 1e-4, f"model does not reproduce its own demo: {sanity:.3e}"

    # --- STEP 1: fit ------------------------------------------------------
    rng = np.random.default_rng(seed + 1)
    z_zero = jnp.zeros(M.K, jnp.float32)
    starts = [z_zero] + [jnp.asarray(rng.normal(0, 1, M.K), jnp.float32)
                         for _ in range(n_starts - 1)]
    t0 = time.perf_counter()
    best = E.fit_z(gf, starts, n_steps=n_steps)
    print(f"[fit] {time.perf_counter()-t0:.1f}s  loss {best['l0']:.3e} -> "
          f"{best['lN']:.3e}", flush=True)

    theta_hat = np.asarray(jax.nn.softmax(best["z"]), float)
    theta_star = np.asarray(jax.nn.softmax(E.Z_STAR), float)
    delta = theta_hat - theta_star

    # --- STEP 2: Gram at theta*, over the demonstrations that were fit ----
    t0 = time.perf_counter()
    xs, full_sc = solve_at(E, M, built, fit, E.Z_STAR)
    cols = GR.feature_columns(built["prob"].full_residual_fn(), xs, full_sc,
                              M.K, built["full_scales"])
    G = GR.gram(cols)
    eigvals, eigvecs, top_idx, null_idx, k = GR.spectrum(G)
    coll = GR.collinearity(G)
    print(f"[gram] {time.perf_counter()-t0:.1f}s  k={k}/{M.K} identifiable  "
          f"eff_rank={coll['eff_rank']:.2f}  max|cos|={coll['max_cos']:.4f}",
          flush=True)

    raw_err = float(np.linalg.norm(delta))
    top_err = GR.project(delta, eigvecs, top_idx)
    null_err = GR.project(delta, eigvecs, null_idx)
    test_rmse = float(jnp.sqrt(jt(best["z"])))

    ev_desc = eigvals[np.argsort(eigvals)[::-1]]
    print(f"\neigenvalues (trace-normalised, descending): {np.round(ev_desc, 8)}")
    print(f"raw  ||theta_hat - theta*||      = {raw_err:.4f}")
    print(f"top-{k} (identifiable) component = {top_err:.4f}")
    print(f"null ({len(null_idx)}-dim) component       = {null_err:.4f}")
    print(f"held-out EE RMSE                 = {test_rmse:.5f}")
    print(f"(orthonormal check: {np.sqrt(top_err**2 + null_err**2):.4f} "
          f"should equal raw)")

    return dict(
        domain=domain, order=np.array(E.PARAM_NAMES, dtype=object),
        G=G, eigvals=eigvals, eigvecs=eigvecs,
        top_idx=np.asarray(top_idx), null_idx=np.asarray(null_idx), k=k,
        theta_hat=theta_hat, theta_star=theta_star, delta=delta,
        raw_err=raw_err, top_err=top_err, null_err=null_err,
        ee_rmse=test_rmse, n_demos=int(n_scenes),
        max_cos=coll["max_cos"], cond=coll["cond"], eff_rank=coll["eff_rank"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", choices=DOMAINS, default="tower")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-scenes", type=int, default=6)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--stack-level", type=int, default=None,
                    help="tower: pin every demo to this level (default: walk)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.setup()
    res = collect(domain=args.domain, seed=args.seed, n_scenes=args.n_scenes,
                  n_iters=args.n_iters, n_steps=args.n_steps,
                  n_starts=args.n_starts, stack_level=args.stack_level)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(args.out, allow_pickle=True, **res)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
