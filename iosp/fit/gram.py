"""Feature-gradient Gram matrix and the identifiable/null split it induces.

What a demonstration set can teach you about a cost is set by how the features
move the SOLUTION, not by how big the features are.  Column `i` here is

    g_i = d/dx  ||r_i(x)||^2     evaluated at the solution x*(theta*),

the direction in trajectory space that feature `i` pulls, and

    G_ij = sum_over_demos  <g_i, g_j>

is the Gram matrix of those pulls.  Two features whose columns are nearly
parallel are nearly the same knob: the demonstrations cannot tell which of them
produced the trajectory, and the corresponding eigendirection of `G` is flat.
A feature whose column is zero -- `skeleton` on the tower, whose rows are PINNED
by `pinned_rows` and therefore have identically zero residual -- is not merely
weakly identified but structurally invisible, and shows up as an exact zero
eigenvalue.

This is the construction `checks/identifiability.py` and `e0d_eigen_projection`
use, lifted off pick-place's hardcoded scene so it can run against any of the
curated domains (`e8_tetris`, `e9_tower`), whose `Z_STAR` is a KNOWN ground
truth rather than a fit.  Summing over demos rather than using one is what lets
a demo set be evaluated AS A SET -- see `e12_demo_diversity`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def feature_columns(residual_fn, xs, scenes, n_features, scales):
    """-> (n_demos, n_features, n_X) gradient columns at the given solutions.

    `residual_fn(x_flat, scene)` is a SINGLE-scene residual returning one array
    per feature (the `full_residual_fn` of a domain problem); `xs` is the
    solved `(n_demos, n_X)` stack and `scenes` the matching batched scene.

    `scales` are the calibration scales the inner solver divides by, i.e. the
    nominal size of `sum(r_i**2)` for each feature.  They are NOT optional and
    they are not cosmetic: the cost the demonstrations were generated from is
    `theta_i * sum(r_i**2) / scales_i`, so the feature that actually moves the
    solution is the scaled one.  Built on raw residuals instead, this Gram
    reports which feature has the largest units -- measured on the tower, that
    put 100% of the trace on one direction and reported every other feature as
    structurally invisible, which is a statement about metres-versus-radians
    rather than about the demonstrations.
    """
    s = jnp.asarray(scales)

    def col(x, sc, i):
        return jax.grad(lambda z: jnp.sum(residual_fn(z, sc)[i] ** 2) / s[i])(x)

    cols = [jax.vmap(lambda x, sc, i=i: col(x, sc, i))(xs, scenes)
            for i in range(n_features)]
    return jnp.stack(cols, axis=1)


def gram(columns):
    """-> (n_features, n_features) Gram matrix summed over demonstrations."""
    g = jnp.asarray(columns, jnp.float64 if columns.dtype == jnp.float64
                    else jnp.float32)
    return np.asarray(jnp.einsum("dix,djx->ij", g, g), float)


def spectrum(G, frac=0.95):
    """Eigendecompose a trace-normalised `G` and split it at `frac` of trace.

    Returns `(eigvals, eigvecs, top_idx, null_idx, k)` with eigvals ASCENDING,
    matching `np.linalg.eigh` and what `viz.fig_eigenspectrum` expects.  The
    split rule is the one used throughout: keep the fewest descending
    eigendirections whose cumulative trace reaches `frac`.
    """
    tr = np.trace(G)
    Gn = G / tr if tr > 0 else G
    eigvals, eigvecs = np.linalg.eigh(Gn)
    desc = np.argsort(eigvals)[::-1]
    ev = eigvals[desc]
    tot = ev.sum()
    if tot <= 0:
        return eigvals, eigvecs, desc[:1], desc[1:], 1
    cum = np.cumsum(ev) / tot
    k = int(np.searchsorted(cum, frac) + 1)
    return eigvals, eigvecs, desc[:k], desc[k:], k


def project(delta, eigvecs, idx):
    """Norm of `delta` projected onto the eigendirections in `idx`."""
    if len(idx) == 0:
        return 0.0
    return float(np.linalg.norm(eigvecs[:, idx].T @ np.asarray(delta, float)))


def collinearity(G):
    """How redundant the feature set is, as the demos see it.

    `max_cos` is the largest absolute cosine between two DISTINCT feature
    columns -- the sharpest "these two are the same knob" pair.  `cond` is the
    spectral condition number over the non-degenerate directions, and
    `eff_rank` the participation ratio `(sum l)^2 / sum l^2`, a continuous
    stand-in for rank that does not need a threshold.
    """
    d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    C = G / np.outer(d, d)
    off = np.abs(C - np.diag(np.diag(C)))
    ev = np.linalg.eigvalsh(G / max(np.trace(G), 1e-300))
    ev = np.clip(ev, 0.0, None)
    pos = ev[ev > ev.max() * 1e-12] if ev.max() > 0 else ev
    return dict(
        max_cos=float(off.max()) if off.size else 0.0,
        max_cos_pair=tuple(int(v) for v in
                           np.unravel_index(int(np.argmax(off)), off.shape))
        if off.size else (0, 0),
        cond=float(pos.max() / pos.min()) if pos.size and pos.min() > 0
        else float("inf"),
        eff_rank=float(ev.sum() ** 2 / np.sum(ev ** 2)) if np.any(ev > 0) else 0.0,
    )
