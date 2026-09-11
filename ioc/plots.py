"""Figure generation for the IOC study, formatted for IEEE T-RO.

Figures are sized to the IEEE two-column grid (3.5 in single, 7.16 in double),
set in a Times-compatible serif at the caption's point size, and written as both
PDF (vector, for LaTeX) and PNG (for review).  Titles are omitted -- T-RO figures
carry their text in the caption -- and panels are labelled (a), (b), ... instead.

Colour is the validated Okabe-Ito subset: every adjacent pair clears the
colour-vision-deficiency separation floor, and the three-series primary set used
for the main result also clears 3:1 contrast against the page.  Every series
additionally carries a distinct marker and dash pattern, so the figures survive
greyscale printing and photocopying, where hue carries nothing.

    python -m ioc.plots                 # all figures
    python -m ioc.plots --only scaling
    python -m ioc.plots --only scaling,ambiguity,recovery   # curated slate
"""

import dataclasses
import glob
import json
import os

import jax

# `ioc.collect` runs every data-generating stage under JAX_ENABLE_X64=1 (the
# lone deliberate exception is the float32 E3 ablation), but fig2/fig3 build
# their environments and solve demonstrations *inline* here rather than from
# collected JSON, so they inherit nothing from that env default. Under the
# JAX default (float32) the GN forward solver's noise floor sits around 1e-5,
# which is above `bench2d.run`'s 1e-6 demonstration-stationarity screen: on
# the multimodal 6-bump `field` benchmark this makes `_screen_demos` raise
# outright, and on benchmarks where it happens not to raise it silently
# degrades the implicit adjoint the same way documented for `ioc.robot`
# (cos agreement 0.9999 -> 0.59). Match the robot experiments' and
# `collect.py`'s convention here instead of relying on the caller's shell.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
import matplotlib.patheffects as pe
from matplotlib.patches import Circle

from ioc import analytic, outer as outer_opt
from ioc.bench2d import problems as b2d
from ioc.bench2d import run as b2d_run
from ioc.bench2d.run import build_solver

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
FIGDATA = os.path.join(DATA, "figdata")
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# Figure-data persistence
#
# Several figures below build their environments and *solve* their
# demonstrations/fits inline rather than reading collected JSON. That coupling
# meant an expensive multi-method fit only ever lived inside a matplotlib
# process: a plotting bug, a font error, or an OOM lost the numbers. Each such
# figure is now split into a `_<name>_compute()` that returns a plain
# dict of numpy arrays and a `_<name>_render(data)` that draws from it. The
# public `fig_<name>` computes (unless a cached file exists and recompute is
# off), writes the dict to `data/figdata/<name>.npz`, then renders *from the
# on-disk file* -- so the persisted data is always exactly what the panel
# shows, and re-rendering never needs the solver.
# ---------------------------------------------------------------------------


def _figdata_path(name):
    return os.path.join(FIGDATA, f"{name}.npz")


def save_figdata(name, data):
    """Persist an arbitrary dict of numpy arrays / nested dicts for a figure."""
    os.makedirs(FIGDATA, exist_ok=True)
    np.savez(_figdata_path(name), payload=np.array(data, dtype=object))
    print(f"  [figdata] wrote data/figdata/{name}.npz")


def load_figdata(name):
    p = _figdata_path(name)
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)["payload"].item()


def _cached(name, compute, recompute):
    """Ensure `data/figdata/<name>.npz` is current, then return its contents.

    Rendering always reads back from disk so the persisted file is verified to
    round-trip and is guaranteed identical to what the figure draws.
    """
    if recompute or load_figdata(name) is None:
        save_figdata(name, compute())
    return load_figdata(name)


class _FieldCtx:
    """Minimal stand-in for a `bench2d` context exposing just what the field
    renderers (`_field_grid`, `_draw_field_background`) read, so a saved
    (centers, widths) pair can be drawn without rebuilding a full context."""

    def __init__(self, centers, widths):
        self.centers = np.asarray(centers)
        self.widths = np.asarray(widths)

# --- T-RO page geometry ------------------------------------------------------
COL1, COL2 = 3.5, 7.16  # inches

# --- validated palette (Okabe-Ito subset; see scripts/validate_palette.js) ----
# Order is fixed: a series keeps its hue no matter which others are plotted.
STYLE = {
    "implicit": dict(color="#0072B2", marker="o", ls="-", label="Implicit (ours)"),
    "fd":       dict(color="#D55E00", marker="s", ls="--", label="Finite diff."),
    "cmaes":    dict(color="#009E73", marker="^", ls="-.", label="CMA-ES"),
    "cioc":     dict(color="#E69F00", marker="D", ls=(0, (3, 1, 1, 1)), label="CIOC [Levine '12]"),
    "eiv":      dict(color="#F0E442", marker="H", ls=(0, (4, 1, 1, 1, 1, 1)), label="EIV-TLS [Rickenbach '24]"),
    "kkt":      dict(color="#CC79A7", marker="v", ls=":", label="Inverse KKT"),
    "kkt_seed": dict(color="#CC79A7", marker="v", ls="-", label=r"Implicit, KKT-seeded $z_0$"),
    "unrolled": dict(color="#56B4E9", marker="P", ls=(0, (5, 2)), label="Unrolled"),
    "random":   dict(color="0.55", marker="", ls=(0, (1, 2)), label="Random weights"),
}
DEMO_C, FIT_C, OBS_C = "#222222", "#0072B2", "#c9c9c9"

# Smallest regret the log axes will show.  Regret is a cost difference and dips
# marginally below zero when a fit beats the reference solve by solver noise.
REGRET_FLOOR = 1e-9


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.color": "#d9d9d9",
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # editable/embeddable type-42 fonts, as IEEE requires
        "ps.fonttype": 42,
    })


def finish(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=400)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf|png")


def panel_label(ax, text, dx=-0.16, dy=1.04, fontsize=None):
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=8 if fontsize is None else fontsize, fontweight="bold",
            va="top", ha="left")


def tidy(ax):
    ax.grid(True, alpha=0.3, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def solves_to(trace, target):
    return next((s for s, l in trace if l < target), None)


# --- shared layout for the single-column line figures (fig. 1, fig. 4) -------
# Both figures carry 4-8 series on a 3.5 in column: no in-axes corner is large
# enough to hold the key without covering curves or IQR bands, and inflating
# the data range to manufacture one wastes the panel.  So the key goes outside,
# under the axes, and the layout is constrained rather than tight -- only the
# constrained engine measures the suptitle and an outside legend, so the gaps
# above and below the axes come out the same on both figures instead of being
# hand-tuned per figure.
#
# One order for every line figure's series, so a reader moving between fig. 1
# and fig. 4 finds a method in the same slot of the key with the same colour,
# marker and dash pattern.  Curves are drawn in the reverse of this order,
# which puts the primary series on top of the baselines it is compared against.
LEGEND_ORDER = ("implicit", "unrolled", "fd", "cmaes",
                "kkt", "cioc", "eiv", "random")


def draw_order(methods):
    """`methods` sorted so the topmost drawn curve is the key's first entry."""
    return [m for m in reversed(LEGEND_ORDER) if m in methods]


def line_figure(height):
    """A single-column line-figure axes laid out for a title and outside key."""
    fig, ax = plt.subplots(1, 1, figsize=(COL1, height), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.05)
    return fig, ax


def line_figure_finish(fig, ax, methods, title, name, ncol=2):
    """Title, legend and save, shared by fig. 1 and fig. 4.

    `methods` fixes the legend order (STYLE keys); entries absent from the axes
    -- a method missing from the collected data -- are skipped rather than
    faked, so the key never advertises a curve that was not drawn.
    """
    got = {lab: h for h, lab in zip(*ax.get_legend_handles_labels())}
    keys = [m for m in LEGEND_ORDER if m in methods and STYLE[m]["label"] in got]
    fig.suptitle(title, fontsize=9, fontweight="bold")
    fig.legend([got[STYLE[m]["label"]] for m in keys],
               [STYLE[m]["label"] for m in keys],
               loc="outside lower center", ncol=ncol, frameon=False,
               handlelength=2.2, columnspacing=1.2, handletextpad=0.5)
    finish(fig, name)


# ---------------------------------------------------------------------------
# Fig. 1 - cost of a fit vs cost dimension  (the main result)
# ---------------------------------------------------------------------------


def fig_scaling():
    files = sorted(glob.glob(os.path.join(DATA, "bench2d", "bench2d_seg_K*.json")),
                   key=lambda f: int(f.split("K")[-1].split(".")[0]))
    if not files:
        print("  [skip] no segments sweep data")
        return
    METHODS = ("implicit", "unrolled", "fd", "cmaes")
    Ks = []
    series = {m: [] for m in METHODS}
    lo = {m: [] for m in METHODS}
    hi = {m: [] for m in METHODS}
    censored = []  # (K, method, n_reached, n_seeds) wherever a seed never got there
    for f in files:
        d = json.load(open(f))
        r = d["results"]
        Ks.append(d["K"])
        for m in METHODS:
            # A file collected before a method existed simply has no entry;
            # treat it as censored rather than crashing the whole figure.
            vals = [solves_to(r[s][m]["trace"], 1e-2) for s in r if m in r[s]]
            if not vals:
                vals = [None] * len(r)
            ok = [v for v in vals if v]
            if len(ok) < len(vals):
                censored.append((d["K"], m, len(ok), len(vals)))
            if len(ok) >= 1:
                series[m].append(np.median(ok))
                lo[m].append(np.percentile(ok, 25) if len(ok) >= 2 else np.nan)
                hi[m].append(np.percentile(ok, 75) if len(ok) >= 2 else np.nan)
            else:
                series[m].append(np.nan)
                lo[m].append(np.nan)
                hi[m].append(np.nan)
    Ks = np.array(Ks)
    # Seeds that never reach the target are dropped, so a median over the
    # survivors *flatters* whichever method failed more often.  Say so out loud
    # rather than let the curve imply every seed converged.
    if censored:
        print("  [fig1] censored (target never reached): " + ", ".join(
            f"K={k} {m} {n}/{tot} seeds" for k, m, n, tot in censored))
    else:
        print("  [fig1] no censoring: every seed reached the target")

    fig, ax = line_figure(2.5)
    for m in draw_order(METHODS):
        st = STYLE[m]
        ax.plot(Ks, series[m], marker=st["marker"], ls=st["ls"], color=st["color"],
                label=st["label"], clip_on=False, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("cost parameters $K$")
    ax.set_ylabel(r"solves to reach $L<10^{-2}$")
    ax.set_xticks(Ks)
    # Keep the K=6 and K=48 markers off the spines; `clip_on=False` would
    # otherwise draw them half outside the frame.
    ax.margins(x=0.06)
    tidy(ax)
    line_figure_finish(fig, ax, METHODS,
                       "Sample Efficiency vs. Cost Dimension",
                       "fig1_scaling", ncol=2)


# ---------------------------------------------------------------------------
# Fig. 2 - the benchmark environments, with demonstrations
# ---------------------------------------------------------------------------


def _make_env(benchmark, k_bumps, M, T, seed, bump_width, theta=None,
              n_iter=None, n_obstacles=None, shared_obstacle=None,
              shared_bumps=False, bump_layout=None, demo_pairs=None,
              unit_scales=False, demo_noise=0.0, n_restarts=1, topo_restarts=False):
    """Build one benchmark's contexts and solve demonstrations under `theta`.

    `theta` is given explicitly for the environment plates: a Dirichlet draw can
    leave a feature almost unweighted, and a plate drawn that way misrepresents
    the environment (with a near-zero boundary weight the racing demonstrations
    cut straight across the infield, which is not what the benchmark poses).
    """
    res_fn, names, d = b2d.BENCHMARKS[benchmark]
    cfg = b2d.default_cfg(benchmark, bump_width=bump_width)
    if n_obstacles is not None:
        cfg["n_obstacles"] = n_obstacles
    if n_iter is None:
        n_iter = b2d_run.DEMO_N_ITER[benchmark]
    names = b2d.benchmark_names(benchmark, k_bumps, cfg)
    K = len(names)
    rng = np.random.default_rng(seed)
    if theta is None:
        th = np.maximum(rng.dirichlet(np.full(K, 0.7)), 0.01)
        th = th / th.sum()
    else:
        th = np.asarray(theta, float)
        th = th / th.sum()
    th = jnp.asarray(th)
    ctxs = b2d.sample_contexts(rng, M, benchmark, T, d, k_bumps, cfg)
    if shared_obstacle is not None:
        # One scene, several start/goal pairs: the plate then reads as a single
        # environment rather than M overlaid ones.  Accepts a single (x, y, r)
        # or a full (n_obs, 3) set.
        so = jnp.atleast_2d(jnp.asarray(shared_obstacle, float))
        ctxs = dataclasses.replace(ctxs, obstacles=jnp.tile(so[None], (M, 1, 1)))
    if shared_bumps:
        # `sample_contexts` draws a fresh bump layout per context, but a field
        # figure only ever renders context 0's field.  Without this, most of
        # the plotted demonstrations are optimal for a *different*, unseen
        # field and the panel does not show what it claims to: a trajectory
        # bending around the drawn cost hills.
        ctxs = dataclasses.replace(
            ctxs,
            centers=jnp.tile(ctxs.centers[:1], (M, 1, 1)),
            widths=jnp.tile(ctxs.widths[:1], (M, 1)),
        )
    if bump_layout is not None:
        # A hand-placed field, not a random draw: random layouts here are both
        # visually incoherent (bumps overlap or scatter with no readable
        # structure) and, measured directly, poorly conditioned for recovery
        # (Gram lambda_2/lambda_K ~1e-2, implicit_L1 1.5-1.7 across several
        # seeds) -- a hand-placed layout can guarantee well-separated bumps
        # every demo actually gets exposed to.
        bc, bw = bump_layout
        ctxs = dataclasses.replace(
            ctxs,
            centers=jnp.tile(jnp.asarray(bc, float)[None], (M, 1, 1)),
            widths=jnp.tile(jnp.asarray(bw, float)[None], (M, 1)),
        )
    if demo_pairs is not None:
        # Explicit start/goal pairs so the illustration can guarantee specific
        # homotopy classes (e.g. above vs. below a blocking bump) instead of
        # hoping a random sample lands on both sides.
        starts, goals = zip(*demo_pairs)
        ctxs = dataclasses.replace(
            ctxs,
            start=jnp.asarray(np.stack(starts), float),
            goal=jnp.asarray(np.stack(goals), float),
        )
    scales = b2d.calibrate(res_fn, ctxs, T, d, cfg, jax.random.key(seed), K)
    if unit_scales:
        scales = jnp.ones_like(scales)
    inner = build_solver(res_fn, scales, T, d, cfg, n_iter, 1e-2, 3, 1e-9,
                         n_restarts=n_restarts, topo_restarts=topo_restarts)
    si, cost = inner.solve_implicit, inner.cost
    x0 = jax.vmap(lambda c: b2d.seed_path(c, T, d, cfg))(ctxs)

    # Demonstrations from a converged solve, as in `bench2d.run` -- a figure
    # must not depict a "demonstration" that is not a local optimum, and the
    # solver the *learner* uses is deliberately cheaper than the demonstrator's.
    demo_iter = max(n_iter, b2d_run.DEMO_N_ITER[benchmark])
    demo_solver = build_solver(res_fn, scales, T, d, cfg, demo_iter, 1e-2, 3,
                               1e-9, n_restarts=n_restarts, topo_restarts=topo_restarts)
    xs = jax.vmap(lambda x, c: demo_solver.solve_implicit(x, th, c))(x0, ctxs)
    b2d_run._screen_demos(
        jax.vmap(lambda x, c: b2d_run.relative_stationarity(
            demo_solver.features, demo_solver.grad_x, x, th, c, K))(xs, ctxs),
        benchmark, seed, demo_iter)
    demos = jax.vmap(lambda x, c: b2d.unpack(x, c, T, d))(xs, ctxs)
    if demo_noise > 0:
        # Matches `bench2d.run`: endpoints are fixed by the context, so noise is
        # applied only to the interior.  Any figure that *compares methods* must
        # set this -- on noiseless demonstrations Inverse KKT is exact by
        # construction (it fits grad_x J = 0, which holds exactly), so it wins
        # trivially and the comparison says nothing about the noisy regime every
        # other result in the study is reported in.
        nz = jnp.asarray(rng.normal(scale=demo_noise, size=demos.shape))
        demos = demos + nz.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return dict(cfg=cfg, ctxs=ctxs, demos=np.asarray(demos), theta=th, K=K,
                solver=si, x0=x0, T=T, d=d, res_fn=res_fn, scales=scales,
                names=names, cost=cost, inner=inner)


def _field_grid(ctx, theta, extent=(-2.5, 2.5, -1.9, 1.9), n=240):
    cen, wid = np.asarray(ctx.centers), np.asarray(ctx.widths)
    gx, gy = np.meshgrid(np.linspace(extent[0], extent[1], n),
                         np.linspace(extent[2], extent[3], n))
    pts = np.stack([gx.ravel(), gy.ravel()], -1)
    f = np.zeros(pts.shape[0])
    for k in range(cen.shape[0]):
        f += float(theta[k + 2]) * np.exp(
            -((pts - cen[k]) ** 2).sum(-1) / (2 * wid[k] ** 2))
    return gx, gy, f.reshape(gx.shape)


def _draw_field_background(ax, ctx, theta, extent=(-2.5, 2.5, -1.9, 1.9), n=240,
                           vmax=None):
    """Render the cost field.

    `vmax` must be passed explicitly whenever two fields are meant to be
    *compared* (true vs recovered).  Letting each panel normalize to its own
    maximum makes a field that is uniformly too weak look identical to the
    correct one, which is precisely the error the figure exists to show.
    """
    gx, gy, f = _field_grid(ctx, theta, extent, n)
    # Sequential, single hue, light->dark.  The field is a *cost* (positive
    # weights on positive residuals): it has magnitude but no polarity, so a
    # diverging map would imply a sign change that does not exist.
    im = ax.pcolormesh(gx, gy, f, cmap="Reds", vmin=0.0,
                       vmax=(f.max() if vmax is None else vmax) + 1e-12,
                       shading="auto", rasterized=True, zorder=0)
    # Contours give the reader the shape of the field independent of the
    # colour ramp, which matters in greyscale reproduction.
    ax.contour(gx, gy, f, levels=6, colors="k", linewidths=0.25, alpha=0.35,
               zorder=1)
    return im


def _draw_circuit(ax, cfg, lw=0.7):
    """The racing circuit: filled corridor, walls, and dashed centreline."""
    a = np.linspace(0, 2 * np.pi, 800)
    r = np.asarray(b2d.track_radius_at(jnp.asarray(a), cfg))
    W = cfg["track_halfwidth"]
    inner, outer = r - W, r + W
    # The inner boundary must be traversed backwards *as a curve* -- reversing
    # the angles while leaving the radii in forward order pairs r(a) with the
    # wrong a and draws a corridor that is not the one the cost uses.
    ax.fill(np.concatenate([outer * np.cos(a), (inner * np.cos(a))[::-1]]),
            np.concatenate([outer * np.sin(a), (inner * np.sin(a))[::-1]]),
            color="#efefef", zorder=0)
    for rad in (inner, outer):
        ax.plot(rad * np.cos(a), rad * np.sin(a), color="0.55", lw=lw, zorder=1)
    ax.plot(r * np.cos(a), r * np.sin(a), color="0.7", lw=0.6, ls=(0, (4, 3)),
            zorder=1)
    return float((r + W).max())


# A thin white casing under every demonstration.  The racing line correctly
# hugs the inside of the turns, which puts it directly on top of the drawn
# track wall; without the casing the two merge and the demonstration reads as
# leaving the corridor when it never does (measured: 0/30 points outside).
HALO = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]


def _plate(ax, demos, n_show, lw=1.15):
    for i in range(min(n_show, len(demos))):
        p = demos[i][:, :2]
        ax.plot(p[:, 0], p[:, 1], color=DEMO_C, lw=lw, alpha=0.95, zorder=3,
                solid_capstyle="round", path_effects=HALO)
        ax.plot(*p[0], marker="o", ms=2.8, color=DEMO_C, zorder=4)
        ax.plot(*p[-1], marker="*", ms=5.4, color=DEMO_C, zorder=4)


def _environments_compute():
    """Solve the three benchmark plates' demonstrations and return, per panel,
    the scene geometry and demonstrated trajectories."""
    T, M = 30, 4

    # (a) racing circuit: weights chosen so the boundary binds and the line has
    # to compromise (a near-zero boundary weight lets demos cut the infield).
    racing = _make_env("racing", 1, M, T, 3, 0.45,
                       theta=[0.30, 0.40, 0.15, 0.15])  # time, boundary, smooth, curv

    # (b) cost field: hand-placed layout (see rationale in the render).
    _field_centers = [
        (0.0, 0.0), (0.2, 0.1),
        (-2.3, 1.75), (2.3, 1.75), (-2.3, -1.75), (2.3, -1.75),
    ]
    _field_widths = [0.4, 0.35, 0.4, 0.4, 0.4, 0.4]
    _field_pairs = [
        ((-2.2, 0.9), (2.2, 0.9)),
        ((-2.2, 0.6), (2.2, 1.1)),
        ((-2.2, -0.9), (2.2, -0.9)),
        ((-2.2, -0.6), (2.2, -1.1)),
    ]
    th_field = [0.16, 0.10, 0.30, 0.10, 0.085, 0.085, 0.085, 0.085]
    field = _make_env("field", 6, M, T, 0, 0.4, theta=th_field,
                      bump_layout=(_field_centers, _field_widths),
                      demo_pairs=_field_pairs)
    fc0 = jax.tree.map(lambda x: x[0], field["ctxs"])

    # (c) car / unicycle: one shared slalom, several start/goal pairs.
    uni = _make_env("unicycle", 1, M, T, 2, 0.45,
                    theta=[0.20, 0.15, 0.10, 0.35, 0.20], n_obstacles=3,
                    shared_obstacle=[(-1.05, 0.22, 0.42), (0.0, -0.26, 0.44),
                                     (1.05, 0.22, 0.42)])
    return dict(
        T=T, M=M,
        racing_cfg=racing["cfg"],
        racing_demos=np.asarray(racing["demos"]),
        field_centers=np.asarray(fc0.centers), field_widths=np.asarray(fc0.widths),
        field_theta=np.asarray(field["theta"]), field_demos=np.asarray(field["demos"]),
        uni_obstacles=np.asarray(uni["ctxs"].obstacles[0]),
        uni_demos=np.asarray(uni["demos"]),
    )


def _environments_render(data):
    """Environment plates in the style of Levine & Koltun (2012, Fig. 3-5):
    the scene geometry with demonstrations drawn on it, one context per panel."""
    M = int(data["M"])
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.25),
                             gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.25]))

    # (a) racing circuit ------------------------------------------------------
    ax = axes[0]
    rmax = _draw_circuit(ax, data["racing_cfg"])
    # Three stints, not four: each spans most of a lap, so a fourth adds
    # overlap without adding information.
    _plate(ax, data["racing_demos"], 3)
    ax.set_xlim(-rmax - 0.12, rmax + 0.12)
    ax.set_ylim(-rmax - 0.12, rmax + 0.12)

    # (b) cost field ----------------------------------------------------------
    # A hand-placed field rather than a random draw. Two reasons:
    #  1. `sample_contexts` draws an independent bump layout per demonstration,
    #     but the panel can only render one field, so an unshared/random layout
    #     leaves most plotted paths bending around bumps that are not the ones
    #     drawn -- the "why doesn't this look optimal" confusion the panel
    #     exists to avoid.
    #  2. A single blocking bump dead centre, flanked by four small corner
    #     bumps that stay clear of the start-goal corridor, gives every
    #     demonstration one obstacle to react to instead of several
    #     overlapping ones. Paired with start/goal offsets above and below
    #     y=0, this reads as one obvious story: both homotopy classes (over
    #     the top, under the bottom) arcing back through the panel's
    #     horizontal centre, with a genuine near-zero-cost gap on each side
    #     (bump width 0.4 at a lane offset of ~1.0 puts exposure at <1% of
    #     peak) rather than a path grazing the cost hill's edge.
    ax = axes[1]
    c0 = _FieldCtx(data["field_centers"], data["field_widths"])
    im = _draw_field_background(ax, c0, data["field_theta"])
    ax.scatter(np.asarray(c0.centers)[:, 0], np.asarray(c0.centers)[:, 1],
               s=4, c="0.25", zorder=2, linewidths=0)
    _plate(ax, data["field_demos"], M)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.9, 1.9)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("cost", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)

    # (c) car / unicycle ------------------------------------------------------
    # One shared slalom, several start/goal pairs: the plate then reads as a
    # single environment rather than four overlaid ones.
    ax = axes[2]
    for ox, oy, orr in np.asarray(data["uni_obstacles"]):
        ax.add_patch(Circle((ox, oy), orr, facecolor=OBS_C, edgecolor="0.5",
                            lw=0.5, zorder=1))
    for i in range(M):
        s_ = data["uni_demos"][i]
        p, phi = s_[:, :2], s_[:, 2]
        ax.plot(p[:, 0], p[:, 1], color=DEMO_C, lw=1.15, alpha=0.95, zorder=3,
                solid_capstyle="round", path_effects=HALO)
        # heading arrows make the nonholonomic state visible, as in the
        # car-navigation plates of Levine & Koltun
        for t in range(2, len(p) - 2, 9):
            ax.arrow(p[t, 0], p[t, 1], 0.20 * np.cos(phi[t]), 0.20 * np.sin(phi[t]),
                     head_width=0.09, head_length=0.09, fc=DEMO_C, ec=DEMO_C,
                     lw=0.4, length_includes_head=True, zorder=4)
        ax.plot(*p[0], marker="o", ms=2.8, color=DEMO_C, zorder=5)
        ax.plot(*p[-1], marker="*", ms=5.4, color=DEMO_C, zorder=5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.9, 1.9)

    for ax, lab, tag in zip(axes,
                            ("racing corridor", "cost field", "car (unicycle)"),
                            ("(a)", "(b)", "(c)")):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(lab, labelpad=2)
        panel_label(ax, tag, dx=-0.02, dy=1.11)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=DEMO_C, lw=1.15, label="demonstration"),
        Line2D([], [], color=DEMO_C, marker="o", ls="", ms=2.8, label="start"),
        Line2D([], [], color=DEMO_C, marker="*", ls="", ms=5.4, label="goal"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Benchmark Environments and Demonstrations", fontsize=9, y=1.02,
                fontweight="bold")
    fig.tight_layout(w_pad=1.0, rect=(0, 0.05, 1, 0.93))
    finish(fig, "fig2_environments")


def fig_environments(recompute=True):
    _environments_render(_cached("fig2_environments", _environments_compute,
                                 recompute))


# ---------------------------------------------------------------------------
# Fig. 2b - IOC ambiguity: one start/goal, several cost-explained paths
# ---------------------------------------------------------------------------

# theta order is UNICYCLE_NAMES: speed, turn, steer_smooth, obstacle, nonholonomic
_AMBIG_THETAS = {
    "aggressive": ([0.60, 0.03, 0.02, 0.05, 0.30], "cuts close, direct"),
    "smooth":     ([0.05, 0.20, 0.10, 0.55, 0.20], "wide clearance margin"),
    "balanced":   ([0.20, 0.15, 0.10, 0.35, 0.20], "balanced tradeoff"),
}
_AMBIG_STYLE = {
    "aggressive": dict(color="#D55E00", ls="--"),
    "smooth":     dict(color="#0072B2", ls="-"),
    "balanced":   dict(color="#009E73", ls="-."),
}


def _ambiguity_compute():
    """Solve `bench2d`'s unicycle under each hand-chosen theta; return the
    demonstrated states (x, y, phi) per cost, plus the shared scene geometry."""
    T, M = 30, 1
    start_goal = [((-2.3, 0.0, 0.0), (2.3, 0.0, 0.0))]
    obstacles = [(-1.05, 0.22, 0.42), (0.0, -0.26, 0.44), (1.05, 0.22, 0.42)]
    paths = {}
    for key, (theta, desc) in _AMBIG_THETAS.items():
        env = _make_env("unicycle", 1, M, T, 2, 0.45, theta=theta,
                        n_obstacles=3, shared_obstacle=obstacles,
                        demo_pairs=start_goal)
        paths[key] = np.asarray(env["demos"][0])  # (T, 3): x, y, phi
    return dict(paths=paths, obstacles=np.asarray(obstacles, float),
                start_goal=np.asarray([(*s, *g) for s, g in start_goal], float))


def _ambiguity_render(data):
    obstacles = data["obstacles"]
    start_goal = [(tuple(r[:3]), tuple(r[3:])) for r in data["start_goal"]]
    paths = data["paths"]

    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.78))
    for ox, oy, orr in obstacles:
        ax.add_patch(Circle((ox, oy), orr, facecolor=OBS_C, edgecolor="0.5",
                            lw=0.5, zorder=1))

    for key, (theta, desc) in _AMBIG_THETAS.items():
        st = _AMBIG_STYLE[key]
        s_ = paths[key]
        p, phi = s_[:, :2], s_[:, 2]
        ax.plot(p[:, 0], p[:, 1], color=st["color"], ls=st["ls"], lw=1.5,
                alpha=0.95, zorder=3, solid_capstyle="round", path_effects=HALO,
                label=f"{key} ({desc})")
        for t in range(2, len(p) - 2, 9):
            ax.arrow(p[t, 0], p[t, 1], 0.20 * np.cos(phi[t]), 0.20 * np.sin(phi[t]),
                     head_width=0.08, head_length=0.08, fc=st["color"],
                     ec=st["color"], lw=0.4, length_includes_head=True, zorder=4)

    s0, g0 = start_goal[0]
    ax.plot(*s0[:2], marker="o", ms=4.5, color="#222222", zorder=5)
    ax.plot(*g0[:2], marker="*", ms=8, color="#222222", zorder=5)
    ax.annotate("start", s0[:2], xytext=(-4, -12), textcoords="offset points",
               fontsize=6.5, ha="center")
    ax.annotate("goal", g0[:2], xytext=(4, -12), textcoords="offset points",
               fontsize=6.5, ha="center")

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="upper center", fontsize=6, ncol=1,
             bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cost Ambiguity: Multiple Optima from One Demonstration",
                fontsize=9, y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    finish(fig, "fig2b_ambiguity")


def fig_ambiguity(recompute=True):
    """One demonstration scene, several plausible costs.

    The identifiability problem this study addresses is not abstract: the same
    start, goal and obstacles admit distinct locally-optimal paths depending on
    the (unknown) weighting between speed, turn, steering-smoothness, obstacle
    clearance and the nonholonomic residual. Each curve here is a real solve of
    `bench2d`'s unicycle benchmark under one hand-chosen theta -- not an
    illustrative sketch -- sharing the scene from `fig_environments`' car panel
    so the two figures are directly comparable.
    """
    _ambiguity_render(_cached("fig2b_ambiguity", _ambiguity_compute, recompute))


# ---------------------------------------------------------------------------
# Fig. 3 - recovered reward field and recovered trajectories
# ---------------------------------------------------------------------------


def _match_aspect(ext, ratio):
    """Grow `ext` symmetrically until its width:height equals `ratio`.

    Panels drawn with `aspect="equal"` in equal-width gridspec cells only come
    out the same height if their *data* extents share an aspect ratio.  When
    they do not, matplotlib centres the shorter axes in its cell and its title
    and xlabel drift out of line with the row.  Growing (never cropping) keeps
    everything that was inside the extent inside it.
    """
    x0, x1, y0, y1 = ext
    w, h = x1 - x0, y1 - y0
    if w / h < ratio:
        grow = (ratio * h - w) / 2.0
        x0, x1 = x0 - grow, x1 + grow
    else:
        grow = (w / ratio - h) / 2.0
        y0, y1 = y0 - grow, y1 + grow
    return (x0, x1, y0, y1)


ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _text_width_in(text, size, weight="normal"):
    """Rendered width of `text` in inches, at the current rcParams family.

    Laid out on a throwaway canvas: a layout that has to know whether a label
    fits its column cannot get that from the point size alone, since the same
    9 pt buys very different widths for "CIOC" and for a norm in mathtext.
    """
    fig = plt.figure(figsize=(1, 1))
    t = fig.text(0, 0, text, fontsize=size, fontweight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    plt.close(fig)
    return w


def _tag_inside(ax, text, size=7.5):
    """Panel letter inside the axes.

    Image panels are drawn edge to edge, so a tag placed above the frame either
    collides with the panel title or forces dead space between rows.
    """
    ax.text(0.035, 0.96, text, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="top", ha="left", zorder=6,
            bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="none",
                      alpha=0.82))


def _recovery_compute(budget=8000, lr=0.1, n_iter=800, demo_noise=0.02,
                      name="fig3_recovery", n_show=4, n_restarts=7, fd_eps=1e-4):
    """Fit every method on the multimodal `field` benchmark and return the
    recovered weight vectors, the recovered trajectories for the shown demos,
    the demonstrations, the field geometry, and the Gram spectrum -- everything
    `_recovery_render` needs, with no solver in the loop."""
    T, M = 30, 8
    # n_restarts>1 with topo_restarts=True replaces i.i.d. jitter multistart
    # with structured lateral-detour seeds (see `pb.make_topo_seed_fn`): on
    # this 6-bump field the inner problem is genuinely multimodal, and a
    # single-start solve lets x*(theta) jump basins as theta moves, which
    # breaks the implicit adjoint (see `ioc.inner.InnerSolver.solve`) while
    # leaving the zero-solve analytic baselines (kkt, cioc) unaffected --
    # exactly the asymmetry this panel was showing without diagnosing it.
    # A hand-placed field, same rationale as `fig_environments`'s field panel:
    # a random bump layout was measured poorly conditioned regardless of seed
    # (Gram lambda_2/lambda_K ~1-5e-2, implicit_L1 1.5-1.7 across several
    # seeds tried), because a random draw can leave a bump nearly un-excited
    # by every demonstration -- unidentifiable however good the solver is.
    # This layout reuses fig2's central blocker (two homotopy lanes) plus five
    # more bumps, each placed on one specific demo's path so every feature is
    # excited by at least one demonstration; `demo_pairs` spans both homotopy
    # classes at three lane offsets each.
    _rec_centers = [
        (0.0, 0.0),
        (0.0, 1.35), (0.0, -1.35),
        (-1.6, 1.65), (1.6, 1.65),
        (-1.6, -1.65),
    ]
    _rec_widths = [0.45, 0.35, 0.35, 0.4, 0.4, 0.4]
    _rec_pairs = [
        ((-2.2, 0.9), (2.2, 0.9)), ((-2.2, 0.6), (2.2, 1.1)),
        ((-2.2, -0.9), (2.2, -0.9)), ((-2.2, -0.6), (2.2, -1.1)),
        ((-2.2, 1.3), (2.2, 1.6)), ((-2.2, 1.6), (2.2, 1.3)),
        ((-2.2, -1.3), (2.2, -1.6)), ((-2.2, -1.6), (2.2, -1.3)),
    ][:M]
    _rec_theta = [0.12, 0.08, 0.18, 0.10, 0.14, 0.16, 0.12, 0.10]
    env = _make_env("field", 6, M, T, 0, 0.4, theta=_rec_theta, n_iter=n_iter,
                    demo_noise=demo_noise, n_restarts=n_restarts,
                    topo_restarts=n_restarts > 1,
                    bump_layout=(_rec_centers, _rec_widths),
                    demo_pairs=_rec_pairs)
    si, ctxs, demos = env["solver"], env["ctxs"], env["demos"]
    x0, d, K = env["x0"], env["d"], env["K"]
    inner = env["inner"]

    def loss_of(solver):
        def loss(z):
            t = jax.nn.softmax(z)

            def one(c, dm, xx):
                p = b2d.unpack(solver(xx, t, c), c, T, d)[:, :2]
                return jnp.mean(jnp.sum((p - dm[:, :2]) ** 2, axis=-1))

            return jnp.mean(jax.vmap(one)(ctxs, demos, x0))

        return loss

    per_solve = M * n_restarts

    def fit_adam(solver):
        li = jax.jit(loss_of(solver))
        gi = jax.jit(jax.value_and_grad(li))
        z, _ = outer_opt.adam(gi, jnp.zeros(K), lr=lr, budget_solves=budget,
                              solves_per_step=per_solve, trace_best=True)
        return z, li

    # Restarts are charged to the solve budget like every other method pays
    # for its extra work (see `ioc.outer` module docstring).
    z, li = fit_adam(si)
    z_unrolled, _ = fit_adam(inner.solve_unrolled)

    z_fd, _ = outer_opt.adam(outer_opt.fd_grad_fn(li, fd_eps), jnp.zeros(K), lr=lr,
                             budget_solves=budget, solves_per_step=(K + 1) * per_solve,
                             trace_best=True)
    z_cmaes, _ = outer_opt.cma_es(li, jnp.zeros(K), budget_solves=budget,
                                  solves_per_eval=per_solve, seed=0, trace_best=True)

    # The zero-solve baselines, fitted on exactly the same demonstrations.
    z_kkt = analytic.kkt_fit(inner.grad_x, ctxs, demos, K, n_steps=600)
    z_cioc = analytic.cioc_fit(inner.grad_x, inner.gn_system, ctxs, demos, K,
                               n_steps=600)
    z_eiv = analytic.eiv_fit(inner.grad_x, ctxs, demos, K)
    _, G = analytic.kkt_fit(inner.grad_x, ctxs, demos, K, n_steps=1, lr=0.0,
                            return_gram=True)
    ev = np.linalg.eigvalsh(np.asarray(G) / np.trace(np.asarray(G)) * K)

    fits = [("implicit", jax.nn.softmax(z)),
            ("unrolled", jax.nn.softmax(z_unrolled)),
            ("fd", jax.nn.softmax(z_fd)),
            ("cmaes", jax.nn.softmax(z_cmaes)),
            ("kkt", jax.nn.softmax(z_kkt)),
            ("cioc", jax.nn.softmax(z_cioc)),
            ("eiv", jax.nn.softmax(z_eiv))]
    th_hat = fits[0][1]
    xhat = jax.vmap(lambda x, c: si(x, th_hat, c))(x0, ctxs)

    c0 = jax.tree.map(lambda x: x[0], ctxs)
    errs = {m: float(jnp.sum(jnp.abs(th - env["theta"]))) for m, th in fits}
    best = min(errs, key=errs.get)

    show = list(range(min(n_show, M)))
    paths = [np.asarray(b2d.unpack(xhat[i], jax.tree.map(lambda x: x[i], ctxs),
                                   T, d))[:, :2] for i in show]

    print(f"  [{name}] sigma={demo_noise:g}  "
          f"Gram lambda_2/lambda_K = {ev[1] / ev[-1]:.2e}  "
          + "  ".join(f"{m}_L1={errs[m]:.3f}" for m, _ in fits)
          + f"  best={best}")
    return dict(
        name=name, demo_noise=demo_noise, n_show=n_show, M=M,
        centers=np.asarray(c0.centers), widths=np.asarray(c0.widths),
        theta_star=np.asarray(env["theta"]),
        fit_methods=[m for m, _ in fits],
        fit_thetas=np.asarray([np.asarray(th) for _, th in fits]),
        errs={m: errs[m] for m in errs}, best=best,
        demos=np.asarray(demos), paths=np.asarray(paths),
        gram_ev=np.asarray(ev),
    )


def _recovery_render(data):
    """Recovered cost field, for the implicit method and both zero-solve baselines.

    Three things this figure has to get right, none of which are cosmetic:

    * **Name the method.**  A panel captioned only "recovered" leaves the reader
      guessing which of six methods produced it.  Every recovery panel is
      labelled, and the analytic baselines are shown beside ours on identical
      data so the comparison is visible rather than asserted.
    * **One colour scale.**  The panels share a single `vmax` and one colorbar.
      Per-panel normalization rescales a recovered field to fill the same ramp
      as the truth, so a uniformly-too-weak fit is drawn identically to a
      correct one -- it hides exactly the error the figure exists to show.
    * **The noisy regime.**  Demonstrations carry the same sigma noise as every
      dataset in the study; on noiseless demonstrations Inverse KKT is exact by
      construction and the panel says nothing about the regime the paper's
      claims live in.
    """
    name = str(data["name"])
    demo_noise = float(data["demo_noise"])
    n_show, M = int(data["n_show"]), int(data["M"])
    c0 = _FieldCtx(data["centers"], data["widths"])
    theta_star = np.asarray(data["theta_star"])
    d = 2  # only the (x, y) columns of the demonstrations/paths are drawn
    fits = list(zip([str(m) for m in data["fit_methods"]],
                    [np.asarray(t) for t in data["fit_thetas"]]))
    errs, best = data["errs"], str(data["best"])
    demos, paths = np.asarray(data["demos"]), [np.asarray(p) for p in data["paths"]]

    # Shared ceiling across every field panel, including the recovered ones.
    vmax = max(_field_grid(c0, th)[2].max()
               for th in [theta_star] + [t for _, t in fits])

    panels = [(theta_star, "ground truth", None, False)] + [
        (th, STYLE[m]["label"], errs[m], m == best) for m, th in fits]
    n_panels = len(panels)  # 8: ground truth + 7 methods
    n_cols, n_rows = 4, 2

    fig = plt.figure(figsize=(COL2, 3.4))
    gs = fig.add_gridspec(n_rows, n_cols + 2,
                          width_ratios=[1] * n_cols + [0.05, 1.4],
                          hspace=0.32, wspace=0.06)
    axes = []
    for idx, (th, lab, err, is_best) in enumerate(panels):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)
        im = _draw_field_background(ax, c0, th, vmax=vmax)
        ax.scatter(np.asarray(c0.centers)[:, 0], np.asarray(c0.centers)[:, 1],
                   s=5, c="k", zorder=3, linewidths=0)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.9, 1.9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lab, fontsize=7, pad=3)
        if err is not None:
            ax.set_xlabel(rf"$\|\hat\theta-\theta^\star\|_1={err:.2f}$",
                          labelpad=3, fontsize=7,
                          fontweight="bold" if is_best else "normal")
    cb = fig.colorbar(im, cax=axes[-1].inset_axes([1.08, 0.0, 0.06, 1.0]))
    cb.set_label("cost", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)

    # Trajectory panel spanning both rows on the right.
    axt = fig.add_subplot(gs[:, n_cols + 1])
    show = list(range(min(n_show, M)))
    allp = np.concatenate([demos[show][:, :, :2].reshape(-1, 2)] + paths)
    pad = 0.12
    ext = _match_aspect(
        (min(-2.5, allp[:, 0].min() - pad), max(2.5, allp[:, 0].max() + pad),
         min(-1.9, allp[:, 1].min() - pad), max(1.9, allp[:, 1].max() + pad)),
        5.0 / 3.8)
    _draw_field_background(axt, c0, theta_star, extent=ext, vmax=vmax)
    for j, i in enumerate(show):
        dm = demos[i][:, :2]
        axt.plot(dm[:, 0], dm[:, 1], color=DEMO_C, lw=0.7, alpha=0.55,
                 zorder=3, label="demonstration" if j == 0 else None)
        axt.scatter(dm[:, 0], dm[:, 1], s=1.6, color=DEMO_C, alpha=0.85,
                    linewidths=0, zorder=4)
        axt.plot(paths[j][:, 0], paths[j][:, 1], color=FIT_C, lw=1.15,
                 ls=(0, (3, 2)), path_effects=HALO,
                 label="implicit fit" if j == 0 else None, zorder=5)
    axt.set_xlim(ext[0], ext[1])
    axt.set_ylim(ext[2], ext[3])
    axt.set_aspect("equal")
    axt.set_xticks([])
    axt.set_yticks([])
    axt.set_title("trajectories", fontsize=7, pad=3)
    axt.set_xlabel(rf"$\sigma={demo_noise:g}$", labelpad=3, fontsize=7)
    axt.legend(loc="lower left", ncol=1, fontsize=5.8, handlelength=1.6,
               borderpad=0.25, labelspacing=0.25,
               bbox_to_anchor=(0.005, 0.005), framealpha=0.82, frameon=True)
    axt.get_legend().get_frame().set_linewidth(0)

    for ax, tag in zip(axes + [axt],
                      (f"({c})" for c in "abcdefghijklmnop")):
        _tag_inside(ax, tag)
    noise_lab = "High" if "highnoise" in name else "Low"
    fig.suptitle(f"Recovered Cost Fields Under {noise_lab} Demonstration Noise "
                f"($\\sigma={demo_noise:g}$)", fontsize=9, y=0.99, fontweight="bold")
    fig.subplots_adjust(left=0.002, right=0.998, top=0.92, bottom=0.04)
    finish(fig, name)
    return errs


# ---------------------------------------------------------------------------
# Fig. 4 - robustness to demonstration suboptimality (robot, E1)
# ---------------------------------------------------------------------------


def fig_recovery(recompute=True, name="fig3_recovery", **kw):
    """Recovered cost fields (implicit vs. baselines) on the multimodal field.

    Solves inline, persists the recovered weights / trajectories / geometry to
    `data/figdata/<name>.npz`, then draws from that file. The Gram lambda_2 and
    the identifiability rationale are documented on `_recovery_render`.
    """
    data = _cached(name, lambda: _recovery_compute(name=name, **kw), recompute)
    return _recovery_render(data)


def fig_recovery_highnoise(recompute=True, sigma=0.05, **kw):
    """The same recovery panel in the high-noise regime, where the bilevel
    methods separate from the analytic ones.

    This is not a second look at the same result.  The two zero-solve baselines
    fit the *stationarity residual at the demonstration*, so they assume the
    demonstration is an exact optimum.  It is at sigma=0 (relative stationarity
    ~1e-9) and it is emphatically not once noise is added (0.49 at sigma=0.02),
    and nothing in either method lets it trade a stationarity violation for
    behaviour that matches.  A rollout-based loss can, because it re-solves.
    So the ordering is expected to invert as sigma grows -- and at sigma=0 it
    runs the other way, with Inverse KKT exact at zero solves.  Both regimes are
    reported (`fig3_recovery` at sigma=0.02) rather than only the flattering one;
    the crossover is the result, not the win.
    """
    return fig_recovery(recompute=recompute, demo_noise=sigma,
                        name="fig3b_recovery_highnoise", **kw)


def fig_recovery_whole(font_scale=1.3, title="Recovered Cost Fields by Method"):
    """Combined low- and high-noise recovery in one figure, from cached data.

    Layout: the methods in `ORDER` split over 2 rows per half, plus a
    trajectory column spanning both; the two halves are separated by a thin
    rule, with a horizontal colorbar across the bottom.

    `font_scale` multiplies every type size.  The whole layout is solved from
    the resulting text metrics rather than from fixed fractions, so the row
    gaps, margins and figure height follow the type instead of having to be
    retuned by hand -- see the geometry block below.
    """
    from matplotlib.gridspec import GridSpec

    data_lo = load_figdata("fig3_recovery")
    data_hi = load_figdata("fig3b_recovery_highnoise")
    if data_lo is None or data_hi is None:
        print("  [skip] cached data missing for fig3 or fig3b")
        return

    # Type sizes at scale 1.0: the serif/STIX family and scale of fig4, one
    # notch up, since this figure runs the full 7.16 in page width across 18
    # panels.  `font_scale` takes the whole set up or down together.
    s = float(font_scale)
    FS_SUP, FS_TITLE, FS_LABEL = 10.5 * s, 9 * s, 8.5 * s
    FS_TAG, FS_LEGEND, FS_ROW, FS_TICK = 8.5 * s, 8 * s, 10 * s, 8 * s

    # Panel order: the baselines that attack the inverse problem directly on
    # the top row, the methods that differentiate through the solve on the
    # bottom, so the crossover between the two families reads down the rows.
    # The split is at the halfway point, so keep the two rows the same length.
    ORDER = ("kkt", "eiv", "cioc",
             "implicit", "fd", "cmaes")
    # Figure-local titles.  A method name has a fifth of the page width to fit
    # in, so the citations stay in the caption; `STYLE` keeps the cited forms
    # for the legends of the line figures, which have room for them.
    LABELS = {"cioc": "CIOC", "eiv": "IOC-EIV",
              "implicit": "Implicit diff.", "unrolled": "Unrolled diff."}

    def _unpack(data):
        c0 = _FieldCtx(data["centers"], data["widths"])
        theta_star = np.asarray(data["theta_star"])
        fits = list(zip([str(m) for m in data["fit_methods"]],
                        [np.asarray(t) for t in data["fit_thetas"]]))
        errs, best = data["errs"], str(data["best"])
        demos = np.asarray(data["demos"])
        paths = [np.asarray(p) for p in data["paths"]]
        n_show, M = int(data["n_show"]), int(data["M"])
        demo_noise = float(data["demo_noise"])
        vmax = max(_field_grid(c0, th)[2].max()
                   for th in [theta_star] + [t for _, t in fits])
        by_method = dict(fits)
        panels = [(by_method[m], LABELS.get(m, STYLE[m]["label"]), errs[m],
                   m == best)
                  for m in ORDER if m in by_method]
        return c0, theta_star, fits, errs, best, demos, paths, n_show, M, \
            demo_noise, vmax, panels

    lo = _unpack(data_lo)
    hi = _unpack(data_hi)
    # Fixed 0-1 colour scale rather than the observed maximum, so a shade means
    # the same cost across every panel and across regenerations of the figure.
    vmax = 1.0
    vmax_data = max(lo[10], hi[10])
    if vmax_data > vmax:
        print(f"  [warn] field maximum {vmax_data:.3f} exceeds the fixed "
              f"colour limit {vmax:g}; those panels clip")

    # --- page geometry, solved from the type ---------------------------------
    # Everything below is in inches, then converted to figure fractions at the
    # end.  Nothing is a tuned constant that `font_scale` would silently
    # invalidate: the gaps and margins come from the measured height of the
    # text that has to fit in them, and the trajectory column is narrowed just
    # far enough that the widest panel subtitle clears its neighbour.
    FIELD_EXT = (-2.5, 2.5, -1.9, 1.9)
    field_ar = (FIELD_EXT[1] - FIELD_EXT[0]) / (FIELD_EXT[3] - FIELD_EXT[2])
    fig_w = COL2
    WSP = 0.10

    def _line_h(pt):
        """Height of one line of `pt` type, with room for descenders."""
        return pt * 1.55 / 72.0

    # Two equal rows of method panels per half; the grid follows the number of
    # methods in ORDER rather than a hard-coded column count.
    n_cols = (len(lo[11]) + 1) // 2

    all_panels = list(lo[11]) + list(hi[11])
    widest_sub = max(
        _text_width_in(rf"$\|\hat\theta-\theta^\star\|_1={err:.2f}$",
                       FS_LABEL, "bold")
        for _, _, err, _ in all_panels if err is not None)
    widest_title = max(_text_width_in(lab, FS_TITLE, "bold")
                       for _, lab, _, _ in all_panels)

    # Row gaps: each carries the xlabel of the row above and the title of the
    # row below; the separator gap also carries the rule between halves.
    gap_t = _line_h(FS_LABEL) + _line_h(FS_TITLE) + 0.05
    gap_s = gap_t + _line_h(FS_TITLE) * 0.55

    # Margins: whatever the text outside the panel block needs.
    m_left = _line_h(FS_ROW) + 0.04            # rotated row label
    # The colorbar runs to the block's right edge and its last tick label is
    # centred there, so half of it hangs past; leave room or the tight bbox
    # pushes the saved figure over the 7.16 in column width.
    m_right = _text_width_in(f"{vmax:g}", FS_TICK) / 2 + 0.02
    m_top = _line_h(FS_SUP) + _line_h(FS_TITLE) + 0.08 if title \
        else _line_h(FS_TITLE) + 0.03
    cb_h = max(0.075, 0.09 * s)
    m_bot = (_line_h(FS_LABEL) + 0.13 + cb_h + _line_h(FS_TICK)
             + _line_h(FS_LABEL) + 0.04)

    avail_w = fig_w - m_left - m_right
    # wspace is a fraction of the *mean* column width; with four gaps over five
    # columns that inflates the block by this factor whatever the ratios are.
    k_wsp = 1 + WSP * n_cols / (n_cols + 1)

    # Field-column width `u` as a function of the trajectory cell's aspect:
    # the trajectory cell spans two field rows plus the gap between them, so
    # its width is traj_ar * (2v + gap_t) and the block must still fit
    # `avail_w`.  Solving 4u + traj_ar*(2u/field_ar + gap_t) = avail_w/k_wsp:
    def _col_width(traj_ar):
        return ((avail_w / k_wsp - traj_ar * gap_t)
                / (n_cols + 2 * traj_ar / field_ar))

    # A subtitle wider than its column pitch collides with its neighbour's.
    # Give the trajectory panel the most width that still leaves the pitch
    # clear, walking down from a near-square cell.
    TRAJ_AR_MAX, TRAJ_AR_MIN = 1.05, 0.60
    traj_ar = TRAJ_AR_MAX
    while traj_ar > TRAJ_AR_MIN:
        u = _col_width(traj_ar)
        traj_w = traj_ar * (2 * u / field_ar + gap_t)
        pitch = u + WSP * (n_cols * u + traj_w) / (n_cols + 1)
        if pitch >= widest_sub + 0.03 and pitch >= widest_title + 0.03:
            break
        traj_ar -= 0.01
    else:
        u = _col_width(TRAJ_AR_MIN)
        print(f"  [warn] at font_scale={s:g} the widest panel label "
              f"({max(widest_sub, widest_title):.2f} in) does not fit the "
              f"column pitch; labels will crowd")

    v = u / field_ar
    traj_w = traj_ar * (2 * v + gap_t)
    block_h = 4 * v + 2 * gap_t + gap_s
    fig_h = m_top + block_h + m_bot

    L, R = m_left / fig_w, 1 - m_right / fig_w
    TOP, BOT = 1 - m_top / fig_h, m_bot / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    # 5 content columns (4 field + 1 trajectory); 4 content rows (2 per half)
    # with explicit spacer rows, so each gap is set independently and the
    # colorbar is placed in the bottom margin rather than a grid row.  Ratios
    # are passed in inches; GridSpec normalizes them.
    gs = GridSpec(7, n_cols + 1, figure=fig,
                  height_ratios=[v, gap_t, v, gap_s, v, gap_t, v],
                  width_ratios=[u] * n_cols + [traj_w],
                  hspace=0.0, wspace=WSP,
                  left=L, right=R, top=TOP, bottom=BOT)

    row_map = {0: (0, 2), 1: (4, 6)}
    half_labels = {0: "Low noise", 1: "High noise"}
    all_field_axes = []
    traj_axes = []
    im = None

    for half, (c0, theta_star, fits, errs, best, demos, paths, n_show, M,
               demo_noise, _, panels) in enumerate([lo, hi]):
        r0, r1 = row_map[half]

        axes = []
        for idx, (th, lab, err, is_best) in enumerate(panels):
            r_local, c = divmod(idx, n_cols)
            ax = fig.add_subplot(gs[r0 + 2 * r_local, c])
            axes.append(ax)
            im = _draw_field_background(ax, c0, th, extent=FIELD_EXT, vmax=vmax)
            ax.scatter(np.asarray(c0.centers)[:, 0],
                       np.asarray(c0.centers)[:, 1],
                       s=5, c="k", zorder=3, linewidths=0)
            ax.set_xlim(FIELD_EXT[0], FIELD_EXT[1])
            ax.set_ylim(FIELD_EXT[2], FIELD_EXT[3])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(lab, fontsize=FS_TITLE, pad=2.5, fontweight="bold")
            if err is not None:
                ax.set_xlabel(rf"$\|\hat\theta-\theta^\star\|_1={err:.2f}$",
                              labelpad=2, fontsize=FS_LABEL,
                              fontweight="bold")
        all_field_axes.extend(axes)

        # Trajectory panel spanning both rows of this half.
        axt = fig.add_subplot(gs[r0:r1 + 1, n_cols])
        show = list(range(min(n_show, M)))
        allp = demos[show][:, :, :2].reshape(-1, 2)
        pad = 0.12
        # Match the extent to the cell the panel actually got, so the panel
        # fills it.  Read from the layout rather than from TRAJ_AR, so tuning
        # any geometry constant above cannot reintroduce dead space.
        cell = axt.get_position()
        ext = _match_aspect(
            (min(FIELD_EXT[0], allp[:, 0].min() - pad),
             max(FIELD_EXT[1], allp[:, 0].max() + pad),
             min(FIELD_EXT[2], allp[:, 1].min() - pad),
             max(FIELD_EXT[3], allp[:, 1].max() + pad)),
            (cell.width * fig_w) / (cell.height * fig_h))
        _draw_field_background(axt, c0, theta_star, extent=ext, vmax=vmax)
        # Demonstrations only.  The panel's job is now to show how much the
        # noise roughens the demonstrations between (i) and (r), so draw them
        # opaque, at full width, and mark every knot: the jaggedness lives in
        # the per-knot deviation, which a thin translucent line hides.
        for j, i in enumerate(show):
            dm = demos[i][:, :2]
            axt.plot(dm[:, 0], dm[:, 1], color=DEMO_C, lw=1.0, alpha=1.0,
                     solid_joinstyle="miter", zorder=3,
                     label="demonstration" if j == 0 else None)
            axt.scatter(dm[:, 0], dm[:, 1], s=3.2, color=DEMO_C,
                        linewidths=0, zorder=4)
        axt.set_xlim(ext[0], ext[1])
        axt.set_ylim(ext[2], ext[3])
        axt.set_aspect("equal")
        axt.set_xticks([])
        axt.set_yticks([])
        axt.set_title("Trajectories", fontsize=FS_TITLE, pad=2.5,
                      fontweight="bold")
        axt.set_xlabel(rf"$\sigma={demo_noise:g}$", labelpad=2,
                       fontsize=FS_LABEL, fontweight="bold")
        traj_axes.append(axt)

        # Letters run straight through both halves, trajectory panel included.
        per_half = len(axes) + 1
        letters = ALPHABET[half * per_half:(half + 1) * per_half]
        for ax, tag in zip(axes + [axt],
                           (f"({c})" for c in letters)):
            _tag_inside(ax, tag, size=FS_TAG)

        # Row label, centred in the left margin the grid leaves for it.
        noise_lab = f"{half_labels[half]} ($\\sigma={demo_noise:g}$)"
        mid_y = (axes[0].get_position().y1 + axes[-1].get_position().y0) / 2
        fig.text(L / 2, mid_y, noise_lab, rotation=90, ha="center",
                 va="center", fontsize=FS_ROW, fontstyle="italic")

    # Thin rule between halves, run to the panel block's own edges.
    n_lo = len(lo[11])
    sep_y = (all_field_axes[n_lo - 1].get_position().y0
             + all_field_axes[n_lo].get_position().y1) / 2
    fig.add_artist(plt.Line2D([L, R], [sep_y, sep_y],
                   transform=fig.transFigure, color="#cccccc",
                   linewidth=0.5, zorder=0))

    # Horizontal colorbar spanning the full panel block along the bottom, sat
    # on top of the space its own ticks and label need.
    cb_y = (_line_h(FS_TICK) + _line_h(FS_LABEL) + 0.04) / fig_h
    cax = fig.add_axes([L, cb_y, R - L, cb_h / fig_h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Cost", fontsize=FS_LABEL, labelpad=2, fontweight="bold")
    cb.ax.tick_params(labelsize=FS_TICK, width=0.5, length=2.0, pad=1.5)
    cb.outline.set_linewidth(0.4)

    if title:
        fig.suptitle(title, fontsize=FS_SUP, fontweight="bold",
                     y=1 - 0.35 * _line_h(FS_SUP) / fig_h, va="center")

    finish(fig, "fig3_recovery_whole")


def fig_noise():
    files = sorted(glob.glob(os.path.join(DATA, "robot", "e1_sigma*.json")))
    if not files:
        print("  [skip] no E1 noise data")
        return
    rows = {}
    for f in files:
        d = json.load(open(f))
        rows[d["demo_noise"]] = d["results"]
    sig = np.array(sorted(rows))
    methods = ("implicit", "unrolled", "fd", "cmaes", "kkt", "cioc", "eiv", "random")
    # Implicit and finite differences agree to 3-4 significant figures here, so
    # one curve would sit exactly on top of the other and vanish.  Dodge the
    # markers slightly along x so both remain visible; the lines still coincide,
    # which is the point being made.
    dodge = {"implicit": -0.0006, "fd": 0.0006}
    floor = 1e-7  # KKT is exact at sigma=0 (regret ~1e-19); an unclipped log
    # axis would span 12 decades of empty space and crush the informative range.

    fig, ax = line_figure(3.0)
    for m in draw_order(methods):
        # Skip methods absent from the data (e.g. "eiv" before re-collection).
        if any(m not in t["methods"] for s in sig for t in rows[s].values()):
            continue
        st = STYLE[m]
        vals = [[t["methods"][m]["theta_l1"] for t in rows[s].values()] for s in sig]
        med = np.array([np.median(v) for v in vals])
        q1 = np.array([np.percentile(v, 25) for v in vals])
        q3 = np.array([np.percentile(v, 75) for v in vals])
        x = sig + dodge.get(m, 0.0)
        ax.plot(x, med, marker=st["marker"], ls=st["ls"], color=st["color"],
                label=st["label"], clip_on=False, zorder=3)
        ax.fill_between(x, q1, q3, color=st["color"], alpha=0.12, lw=0, zorder=2)
    ax.set_xlabel(r"demonstration noise $\sigma$ [rad]")
    ax.set_ylabel(r"$\|\hat\theta-\theta^\star\|_1$")
    ax.set_xticks(sig)
    ax.margins(x=0.06)
    tidy(ax)
    line_figure_finish(fig, ax, methods,
                       "Weight Recovery vs. Demonstration Noise (Robot)",
                       "fig4_noise_robot", ncol=2)


# ---------------------------------------------------------------------------
# Fig. 4b - KKT-seeding ablation (implicit only): does a free z0 from
# Inverse-KKT survive demonstration noise and landscape multimodality?
# ---------------------------------------------------------------------------

# Local to this ablation: the "random" arm here is z0 ~ N(0, 0.5) *fed through
# the same implicit-adjoint Adam fit* as the KKT-seeded arm, not the
# unoptimized `STYLE["random"]` baseline used elsewhere -- same optimizer,
# same budget, different starting point only.
_KKT_SEED_STYLE = {
    "random":   dict(color="0.45", marker="o", ls="--", label=r"Implicit, random $z_0$"),
    "kkt_seed": STYLE["kkt_seed"],
}


def fig_kkt_seed():
    """Regret vs demonstration noise, split by bump width (uni/multimodal
    `field` regime) and inner-solver restarts, comparing an implicit-adjoint
    Adam fit started from a random z0 against one seeded by `analytic.kkt_fit`
    (zero extra forward solves). Every other setting -- contexts, demos,
    budget -- is identical between the two lines in a panel; only z0 differs.
    """
    path = os.path.join(DATA, "bench2d", "bench2d_kkt_seed_dynamics.json")
    if not os.path.exists(path):
        print("  [skip] no kkt-seed ablation data")
        return
    data = json.load(open(path))

    import re
    parsed = {}
    for tag, d in data.items():
        m = re.match(r"bw([\d.]+)_R(\d+)_sigma([\d.]+)", tag)
        bw, R, sigma = m.group(1), int(m.group(2)), float(m.group(3))
        parsed[(bw, R, sigma)] = d["results"]
    bws = sorted({k[0] for k in parsed}, reverse=True)   # wide (unimodal) first
    Rs = sorted({k[1] for k in parsed})
    sigmas = sorted({k[2] for k in parsed})

    regime_name = {bws[0]: "wide bumps\n(unimodal)", bws[1]: "narrow bumps\n(multimodal)"}

    # This is the one double-column figure in the slate whose axes are a 2x2
    # grid, so the column-figure point sizes set in `set_style()` come out
    # roughly half the apparent size of fig. 1's when the panel is printed at
    # 7.16 in.  Scale the text up locally -- same family, same weights, same
    # palette -- so a reader moving between the figures sees one style at one
    # legible size rather than two.
    with plt.rc_context({
        "font.size": 11.7,
        "axes.labelsize": 12.5,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.6,
    }):
        # Constrained layout, as in fig. 1 and fig. 4: it is the only engine
        # that measures the suptitle and the outside key, so the gaps above and
        # below the grid come out matched instead of hand-tuned.
        fig, axes = plt.subplots(len(bws), len(Rs), figsize=(COL2, 2.15 * len(bws)),
                                 squeeze=False, sharex=True, sharey="row",
                                 layout="constrained")
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.05, wspace=0.03, hspace=0.04)
        for i, bw in enumerate(bws):
            for j, R in enumerate(Rs):
                ax = axes[i][j]
                meds = {}
                for arm in ("random", "kkt_seed"):
                    st = _KKT_SEED_STYLE[arm]
                    vals = [[t["arms"][arm]["l1"] for t in parsed[(bw, R, s)].values()]
                            for s in sigmas if (bw, R, s) in parsed]
                    med = np.array([np.median(v) for v in vals])
                    q1 = np.array([np.percentile(v, 25) for v in vals])
                    q3 = np.array([np.percentile(v, 75) for v in vals])
                    meds[arm] = med
                    ax.plot(sigmas, med, marker=st["marker"], ls=st["ls"],
                            color=st["color"], label=st["label"], clip_on=False,
                            zorder=3)
                    ax.fill_between(sigmas, q1, q3, color=st["color"], alpha=0.15,
                                    lw=0, zorder=2)
                ax.set_xticks(sigmas)
                ax.margins(x=0.06)
                tidy(ax)
                panel_label(ax, "(" + "abcdefgh"[i * len(Rs) + j] + ")",
                            dx=-0.13 if j == 0 else -0.04, dy=1.10, fontsize=13)
                # Headline number per panel: the noiseless-case ratio, where the
                # comparison is cleanest.  Placed in the low corner shared by
                # both curves' right tail, which stays clear across all panels.
                gain = meds["random"][0] / meds["kkt_seed"][0]
                txt = rf"${gain:.1f}\times$ at $\sigma{{=}}0$" if gain < 10 else \
                      rf"${gain:,.0f}\times$ at $\sigma{{=}}0$"
                ax.text(0.97, 0.06, txt, transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=11, fontweight="bold",
                        color=_KKT_SEED_STYLE["kkt_seed"]["color"],
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                                  alpha=0.85))
                if i == 0:
                    ax.set_title(f"$R={R}$ restart" + ("s" if R != 1 else ""))
                if i == len(bws) - 1:
                    ax.set_xlabel(r"demonstration noise $\sigma$ [rad]")
                if j == 0:
                    ax.set_ylabel(r"$\|\hat\theta-\theta^\star\|_1$")
                    # Row label rides outside the y-axis label in axes
                    # coordinates rather than figure coordinates, so it tracks
                    # the row wherever the layout engine puts it.
                    ax.text(-0.30, 0.5, regime_name[bw], transform=ax.transAxes,
                            rotation=90, ha="center", va="center", fontsize=12.5,
                            fontweight="bold")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.suptitle(r"Weight Recovery vs. Demonstration Noise: KKT-Seeded vs. Random $z_0$",
                     fontsize=14, fontweight="bold")
        fig.legend(handles, labels, loc="outside lower center", ncol=2,
                   fontsize=12.5, frameon=False, handlelength=2.4,
                   columnspacing=1.6, handletextpad=0.5)
        finish(fig, "fig4b_kkt_seed")


def fig_kkt_seed_trace():
    """Convergence traces (best loss vs forward solves) at sigma=0 -- the free,
    noiseless case where the head start should be largest -- one seed per
    (bump width, restarts) cell, chosen as the seed closest to that cell's
    median final regret so the picked trace is representative, not cherry-picked.
    """
    path = os.path.join(DATA, "bench2d", "bench2d_kkt_seed_dynamics.json")
    if not os.path.exists(path):
        print("  [skip] no kkt-seed ablation data")
        return
    data = json.load(open(path))

    import re
    cells = {}
    for tag, d in data.items():
        m = re.match(r"bw([\d.]+)_R(\d+)_sigma0\.0$", tag)
        if m:
            cells[(m.group(1), int(m.group(2)))] = d["results"]
    if not cells:
        print("  [skip] no sigma=0 cells in kkt-seed ablation data")
        return
    bws = sorted({k[0] for k in cells}, reverse=True)  # wide (unimodal) first
    Rs = sorted({k[1] for k in cells})
    regime_name = {bws[0]: "wide bumps, unimodal", bws[-1]: "narrow bumps, multimodal"} \
        if len(bws) > 1 else {bws[0]: ""}

    fig, axes = plt.subplots(len(bws), len(Rs), figsize=(COL2, 2.3 * len(bws)),
                              squeeze=False, sharex=True, sharey=True)
    for i, bw in enumerate(bws):
        for j, R in enumerate(Rs):
            ax = axes[i][j]
            results = cells.get((bw, R))
            if results is None:
                ax.axis("off")
                continue
            finals = np.array([t["arms"]["kkt_seed"]["regret"] for t in results.values()])
            rep = list(results.keys())[int(np.argsort(finals)[len(finals) // 2])]
            for arm in ("random", "kkt_seed"):
                st = _KKT_SEED_STYLE[arm]
                solves, loss = zip(*results[rep]["arms"][arm]["trace"])
                ax.plot(solves, np.maximum(loss, REGRET_FLOOR), ls=st["ls"],
                        color=st["color"], label=st["label"], lw=1.1, zorder=3)
            ax.set_yscale("log")
            tidy(ax)
            if i == 0:
                ax.set_title(f"$R={R}$ restarts", fontsize=8)
            if i == len(bws) - 1:
                ax.set_xlabel("forward solves")
            if j == 0:
                ax.set_ylabel("best loss so far")
    for i, bw in enumerate(bws):
        y0 = axes[i][0].get_position().y0
        y1 = axes[i][0].get_position().y1
        fig.text(0.005, (y0 + y1) / 2, regime_name[bw], rotation=90,
                  ha="left", va="center", fontsize=7.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.55, 0.94), frameon=False)
    fig.tight_layout(rect=(0.03, 0, 1, 0.88))
    finish(fig, "fig4c_kkt_seed_trace")


# ---------------------------------------------------------------------------
# Fig. 5 - the noise/optimality tradeoff on a multimodal 2D field
# ---------------------------------------------------------------------------

# Same hand-placed, identifiability-friendly layout `fig_recovery` validated:
# well-separated bumps every demonstration is actually exposed to, plus
# `topo_restarts=True` structured lateral-detour multistart (the i.i.d.-jitter
# default was measured to leave `implicit` stuck in outer-loss local minima on
# this multimodal field regardless of budget -- see the regime-figure retune
# notes; jitter resamples one basin instead of covering the distinct ones a
# multimodal context has, per `ioc.inner.InnerSolver.solve`'s docstring).
_REGIME_CENTERS = [
    (0.0, 0.0),
    (0.0, 1.35), (0.0, -1.35),
    (-1.6, 1.65), (1.6, 1.65),
    (-1.6, -1.65),
]
_REGIME_WIDTHS = [0.45, 0.35, 0.35, 0.4, 0.4, 0.4]
_REGIME_PAIRS = [
    ((-2.2, 0.9), (2.2, 0.9)), ((-2.2, 0.6), (2.2, 1.1)),
    ((-2.2, -0.9), (2.2, -0.9)), ((-2.2, -0.6), (2.2, -1.1)),
    ((-2.2, 1.3), (2.2, 1.6)), ((-2.2, 1.6), (2.2, 1.3)),
    ((-2.2, -1.3), (2.2, -1.6)), ((-2.2, -1.6), (2.2, -1.3)),
]
_REGIME_THETA = [0.12, 0.08, 0.18, 0.10, 0.14, 0.16, 0.12, 0.10]


def _regime_trial(seed, demo_noise, budget=8000, n_iter=800, n_restarts=7,
                  fd_eps=1e-4):
    """One (sigma, seed) draw: fit all six baseline methods on the multimodal
    field, identical recipe to `fig_recovery`, varying only the noise
    seed/level. Returns {method: (theta_l1, regret)}."""
    M, T = 8, 30
    env = _make_env("field", 6, M, T, seed, 0.4, theta=_REGIME_THETA,
                    n_iter=n_iter, demo_noise=demo_noise, n_restarts=n_restarts,
                    topo_restarts=True, bump_layout=(_REGIME_CENTERS, _REGIME_WIDTHS),
                    demo_pairs=_REGIME_PAIRS)
    si, ctxs, demos = env["solver"], env["ctxs"], env["demos"]
    x0, d, K, inner, theta_star = env["x0"], env["d"], env["K"], env["inner"], env["theta"]

    def loss_of(solver):
        def loss(z):
            t = jax.nn.softmax(z)

            def one(c, dm, xx):
                p = b2d.unpack(solver(xx, t, c), c, T, d)[:, :2]
                return jnp.mean(jnp.sum((p - dm[:, :2]) ** 2, axis=-1))

            return jnp.mean(jax.vmap(one)(ctxs, demos, x0))

        return loss

    per_solve = M * n_restarts

    def fit_adam(solver):
        li = jax.jit(loss_of(solver))
        gi = jax.jit(jax.value_and_grad(li))
        z, _ = outer_opt.adam(gi, jnp.zeros(K), lr=0.1, budget_solves=budget,
                              solves_per_step=per_solve, trace_best=True)
        return z, li

    z, li = fit_adam(si)
    z_unrolled, _ = fit_adam(inner.solve_unrolled)
    z_fd, _ = outer_opt.adam(outer_opt.fd_grad_fn(li, fd_eps), jnp.zeros(K), lr=0.1,
                             budget_solves=budget, solves_per_step=(K + 1) * per_solve,
                             trace_best=True)
    z_cmaes, _ = outer_opt.cma_es(li, jnp.zeros(K), budget_solves=budget,
                                  solves_per_eval=per_solve, seed=seed, trace_best=True)
    z_kkt = analytic.kkt_fit(inner.grad_x, ctxs, demos, K, n_steps=600)
    z_cioc = analytic.cioc_fit(inner.grad_x, inner.gn_system, ctxs, demos, K,
                               n_steps=600)
    z_eiv = analytic.eiv_fit(inner.grad_x, ctxs, demos, K)
    # Do-nothing baseline: an unfit random draw, not the zeros(K) the other
    # methods start their optimization from -- matches `bench2d.run`'s own
    # "random" convention (`z0 = rng.normal(scale=0.5, size=K)`).
    z_random = 0.5 * jax.random.normal(jax.random.key(seed), (K,))
    fits = {"implicit": jax.nn.softmax(z), "unrolled": jax.nn.softmax(z_unrolled),
            "fd": jax.nn.softmax(z_fd), "cmaes": jax.nn.softmax(z_cmaes),
            "kkt": jax.nn.softmax(z_kkt), "cioc": jax.nn.softmax(z_cioc),
            "eiv": jax.nn.softmax(z_eiv),
            "random": jax.nn.softmax(z_random)}

    x_star = jax.vmap(lambda x, c: si(x, theta_star, c))(x0, ctxs)
    ref = jax.vmap(lambda x, c: inner.cost(x, theta_star, c))(x_star, ctxs)
    out = {}
    for m, th in fits.items():
        xh = jax.vmap(lambda x, c: si(x, th, c))(x0, ctxs)
        c = jax.vmap(lambda x, cc: inner.cost(x, theta_star, cc))(xh, ctxs)
        out[m] = (float(jnp.sum(jnp.abs(th - theta_star))), float(jnp.mean(c - ref)))
    return out


_REGIME_METHODS = ("implicit", "unrolled", "fd", "cmaes", "kkt", "cioc", "eiv",
                   "random")


def _regime_compute(sigmas=(0.0, 0.01, 0.02, 0.05, 0.08), n_seeds=5):
    """Sweep demonstration noise, fitting every method over `n_seeds` draws per
    sigma; return the per-(sigma, seed) L1 weight error and cost regret."""
    methods = _REGIME_METHODS
    sig = np.asarray(sigmas, float)
    l1 = {m: [] for m in methods}
    regret = {m: [] for m in methods}
    for s in sig:
        trials = [_regime_trial(seed, float(s)) for seed in range(n_seeds)]
        for m in methods:
            l1[m].append([t[m][0] for t in trials])
            regret[m].append([t[m][1] for t in trials])
        print(f"  [fig5] sigma={s:g}  " + "  ".join(
            f"{m}_l1={np.median([t[m][0] for t in trials]):.3f}" for m in methods))
    return dict(sig=sig, methods=list(methods),
                l1={m: np.asarray(l1[m]) for m in methods},
                regret={m: np.asarray(regret[m]) for m in methods})


def _regime_render(data):
    """Distribution of recovery error vs. demonstration noise, on the same
    multimodal field `fig_recovery`/`fig_recovery_highnoise` use at two fixed
    sigma points.  This sweeps sigma continuously (several seeds per point) to
    show the crossover documented in `fig_recovery_highnoise` as a trend
    rather than a before/after snapshot: `kkt`/`cioc` fit stationarity/Laplace
    quantities *at the demonstration itself*, which is only justified while
    the demonstration stays near-optimal; `implicit` re-solves and fits
    rollout behaviour, which degrades far more gracefully as sigma grows.
    """
    methods = [str(m) for m in data["methods"]]
    sig = np.asarray(data["sig"])

    fig, ax = plt.subplots(1, 1, figsize=(COL1, 2.4))
    for m in methods:
        st = STYLE[m]
        vals = np.asarray(data["l1"][m])  # (n_sigma, n_seeds)
        med = np.array([np.median(v) for v in vals])
        q1 = np.array([np.percentile(v, 25) for v in vals])
        q3 = np.array([np.percentile(v, 75) for v in vals])
        ax.plot(sig, med, marker=st["marker"], ls=st["ls"], color=st["color"],
                label=st["label"], clip_on=False, zorder=3)
        ax.fill_between(sig, q1, q3, color=st["color"], alpha=0.12, lw=0, zorder=2)
    ax.set_xlabel(r"demonstration noise $\sigma$")
    ax.set_ylabel(r"$\|\hat\theta-\theta^\star\|_1$")
    ax.set_xticks(sig)
    tidy(ax)
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    fig.suptitle("Weight Recovery vs. Demonstration Noise (Multimodal Field)",
                fontsize=9, y=1.03, fontweight="bold")
    fig.tight_layout()
    finish(fig, "fig5_noise_field")


def fig_regime(recompute=True, sigmas=(0.0, 0.01, 0.02, 0.05, 0.08), n_seeds=5):
    """Recovery error vs. demonstration noise on the multimodal field. Sweeps
    inline, persists the per-(sigma, seed) errors to
    `data/figdata/fig5_noise_field.npz`, then draws from that file."""
    data = _cached("fig5_noise_field",
                   lambda: _regime_compute(sigmas=sigmas, n_seeds=n_seeds),
                   recompute)
    return _regime_render(data)


def main(only: str = "", recompute: bool = True):
    """Render the paper figures.

    `--no-recompute` renders the inline-built figures (ambiguity, environments,
    recovery, recovery_highnoise, regime) from their cached
    `data/figdata/<name>.npz` instead of re-solving -- fast, and the default
    once the data has been generated once. The disk-backed figures (scaling,
    noise, kkt_seed, kkt_seed_trace) always read their collected JSON and ignore
    this flag.
    """
    set_style()
    todo = {
        "scaling": fig_scaling,
        "environments": fig_environments,
        "ambiguity": fig_ambiguity,
        "recovery": fig_recovery,
        "recovery_highnoise": fig_recovery_highnoise,
        "noise": fig_noise,
        "kkt_seed": fig_kkt_seed,
        "kkt_seed_trace": fig_kkt_seed_trace,
        "regime": fig_regime,
    }
    # Figures that build their data inline accept `recompute`; the JSON-backed
    # ones do not (they always render from their collected file).
    cacheable = {"environments", "ambiguity", "recovery", "recovery_highnoise",
                 "regime"}
    # `--only` takes a comma-separated list: the curated paper slate is a
    # subset of `todo`, and regenerating it should be one command.
    want = [n.strip() for n in only.split(",") if n.strip()] if only else []
    unknown = [n for n in want if n not in todo]
    if unknown:
        raise SystemExit(
            f"unknown figure(s) {unknown}; available: {list(todo)}")
    for name, fn in todo.items():
        if want and name not in want:
            continue
        print(f"[{name}]")
        fn(recompute=recompute) if name in cacheable else fn()


if __name__ == "__main__":
    tyro.cli(main)
