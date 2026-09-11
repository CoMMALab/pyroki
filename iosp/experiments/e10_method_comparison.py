"""E10 method comparison: CMA-ES, unrolled autodiff, implicit diff, and FD
on the human teleop episodes, with a CROSS-SESSION fit/held-out split: the 27
episodes of the 2026-09-03 session are fitted, the 10 of the 2026-09-02 session
(the original set, which used to be split 8/2 within itself) are held out in
full.  See `iosp.fit.teleop.FIT_DEMO_DIR`.

Saves structured results (JSON + NPZ) for later visualization and tables.

Usage:
    python -m iosp.experiments.e10_method_comparison [--compile-timeout 1800]
"""
import argparse
import dataclasses
import json
import os
import pathlib
import signal
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from iosp import config
config.enable_compilation_cache()

from ioc import identifiability as ident
from ioc import outer as outer_opt
from iosp.fit.teleop import build_teleop, measure_standoffs, z_prior, TCP_OFFSET_M
from iosp.fit.parametric import _build_inner, screen_stationarity
from iosp.fit.params import z_scale
from iosp.model import fr3, pickplace as pp
from iosp.model.pickplace import split_trajopt as _split_trajopt
from iosp.model.pickplace import split_trajopt_perseg as _split_trajopt_perseg
from ioc.robot.problem import Scene

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "results" / "e10_methods"
N_OUTER_STEPS = 40
LR = 0.05
FD_EPS = 1e-3
CMA_BUDGET_SOLVES = 500


def _metrics(built, u):
    return dict(
        joint_rmse_fit=built["rmse_a"](u),
        joint_rmse_gen=built["rmse_b"](u),
        ee_rmse_fit=built["ee_rmse_a"](u),
        ee_rmse_gen=built["ee_rmse_b"](u),
        loss=float(built["gf"](u)[0]),
    )


def _check_trace(res):
    """A non-finite terminal loss is not a result.  The 5-seed run recorded
    `unrolled` with `trace = [(40, nan)]` and still wrote its theta into
    `summary.json` and the comparison table, where it read as a converged fit
    that merely performed poorly.  Flag it on the record instead."""
    tail = res.get("trace") or [(0, float("nan"))]
    val = float(tail[-1][1])
    if not np.isfinite(val):
        print(f"  WARNING: {res['method']} terminal loss is {val} -- "
              "the fit DIVERGED; theta below is not a result", flush=True)
        res["diverged"] = True
    return res


def _theta_dict(built, u):
    theta = built["theta_of"](u)
    return {n: float(v) for n, v in zip(built["names"], theta)}


def _gram(built, u):
    t0 = time.perf_counter()
    eigvals, eigvecs = ident.sensitivity_spectrum(built["jac_fn"], u)
    retained, discarded, r = ident.select_rank(eigvals, rule="gap")
    t_gram = time.perf_counter() - t0
    return dict(
        eigvals=eigvals.tolist(),
        rank=int(r),
        retained=[int(i) for i in retained],
        discarded=[int(i) for i in discarded],
        wall_gram_s=t_gram,
    )


def _chunked_eval(vg, U, chunk):
    """`vg(U)` for a (S, K) seed batch, evaluated `chunk` rows at a time.

    EXACTLY equivalent to `vg(U)`: each row's (value, gradient) is independent
    -- the seeds never interact inside the forward map -- so splitting the
    batch changes only peak memory, never a number.

    It is needed because the whole seed batch does not fit.  MEASURED on a
    24 GiB A5000 at 25 fit + 5 held episodes and K=28: the 3-seed
    `jit_loss_a` asks for 10.33 GiB on top of its live buffers and dies with
    RESOURCE_EXHAUSTED, where one seed at a time fits comfortably.  It costs
    almost nothing in wall-clock: a single 30-episode solve already saturates
    the card, so the seed axis was never actually running in parallel --
    measured, batching 2 candidates instead of 1 bought 1.11x, and 4 bought
    1.17x.  The batch was buying memory pressure, not throughput.

    The last chunk is PADDED to a full `chunk` rows (by repeating the final
    seed) so every call has one shape and XLA compiles the program once.
    """
    S = U.shape[0]
    if chunk is None or chunk >= S:
        return vg(U)
    vals, grads = [], []
    for i in range(0, S, chunk):
        block = U[i:i + chunk]
        pad = chunk - block.shape[0]
        if pad:
            block = jnp.concatenate([block, jnp.repeat(block[-1:], pad, axis=0)])
        v, g = vg(block)
        if pad:
            v, g = v[:chunk - pad], g[:chunk - pad]
        vals.append(v)
        grads.append(g)
    return jnp.concatenate(vals), jnp.concatenate(grads)


def _batched_adam(vg, seeds, lr, n_steps, chunk=None):
    """One Adam over a (S, K) seed batch driven by a vmapped `(value, grad)` fn;
    returns (u_hat, best_vals, hist, t_compile, t_infer).  Each row gets its own
    exact gradient (no interaction), and the winner is the best TRAINING loss
    across seeds -- rollout success is never consulted.

    `chunk` caps how many seeds are evaluated at once; see `_chunked_eval`.
    The Adam update itself stays over the full (S, K) array, so the optimiser
    state and the step arithmetic are unchanged."""
    import optax

    U = jnp.stack([jnp.asarray(s, jnp.float32) for s in seeds])   # (S, K)
    t0 = time.perf_counter()
    v0, _ = _chunked_eval(vg, U, chunk); v0.block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"  compile: {t_compile:.1f}s  ({U.shape[0]} seeds"
          f"{f', {chunk} at a time' if chunk and chunk < U.shape[0] else ' batched'})",
          flush=True)

    opt = optax.adamw(lr, weight_decay=0.0)
    state = opt.init(U)
    best_U = U
    best_vals = jnp.asarray(np.full(U.shape[0], np.inf, np.float32))
    hist = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        vals, grads = _chunked_eval(vg, U, chunk)
        improved = vals < best_vals
        best_U = jnp.where(improved[:, None], U, best_U)
        best_vals = jnp.where(improved, vals, best_vals)
        hist.append(float(jnp.min(vals)))
        updates, state = opt.update(grads, state, U)
        U = optax.apply_updates(U, updates)
    t_infer = time.perf_counter() - t0
    i = int(jnp.argmin(best_vals))
    return np.asarray(best_U[i]), np.asarray(best_vals), hist, t_compile, t_infer


def run_implicit_parallel(built, seeds, lr=LR, n_steps=N_OUTER_STEPS, chunk=None):
    """Implicit-diff multistart as ONE batched program: `value_and_grad` vmapped
    over the (S, K) seeds, one batched Adam.  The IK `custom_vmap` folds the seed
    axis into the kernel's problem batch, so S seeds cost ~one seed."""
    print("\n=== Implicit diff (parallel multistart) ===", flush=True)
    vg = jax.jit(jax.vmap(built["gf"]))
    u_hat, best_vals, hist, tc, ti = _batched_adam(vg, seeds, lr, n_steps, chunk)
    starts = [round(float(v), 4) for v in best_vals]
    print(f"  fit: {ti:.1f}s  winner loss {min(starts):.4f} "
          f"(per-seed best: {starts})", flush=True)
    return dict(method="implicit", u_hat=u_hat,
                trace=[(0, hist[0]), (n_steps, hist[-1])],
                wall_compile_s=tc, wall_infer_s=ti, wall_total_s=tc + ti,
                n_starts=len(seeds), start_losses=starts)


def run_fd_parallel(built, seeds, lr=LR, n_steps=N_OUTER_STEPS, chunk=None):
    """Finite-difference multistart, batched: the value-only loss's FD gradient
    (K+1 probes) is vmapped over the seeds and one batched Adam updates them.
    Same eager-loop reasoning as `run_fd` (no scan inlining the K+1 solves), but
    now the seed axis rides along inside each already-batched forward."""
    print("\n=== Finite differences (parallel multistart) ===", flush=True)
    fd_gf = outer_opt.fd_grad_fn(built["loss"], FD_EPS, batched=False)
    vg = jax.jit(jax.vmap(fd_gf))
    u_hat, best_vals, hist, tc, ti = _batched_adam(vg, seeds, lr, n_steps, chunk)
    starts = [round(float(v), 4) for v in best_vals]
    print(f"  fit: {ti:.1f}s  winner loss {min(starts):.4f} "
          f"(per-seed best: {starts})", flush=True)
    return dict(method="fd", u_hat=u_hat,
                trace=[(0, hist[0]), (n_steps, hist[-1])],
                wall_compile_s=tc, wall_infer_s=ti, wall_total_s=tc + ti,
                n_starts=len(seeds), start_losses=starts)


def run_implicit(built, u0):
    print("\n=== Implicit diff ===", flush=True)
    gf = built["gf"]

    # Warm up (first call triggers XLA compile)
    t_compile_start = time.perf_counter()
    _ = gf(u0)
    t_compile = time.perf_counter() - t_compile_start
    print(f"  compile: {t_compile:.1f}s", flush=True)

    t0 = time.perf_counter()
    u_hat, trace = ident.wide_fit(gf, u0, lr=LR, n_steps=N_OUTER_STEPS)
    t_infer = time.perf_counter() - t0
    best_loss = min(v for _, v in trace)
    print(f"  fit: {t_infer:.1f}s  loss {trace[0][1]:.4f} → {best_loss:.4f} (best of {len(trace)} steps)", flush=True)

    return dict(
        method="implicit",
        u_hat=np.asarray(u_hat),
        trace=trace,
        wall_compile_s=t_compile,
        wall_infer_s=t_infer,
        wall_total_s=t_compile + t_infer,
    )


def run_fd(built, u0):
    print("\n=== Finite differences ===", flush=True)
    K = built["K"]
    loss_fn = built["loss"]  # value-only: FD must not build the adjoint Hessian

    fd_gf = outer_opt.fd_grad_fn(loss_fn, FD_EPS, batched=False)

    t_compile_start = time.perf_counter()
    _ = fd_gf(u0)  # warm the single value-only loss compile (then cached)
    t_compile = time.perf_counter() - t_compile_start
    print(f"  compile: {t_compile:.1f}s", flush=True)

    # EAGER Adam outer loop -- NOT `outer_opt.adam`, whose `lax.scan` inlines all
    # K+1 forward solves into one scan body: that compile alone consumed >90 GB
    # host RAM and OOM'd.  Run eagerly instead: `fd_gf` executes its K+1 probes
    # one at a time (each a call into the already-compiled value-only loss), so
    # only one forward solve is ever live.  The AdamW math and step count are
    # identical to `outer_opt.adam`, so the fitted u is unchanged.
    import optax
    opt = optax.adamw(LR, weight_decay=0.0)
    z = u0
    state = opt.init(z)
    best_z, best_val = z, float("inf")
    trace = []
    t0 = time.perf_counter()
    for step in range(N_OUTER_STEPS):
        val, g = fd_gf(z)
        val = float(val)
        if val < best_val:
            best_val, best_z = val, z
        updates, state = opt.update(g, state, z)
        z = optax.apply_updates(z, updates)
        trace.append((int((step + 1) * (K + 1)), val))
    u_hat = best_z
    t_infer = time.perf_counter() - t0
    best_loss = min(v for _, v in trace)
    print(f"  fit: {t_infer:.1f}s  loss {trace[0][1]:.4f} → {best_loss:.4f} (best of {len(trace)} steps)", flush=True)

    return dict(
        method="fd",
        u_hat=np.asarray(u_hat),
        trace=trace,
        wall_compile_s=t_compile,
        wall_infer_s=t_infer,
        wall_total_s=t_compile + t_infer,
    )


def run_cmaes(built, u0):
    print("\n=== CMA-ES ===", flush=True)
    loss_fn = built["loss"]  # value-only: no gradient/Hessian needed

    t_compile_start = time.perf_counter()
    _ = loss_fn(u0)
    t_compile = time.perf_counter() - t_compile_start
    print(f"  compile: {t_compile:.1f}s", flush=True)

    t0 = time.perf_counter()
    # Pass the loss straight through: cma_es evaluates it inside `jax.lax.map`,
    # which TRACES it, so a `float(...)` wrapper raises ConcretizationTypeError.
    # `built["loss"]` already returns a jax scalar.
    u_hat, trace = outer_opt.cma_es(
        loss_fn,
        np.asarray(u0),
        sigma0=0.5,
        budget_solves=CMA_BUDGET_SOLVES,
        seed=0,
        batched_eval=False,
    )
    t_infer = time.perf_counter() - t0
    u_hat = jnp.asarray(u_hat, dtype=jnp.float32)
    best_loss = min(v for _, v in trace)
    print(f"  fit: {t_infer:.1f}s  loss {trace[0][1]:.4f} → {best_loss:.4f} (best of {len(trace)} steps)", flush=True)

    return dict(
        method="cmaes",
        u_hat=np.asarray(u_hat),
        trace=trace,
        wall_compile_s=t_compile,
        wall_infer_s=t_infer,
        wall_total_s=t_compile + t_infer,
    )


def run_cmaes_parallel(built, seeds):
    """CMA-ES multistart as ONE batched program via `outer.cma_es_multi`: C
    independent CMA-ES runs (one per seed) whose whole populations are scored in
    a single vmapped forward per generation, with the strategy recursion vmapped
    over runs.  All runs share the fit episodes, so the population loss is just
    `vmap(loss)` over the flat candidate stack.  Winner by best training loss."""
    print("\n=== CMA-ES (parallel multistart) ===", flush=True)
    loss = built["loss"]
    Z0 = jnp.stack([jnp.asarray(s, jnp.float32) for s in seeds])   # (C, K)
    loss_rows = jax.jit(jax.vmap(loss))

    t0 = time.perf_counter()
    best_Z, _ = outer_opt.cma_es_multi(loss_rows, Z0, sigma0=0.5,
                                       budget_solves=CMA_BUDGET_SOLVES, seed=0)
    best_Z = jnp.asarray(best_Z).block_until_ready()
    t_infer = time.perf_counter() - t0

    vals = np.asarray(jax.vmap(loss)(best_Z))
    i = int(vals.argmin())
    starts = [round(float(v), 4) for v in vals]
    print(f"  fit: {t_infer:.1f}s  winner loss {min(starts):.4f} "
          f"({len(seeds)} runs batched; per-run: {starts})", flush=True)
    return dict(method="cmaes", u_hat=np.asarray(best_Z[i]),
                trace=[(0, float(max(vals))), (CMA_BUDGET_SOLVES, float(min(vals)))],
                wall_compile_s=0.0, wall_infer_s=t_infer, wall_total_s=t_infer,
                n_starts=len(seeds), start_losses=starts)


def run_unrolled(built, seeds, compile_timeout=1800, freeze_ik=False, ee_weight=0.0,
                 pin_ik=None, upright_floor=0.0, free_space_only=False,
                 per_segment=False, chunk=None):
    """Unrolled autodiff through the last `unroll_tail` solver iterations.

    Requires rebuilding the inner solvers with a differentiable forward solver.
    If XLA compilation exceeds `compile_timeout` seconds, returns inf metrics.
    `seeds` is a list of outer inits; >1 runs a vmapped batched multistart (each
    seed's `value_and_grad` vmapped, one batched Adam) -- memory permitting, the
    unrolled Jacobians are the heaviest so this is where an OOM would show.
    """
    seeds = [jnp.asarray(s, jnp.float32) for s in seeds]
    u0 = seeds[0]
    multi = len(seeds) > 1
    print(f"\n=== Unrolled autodiff{' (parallel multistart)' if multi else ''} ===",
          flush=True)
    print(f"  compile timeout: {compile_timeout}s", flush=True)

    try:
        from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt
    except ImportError:
        print("  SKIP: pyroffi optimization_engines not available", flush=True)
        return _failed_result("unrolled", "import_error")

    # Build a differentiable forward solver with unroll_tail > 0
    unroll_tail = 10
    opt_cfg = DynamicsTrajOptConfig(
        n_iters=60, early_stop=False, unroll_tail=unroll_tail,
        soft_line_search=True, soft_curvature_gate=True,
    )
    unrolled_fwd = lambda x0, cost_fn: dynamics_trajopt(x0, cost_fn, opt_cfg)

    # Rebuild inner solvers with the unrolled forward solver
    prob = built["prob"]
    scenes = built["scenes"]
    fit_idx = built["fit_idx"]
    fit_scenes = jax.tree.map(lambda a: a[fit_idx], scenes)
    K = built["K"]
    S = z_scale(K, pp.K_IK)
    standoffs = built["standoff_prior"]
    P = z_prior(K, pp.K_IK, standoffs)
    z_of = lambda u: P + S * u

    # Build inner solvers: stock forward solver for x*, unrolled for differentiation.
    # _build_inner handles per-phase Scene construction and calibration correctly.
    stock_fwd = pp.make_composed_forward_solver(n_iters=60)
    theta_ik_init = z_of(jnp.zeros(K))[:pp.K_IK]
    x0_cal, phase_scenes_cal, _, _ = prob.seeds(fit_scenes, theta_ik_init)

    from ioc.inner import make_inner_solver
    inner_unrolled = {}
    for phase in pp.PHASES:
        residual_fn, _ = prob.make_segment_inner(phase, stock_fwd)
        cal_scales = prob.calibrate_segment(phase, residual_fn,
                                            phase_scenes_cal[phase],
                                            jax.random.PRNGKey(0))
        inner_unrolled[phase] = make_inner_solver(
            residual_fn, cal_scales,
            forward_solver=stock_fwd,
            unrolled_forward_solver=unrolled_fwd,
        )

    demo = built["demo_paths"]

    theta_ik_frozen = P[:pp.K_IK]
    # Mirror build_teleop's principled constraints on the unrolled path (see
    # iosp.fit.teleop).  Kept in lock-step so an unrolled run scores the same
    # constrained forward map the other methods do.
    upright_floor = 0.0  # retired with the `upright` weight; see iosp.fit.teleop
    EVENT_ROWS = sorted(set(pp.SKELETON_PICK) | set(pp.SKELETON_PLACE))
    FREE_ROWS = jnp.asarray([r for r in range(pp.N_FULL) if r not in EVENT_ROWS])
    theta_ik_bucket = jnp.asarray([standoffs[0], standoffs[1], 0.0, 0.0],
                                  dtype=jnp.float32)
    if pin_ik == "bucket":
        theta_ik_pinned = theta_ik_bucket
    elif pin_ik == "measured" or freeze_ik:
        theta_ik_pinned = theta_ik_frozen
    else:
        theta_ik_pinned = None

    _nf = pp.N_FEAT_PER_SEG

    def _weights(z_traj):
        if per_segment:
            return jnp.concatenate([jax.nn.softmax(z_traj[i * _nf:(i + 1) * _nf])
                                    for i in range(len(pp.PHASES))])
        return jax.nn.softmax(z_traj)

    _do_split = _split_trajopt_perseg if per_segment else _split_trajopt

    def _rollout_unrolled(u):
        z = z_of(u)
        theta_ik = theta_ik_pinned if theta_ik_pinned is not None else z[:pp.K_IK]
        z_traj = z[pp.K_IK:]
        x0, phase_sc, _, _ = prob.seeds(scenes, theta_ik)
        by_phase = _do_split(_weights(z_traj))
        xs = {}
        for phase in pp.PHASES:
            xs[phase] = jax.vmap(inner_unrolled[phase].solve_unrolled,
                                  in_axes=(0, None, 0))(
                x0[phase], by_phase[phase], phase_sc[phase])
        return xs, phase_sc

    ee_demo = built["ee_demo_paths"]

    def loss_unrolled(u):
        xs, phase_sc = _rollout_unrolled(u)
        paths = prob.full_joint_paths(scenes, xs, phase_sc)
        rows = FREE_ROWS if free_space_only else slice(None)
        loss = jnp.mean(jnp.sum(
            (paths[fit_idx][:, rows] - demo[fit_idx][:, rows]) ** 2, axis=-1))
        if ee_weight:
            ee = prob.ee_positions(paths)
            loss = loss + ee_weight * jnp.mean(jnp.sum(
                (ee[fit_idx][:, rows] - ee_demo[fit_idx][:, rows]) ** 2, axis=-1))
        return loss

    gf_unrolled = jax.jit(jax.value_and_grad(loss_unrolled))
    vg = jax.jit(jax.vmap(gf_unrolled)) if multi else None
    U0 = jnp.stack(seeds) if multi else None

    # Try to compile with a timeout
    class CompileTimeout(Exception):
        pass

    def _alarm_handler(signum, frame):
        raise CompileTimeout()

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(compile_timeout)
    try:
        t_compile_start = time.perf_counter()
        _ = (_chunked_eval(vg, U0, chunk) if multi else gf_unrolled(u0))
        t_compile = time.perf_counter() - t_compile_start
        signal.alarm(0)
    except CompileTimeout:
        signal.signal(signal.SIGALRM, old_handler)
        print(f"  TIMEOUT: XLA compile exceeded {compile_timeout}s", flush=True)
        return _failed_result("unrolled", "compile_timeout")
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc()
        return _failed_result("unrolled", f"error: {e}")
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    print(f"  compile: {t_compile:.1f}s"
          f"{f'  ({len(seeds)} seeds batched)' if multi else ''}", flush=True)

    if multi:
        u_hat, best_vals, hist, _, t_infer = _batched_adam(vg, seeds, LR, N_OUTER_STEPS, chunk)
        starts = [round(float(v), 4) for v in best_vals]
        trace = [(0, hist[0]), (N_OUTER_STEPS, hist[-1])]
        print(f"  fit: {t_infer:.1f}s  winner loss {min(starts):.4f} "
              f"(per-seed best: {starts})", flush=True)
        extra = dict(n_starts=len(seeds), start_losses=starts)
    else:
        t0 = time.perf_counter()
        u_hat, trace = ident.wide_fit(gf_unrolled, u0, lr=LR, n_steps=N_OUTER_STEPS)
        t_infer = time.perf_counter() - t0
        print(f"  fit: {t_infer:.1f}s  loss {trace[0][1]:.4f} → {trace[-1][1]:.4f}",
              flush=True)
        extra = {}

    return dict(
        method="unrolled",
        u_hat=np.asarray(u_hat),
        trace=trace,
        wall_compile_s=t_compile,
        wall_infer_s=t_infer,
        wall_total_s=t_compile + t_infer,
        **extra,
    )


def rollout_success(built, u_hat, label=""):
    """Physics rollout success for a fitted construction, per episode.

    EE RMSE says how close the reconstructed path is to the demonstration; this
    says whether that path, executed under contact physics in each episode's own
    scene, actually PICKS the cube and DROPS it in the bucket.  Uses the same
    joint paths the experiment fits (`paths_fn`, the two-stage teleop map),
    driven through MuJoCo by `iosp.viz.e10_spasm_sim`.
    """
    from iosp.viz.e10_spasm_sim import physics_success
    J = np.asarray(built["paths_fn"](jnp.asarray(u_hat)))   # (B, T, dof)
    fit_idx, gen_idx = list(built["fit_idx"]), list(built["gen_idx"])
    # Each row is scored in ITS OWN episode's scene.  Passing the path rather
    # than the row index matters as soon as the batch is not one directory in
    # order -- see `physics_success`.
    ep_paths = built.get("episode_paths")
    per = []
    t0 = time.perf_counter()
    for i in range(J.shape[0]):
        r = physics_success(i, J[i], verbose=False,
                            episode_path=(ep_paths[i] if ep_paths else None))
        per.append(r)
        tag = "fit " if i in fit_idx else "held"
        print(f"    [{label}] ep{i} {tag} {'OK  ' if r['success'] else 'FAIL'} "
              f"dxy {r['dxy']*1000:5.0f}mm  dz {r['dz']*1000:+5.0f}mm", flush=True)
    fit_succ = sum(per[i]["success"] for i in fit_idx)
    gen_succ = sum(per[i]["success"] for i in gen_idx)
    print(f"    [{label}] success  fit {fit_succ}/{len(fit_idx)}  "
          f"held {gen_succ}/{len(gen_idx)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    return dict(
        per_episode=[bool(p["success"]) for p in per],
        fit_success=int(fit_succ), fit_total=len(fit_idx),
        gen_success=int(gen_succ), gen_total=len(gen_idx),
        details=[{k: v for k, v in p.items() if k != "cube"} for p in per],
    )


def _json_default(o):
    """`json.dumps` fallback.  Non-finite floats become `null`: a literal
    `Infinity` is what `json` emits by default and it is not valid JSON, so a
    diverged fit would otherwise produce a results file nothing can read back."""
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return None


def _failed_result(method, reason):
    return dict(
        method=method,
        u_hat=None,
        trace=[],
        wall_compile_s=float("inf"),
        wall_infer_s=float("inf"),
        wall_total_s=float("inf"),
        failure=reason,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-timeout", type=int, default=1800,
                        help="max seconds for unrolled XLA compile (default 1800)")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--methods", type=str, default="implicit,fd,cmaes,unrolled",
                        help="comma-separated list of methods to run")
    parser.add_argument("--freeze-ik", action="store_true",
                        help="hold theta_ik at the measured standoff prior and fit "
                             "only theta_trajopt (keeps the events physically right)")
    parser.add_argument("--ee-weight", type=float, default=0.0,
                        help="add ee_weight * EE-RMSE^2 to the joint loss (0 = joint only)")
    parser.add_argument("--pin-ik", type=str, default=None,
                        choices=(None, "bucket", "measured", "feasible"),
                        help="pin the release/grasp events as constraints: 'bucket' "
                             "= release above bucket centre (task goal); 'measured' "
                             "= demo's own offsets (fits best but lands on the rim, "
                             "0/10 -- see iosp.fit.teleop.release_offset_cap); "
                             "'feasible' = the demo's offsets projected into the "
                             "bucket, which keeps both; None = fit theta_ik")
    parser.add_argument("--upright-floor", type=float, default=0.0,
                        help="floor on transport.upright weight (grasp-retention "
                             "constraint); 0 = off")
    parser.add_argument("--free-space-only", action="store_true",
                        help="fit the loss on free-space rows only (exclude event "
                             "rows 7/10/19 that the constraints fix)")
    parser.add_argument("--n-starts", type=int, default=3,
                        help="outer multistart: fit each method from N seeds "
                             "(u=0 plus N-1 random) and keep the best by TRAINING "
                             "loss (never rollout success). 1 = single start.")
    parser.add_argument("--hard-upright", type=str, default="",
                        help="comma-separated phases to put a HARD (AL) grasp-"
                             "maintenance constraint on, e.g. 'transport' or "
                             "'grasp,transport' (empty = soft only)")
    parser.add_argument("--per-segment", action="store_true",
                        help="learn cost weights per segment (4x features) "
                             "instead of shared across all phases")
    parser.add_argument("--n-restarts", type=int, default=3,
                        help="inner solver restarts per segment (>1 stabilises "
                             "multimodal segments like transport)")
    args = parser.parse_args()
    hard_upright = tuple(p.strip() for p in args.hard_upright.split(",") if p.strip())

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Building teleop forward map... (freeze_ik={args.freeze_ik}, "
          f"ee_weight={args.ee_weight}, pin_ik={args.pin_ik}, "
          f"upright_floor={args.upright_floor}, free_space_only={args.free_space_only}, "
          f"per_segment={args.per_segment}, n_restarts={args.n_restarts})",
          flush=True)
    built = build_teleop(n_iters=600, space="joint", freeze_ik=args.freeze_ik,
                         ee_weight=args.ee_weight, pin_ik=args.pin_ik,
                         upright_floor=args.upright_floor,
                         free_space_only=args.free_space_only,
                         hard_upright=hard_upright,
                         per_segment=args.per_segment,
                         n_restarts=args.n_restarts)
    K = built["K"]
    u0 = jnp.zeros(K, dtype=jnp.float32)

    print(f"\n{len(built['episodes'])} episodes: {built['n_fit']} fit / "
          f"{len(built['gen_idx'])} held out")

    # Baseline: random init (normal draw) to show what un-optimised looks like
    rng_init = np.random.default_rng(42)
    u_rand = jnp.asarray(rng_init.normal(0, 1.0, K), jnp.float32)
    init_metrics = _metrics(built, u_rand)
    print(f"\nBaseline (random): loss={init_metrics['loss']:.4f}  "
          f"ee_fit={init_metrics['ee_rmse_fit']:.4f}m  "
          f"ee_gen={init_metrics['ee_rmse_gen']:.4f}m")

    # Outer multistart seeds: u=0 (the measured prior) plus N-1 random starts.
    # Each method is fit from every seed and the winner is chosen by TRAINING
    # loss only -- rollout success is never consulted for selection.
    rng = np.random.default_rng(0)
    seeds = [u0] + [jnp.asarray(rng.normal(0, 0.5, K), jnp.float32)
                    for _ in range(max(args.n_starts, 1) - 1)]
    if len(seeds) > 1:
        print(f"\nOuter multistart: {len(seeds)} starts per method "
              f"(winner by training loss)", flush=True)

    # Run methods.  Every method multistarts via VMAP -- all S seeds in ONE
    # batched program: the gradient methods (implicit/fd/unrolled) vmap their
    # value-and-grad over the (S, K) seed array and run one batched Adam;
    # CMA-ES uses `outer.cma_es_multi`, which vmaps C independent CMA runs and
    # scores every run's whole population in one forward per generation.  Winner
    # is best TRAINING loss across seeds; rollout success is never consulted.
    unroll_kw = dict(compile_timeout=args.compile_timeout, freeze_ik=args.freeze_ik,
                     ee_weight=args.ee_weight, pin_ik=args.pin_ik,
                     upright_floor=args.upright_floor, free_space_only=args.free_space_only,
                     per_segment=args.per_segment)
    multi = len(seeds) > 1

    def dispatch(name):
        if name == "implicit":
            return run_implicit_parallel(built, seeds) if multi else run_implicit(built, u0)
        if name == "fd":
            return run_fd_parallel(built, seeds) if multi else run_fd(built, u0)
        if name == "cmaes":
            return run_cmaes_parallel(built, seeds) if multi else run_cmaes(built, u0)
        if name == "unrolled":
            return run_unrolled(built, seeds if multi else [u0], **unroll_kw)
        return None

    results = {}
    run_methods = [m.strip() for m in args.methods.split(",")]
    for name in run_methods:
        r = dispatch(name)
        if r is None:
            print(f"Unknown method {name}, skipping", flush=True)
            continue
        _check_trace(r)
        if r["u_hat"] is not None:
            r["metrics"] = _metrics(built, jnp.asarray(r["u_hat"]))
            r["theta"] = _theta_dict(built, jnp.asarray(r["u_hat"]))
        else:
            r["metrics"] = {k: float("inf") for k in init_metrics}
            r["theta"] = {}
        results[name] = r

    # Gram eigendecomposition at each fitted point
    print("\n=== Gram eigendecomposition ===", flush=True)
    for name, r in results.items():
        if r["u_hat"] is not None:
            print(f"  {name}...", flush=True)
            r["gram"] = _gram(built, jnp.asarray(r["u_hat"]))
            print(f"    rank={r['gram']['rank']}  "
                  f"top3 eigvals={r['gram']['eigvals'][:3]}", flush=True)
        else:
            r["gram"] = None

    # Reconstruction ROLLOUT SUCCESS: execute each fitted construction in
    # contact physics and count cube-in-bucket, per episode (fit + held out).
    print("\n=== Rollout success (physics) ===", flush=True)
    init_roll = rollout_success(built, u_rand, label="init")
    for name, r in results.items():
        if r["u_hat"] is not None:
            r["rollout"] = rollout_success(built, jnp.asarray(r["u_hat"]), label=name)
        else:
            r["rollout"] = None

    # Save paths for visualization
    print("\n=== Saving paths ===", flush=True)
    path_data = {"demo": np.asarray(built["ee_demo_paths"])}
    for name, r in results.items():
        if r["u_hat"] is not None:
            path_data[name] = np.asarray(built["ee_paths_fn"](jnp.asarray(r["u_hat"])))
    np.savez_compressed(out / "paths.npz", **path_data)

    # Save joint paths too
    joint_path_data = {"demo": np.asarray(built["demo_paths"])}
    for name, r in results.items():
        if r["u_hat"] is not None:
            joint_path_data[name] = np.asarray(built["paths_fn"](jnp.asarray(r["u_hat"])))
    np.savez_compressed(out / "joint_paths.npz", **joint_path_data)

    # Summary table
    print("\n" + "=" * 80)
    def _succ(roll):
        if roll is None:
            return "--"
        return f"{roll['fit_success']}/{roll['fit_total']},{roll['gen_success']}/{roll['gen_total']}"

    print(f"{'Method':<12} {'EE fit':>8} {'EE gen':>8} "
          f"{'J fit':>8} {'J gen':>8} {'Succ f,h':>10} "
          f"{'Compile':>10} {'Infer':>10} {'Total':>10}")
    print("-" * 90)
    print(f"{'init':<12} "
          f"{init_metrics['ee_rmse_fit']:8.4f} {init_metrics['ee_rmse_gen']:8.4f} "
          f"{init_metrics['joint_rmse_fit']:8.4f} {init_metrics['joint_rmse_gen']:8.4f} "
          f"{_succ(init_roll):>10} {'--':>10} {'--':>10} {'--':>10}")
    for name in run_methods:
        r = results[name]
        m = r["metrics"]
        def _fmt_time(t):
            return f"{t:.1f}s" if np.isfinite(t) else "inf"
        print(f"{name:<12} "
              f"{m['ee_rmse_fit']:8.4f} {m['ee_rmse_gen']:8.4f} "
              f"{m['joint_rmse_fit']:8.4f} {m['joint_rmse_gen']:8.4f} "
              f"{_succ(r.get('rollout')):>10} "
              f"{_fmt_time(r['wall_compile_s']):>10} "
              f"{_fmt_time(r['wall_infer_s']):>10} "
              f"{_fmt_time(r['wall_total_s']):>10}")
    print("=" * 90)
    print("Succ f,h = cube-in-bucket count, fit,held-out episodes")

    # Save JSON summary
    summary = {
        "init_metrics": init_metrics,
        "init_rollout": init_roll,
        "episodes": built["episodes"],
        "n_fit": int(built["n_fit"]),
        "K": K,
        "n_outer_steps": N_OUTER_STEPS,
        "lr": LR,
        "fd_eps": FD_EPS,
        "cma_budget_solves": CMA_BUDGET_SOLVES,
        "names": built["names"],
        "methods": {},
    }
    for name, r in results.items():
        entry = {
            "metrics": r["metrics"],
            "theta": r.get("theta", {}),
            "wall_compile_s": r["wall_compile_s"],
            "wall_infer_s": r["wall_infer_s"],
            "wall_total_s": r["wall_total_s"],
            "trace": r["trace"],
            "gram": r.get("gram"),
            "rollout": r.get("rollout"),
        }
        if "failure" in r:
            entry["failure"] = r["failure"]
        if r.get("diverged"):
            entry["diverged"] = True
        summary["methods"][name] = entry
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    # Save u_hat arrays
    u_hats = {}
    for name, r in results.items():
        if r["u_hat"] is not None:
            u_hats[name] = r["u_hat"]
    np.savez(out / "u_hats.npz", **u_hats)

    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
