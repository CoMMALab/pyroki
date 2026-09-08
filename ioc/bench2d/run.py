"""Fit every method on a 2D benchmark under a matched forward-solve budget.

The point of doing this in 2D is that a solve costs milliseconds, so the
comparison can be run the way it should be: every method gets the *same* number
of forward trajectory-optimization solves and is scored on what it did with
them.  That is the only fair currency, because the per-step costs differ by
construction (1 solve for a gradient step, K+1 for finite differences, a
population for CMA-ES), and it is what turns "the adjoint is cheaper" into a
measurable statement.

Each method returns a `(solves, best_loss)` trace, from which the headline
number -- solves to reach a target outer loss -- is read off directly.

    python -m ioc.bench2d.run --benchmark field --k-bumps 8
"""

import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from ioc import analytic, outer as outer_opt
from ioc.inner import make_inner_solver
from ioc.bench2d import bases2d, problems as pb, robot2d
from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt


def make_dynamics_forward_solver(opt_cfg=DynamicsTrajOptConfig(n_iters=400)):
    """Wrap pyroffi's dynamics-aware L-BFGS engine as an `inner.forward_solver`."""

    def forward_solver(x0, cost_fn):
        return dynamics_trajopt(x0, cost_fn, opt_cfg)

    return forward_solver


def build_solver(res_fn, scales, T, d, cfg, n_iter, damping, unroll_tail, ridge,
                 n_restarts=1, topo_restarts=False, dynamics=False):
    """Adapt a benchmark's `res_fn(x, ctx, T, cfg)` to the shared inner solver.

    `methods` selects which methods to run, as a comma-separated list
    ("all" runs every one).  A subset run MERGES into `out` if it already
    exists, so a single series can be re-run or added without redoing the
    sweep:

        python -m ioc.bench2d.run --benchmark segments --methods unrolled \
            --out bench2d/bench2d_seg_K12.json ...

    Every method sees the same problem either way -- theta*, contexts,
    demonstration noise and z0 are drawn host-side from `seed` alone.

    `topo_restarts` swaps the default i.i.d.-jitter multistart for the
    structured lateral-detour seeding in `pb.make_topo_seed_fn` -- only
    meaningful for `n_restarts > 1` on a 2D (d=2) benchmark.

    `dynamics=True` drives the benchmark through a synthetic GRiD-backed robot
    (`ioc.bench2d.robot2d`) instead of pure point-mass/unicycle kinematics:
    `res_fn`'s residuals are wrapped with `bases2d.dynamic` to append an RNEA
    torque feature.  The forward solve always runs on pyroffi's dynamics_trajopt
    L-BFGS engine, regardless of `dynamics`.  The implicit adjoint's exact
    Hessian (`ioc.inner`'s `adjoint_hessian="jax"`, the only mode) works
    straight through the GRiD torque feature -- `GRiDDynamics`'s analytic
    kernels carry their own `custom_jvp` from GRiD's `idsva_so` second-order
    kernel (see `pyroffi.dynamics._grid_dynamics`) -- so `dynamics=True` needs
    no separate curvature path any more. `unrolled` uses the same engine in
    its fixed-iteration, reverse-mode-differentiable form (`early_stop=False`)
    instead of `implicit`'s early-stopping form.
    """
    del damping  # no longer meaningful: the internal GN loop it damped is gone
    if dynamics:
        problem = robot2d.Robot2DProblem.load(d)
        residual_fn = bases2d.dynamic(problem, T, cfg, res_fn, torque_backend="grid")
    else:
        residual_fn = functools.partial(_residuals, res_fn, T, cfg)
    forward_solver = make_dynamics_forward_solver(DynamicsTrajOptConfig(
        n_iters=n_iter, grad_tol=1e-9))
    unrolled_forward_solver = make_dynamics_forward_solver(DynamicsTrajOptConfig(
        n_iters=n_iter, early_stop=False, unroll_tail=unroll_tail))
    return make_inner_solver(
        residual_fn,
        scales,
        adjoint_ridge=ridge,
        n_restarts=n_restarts,
        restart_seed_fn=pb.make_topo_seed_fn(T, d) if topo_restarts else None,
        forward_solver=forward_solver,
        unrolled_forward_solver=unrolled_forward_solver,
    )


def _calibrate_dynamics(residual_fn, ctxs, T, d, cfg, key, K, n_probe=12, jitter=0.25):
    """Like `pb.calibrate`, but for a `residual_fn(x, ctx)` already wrapped by
    `bases2d.dynamic` (K includes the appended torque feature)."""
    keys = jax.random.split(key, n_probe)

    def raw(ctx, k):
        x0 = pb.seed_path(ctx, T, d, cfg)
        x = x0 + jitter * jax.random.normal(k, x0.shape)
        rs = residual_fn(x, ctx)
        return jnp.stack([jnp.sum(r**2) for r in rs])

    vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)), in_axes=(0, None))(ctxs, keys)
    return jnp.maximum(jnp.mean(jnp.abs(vals.reshape(-1, K)), axis=0), 1e-8)


def _residuals(res_fn, T, cfg, x, ctx):
    return res_fn(x, ctx, T, cfg)


DEMO_STATIONARITY_TOL = 1e-6

# Inner iterations needed to drive the *demonstrations* to that tolerance.  These
# differ by an order of magnitude between benchmarks, which is exactly why a
# single shared default silently produced non-optimal demonstrations.  Only the
# one-off demonstration solve pays this; the bilevel fit uses `--n-iter`.
DEMO_N_ITER = {"racing": 400, "field": 2500, "unicycle": 4000, "segments": 6000}

# Extra headroom for the dynamics_trajopt L-BFGS forward solve (used whenever
# `dynamics=True`): its landscape is a harder RNEA-augmented cost than the
# kinematic default, so it benefits from a higher demo iteration count even
# though `dynamics_trajopt` now correctly returns the point with the smallest
# measured gradient rather than the lowest-cost line-search trial (see
# `pyroffi.optimization_engines._dynamics_trajopt`) -- that bug, not slow
# convergence, was the original cause of demos failing `_screen_demos`.
DEMO_N_ITER_DYNAMICS = {"field": 6000}


def relative_stationarity(features, grad_x, x, theta, ctx, K):
    """||sum_k th_k grad phi_k|| / sum_k th_k ||grad phi_k||, in [0, 1].

    The scale-free version of `inner.stationarity`.  The raw gradient norm is
    not comparable across benchmarks -- it carries the units of the cost, which
    whitening sets per benchmark -- so an absolute threshold is either vacuous
    on one benchmark or unreachable on another.  This ratio asks the question
    that actually matters: how completely do the weighted feature gradients
    cancel?  0 is an exact KKT point; 1 means they do not cancel at all.
    """
    # The K per-feature gradients are a vmapped probe axis rather than an
    # unrolled Python loop; identical values, one call wide in the graph.
    gs = jax.vmap(grad_x, in_axes=(None, 0, None))(x, jnp.eye(K), ctx)
    num = jnp.linalg.norm(theta @ gs)
    return num / (jnp.sum(theta * jnp.linalg.norm(gs, axis=-1)) + 1e-300)


def _screen_demos(rel, benchmark, seed, n_iter):
    """Refuse to build a dataset on demonstrations that are not local optima.

    Every method here is scored on recovering the theta that *generated* the
    demonstrations, and that ground truth only means anything if the
    demonstration really is a minimizer of the corresponding cost.  Inverse KKT
    assumes it exactly (it fits grad_x J = 0 at the demonstration), so an
    unconverged demonstration silently handicaps one baseline and biases the
    weights recovered by all the others.

    This was not hypothetical.  At the shipped iteration counts the
    demonstrations sat at relative stationarity 1.1e-3 (`racing`), 2.0e-3
    (`field`) and 2.5e-2 (`unicycle`) -- the unicycle demonstrations were 2.5%
    away from being optimal for the cost they were labelled with.  The
    diagnostic was already being computed and stored as metadata, and simply
    never looked at; hence a hard check rather than a printed warning.
    """
    worst = float(jnp.max(rel))
    if not np.isfinite(worst) or worst > DEMO_STATIONARITY_TOL:
        n_bad = int(jnp.sum(rel > DEMO_STATIONARITY_TOL))
        raise RuntimeError(
            f"[{benchmark} seed {seed}] {n_bad}/{rel.shape[0]} demonstrations are "
            f"not local optima: max relative stationarity = {worst:.2e} > "
            f"{DEMO_STATIONARITY_TOL:.0e} after {n_iter} inner iterations. "
            "The recovered costs would not be valid; raise the demonstration "
            "iteration count (ioc.bench2d.run.DEMO_N_ITER)."
        )


def _seed_dataset(res_fn, theta_stars, ctxs, keys, T, d, cfg, K, demo_iter,
                  damping, unroll_tail, ridge, n_restarts, dynamics):
    """Every seed's calibration, demonstrations and screening statistic, batched.

    Seeds are independent by construction -- different theta*, different
    contexts, different calibration key -- so the whole data-generation stage
    is one `vmap` over the seed axis on top of the existing `vmap` over
    contexts, instead of a Python loop that dispatches it once per seed.  Only
    the *fitting* stage stays serial (see `run`).

    Batching seeds into the same solve is safe despite `dynamics_trajopt`'s
    early-stopping `while_loop`: under `vmap` the loop runs until *every*
    element has converged, and the solver returns `best_x`, the iterate with
    the smallest gradient norm seen.  Extra iterations therefore cannot make
    any seed's demonstration worse -- they can only lower its stationarity,
    which is the direction `_screen_demos` wants.

    The seed axis multiplies device memory for the demonstration solve.  It is
    the one place in this function where that is true; if it becomes the
    binding constraint, chunk this call the way
    `ioc.robot.problem.screen_scenes` chunks its pool.

    Returns (scales, x0s, x_star, demos, ref, rel), each with a leading seed axis.
    """

    def one_seed(theta_star, ctx, key):
        if dynamics:
            residual_fn = bases2d.dynamic(
                robot2d.Robot2DProblem.load(d), T, cfg, res_fn)
            scales = _calibrate_dynamics(residual_fn, ctx, T, d, cfg, key, K)
        else:
            scales = pb.calibrate(res_fn, ctx, T, d, cfg, key, K)
        # Demonstrations come from a *converged* solve, not from the same
        # truncated solver the learner uses.  This is both more correct (the
        # ground-truth theta labels a genuine local optimum) and more faithful
        # to the setting: the demonstrator is an expert, the learner's forward
        # model is an approximation of it.  Solved once per seed, so the high
        # iteration count costs nothing against the fitting budget.
        demo_solver = build_solver(res_fn, scales, T, d, cfg, demo_iter, damping,
                                   unroll_tail, ridge, n_restarts,
                                   dynamics=dynamics)
        x0s = jax.vmap(lambda c: pb.seed_path(c, T, d, cfg))(ctx)
        x_star = jax.vmap(
            lambda x, c: demo_solver.solve_implicit(x, theta_star, c))(x0s, ctx)
        rel = jax.vmap(lambda x, c: relative_stationarity(
            demo_solver.features, demo_solver.grad_x, x, theta_star, c, K)
        )(x_star, ctx)
        demos = jax.vmap(lambda x, c: pb.unpack(x, c, T, d))(x_star, ctx)
        # `cost` depends only on the whitening scales, not on the iteration
        # count, so the demo solver's cost is the learner's cost.
        ref = jax.vmap(lambda x, c: demo_solver.cost(x, theta_star, c))(x_star, ctx)
        return scales, x0s, x_star, demos, ref, rel

    return jax.vmap(one_seed)(theta_stars, ctxs, keys)


def run(benchmark, n_contexts, n_seeds, T, n_iter, budget, k_bumps, demo_noise,
        damping, unroll_tail, ridge, fd_eps, lr, cfg, out, n_restarts=1,
        dynamics=False, topo_restarts=False, methods=None):
    res_fn, _, d = pb.BENCHMARKS[benchmark]
    names = pb.benchmark_names(benchmark, k_bumps, cfg)
    if dynamics:
        # Demonstrated end to end on one benchmark; the pattern (wrap res_fn
        # with bases2d.dynamic, drive the forward solve through
        # dynamics_trajopt) generalizes trivially to the others once proven.
        assert benchmark == "field", "--dynamics is demonstrated on 'field' only"
        names = names + ("torque",)
    K = len(names)
    n_bump = k_bumps if benchmark == "field" else 1
    print(f"[{benchmark}] K={K} d={d} T={T} M={n_contexts} budget={budget} solves"
          + (" [dynamics]" if dynamics else ""))

    # Host-side per-seed draws, in the same order the old serial loop drew
    # them, so `seed` still indexes the same stream: theta*, contexts, the
    # demonstration noise, then z0.  None of these depend on device results,
    # which is what lets the device stage below be a single batched call.
    theta_stars, ctx_list, noises, z0s = [], [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        th = np.maximum(rng.dirichlet(np.full(K, 0.7)), 0.01)
        theta_stars.append(th / th.sum())
        ctx_list.append(pb.sample_contexts(rng, n_contexts, benchmark, T, d,
                                           n_bump, cfg))
        noises.append(rng.normal(scale=demo_noise, size=(n_contexts, T, d))
                      if demo_noise > 0 else None)
        z0s.append(jnp.asarray(rng.normal(scale=0.5, size=K)))

    theta_stars = jnp.asarray(np.stack(theta_stars))
    ctxs_all = jax.tree.map(lambda *a: jnp.stack(a), *ctx_list)
    demo_counts = DEMO_N_ITER_DYNAMICS if dynamics else DEMO_N_ITER
    demo_iter = max(n_iter, demo_counts.get(benchmark, n_iter))
    scales_all, x0s_all, _, demos_all_raw, ref_all, rel_all = _seed_dataset(
        res_fn, theta_stars, ctxs_all, jax.vmap(jax.random.key)(jnp.arange(n_seeds)),
        T, d, cfg, K, demo_iter, damping, unroll_tail, ridge, n_restarts, dynamics,
    )

    # Demonstration noise is applied per seed on the host, exactly as before.
    demos_all = demos_all_raw
    if demo_noise > 0:
        nz = jnp.asarray(np.stack(noises))
        demos_all = demos_all + nz.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

    for seed in range(n_seeds):
        _screen_demos(rel_all[seed], benchmark, seed, demo_iter)
    conv = [float(jnp.median(rel_all[s])) for s in range(n_seeds)]

    # ---- the flattened fit -------------------------------------------------
    # ONE inner solver, built with UNIT scales, serves every seed.  The
    # whitening does not disappear: `theta_k / s_k` is exactly what the
    # whitened solver's cost already computes
    #
    #     C(x) = sum_k (theta_k / s_k) * ||r_k(x)||^2,
    #
    # so passing `softmax(z) / scales` into a unit-scale solver is the SAME
    # function of x -- same cost, same Hessian, same adjoint ridge, same
    # solution -- with the per-seed part moved from the solver's closure into a
    # per-row argument.  That is what makes one solver serve all seeds, and it
    # is the precondition for folding the seed axis into the context batch.
    inner_unit = build_solver(res_fn, jnp.ones(K, dtype=scales_all.dtype), T, d,
                              cfg, n_iter, damping, unroll_tail, ridge,
                              n_restarts, topo_restarts=topo_restarts,
                              dynamics=dynamics)

    def rows_loss(solver):
        """`(Z, sidx) -> (R,)`: row r's outer loss, under fit `sidx[r]`'s data.

        The flattening.  Rows are whatever the caller folded in -- one per seed
        for a gradient step, K+1 per seed for finite differences, lambda per
        seed for a CMA-ES generation -- and each row is paired with its own
        fit's contexts, demonstrations and seeds via `sidx`.  The (R, M) grid is
        then solved as ONE batch of R*M trajectory optimizations.

        Nothing about the *cost* of a method changes: a solve is still a solve,
        and every method is still charged `n_contexts * n_restarts` per fit per
        step.  What changes is occupancy.  MEASURED on this benchmark (field,
        M=8, K=8), one value-and-grad step:

            1 seed  (8 rows)   22.2 ms
            2 seeds (16 rows)  22.4 ms
            4 seeds (32 rows)  22.9 ms
            8 seeds (64 rows)  22.8 ms

        i.e. the 2D benchmark leaves the GPU almost entirely idle at one seed's
        width, so seeds are close to free.  (`ioc.diagnostics` records the
        opposite regime on the robot problem, where ~10 concurrent rollouts
        already saturate the device and widening buys ~1.5x at best -- the trick
        is worth what the occupancy headroom is worth, no more.)
        """

        def loss(Z, sidx):
            R_ = Z.shape[0]
            th = jax.nn.softmax(Z, axis=-1) / scales_all[sidx]     # (R, K)
            flat = lambda a: a.reshape((R_ * M_ctx,) + a.shape[2:])
            ctx = jax.tree.map(lambda a: flat(a[sidx]), ctxs_all)
            x0 = flat(x0s_all[sidx])
            dm = flat(demos_all[sidx])
            thr = jnp.repeat(th, M_ctx, axis=0)                    # (R*M, K)

            def one(x0_, t, c, m):
                p = pb.unpack(solver(x0_, t, c), c, T, d)[:, :2]
                return jnp.mean(jnp.sum((p - m[:, :2]) ** 2, axis=-1))

            return jax.vmap(one)(x0, thr, ctx, dm).reshape(R_, M_ctx).mean(axis=1)

        return loss

    M_ctx = n_contexts
    sidx_1 = jnp.arange(n_seeds)

    def score_rows(Z):
        """Per-seed weight error and regret under each seed's TRUE cost."""
        th = jax.nn.softmax(Z, axis=-1)                            # (S, K)
        R_ = Z.shape[0]
        flat = lambda a: a.reshape((R_ * M_ctx,) + a.shape[2:])
        ctx = jax.tree.map(lambda a: flat(a[sidx_1]), ctxs_all)
        x0 = flat(x0s_all[sidx_1])
        thr = jnp.repeat(th / scales_all, M_ctx, axis=0)
        star = jnp.repeat(theta_stars / scales_all, M_ctx, axis=0)
        xh = jax.vmap(inner_unit.solve_implicit)(x0, thr, ctx)
        c = jax.vmap(inner_unit.cost)(xh, star, ctx).reshape(R_, M_ctx)
        l1 = jnp.sum(jnp.abs(th - theta_stars), axis=-1)
        return l1, jnp.mean(c - ref_all, axis=-1), th

    score_j = jax.jit(score_rows)

    # Every method is charged the same way: one "solve" is one context, times
    # the number of inner restarts, which restarts are charged for too.  This is
    # a PER-SEED budget and stays per-seed under flattening -- batching changes
    # how the solves are dispatched, never how many each fit is allowed.
    per_solve = n_contexts * n_restarts
    Z0 = jnp.stack(z0s)                                            # (S, K)

    # Methods can be run independently and merged into an existing results
    # file (`methods=("unrolled",)` re-runs just that series and leaves every
    # other method's numbers on disk untouched).  Seeding, contexts and
    # demonstrations are drawn host-side from `seed` alone above, so a method
    # run on its own sees exactly the problem it would have seen in a full
    # sweep -- the merge is sound.
    want = (lambda m: True) if not methods else (lambda m: m in set(methods))
    prev = {}
    if os.path.exists(out):
        try:
            prev = json.load(open(out)).get("results", {})
        except (json.JSONDecodeError, OSError):
            prev = {}
    all_res = {f"s{seed}": dict(prev.get(f"s{seed}", {}))
               for seed in range(n_seeds)}

    def record(name, Z, traces, wall, keep_theta=False):
        l1, rg, th = score_j(Z)
        l1, rg, th = np.asarray(l1), np.asarray(rg), np.asarray(th)
        for seed in range(n_seeds):
            r = dict(l1=float(l1[seed]), regret=float(rg[seed]), wall=wall,
                     trace=traces[seed] if traces else [])
            if keep_theta:
                r["theta"] = [float(v) for v in th[seed]]
            all_res[f"s{seed}"][name] = r

    def timed(fn):
        """Wall-clock for one method over ALL seeds, warm-up excluded.

        `wall` is now the batched cost of running every seed of that method, not
        a single seed's -- the seed axis lives inside one program, so a per-seed
        wall no longer exists as a measurable quantity.  The column's purpose is
        unaffected: it exists to compare METHODS against each other, and every
        method here is measured the same way over the same seeds.  It is not
        comparable to `wall` from runs recorded before the flattening.
        """
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out[0])
        return out, time.perf_counter() - t0

    # `li` also backs `fd` and `cmaes`, so it is built whenever any of the
    # three is selected, not only for `implicit` itself.
    li = rows_loss(inner_unit.solve_implicit)
    if want("implicit"):
        gi = outer_opt.summed_grad_fn(lambda Z: li(Z, sidx_1))
        jax.block_until_ready(gi(Z0)[0])
        (Z, trs), w = timed(lambda: outer_opt.adam_multi(
            gi, Z0, lr=lr, budget_solves=budget, solves_per_step=per_solve,
            trace_best=True))
        record("implicit", Z, trs, w, keep_theta=True)

    if want("unrolled"):
        lu = rows_loss(inner_unit.solve_unrolled)
        gu = outer_opt.summed_grad_fn(lambda Z: lu(Z, sidx_1))
        jax.block_until_ready(gu(Z0)[0])
        (Z, trs), w = timed(lambda: outer_opt.adam_multi(
            gu, Z0, lr=lr, budget_solves=budget, solves_per_step=per_solve,
            trace_best=True))
        record("unrolled", Z, trs, w)

    if want("fd"):
        fd_idx = jnp.asarray(outer_opt.fit_index_for(n_seeds, K + 1))
        gfd = jax.jit(outer_opt.fd_grad_multi_fn(
            lambda P: li(P, fd_idx), fd_eps, n_seeds, K))
        jax.block_until_ready(gfd(Z0)[0])
        (Z, trs), w = timed(lambda: outer_opt.adam_multi(
            gfd, Z0, lr=lr, budget_solves=budget,
            solves_per_step=(K + 1) * per_solve, trace_best=True))
        record("fd", Z, trs, w)

    if want("cmaes"):
        lam = outer_opt.cma_population_size(K)
        cma_idx = jnp.asarray(outer_opt.fit_index_for(n_seeds, lam))
        cma_rows = jax.jit(lambda X: li(X, cma_idx))
        jax.block_until_ready(cma_rows(jnp.repeat(Z0, lam, axis=0)))
        (Z, trs), w = timed(lambda: outer_opt.cma_es_multi(
            cma_rows, Z0, budget_solves=budget, solves_per_eval=per_solve,
            seed=0, trace_best=True))
        record("cmaes", Z, trs, w)

    # The analytic baselines share the same unit-scale solver; each seed's
    # whitening rides in its own probe basis (see `analytic.kkt_fit`), which is
    # what lets a whole seed's fit be a `vmap` axis rather than a Python loop.
    bases = jax.vmap(lambda sc: jnp.diag(1.0 / sc))(scales_all)      # (S, K, K)
    if want("kkt"):
        kkt_all = jax.jit(jax.vmap(
            lambda ct, dm, b: analytic.kkt_fit(inner_unit.grad_x, ct, dm, K,
                                               n_steps=600, basis=b)))
        (Z,), w = timed(lambda: (kkt_all(ctxs_all, demos_all, bases),))
        record("kkt", Z, None, w)

    if want("cioc"):
        cioc_all = jax.jit(jax.vmap(
            lambda ct, dm, b: analytic.cioc_fit(inner_unit.grad_x,
                                                inner_unit.gn_system, ct, dm, K,
                                                n_steps=600, basis=b)))
        (Z,), w = timed(lambda: (cioc_all(ctxs_all, demos_all, bases),))
        record("cioc", Z, None, w, keep_theta=True)

    if want("eiv"):
        eiv_all = jax.jit(jax.vmap(
            lambda ct, dm, b: analytic.eiv_fit(inner_unit.grad_x, ct, dm, K,
                                               n_outer=5, n_inner=600, basis=b)))
        (Z,), w = timed(lambda: (eiv_all(ctxs_all, demos_all, bases),))
        record("eiv", Z, None, w)

    if want("random"):
        record("random", Z0, None, 0.0)

    for seed in range(n_seeds):
        all_res[f"s{seed}"]["_meta"] = dict(
            K=K, demo_relative_stationarity=conv[seed], demo_n_iter=demo_iter,
            theta_star=[float(v) for v in theta_stars[seed]])
        print(f"  seed {seed}: " + "  ".join(
            f"{k}={all_res[f's{seed}'][k]['regret']:.2e}"
            for k in ["implicit", "fd", "cmaes", "kkt", "cioc", "eiv", "random"]
            if k in all_res[f"s{seed}"]))

    with open(out, "w") as f:
        json.dump({"benchmark": benchmark, "K": K, "M": n_contexts,
                   "budget": budget, "demo_noise": demo_noise,
                   "results": all_res}, f, indent=2)
    print(f"wrote {out}")

    print(f"\n{'method':>10s} {'theta L1':>10s} {'regret':>12s} "
          f"{'wall':>8s}   (wall = all %d seeds, batched)" % n_seeds)
    for m in ["implicit", "unrolled", "fd", "cmaes", "kkt", "cioc", "eiv", "random"]:
        if any(m not in all_res[f"s{s}"] for s in range(n_seeds)):
            continue
        l1 = np.median([all_res[f"s{s}"][m]["l1"] for s in range(n_seeds)])
        rg = np.median([all_res[f"s{s}"][m]["regret"] for s in range(n_seeds)])
        # `wall` is identical across seeds by construction: the seed axis is
        # inside one batched program, so this is that method's cost for ALL
        # seeds, not a per-seed median.  See `timed` in `run`.
        w = all_res["s0"][m]["wall"]
        print(f"{m:>10s} {l1:>10.3f} {rg:>12.3e} {w:>7.1f}s")
    return all_res


def main(
    benchmark: str = "field",
    n_contexts: int = 8,
    n_seeds: int = 3,
    n_timesteps: int = 30,
    n_iter: int = 60,
    budget: int = 4000,
    k_bumps: int = 6,
    demo_noise: float = 0.02,
    damping: float = 1e-2,
    unroll_tail: int = 3,
    ridge: float = 1e-9,
    fd_eps: float = 1e-4,
    lr: float = 0.1,
    track_radius: float = 1.5,
    track_halfwidth: float = 0.42,
    n_obstacles: int = -1,  # -1: the benchmark's own default
    clearance: float = 0.20,
    nonhol_weight: float = 1.0,
    bump_width: float = 0.45,
    n_restarts: int = 1,
    topo_restarts: bool = False,
    k_segments: int = 2,
    dynamics: bool = False,
    methods: str = "all",
    out: str = "",
):
    """`dynamics` drives the benchmark through a synthetic GRiD-backed robot
    (`ioc.bench2d.robot2d`) instead of pure kinematics, appending an RNEA
    torque feature and running the forward solve on pyroffi's dynamics_trajopt
    L-BFGS engine.  Demonstrated on `field` only; see `run`'s docstring.

    `dynamics=True` works with x64 on or off -- the GRiD torque feature's
    implicit-adjoint curvature is exact either way (see
    `ioc.bench2d.bases2d.dynamic`, `pyroffi.dynamics._grid_dynamics`):

        python -m ioc.bench2d.run --benchmark field --dynamics

    `topo_restarts` swaps the default i.i.d.-jitter `n_restarts>1` multistart
    for `pb.make_topo_seed_fn`'s structured lateral-detour seeds (2D
    benchmarks only) -- the variant `fig_recovery` validated as actually
    escaping basins on a multimodal field, unlike plain jitter (see
    `ioc.inner.InnerSolver.solve`'s docstring on why jitter mostly doesn't).
    """
    cfg = pb.default_cfg(
        benchmark, track_radius=track_radius, track_halfwidth=track_halfwidth,
        clearance=clearance, nonhol_weight=nonhol_weight,
        bump_width=bump_width, k_segments=k_segments,
    )
    if n_obstacles > 0:
        cfg["n_obstacles"] = n_obstacles
    run(benchmark, n_contexts, n_seeds, n_timesteps, n_iter, budget, k_bumps,
        demo_noise, damping, unroll_tail, ridge, fd_eps, lr, cfg,
        out or f"bench2d_{benchmark}.json", n_restarts, dynamics,
        topo_restarts=topo_restarts,
        methods=None if methods == "all" else tuple(
            m.strip() for m in methods.split(",") if m.strip()))


if __name__ == "__main__":
    tyro.cli(main)
