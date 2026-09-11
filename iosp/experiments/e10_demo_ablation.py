"""E10 demo-count ablation: how much does one more demonstration buy?

Fits the pick-and-place cost on the first `n` of the 27 teleop episodes
recorded on 2026-09-03, for every `n` in a grid, and scores each fit on the
SAME ten held-out episodes from the 2026-09-02 session.  One checkpoint
(`theta`, metrics, trace, wall time) is written per (method, n) as soon as it
is produced, so the run is an ablation curve that can be read while it is still
being collected -- and so a kill at hour 20 still leaves everything fitted up
to that point on disk.

Why `n` TRUNCATES rather than down-weighting
--------------------------------------------
Each `n` rebuilds the forward map with only the first `n` fit episodes
(`build_teleop(n_fit_max=n)`); the ten held-out episodes are always present.
The obvious alternative -- keep all 27 in the batch and zero the loss weight of
the excluded ones -- would let a single XLA compile serve the whole grid, but
it does not save the thing that actually costs: the forward map solves every
episode in the batch whether or not the loss looks at it.  MEASURED at 27 fit +
10 held, one `value_and_grad` is 39 s and one loss evaluation 44 s, so a masked
sweep costs 27x a full fit at EVERY grid point (~300 GPU-hours for four
methods) against ~14x in total when truncated.  The price of truncating is a
rebuild (~2 min) and a recompile per point, which is noise against either.

A welcome side effect: the standoff prior (`measure_standoffs`) and the inner
feature-scale calibration are re-measured on exactly the `n` episodes being
fitted, so the n=1 point is an honest n=1 -- nothing in it has seen the other
26.  Neither quantity ever touches the held-out session.

Usage
-----
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -u -m iosp.experiments.e10_demo_ablation \\
            --methods implicit,fd,cmaes,unrolled --n-grid 1-27
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from iosp import config
config.enable_compilation_cache()

from ioc import identifiability as ident
from iosp.experiments import e10_method_comparison as cmp
from iosp.fit.teleop import (build_teleop, find_episodes, mixed_split,
                             FIT_DEMO_DIR, TEST_DEMO_DIR)

OUT_DIR = (pathlib.Path(__file__).resolve().parents[1]
           / "data" / "results" / "e10_demo_ablation")


def parse_grid(spec, n_max):
    """'1-27', '25,20,15,10,5', 'log' or 'all' -> a list of demo counts.

    ORDER IS PRESERVED for an explicit list, and it is a real choice rather
    than a formatting detail: the grid is walked in the order given and a
    checkpoint is written after each point, so a descending grid spends the
    expensive end of the sweep first and a run stopped early still has the
    large-n fits -- the ones an ablation curve is anchored on -- on disk.
    Duplicates are dropped, keeping the first occurrence.
    """
    if spec in ("all", "1-%d" % n_max):
        return list(range(1, n_max + 1))
    if spec == "log":
        # Dense where the return is expected to be steep (1-5), sparse after.
        g = [1, 2, 3, 4, 5, 6, 8, 11, 14, 18, 22, n_max]
        return sorted({min(v, n_max) for v in g})
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        vals = (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1)
                if "-" in part else [int(part)])
        for v in vals:
            if v not in out:
                out.append(v)
    bad = [v for v in out if not 0 < v <= n_max]
    if bad:
        raise ValueError(f"demo counts out of range 1..{n_max}: {sorted(bad)}")
    return out


def _clean(o):
    """Recursively replace non-finite floats with `None`.

    `json.dumps(default=...)` never sees them: a bare `float('inf')` IS
    serializable to Python's encoder, which emits the literal `Infinity` --
    accepted by `json.loads` and by nothing else (jq, JavaScript, and every
    strict JSON reader reject it).  A diverged fit puts `inf` in `trace` and in
    `wall_*`, so without this the checkpoint stream stops being readable
    exactly when something went wrong and you most need to read it.
    """
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _rmse(P, D, idx):
    return float(jnp.sqrt(jnp.mean(jnp.sum((P[idx] - D[idx]) ** 2, axis=-1))))


def _metrics_n(built, u):
    """Fit RMSE over the `n` episodes in this build, held-out RMSE over the ten
    episodes of the other session -- the latter is the ablation curve."""
    u = jnp.asarray(u)
    J, E = built["paths_fn"](u), built["ee_paths_fn"](u)
    Jd, Ed = built["demo_paths"], built["ee_demo_paths"]
    fit, gen = np.asarray(built["fit_idx"]), np.asarray(built["gen_idx"])
    return dict(
        joint_rmse_fit=_rmse(J, Jd, fit), ee_rmse_fit=_rmse(E, Ed, fit),
        joint_rmse_gen=_rmse(J, Jd, gen), ee_rmse_gen=_rmse(E, Ed, gen),
        loss=float(built["loss"](u)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", default="implicit,fd,cmaes,unrolled",
                    help="comma-separated, run IN THE ORDER GIVEN")
    ap.add_argument("--n-grid", default="all",
                    help="'all', 'log', '1-27' or an explicit list like '1,2,3,5,10'")
    ap.add_argument("--loop", default="method", choices=("method", "n"),
                    help="'method' (default): finish every demo count for the "
                         "first method before starting the second, so a run "
                         "killed early still has one COMPLETE curve. 'n': "
                         "finish every method at n=1, then n=2, ... which "
                         "rebuilds the forward map 4x less often (~2 min each) "
                         "but leaves every curve partial if killed.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--n-starts", type=int, default=1,
                    help="outer multistart, winner by TRAINING loss. NOTE: >1 "
                         "REQUIRES --pin-ik: it takes the vmapped path, and "
                         "with theta_ik fitted the IK `ffi_call` gets batched "
                         "and has no vmap rule (NotImplementedError). Pinned, "
                         "theta_ik is constant in u, so it never is. MEASURED "
                         "both ways on a 2-episode build.")
    ap.add_argument("--n-iters", type=int, default=600)
    ap.add_argument("--n-restarts", type=int, default=3)
    ap.add_argument("--compile-timeout", type=int, default=1800)
    ap.add_argument("--pin-ik", default=None,
                    choices=(None, "bucket", "measured", "feasible"))
    ap.add_argument("--per-segment", action="store_true")
    ap.add_argument("--free-space-only", action="store_true",
                    help="fit on free-space rows only; the event rows are "
                         "determined by --pin-ik's constraints")
    ap.add_argument("--rollout-at", default="last",
                    help="'last' (default), 'none', 'all', or a list of demo "
                         "counts at which to also run the physics rollout")
    ap.add_argument("--gram-at", default="last", help="as --rollout-at")
    ap.add_argument("--seed-chunk", type=int, default=1,
                    help="multistart seeds evaluated at once. 1 (default) is "
                         "what fits: at 25 fit episodes and K=28 the 3-seed "
                         "batch asks for 10.3 GiB and dies with "
                         "RESOURCE_EXHAUSTED on a 24 GiB A5000. Costs almost "
                         "no wall-clock -- one solve already saturates the "
                         "card. 0 or >=n-starts means no chunking.")
    ap.add_argument("--n-train", type=int, default=None,
                    help="episodes taken from the 2026-09-03 session "
                         "(default: all of them)")
    ap.add_argument("--n-orig-fit", type=int, default=0,
                    help="episodes of the ORIGINAL 2026-09-02 session moved "
                         "into the fit set; the rest are held out. 0 (default) "
                         "is the pure cross-session split. See "
                         "`iosp.fit.teleop.mixed_split`.")
    ap.add_argument("--tag", default="",
                    help="suffix for the checkpoint file, so several methods "
                         "can collect into one directory from separate GPUs "
                         "without interleaving writes")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f".{args.tag}" if args.tag else ""
    ckpt_path = out / f"checkpoints{suffix}.jsonl"

    fit_eps, held_eps = mixed_split(args.n_train, args.n_orig_fit)
    n_fit_total, n_held = len(fit_eps), len(held_eps)
    grid = parse_grid(args.n_grid, n_fit_total)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    def _at(spec):
        if spec == "none":
            return set()
        if spec == "last":
            # The LARGEST count, not the last one walked: with a descending
            # grid `grid[-1]` is the smallest fit, which is the least
            # interesting place to spend a physics rollout.
            return {max(grid)}
        if spec == "all":
            return set(grid)
        return {int(v) for v in spec.split(",") if v.strip()}
    roll_at, gram_at = _at(args.rollout_at), _at(args.gram_at)

    n_from_orig = args.n_orig_fit
    print(f"fit  : {n_fit_total} episodes "
          f"({n_fit_total - n_from_orig} from 2026-09-03, "
          f"{n_from_orig} from 2026-09-02)", flush=True)
    print(f"held : {n_held} episodes (2026-09-02): "
          f"{[e.name for e in held_eps]}", flush=True)
    print(f"demo-count grid ({len(grid)} points): {grid}")
    print(f"methods in order: {methods}   loop order: {args.loop}")
    print(f"{len(methods) * len(grid)} fits total", flush=True)

    (out / f"meta{suffix}.json").write_text(json.dumps(_clean(dict(
        fit_dir=str(FIT_DEMO_DIR), held_dir=str(TEST_DEMO_DIR),
        fit_episodes=[e.name for e in fit_eps],
        held_episodes=[e.name for e in held_eps],
        n_train=args.n_train, n_orig_fit=args.n_orig_fit,
        n_fit=n_fit_total, n_held=n_held, grid=grid, methods=methods,
        loop=args.loop, n_starts=args.n_starts, n_iters=args.n_iters,
        n_restarts=args.n_restarts, n_outer_steps=cmp.N_OUTER_STEPS,
        lr=cmp.LR, fd_eps=cmp.FD_EPS,
        cma_budget_solves=cmp.CMA_BUDGET_SOLVES, pin_ik=args.pin_ik,
        per_segment=args.per_segment, free_space_only=args.free_space_only,
    )), indent=2, default=cmp._json_default))

    # One build per demo count, memoised: with --loop method the same `n` comes
    # back once per method, and rebuilding costs ~2 min each time.  Only the
    # most recent is kept -- holding 27 built forward maps alive would pin 27
    # sets of compiled executables and demo arrays on the device.
    _cache = {}

    def get_built(n):
        if n not in _cache:
            _cache.clear()
            t = time.perf_counter()
            _cache[n] = build_teleop(demo_dir=fit_eps, held_dir=held_eps,
                                     n_fit_max=n, n_iters=args.n_iters,
                                     space="joint", pin_ik=args.pin_ik,
                                     per_segment=args.per_segment,
                                     free_space_only=args.free_space_only,
                                     n_restarts=args.n_restarts)
            print(f"  [build] n={n} in {time.perf_counter()-t:.0f}s", flush=True)
        return _cache[n]

    def seeds_for(K):
        rng = np.random.default_rng(0)
        u0 = jnp.zeros(K, dtype=jnp.float32)
        return [u0] + [jnp.asarray(rng.normal(0, 0.5, K), jnp.float32)
                       for _ in range(max(args.n_starts, 1) - 1)]

    chunk = args.seed_chunk or None

    def dispatch(name, b, seeds):
        multi = len(seeds) > 1
        u0 = seeds[0]
        if name == "implicit":
            return (cmp.run_implicit_parallel(b, seeds, chunk=chunk) if multi
                    else cmp.run_implicit(b, u0))
        if name == "fd":
            return (cmp.run_fd_parallel(b, seeds, chunk=chunk) if multi
                    else cmp.run_fd(b, u0))
        if name == "cmaes":
            return (cmp.run_cmaes_parallel(b, seeds) if multi
                    else cmp.run_cmaes(b, u0))
        if name == "unrolled":
            return cmp.run_unrolled(b, seeds, chunk=chunk,
                                    compile_timeout=args.compile_timeout,
                                    pin_ik=args.pin_ik,
                                    free_space_only=args.free_space_only,
                                    per_segment=args.per_segment)
        raise SystemExit(f"unknown method {name!r}")

    jobs = ([(m, n) for m in methods for n in grid] if args.loop == "method"
            else [(m, n) for n in grid for m in methods])
    u_hats, t_run0 = {}, time.perf_counter()
    for done, (method, n) in enumerate(jobs, 1):
        el = time.perf_counter() - t_run0
        print("\n" + "#" * 78, flush=True)
        print(f"# [{done}/{len(jobs)}] method={method}  n_demos={n}/{n_fit_total}"
              f"  elapsed {el/3600:.2f} h", flush=True)
        print("#" * 78, flush=True)
        t0 = time.perf_counter()
        try:
            b = get_built(n)
            r = dispatch(method, b, seeds_for(int(b["K"])))
        except Exception as e:                       # noqa: BLE001
            # One diverging or OOMing (method, n) must not cost the rest of
            # the grid; the failure is recorded as a checkpoint of its own.
            import traceback
            traceback.print_exc()
            b, r = _cache.get(n), cmp._failed_result(method, f"error: {e}")
        cmp._check_trace(r)
        wall = time.perf_counter() - t0

        rec = dict(method=method, n_demos=n, n_fit_total=n_fit_total,
                   n_held=n_held, checkpoint=True, wall_s=wall,
                   wall_compile_s=r.get("wall_compile_s"),
                   wall_infer_s=r.get("wall_infer_s"),
                   trace=r.get("trace"), diverged=bool(r.get("diverged")),
                   failure=r.get("failure"),
                   start_losses=r.get("start_losses"),
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        if r["u_hat"] is not None and b is not None:
            u = jnp.asarray(r["u_hat"])
            rec["u_hat"] = np.asarray(r["u_hat"]).tolist()
            rec["theta"] = cmp._theta_dict(b, u)
            rec["metrics"] = _metrics_n(b, u)
            rec["standoff_prior"] = np.asarray(b["standoff_prior"]).tolist()
            rec["episodes"] = list(b["episodes"])
            u_hats[f"{method}_n{n:02d}"] = np.asarray(r["u_hat"])
            if n in gram_at:
                rec["gram"] = cmp._gram(b, u)
            if n in roll_at:
                rec["rollout"] = cmp.rollout_success(b, u, label=f"{method}/n{n}")
            m = rec["metrics"]
            print(f"  -> n={n:2d} {method:9s} loss {m['loss']:.4f}  "
                  f"EE fit {m['ee_rmse_fit']:.4f}  "
                  f"EE held {m['ee_rmse_gen']:.4f}  ({wall:.0f}s)", flush=True)
        else:
            rec["theta"], rec["metrics"] = {}, {}
            print(f"  -> n={n:2d} {method:9s} FAILED ({rec['failure']})",
                  flush=True)

        # Append-and-fsync: the curve is readable, and survives a kill, at
        # every point of the run rather than only at the end.
        with open(ckpt_path, "a") as fh:
            fh.write(json.dumps(_clean(rec), default=cmp._json_default) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if u_hats:
            np.savez(out / f"u_hats{suffix}.npz", **u_hats)

    print(f"\nAll {len(jobs)} fits done in "
          f"{(time.perf_counter()-t_run0)/3600:.2f} h -> {out}", flush=True)


if __name__ == "__main__":
    main()
