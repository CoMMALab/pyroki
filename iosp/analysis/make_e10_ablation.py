"""Read `e10_demo_ablation`'s checkpoint stream into a table and a curve.

The stream is JSONL, appended one line per (method, demo count), so this runs
against a collection that is still in progress -- which is the point: it is how
the 24-hour run is monitored, not only how it is written up.

    python -m iosp.analysis.make_e10_ablation [--results DIR] [--plot OUT.pdf]
"""
import argparse
import json
import pathlib

import numpy as np

DEFAULT = (pathlib.Path(__file__).resolve().parents[1]
           / "data" / "results" / "e10_demo_ablation")


def load(results_dir=DEFAULT):
    d = pathlib.Path(results_dir)
    metas = sorted(d.glob("meta*.json"))
    meta = json.loads(metas[0].read_text()) if metas else {}
    recs = []
    # `checkpoints.jsonl` plus any `checkpoints.<tag>.jsonl` written by a
    # sibling run on another GPU -- one method per GPU is the usual layout, so
    # the curve is assembled from however many streams happen to exist.
    for path in sorted(d.glob("checkpoints*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                # A kill mid-write leaves one truncated trailing line; the
                # complete records before it are still the result.
                print(f"  (skipping truncated trailing record in {path.name})")
    return meta, recs


def table(recs):
    hdr = (f"{'method':<10} {'n':>3} {'loss':>9} {'EE fit':>8} {'EE held':>8} "
           f"{'J held':>8} {'wall':>8}")
    lines = [hdr, "-" * len(hdr)]
    for r in sorted(recs, key=lambda r: (r["method"], r["n_demos"])):
        m = r.get("metrics") or {}
        if not m:
            lines.append(f"{r['method']:<10} {r['n_demos']:>3}   "
                         f"FAILED  {r.get('failure')}")
            continue
        flag = "  DIVERGED" if r.get("diverged") else ""
        lines.append(
            f"{r['method']:<10} {r['n_demos']:>3} {m['loss']:9.4f} "
            f"{m['ee_rmse_fit']:8.4f} {m['ee_rmse_gen']:8.4f} "
            f"{m['joint_rmse_gen']:8.4f} {r['wall_s']/60:7.1f}m{flag}")
    return "\n".join(lines)


def plot(recs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for method in sorted({r["method"] for r in recs}):
        rs = sorted((r for r in recs if r["method"] == method
                     and (r.get("metrics") or {})),
                    key=lambda r: r["n_demos"])
        if not rs:
            continue
        n = [r["n_demos"] for r in rs]
        axes[0].plot(n, [r["metrics"]["ee_rmse_gen"] for r in rs], "o-", label=method)
        axes[1].plot(n, [r["metrics"]["loss"] for r in rs], "o-", label=method)
    axes[0].set_ylabel("held-out EE RMSE [m]")
    axes[1].set_ylabel("training loss")
    for ax in axes:
        ax.set_xlabel("demonstrations used in the fit")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("E10: cost recovery vs. number of demonstrations "
                 "(held out: 10 episodes, different session)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(DEFAULT))
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    meta, recs = load(a.results)
    if meta:
        print(f"{meta.get('n_fit')} fit / {meta.get('n_held')} held out, "
              f"grid {meta.get('grid')}, methods {meta.get('methods')}")
    print(f"{len(recs)} checkpoints\n")
    print(table(recs))
    if a.plot:
        plot(recs, a.plot)


if __name__ == "__main__":
    main()
