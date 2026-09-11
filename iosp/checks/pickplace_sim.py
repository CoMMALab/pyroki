"""Physics validation of pick-and-place using the principled build_pick_and_place pipeline.

Uses `build_pick_and_place` with `pin_ik="bucket"` (release directly above bucket centre,
zero radial/tangential offsets), `freeze_ik=True` (theta_ik held at the measured
standoff prior), and 600 L-BFGS iterations (inner stationarity ~5e-4).

This is the aligned pipeline that achieves 10/10 rollout success on the teleop
episodes.  Separate from `e10_spasm_sim` (which uses `build_from_demos`, the
K=17 multistart model with refine phase and measured radial/tangential offsets).

Usage:
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
        python -m iosp.checks.pickplace_sim [--n-iters 600]
"""
import argparse
import json

import numpy as np


def build_and_rollout(u=None, n_iters=600, pin_ik="bucket"):
    """Joint paths for ALL episodes at outer parameter `u`.

    Returns (paths, built) where paths is (M, N_FULL, 7).
    """
    import jax
    import jax.numpy as jnp
    from iosp.fit.teleop import build_pick_and_place
    from iosp.model import pickplace as pp

    built = build_pick_and_place(n_iters=n_iters, space="joint", freeze_ik=True,
                         pin_ik=pin_ik, fast_forward=True)
    K = built["K"]
    u_vec = jnp.zeros(K, jnp.float32) if u is None else jnp.asarray(u, jnp.float32)
    paths = np.asarray(jax.jit(built["paths_fn"])(u_vec))

    theta = np.asarray(built["theta_of"](u_vec))
    theta_ik = theta[: pp.K_IK]
    w = theta[pp.K_IK:]
    print(f"\ntheta_ik: grasp {theta_ik[0]:.4f}  place {theta_ik[1]:.4f}  "
          f"radial {theta_ik[2]:+.4f}  tangential {theta_ik[3]:+.4f}")
    print("weights:", {n: round(float(v), 3) for n, v in zip(pp.THETA_TRAJOPT_NAMES, w)})
    return paths, built


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-iters", type=int, default=600)
    ap.add_argument("--u", type=str, default=None,
                    help="outer parameter as a JSON list of length K")
    ap.add_argument("--pin-ik", choices=["bucket", "measured", "feasible"],
                    default="bucket")
    args = ap.parse_args()

    u = json.loads(args.u) if args.u else None
    paths, built = build_and_rollout(u=u, n_iters=args.n_iters, pin_ik=args.pin_ik)
    M = paths.shape[0]

    from iosp.viz.e10_spasm_sim import physics_success
    n_ok = 0
    print(f"\n{'='*60}")
    print(f"Testing {M} episodes (pin_ik={args.pin_ik}, n_iters={args.n_iters})")
    print(f"{'='*60}")
    for i in range(M):
        res = physics_success(i, paths[i], verbose=False)
        n_ok += res["success"]
        print(f"  ep {i} ({built['episodes'][i]}): "
              f"{'OK' if res['success'] else 'FAIL'}  "
              f"dxy={res['dxy']*1000:.1f}mm  dz={res['dz']*1000:.1f}mm")
    print(f"\n  => {n_ok}/{M} succeeded")


if __name__ == "__main__":
    main()
