"""Extract a domain's FORWARD-pass joint trajectory + scene targets at Z_STAR.

The demonstrations the inverse pass tries to recover are rollouts of the composed
trajopt at the ground-truth cost `Z_STAR`.  This dumps, for one scene, the joint
path `q` (N_FULL, dof) plus the pick/place targets and block/obstacle geometry, so
a MuJoCo pass can check the trajectory is actually feasible (reachable, collision-
free, and the block ends where the skeleton says).

    python -m iosp.checks.forward_extract tower --out scratch/feas/tower.npz
    python -m iosp.checks.forward_extract tetris --out scratch/feas/tetris.npz
    python -m iosp.checks.forward_extract pickplace --out scratch/feas/pickplace.npz
"""
import argparse
import logging
import os
import pathlib
import sys
import time

# This check only ever runs the forward pass -- nothing here differentiates
# through the solve -- so the fixed-length unroll that exists to make
# q*(theta) differentiable is pure compile cost (~2.4x on tetris and tower).
# Default to the stock early-stopping solver; IOSP_STOCK_SOLVER=0 opts back out.
os.environ.setdefault("IOSP_STOCK_SOLVER", "1")

from iosp import config
config.setup()

import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger("forward_extract")


def _zstar_override(default, names):
    """Env `ZSTAR_SKEL` overrides the skeleton logit for the feasibility test."""
    import os
    boost = os.environ.get("ZSTAR_SKEL")
    if boost is None:
        return default
    z = np.asarray(default, np.float32).copy()
    z[list(names).index("skeleton")] = float(boost)
    print(f"  [Z_STAR override] skeleton logit -> {boost}", flush=True)
    return jnp.asarray(z)


def _tower(n_scenes=6):
    from iosp.experiments import e9_tower as E
    from iosp.model import tower as tw
    log.info("tower: building model (n_iters=60, n_scenes=%d, stack_level=None)",
             n_scenes)
    t0 = time.perf_counter()
    built = E.build(seed=0, n_iters=60, n_scenes=n_scenes, stack_level=None)
    log.info("tower: build done in %.1fs, starting jit(q_of)", time.perf_counter() - t0)
    prob, refine = built["prob"], built["refine"]
    scenes = built["fit"]
    zstar = _zstar_override(E.Z_STAR, tw.FEATURE_NAMES)

    def q_of(z):
        theta = jax.nn.softmax(z)
        xs, seg_scenes, full_sc, _, _ = prob.solve(
            scenes, built["inner_by_phase"], theta[:tw.K_SEG], theta, refine)
        return jax.vmap(prob.seg["full"].unpack)(xs["full"], full_sc)

    t1 = time.perf_counter()
    q = np.asarray(jax.jit(q_of)(zstar))            # (S, N_FULL, dof)
    log.info("tower: jit(q_of) done in %.1fs, q%s", time.perf_counter() - t1, q.shape)
    meta = dict(
        domain="tower", idx_pick=tw.IDX_PICK, idx_place=tw.IDX_PLACE,
        n_full=tw.N_FULL, block_half=tw.BLOCK_HALF,
        pick_pos=np.asarray(scenes.pick_pos), place_pos=np.asarray(scenes.place_pos),
        target_z=np.asarray(scenes.target_z), q_start=np.asarray(scenes.q_start),
        obs_center=np.asarray(scenes.obs_center), obs_radius=np.asarray(scenes.obs_radius),
        **_limit_meta(prob),
    )
    return q, meta


def _tetris(num_blocks=3):
    from iosp.experiments import e8_tetris as E
    from iosp.model import tetris as tt
    log.info("tetris: building model (n_iters=60, n_scenes=6, num_blocks=%d)",
             num_blocks)
    t0 = time.perf_counter()
    # 3 blocks: SPaSM's saved packing skeleton is a 3-block solution, and
    # `sample_tetris_scenes` now walks that skeleton's slots.
    built = E.build(seed=0, n_iters=60, n_scenes=6, num_blocks=num_blocks)
    log.info("tetris: build done in %.1fs, starting jit(q_of)", time.perf_counter() - t0)
    prob, refine = built["prob"], built["refine"]
    scenes = built["fit"]
    zstar = _zstar_override(E.Z_STAR, tt.FEATURE_NAMES)

    def q_of(z):
        theta = jax.nn.softmax(z)
        xs, seg_scenes, full_sc, _, _ = prob.solve(
            scenes, built["inner_by_phase"], theta[:tt.K_SEG], theta, refine)
        return jax.vmap(prob.seg["full"].unpack)(xs["full"], full_sc)

    t1 = time.perf_counter()
    q = np.asarray(jax.jit(q_of)(zstar))
    log.info("tetris: jit(q_of) done in %.1fs, q%s", time.perf_counter() - t1, q.shape)
    # `pick_pos` sits GRASP_OFFSET above the block, so the spawn pose the
    # sampler drew is recoverable as (pick_pos - up, pick_yaw).  Record it: with
    # spawn randomisation on, the rollout scene has to place each block where
    # this plan actually expects to find it.
    up = np.array([0.0, 0.0, tt.GRASP_OFFSET], np.float32)
    pick_pos = np.asarray(scenes.pick_pos)
    spawn_poses = np.concatenate(
        [pick_pos - up, np.asarray(scenes.pick_yaw)[:, None]], axis=1)
    meta = dict(
        domain="tetris", idx_pick=tt.IDX_PICK, idx_place=tt.IDX_PLACE,
        n_full=tt.N_FULL,
        pick_pos=pick_pos, place_pos=np.asarray(scenes.place_pos),
        q_start=np.asarray(scenes.q_start),
        spawn_poses=spawn_poses, slot=np.asarray(scenes.slot),
        place_yaw=np.asarray(scenes.place_yaw),
        num_blocks=num_blocks,
        **_limit_meta(prob),
    )
    return q, meta


def _pickplace():
    from iosp.experiments import e4_three_stage as E
    from iosp.model import pickplace as pp
    log.info("pickplace: building synthetic model (n_iters=60, n_scenes=6)")
    t0 = time.perf_counter()
    built = E.build(seed=0, n_iters=60, n_scenes=6)
    log.info("pickplace: build done in %.1fs, starting jit(q_of)",
             time.perf_counter() - t0)
    prob, refine = built["prob"], built["refine"]
    scenes = built["fit"]
    zstar = _zstar_override(E.Z_STAR, pp.THETA_SHARED_NAMES)

    def q_of(z):
        theta = jax.nn.softmax(z)
        theta_seg, theta_full = prob.split_shared(theta)
        x0, _, q_pick, q_place = prob.seeds(scenes, E.THETA_IK)
        full_sc = prob.full_scenes(scenes, q_pick, q_place)
        _, _, xs, ps = prob.solve(E.THETA_IK, {p: theta_seg for p in pp.PHASES},
                                  scenes, built["inner_by_phase"], x0,
                                  refine=refine, theta_full=theta_full)
        return jax.vmap(prob.seg["full"].unpack)(xs["full"], ps["full"])

    t1 = time.perf_counter()
    q = np.asarray(jax.jit(q_of)(zstar))
    log.info("pickplace: jit(q_of) done in %.1fs, q%s",
             time.perf_counter() - t1, q.shape)
    meta = dict(
        domain="pickplace",
        idx_pick=pp.SKELETON_PICK[1], idx_place=pp.SKELETON_PLACE[0],
        n_full=pp.N_FULL,
        pick_pos=np.asarray(scenes.pick_pos),
        place_pos=np.asarray(scenes.place_pos),
        q_start=np.asarray(scenes.q_start),
        obs_center=np.asarray(scenes.obs_center),
        obs_radius=np.asarray(scenes.obs_radius),
        **_limit_meta(prob),
    )
    return q, meta


# The planner's URDF (`resources/panda/panda_spherized.urdf`) is 4 deg LOOSER
# than the Panda that actually executes the plan (MuJoCo Menagerie / the Franka
# datasheet) on every one of the seven joints: +-170 vs +-166 on 1/3/5/7,
# +-105 vs +-101 on 2, [-180, 5] vs [-176, -4] on 4, [-5, 219] vs [-1, 215] on
# 6.  A plan solved right up against the URDF limit is therefore still up to
# 4 deg outside what the arm can reach, and the servo answers with a permanent
# pose error rather than a tracking lag.  Executable limits are the URDF's
# minus this margin.
EXEC_JOINT_MARGIN = np.radians(4.5)

# `resources/panda/panda_spherized.urdf`, for extracts saved before the limits
# were recorded in the npz.
URDF_Q_LO = np.radians([-170.0, -105.0, -170.0, -180.0, -170.0, -5.0, -170.0])
URDF_Q_HI = np.radians([170.0, 105.0, 170.0, 5.0, 170.0, 219.0, 170.0])


def _limit_meta(prob):
    """URDF joint limits of `prob`'s robot, saved alongside the plan."""
    j = prob.base.robot.joints
    return dict(q_lo=np.asarray(j.lower_limits, np.float64),
                q_hi=np.asarray(j.upper_limits, np.float64))


def clip_to_limits(q, meta=None):
    """Clamp an extracted plan into the EXECUTABLE joint limits, loudly.

    Only the pinned pick/place rows come from IK (which now clips itself); the
    rows between them are free variables of an UNCONSTRAINED lbfgs solve, so
    nothing stops the plan from leaving the limits on its way to a waypoint.
    An out-of-limit row is not executable: the position servo saturates at the
    limit and holds a steady pose error equal to the violation, which no amount
    of slowing the trajectory removes.  Clipping here makes the saved plan mean
    what it claims -- a path a real arm could follow -- and the warning says how
    far the solve had strayed, so a large number is visible rather than silently
    absorbed.  A large number means the PLAN needs fixing, not the clip: the
    clipped row no longer reaches the pose it was solved for.
    """
    meta = meta or {}
    lo = np.asarray(meta.get("q_lo", URDF_Q_LO), np.float64) + EXEC_JOINT_MARGIN
    hi = np.asarray(meta.get("q_hi", URDF_Q_HI), np.float64) - EXEC_JOINT_MARGIN
    v = np.maximum(lo - q, q - hi)
    if v.max() > 0:
        s, t, k = np.unravel_index(np.argmax(v), v.shape)
        n = int((v > 0).any(axis=-1).sum())
        msg = (f"plan left the executable joint limits on {n} row(s); worst "
               f"{np.degrees(v.max()):.2f} deg (scene {s}, row {t}, joint {k}) "
               f"-- clipped")
        print(f"WARNING: {msg}", flush=True)
        log.warning(msg)
    return np.clip(q, lo, hi).astype(q.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["tower", "tetris", "pickplace"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-blocks", type=int, default=3,
                    help="tetris only: which packing skeleton to plan for "
                         "(3, 5 or 8 -- see resources/tetris_packing_*.npy)")
    ap.add_argument("--n-scenes", type=int, default=6,
                    help="tower only: how many STACK LEVELS to plan.  Scene i "
                         "builds level i, so this is the height of the tower "
                         "the rollout can chain -- the default 6 is why a "
                         "replay stops at six blocks")
    ap.add_argument("--log", type=str, default=None,
                    help="write progress to this log file (tailable)")
    args = ap.parse_args()

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log:
        logpath = pathlib.Path(args.log)
        logpath.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logpath, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    log.info("=== forward_extract %s -> %s ===", args.domain, args.out)
    t_start = time.perf_counter()

    kw = ({"num_blocks": args.num_blocks} if args.domain == "tetris" else
          {"n_scenes": args.n_scenes} if args.domain == "tower" else {})
    q, meta = {"tower": _tower, "tetris": _tetris,
               "pickplace": _pickplace}[args.domain](**kw)
    q = clip_to_limits(q, meta)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    str_keys = {k: v for k, v in meta.items() if isinstance(v, str)}
    arr_keys = {k: np.asarray(v) for k, v in meta.items()
                if not isinstance(v, str)}
    np.savez(out, q=q, **arr_keys, **str_keys)
    msg = (f"wrote {out}: q{q.shape}  pick={meta['pick_pos'][0]}  "
           f"place={meta['place_pos'][0]}")
    print(msg, flush=True)
    log.info(msg)
    qmsg = (f"q range: [{q.min():.2f}, {q.max():.2f}] rad  "
            f"(dof={q.shape[-1]}, N_FULL={q.shape[1]})")
    print(qmsg, flush=True)
    log.info(qmsg)
    log.info("=== %s DONE in %.1fs ===", args.domain, time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
