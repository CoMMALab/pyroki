"""Executed-success report: run each domain's plan through the actuated sim.

`iosp.checks.feasibility_report` asks whether a plan's rows are reachable,
collision-free and in-limits.  This asks the harder question -- whether driving
those rows through the Panda's position servos under gravity actually MOVES THE
OBJECT WHERE THE TASK WANTED IT -- by running `iosp.viz.mj_rollout` headless on
each domain and reading the object's pose off the result.

    python -m iosp.checks.dynamic_report scratch/feas/tetris_fresh.npz
    python -m iosp.checks.dynamic_report scratch/feas/*_fresh.npz

Two numbers per scene matter, and they fail for different reasons:

  release error
    how far the object was from its target AT THE MOMENT OF RELEASE.  This is
    the plan's own error, made physical -- it includes the servo tracking that
    a kinematic replay hides.
  settled error
    where the object ended up after falling and coming to rest.  For `tetris`
    and `tower` the target is a surface, so this is the task's real verdict; for
    `pickplace` the target is a point in FREE SPACE (z = 0.3 with no support
    under it), so the object is always going to fall out of it and only the
    release error means anything.  The report says which it scored.
"""
from __future__ import annotations

import argparse

import numpy as np

from iosp.viz import mj_rollout as R
from iosp.viz import mj_scene as M

TOL_M = 0.05


# ---------------------------------------------------------------------------
# SPaSM's own success criteria, applied to what the rollout ACHIEVED
# ---------------------------------------------------------------------------

def spasm_score(domain, achieved_xyzyaw, slot, num_blocks):
    """Re-score SPaSM's task cost with one block moved to where it ended up.

    SPaSM checks its plans kinematically; the whole point here is to apply the
    SAME criterion to an executed result.  So take SPaSM's own solution, replace
    the slot this scene was responsible for with the pose the arm actually left
    the block in, and evaluate SPaSM's own cost.

    Returns (cost_achieved, cost_planned, threshold_or_None).  For tetris the
    threshold is SPaSM's `SpasmParams.cost_thresh` for that block count -- the
    number `solve.py` itself uses to call an instance solved.  Tower defines no
    threshold, so the bar is SPaSM's own planned cost: an execution that scores
    no worse than the plan did is one the dynamics did not break.
    """
    from iosp.model import spasm_tasks as ST

    if domain == "tetris":
        planned = ST.tetris_skeleton(num_blocks)
        got = np.array(planned, copy=True)
        got[slot] = achieved_xyzyaw
        return (ST.tetris_cost(got, num_blocks),
                ST.tetris_cost(planned, num_blocks),
                ST.TETRIS_COST_THRESH.get(num_blocks))
    if domain == "tower":
        planned = ST.tower_skeleton(num_blocks)
        init = ST.tower_init_state(num_blocks)
        got = np.array(planned, copy=True)
        got[slot] = achieved_xyzyaw
        return (ST.tower_cost(got, init, num_blocks),
                ST.tower_cost(planned, init, num_blocks), None)
    return None, None, None


# ---------------------------------------------------------------------------
# Per-domain scene construction
# ---------------------------------------------------------------------------

def _build_tetris(b, d, n, num_blocks=1, **_):
    from iosp.viz import tetris_viser as V
    so = np.array([0.0, 0.0, V.STANDOFF])
    for i in range(n):
        V._add_tetris_env(b, i, num_blocks, pick=d["pick"][i], place=d["place"][i])
        V._carried_tetromino(b, i, d["pick"][i] - so, collide=True)
    gd, gp = V._goal_dims(num_blocks)

    def scored(pos, i):
        inside = (abs(pos[0] - gp[0]) <= gd[0] / 2 and
                  abs(pos[1] - gp[1]) <= gd[1] / 2)
        return inside, ("in the goal" if inside else "outside the goal")

    return dict(grasp_off=so, place_off=so, score_on="settled", extra=scored)


def _build_pickplace(b, d, n, **_):
    from iosp.viz import pickplace_viser as V
    zero = np.zeros(3)
    for i in range(n):
        V._add_pickplace_env(b, i, pick=d["pick"][i], place=d["place"][i])
        pick = np.asarray(d["pick"][i], float)
        ped_h = pick[2] - V.OBJ_SIZE / 2
        if ped_h > 0.005:
            off = b.offset(i)
            ped_dims = (V.OBJ_SIZE + 0.01, V.OBJ_SIZE + 0.01, ped_h)
            ped_pos = off + np.array([pick[0], pick[1], ped_h / 2])
            b.box(f"s{i}_pedestal", tuple(ped_pos), ped_dims,
                  (0.7, 0.7, 0.7, 0.4))
        V._carried_box(b, i, d["pick"][i], collide=True)
    return dict(grasp_off=zero, place_off=zero, score_on="release", extra=None)


def _build_tower(b, d, n, stack_level=None, **_):
    from iosp.viz import tower_viser as V
    so = np.array([0.0, 0.0, V.STANDOFF])
    levels = []
    for i in range(n):
        if stack_level is not None:
            lvl = stack_level
        else:
            rest_z = float(d["place"][i, 2]) - V.STANDOFF
            lvl = int(round(rest_z / V.BLOCK_DIM - 0.5))
        levels.append(lvl)
        base_xy = (float(d["place"][i, 0]), float(d["place"][i, 1]))
        V._add_tower_env(b, i, lvl, pick=d["pick"][i],
                         place=d["place"][i], obs_center=None,
                         skip_spawn_near=d["pick"][i] - so,
                         base_xy=base_xy)
        V._carried_cube(b, i, d["pick"][i] - so, collide=True)

    def scored(pos, i):
        rest_z = V.BLOCK_DIM * (levels[i] + 0.5)
        on_stack = abs(pos[2] - rest_z) < V.BLOCK_HALF
        return on_stack, ("on the stack" if on_stack else "off the stack")

    level_str = (str(levels[0]) if len(set(levels)) == 1
                 else ",".join(str(l) for l in levels))
    return dict(grasp_off=so, place_off=so, score_on="settled", extra=scored,
                note=f"stack levels {level_str}, clearance obstacles not collided")


BUILDERS = {"tetris": _build_tetris, "pickplace": _build_pickplace,
            "tower": _build_tower}


def _load(path):
    d = np.load(path, allow_pickle=True)
    out = {k: np.asarray(d[k]) for k in d.files if d[k].ndim > 0}
    robot = str(d["robot"]) if "robot" in d else "panda"
    episodes = (list(d["episodes"]) if "episodes" in d else None)
    return (str(d["domain"]), dict(q=out["q"], pick=out["pick_pos"],
                                   place=out["place_pos"],
                                   robot=robot, episodes=episodes,
                                   **{k: out[k] for k in ("obs_center",
                                                          "obs_radius")
                                      if k in out}),
            int(d["idx_pick"]), int(d["idx_place"]))


# ---------------------------------------------------------------------------

def _report_fr3_pickplace(path, d, n, quiet=False):
    """FR3 teleop pickplace: use the teleop physics scene (e10_spasm_sim)."""
    from iosp.viz.e10_spasm_sim import physics_success
    q = d["q"][:n]
    n_ok = 0
    eps_label = d.get("episodes") or list(range(n))
    print(f"\n===== pickplace (FR3 teleop): EXECUTED rollout  ({path})  {n} scenes =====")
    print(f"      scored on bucket landing (physics_success)")
    for i in range(n):
        res = physics_success(i, q[i], verbose=not quiet)
        ok = res["success"]
        n_ok += ok
        label = eps_label[i] if i < len(eps_label) else i
        print(f"  scene {i} ({label}): {'SUCCESS' if ok else '   FAIL'}   "
              f"dxy={res['dxy']*1000:.1f}mm  dz={res['dz']*1000:.1f}mm")
    print(f"  ---> {n_ok}/{n} succeeded\n")
    return n_ok, n


def report(path, n_scenes=None, settle=R.DEFAULT_SETTLE, stack_level=3,
           num_blocks=1, quiet=False):
    domain, d, grasp_row, release_row = _load(path)
    if domain not in BUILDERS:
        raise SystemExit(f"no dynamic scene for domain {domain!r}")
    n = d["q"].shape[0] if n_scenes is None else min(n_scenes, d["q"].shape[0])

    if d.get("robot") == "fr3" and domain == "pickplace":
        return _report_fr3_pickplace(path, d, n, quiet=quiet)
    d = {k: (v[:n] if getattr(v, "ndim", 0) and len(v) >= n else v)
         for k, v in d.items()}

    b = M.WorldBuilder(n_scenes=n, spread=1.8, robot="menagerie")
    cfg = BUILDERS[domain](b, d, n, stack_level=stack_level,
                           num_blocks=num_blocks)
    world = b.compile()

    ro = R.run(world, d["q"], grasp_row, release_row,
               carried_names=[f"s{i}_carried" for i in range(n)],
               settle=settle, progress=not quiet)

    # Object pose at release, and after settling.
    def obj_at(frame):
        world.data.qpos[:] = ro.qpos[frame]
        world.forward()
        return np.stack([world.body_pos(f"s{i}_carried") - world.offsets[i]
                         for i in range(n)])

    rel = obj_at(min(ro.release_frame, ro.n_frames - 1))
    fin = obj_at(ro.n_frames - 1)
    targets = d["place"] - cfg["place_off"]

    # Tracking error has two very different causes, and reporting only their
    # sum invites blaming the wrong one.  `limit` is what the plan asks for
    # beyond the joint's hard stop -- a static property of the rows, unavoidable
    # by any controller.  Whatever tracking error EXCEEDS it came from the arm
    # being physically obstructed by the scene on the way.
    lo, hi = world.joint_limits()
    over = np.degrees(np.maximum(0.0, np.maximum(lo - d["q"], d["q"] - hi))
                      ).max(axis=(1, 2))

    rows, n_ok = [], 0
    print(f"\n===== {domain}: EXECUTED rollout  ({path})  {n} scenes =====")
    if cfg.get("note"):
        print(f"      {cfg['note']}")
    print(f"      scored on {cfg['score_on']} position"
          + ("   [place target is in free space -- the object always falls "
             "out of it]" if cfg["score_on"] == "release" else ""))
    for i in range(n):
        e_rel = float(np.linalg.norm(rel[i][:2] - targets[i][:2]))
        e_fin = float(np.linalg.norm(fin[i][:2] - targets[i][:2]))
        fell = bool(fin[i][2] < -0.02)
        extra_ok, where = (cfg["extra"](fin[i], i) if cfg["extra"]
                           else (True, ""))
        err = e_rel if cfg["score_on"] == "release" else e_fin
        ok = (err < TOL_M) and extra_ok and not fell
        n_ok += ok
        worst, mean = ro.tracking_mm_deg(i)
        note = "fell off the table" if fell else where
        blocked = max(worst - over[i], 0.0)
        print(f"  scene {i}: {'SUCCESS' if ok else '   FAIL'}   "
              f"release {e_rel * 100:6.1f} cm   settled {e_fin * 100:7.1f} cm"
              f"{'  ' + note if note else ''};   "
              f"tracking max {worst:5.1f} deg "
              f"(= {over[i]:.0f} past joint limit + {blocked:.0f} obstructed, "
              f"mean {mean:.1f})")
        rows.append(ok)
    print(f"  ---> {n_ok}/{n} succeeded\n")
    return n_ok, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--n-scenes", type=int, default=None)
    ap.add_argument("--settle", type=float, default=R.DEFAULT_SETTLE)
    ap.add_argument("--stack-level", type=int, default=None,
                    help="tower only; inferred from the place target by default")
    ap.add_argument("--num-blocks", type=int, default=1, help="tetris only")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tot_ok = tot = 0
    for p in args.npz:
        ok, n = report(p, args.n_scenes, args.settle, args.stack_level,
                       args.num_blocks, args.quiet)
        tot_ok += ok
        tot += n
    if len(args.npz) > 1:
        print(f"===== overall: {tot_ok}/{tot} scenes execute successfully =====")


if __name__ == "__main__":
    main()
