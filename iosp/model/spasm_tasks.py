"""The SPaSM tetris and tower TASKS: skeletons, and what counts as success.

Geometry and cost functions live in `iosp.model.spasm_costs`, ported into this
repo; this module adds the task SKELETONS (the goal states a demonstration aims
at) and the success tests applied to what a rollout actually achieved.

Nothing here reads the `../spasm` checkout.  It used to import SPaSM's
`Simulation` for its geometry, which dragged in that repo's whole stack --
meshcat, xmltodict, its own kinematics -- so `iosp.viz.tetris_viser`, which
renders through mjviser and touches none of it, would not start in an
environment missing meshcat.  The one remaining bridge is
`iosp.checks.spasm_trajopt`, which exists specifically to run SPaSM's own
trajectory optimizer and is opt-in.

Success is GEOMETRIC and task-level, not SPaSM's cost.  The cost is reported
alongside as a diagnostic because it is a planning objective: its penetration
terms have dead zones (10 mm between blocks, 20 mm to a wall), and for a goal
0.15 m wide against a 0.12 m block the wall term cannot reach zero at all, so a
perfectly good packing still scores well above the threshold.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np

from iosp.model import spasm_costs as C

RESOURCES = pathlib.Path(__file__).resolve().parents[2] / "resources"

TETRIS_COST_THRESH = C.TETRIS_COST_THRESH

# SPaSM sizes the goal at exactly the cell count its blocks occupy -- 6x2 for
# three blocks, the 12 cells O+L+O cover.  That demands a perfect tiling, and
# none exists: two O's leave a 2x2 hole and an L (4 cells in a 2x3 bounding box)
# cannot fill a 2x2.  That is why SPaSM's own saved packing overlaps by 41 mm.
# TWO extra cells is what it takes for the same three shapes, at the same sizes,
# from the same spawn poses, to actually fit: re-solving gives -6.6 mm clearance
# at one extra cell but +16.0 mm at two.
EXTRA_GOAL_CELLS = int(os.environ.get("IOSP_EXTRA_GOAL_CELLS", "2"))


# ---------------------------------------------------------------------------
# Tetris
# ---------------------------------------------------------------------------

def tetris_geometry(num_blocks=3, extra_cells=None):
    """The tetris scene: goal, walls, sphere layout, spawn poses, table."""
    if extra_cells is None:
        extra_cells = EXTRA_GOAL_CELLS
    gd = C.goal_dims(num_blocks, extra_cells)
    bz = C.block_z()
    shapes = C.TETRIS_SHAPES[:num_blocks]
    return dict(
        goal_dims=gd.astype(np.float32),
        goal_position=C.GOAL_POSITION.astype(np.float32),
        goal_walls=C.create_walls(C.GOAL_POSITION, gd).astype(np.float32),
        block_spheres=np.stack([C.create_tetris_spheres(s)
                                for s in shapes]).astype(np.float32),
        block_poses=np.array([[x, y, bz, yaw] for x, y, yaw
                              in C.tetris_spawn_poses(num_blocks)],
                             np.float32),
        block_z=np.float32(bz),
        table_dims=C.TABLE_DIMS.astype(np.float32),
        table_pose=C.TABLE_POSE.astype(np.float32),
    )


def tetris_cost(block_poses, num_blocks=None, extra_cells=None):
    """SPaSM's packing cost."""
    if extra_cells is None:
        extra_cells = EXTRA_GOAL_CELLS
    return C.tetris_cost(block_poses, num_blocks, extra_cells)


def tetris_block_spheres(num_blocks, block_poses, extra_cells=None):
    """(num_blocks, 6, 4) world-frame spheres at `block_poses`."""
    poses = np.asarray(block_poses, float)
    shapes = C.TETRIS_SHAPES[:num_blocks]
    return np.stack([C.block_pose_to_spheres(
        C.create_tetris_spheres(shapes[k]), poses[k]) for k in range(num_blocks)])


def separate_packing(poses, num_blocks=None, extra_cells=None,
                     target_mm=8.0, w_move=0.01, w_yaw=0.05, restarts=8,
                     seed=0, verbose=False):
    """Push a solved packing apart until no two blocks overlap -> (poses, report).

    SPaSM's packing cost cannot do this itself: `sphere_sphere_penetration`
    carries a 10 mm dead zone, so once two blocks are within a centimetre of
    each other the objective is flat and the optimizer has no reason to separate
    them.  With 8 blocks in the goal that band is always occupied by some pair,
    which is why 16 restarts of `solve_packing` never beat -1.5 mm clearance.

    This is a REFINEMENT, not a packer: it keeps the arrangement the solver
    found (slot order, which block sits where) and only nudges each block in
    (x, y, yaw) to buy `target_mm` of clearance, penalising displacement so the
    result stays the same packing.  z is fixed -- every block rests on the floor.

    The penetration terms here have NO dead zone, so unlike SPaSM's cost they
    keep pulling until the spheres are genuinely apart.

    `target_mm` is the clearance asked for and `w_move` how hard displacement is
    resisted; the two trade off directly.  The defaults were swept on the 8-block
    packing, which the goal has ~12 mm of unused wall slack to absorb:

        w_move  target   clearance   max move
          1.0     3 mm     1.62 mm     1.7 mm
          0.01    8 mm     7.95 mm     4.7 mm   <- default
          0.001  12 mm    11.50 mm    17.9 mm

    8 mm at under 5 mm of movement matches the 5-block packing's native 8.83 mm
    without visibly rearranging the solution; buying more costs a centimetre of
    displacement, which is a different packing, not this one separated.
    """
    from scipy.optimize import minimize

    if extra_cells is None:
        extra_cells = EXTRA_GOAL_CELLS
    p0 = np.asarray(poses, float).copy()
    n = p0.shape[0] if num_blocks is None else num_blocks
    p0 = p0[:n]
    g = tetris_geometry(n, extra_cells)
    gc, gd = np.asarray(g["goal_position"], float), np.asarray(g["goal_dims"], float)
    shapes = C.TETRIS_SHAPES[:n]
    local = [C.create_tetris_spheres(s) for s in shapes]
    tgt = target_mm / 1000.0

    def spheres_of(v):
        return [C.block_pose_to_spheres(
            local[k], [v[3 * k], v[3 * k + 1], p0[k, 2], v[3 * k + 2]])
            for k in range(n)]

    def obj(v):
        sp = spheres_of(v)
        cost = 0.0
        for a in range(n):
            A = sp[a][:4]
            for b in range(a + 1, n):
                B = sp[b][:4]
                d = np.linalg.norm(A[:, None, :3] - B[None, :, :3], axis=-1)
                gap = d - (A[:, None, 3] + B[None, :, 3])
                cost += np.sum(np.maximum(0.0, tgt - gap) ** 2)
            # Walls: same target margin, both in-plane axes.
            for d_ in (0, 1):
                lo = (gc[d_] - gd[d_] / 2) - (A[:, d_] - A[:, 3])
                hi = (A[:, d_] + A[:, 3]) - (gc[d_] + gd[d_] / 2)
                cost += np.sum(np.maximum(0.0, lo + tgt) ** 2)
                cost += np.sum(np.maximum(0.0, hi + tgt) ** 2)
        dv = v.reshape(n, 3) - np.stack(
            [p0[:, 0], p0[:, 1], p0[:, 3]], axis=1)
        cost += w_move * np.sum(dv[:, :2] ** 2) + w_yaw * np.sum(dv[:, 2] ** 2)
        return cost * 1e3

    rng = np.random.default_rng(seed)
    v0 = np.stack([p0[:, 0], p0[:, 1], p0[:, 3]], axis=1).ravel()
    best, best_mc = None, -np.inf
    for t in range(restarts):
        start = v0 if t == 0 else v0 + rng.normal(
            0.0, [0.004, 0.004, 0.03] * n)
        res = minimize(obj, start, method="L-BFGS-B",
                       options=dict(maxiter=2000, maxfun=20000))
        cand = p0.copy()
        w = res.x.reshape(n, 3)
        cand[:, 0], cand[:, 1], cand[:, 3] = w[:, 0], w[:, 1], w[:, 2]
        rep = packing_report(cand, n, extra_cells)
        mc = min(rep["min_block_clearance_mm"], rep["min_wall_margin_mm"])
        if verbose:
            print(f"  [separate] restart {t}: clearance "
                  f"{rep['min_block_clearance_mm']:.2f} mm  wall "
                  f"{rep['min_wall_margin_mm']:.2f} mm"
                  f"{'  <- best' if mc > best_mc else ''}", flush=True)
        if mc > best_mc:
            best, best_mc = cand, mc
    return best.astype(np.float32), packing_report(best, n, extra_cells)


def _packing_path(num_blocks, extra_cells=None):
    if extra_cells is None:
        extra_cells = EXTRA_GOAL_CELLS
    return RESOURCES / f"tetris_packing_{num_blocks}b_{extra_cells}x.npy"


def tetris_skeleton(num_blocks=3, extra_cells=None):
    """The packing skeleton, (num_blocks, 4) as [x, y, z, yaw].

    THIS IS THE TASK: tetris is "put block k at the pose the packing solver
    chose for it", not "put a block somewhere inside the goal".  Two of the
    three slots are rotated by more than 2 rad, which a position-only place
    target cannot express.

    Produced by SPaSM's own packing solver against the widened goal and stored
    here, so the task is reproducible without that checkout;
    `iosp.checks.spasm_trajopt --solve-packing` regenerates it.
    """
    p = _packing_path(num_blocks, extra_cells)
    if not p.exists():
        raise SystemExit(f"no packing skeleton at {p}")
    poses = np.load(p).astype(np.float32)
    if poses.shape[0] != num_blocks:
        raise SystemExit(f"{p} holds {poses.shape[0]} blocks, not {num_blocks}")
    return poses


# How much interpenetration counts as "not overlapping".  MuJoCo's contact
# softness lets resting bodies settle a fraction of a millimetre into each
# other, so demanding a strictly non-negative clearance would fail a perfectly
# good packing on solver compliance.  The same tolerance applies to the WALLS:
# a block pushed 0.1 mm past the line by its neighbours is packed, not spilled.
OVERLAP_TOL_MM = 2.0
WALL_TOL_MM = 2.0
TILT_TOL_DEG = 20.0


def packing_report(poses, num_blocks=None, extra_cells=None, slots=None):
    """Geometric truth about a packing: wall margin, overlap, SPaSM's cost.

    `slots` restricts the check to the blocks that are SUPPOSED to be in the
    box.  A single-block demonstration is one step of the packing sequence, so
    the blocks whose turn has not come are still on the table, and scoring them
    against the goal walls would fail every partial packing by construction.
    The goal geometry still comes from the full `num_blocks`.
    """
    n = len(poses) if num_blocks is None else num_blocks
    g = tetris_geometry(n, extra_cells)
    gd = np.asarray(g["goal_dims"], float)
    gc = np.asarray(g["goal_position"], float)
    sp = tetris_block_spheres(n, poses, extra_cells)
    if slots is not None:
        sp = sp[list(slots)]
    n_scored = len(sp)

    wall_margin = np.inf
    for s_ in sp:
        for d in (0, 1):
            wall_margin = min(
                wall_margin,
                float((s_[:, d] - s_[:, 3] - (gc[d] - gd[d] / 2)).min()),
                float(((gc[d] + gd[d] / 2) - s_[:, d] - s_[:, 3]).min()))

    worst = np.inf
    for a in range(n_scored):
        for b in range(a + 1, n_scored):
            A, B = sp[a][:4], sp[b][:4]
            d3 = np.linalg.norm(A[:, None, :3] - B[None, :, :3], axis=-1)
            worst = min(worst, float((d3 - (A[:, None, 3] + B[None, :, 3])).min()))

    return dict(
        inside_walls=bool(wall_margin >= -WALL_TOL_MM / 1000.0),
        min_wall_margin_mm=wall_margin * 1000,
        # With fewer than two blocks scored there is no pair to measure.
        min_block_clearance_mm=(None if not np.isfinite(worst) else worst * 1000),
        goal_dims=gd.tolist(),
        spasm_cost=tetris_cost(poses, n, extra_cells))


def packing_success(poses, num_blocks=None, extra_cells=None, tilt_deg=None,
                    slots=None):
    """Did the executed packing achieve the TASK?  -> (ok, dict).

    Every block in the goal walls, none overlapping another, none tipped over.
    `tilt_deg` is per-block tilt, which only an executed rollout can produce --
    a plan has no way to tip a block, and `matrix_to_xyzyaw` would discard it
    anyway.
    """
    rep = packing_report(poses, num_blocks, extra_cells, slots)
    mc = rep["min_block_clearance_mm"]
    no_overlap = True if mc is None else mc >= -OVERLAP_TOL_MM
    tilt = np.asarray(tilt_deg, float) if tilt_deg is not None else None
    if tilt is not None and slots is not None:
        tilt = tilt[list(slots)]
    upright = True if tilt is None else bool(np.max(tilt) <= TILT_TOL_DEG)
    ok = bool(rep["inside_walls"]) and no_overlap and upright
    return ok, dict(ok=ok, inside_walls=rep["inside_walls"],
                    no_overlap=no_overlap, upright=upright,
                    min_wall_margin_mm=rep["min_wall_margin_mm"],
                    min_block_clearance_mm=mc,
                    max_tilt_deg=(None if tilt is None else float(np.max(tilt))),
                    spasm_cost=rep["spasm_cost"])


# ---------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------

def tower_cost(block_poses, init_poses=None, num_blocks=10, num_obs=10):
    """SPaSM's stacking cost: z-height + stability + penetrations."""
    return C.tower_cost(block_poses, init_poses, num_blocks)


# [m] the solved stack is TRANSLATED this far before anything uses it.
#
# SPaSM solved where to put each block relative to the others; it never had to
# reach them.  Its stack lands 0.25 m from the robot's base, which is close
# enough in that a straight-down gripper has to fold the wrist past joint 5's
# limit to place there: the level-2 placement wanted 221.0 deg against the
# URDF's 219 and the executing Panda's 215, leaving 17 mm of unavoidable
# placement error no amount of IK seeding removed.  Sliding the whole stack
# 5 cm further out costs nothing structurally -- the levels keep their relative
# offsets, so it is still SPaSM's solution -- and takes the worst of ALL TEN
# levels from 16.9 mm to 0.1 mm at the placement and 39.6 mm to 0.3 mm at its
# standoff.  Measured over a 5x5 grid of translations; +5 cm in x is the
# smallest one that clears every level, and it keeps the stack clear of the
# obstacle spheres (+26 mm) and well inside the table.
TOWER_STACK_OFFSET_XY = np.array([0.05, 0.0], dtype=np.float32)


def tower_skeleton_raw(num_blocks=10):
    """SPaSM's solved stack exactly as saved, (num_blocks, 4).

    For comparing against SPaSM's own reported cost.  Everything that has to
    REACH the stack wants `tower_skeleton` instead.
    """
    p = RESOURCES / f"tower_stack_{num_blocks}b.npy"
    if not p.exists():
        raise SystemExit(f"no tower stack at {p}")
    return np.load(p).astype(np.float32)


def tower_skeleton(num_blocks=10):
    """SPaSM's solved stack, moved to where the arm can actually reach it.

    See `TOWER_STACK_OFFSET_XY`.  This is the single point the stack's location
    is defined: the planner's place targets, the pre-placed blocks in the
    MuJoCo scene and the success check all read through here, so they cannot
    disagree about where the tower is.
    """
    s = tower_skeleton_raw(num_blocks)
    s[:, :2] += TOWER_STACK_OFFSET_XY
    return s


def tower_init_state(num_blocks=10):
    """The tower spawn grid."""
    return C.tower_spawn_poses(num_blocks).astype(np.float32)


TOWER_Z_TOL = 0.015
TOWER_SUPPORT_TOL = 0.030


def tower_success(poses, tilt_deg=None, num_blocks=10, num_obs=10):
    """Did the executed stack achieve the TASK?  -> (ok, dict).

    "Stack the blocks": every block at its own level's height, each supported
    by the one beneath it, none tipped over.
    """
    poses = np.asarray(poses, float)
    n = poses.shape[0]
    h = float(C.BLOCK_DIMS[2])

    dz = np.abs(poses[:, 2] - (np.arange(n) * h + h / 2.0))
    at_height = bool((dz <= TOWER_Z_TOL).all())
    offs = np.linalg.norm(poses[1:, :2] - poses[:-1, :2], axis=1)
    supported = bool((offs <= TOWER_SUPPORT_TOL).all()) if n > 1 else True
    upright = True if tilt_deg is None else bool(
        np.max(np.asarray(tilt_deg, float)) <= TILT_TOL_DEG)

    ok = at_height and supported and upright
    return ok, dict(ok=ok, at_height=at_height, supported=supported,
                    upright=upright, max_z_err_mm=float(dz.max()) * 1000,
                    max_stack_offset_mm=(float(offs.max()) * 1000 if n > 1 else 0.0),
                    max_tilt_deg=(None if tilt_deg is None
                                  else float(np.max(np.asarray(tilt_deg, float)))),
                    spasm_cost=tower_cost(poses, tower_init_state(num_blocks),
                                          num_blocks))


def reference_costs(num_blocks_tetris=3, num_blocks_tower=10):
    """What the planners' OWN solutions score -- the bar an execution clears."""
    tet = tetris_skeleton(num_blocks_tetris)
    tow = tower_skeleton(num_blocks_tower)
    return dict(
        tetris=dict(cost=tetris_cost(tet, num_blocks_tetris),
                    thresh=TETRIS_COST_THRESH.get(num_blocks_tetris)),
        tower=dict(cost=tower_cost(tow, tower_init_state(num_blocks_tower),
                                   num_blocks_tower), thresh=None),
    )


if __name__ == "__main__":
    import json
    print(json.dumps(reference_costs(), indent=2, default=float))
