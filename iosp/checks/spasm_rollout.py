"""Build the SPaSM tetris scene in MuJoCo and roll a plan through it.

All three blocks are present as free bodies from the start, at their SPaSM
spawn poses, and stay physical for the whole rollout -- so a block already
packed into the goal can be knocked out of it by the arm coming back for the
next one, which is the failure mode the kinematic check cannot see.

The scene's geometry is SPaSM's, via `iosp.model.spasm_tasks`: goal walls at
`sim.goal_walls`, blocks from `create_tetris_spheres`, resting height
`sim.block_z`.
"""
from __future__ import annotations

import numpy as np

from iosp.viz import mj_rollout as R
from iosp.viz import mj_scene as M

# Arm links whose approach trajectories sweep through the carried object (the
# planner has no hand model).  These are excluded from contact with the target.
# The FINGERS are NOT excluded: the gripper must physically close on the object
# for a physics-only grasp to work.
ARM_LINK_BODIES = [f"link{i}" for i in range(8)]
GRIPPER_BODIES = ARM_LINK_BODIES


class _TowerGeomView:
    """Tower geometry with `TowerSimulation`'s attribute names."""

    def __init__(self):
        from iosp.model import spasm_costs as SC
        self.block_dims = SC.BLOCK_DIMS
        self.block_height = float(SC.BLOCK_DIMS[2])
        self.table_dims = SC.TOWER_TABLE_DIMS
        self.table_pose = SC.TOWER_TABLE_POSE
        self.obstacle_poses = SC.TOWER_OBSTACLE_POSES
        self.obstacle_radii = SC.TOWER_OBSTACLE_RADII


class _GeomView:
    """The cached geometry dict with `Simulation`'s attribute names."""

    def __init__(self, g):
        self.goal_dims = g["goal_dims"]
        self.goal_position = g["goal_position"]
        self.goal_walls = g["goal_walls"]
        self.block_spheres = g["block_spheres"]
        self.block_poses_original = list(g["block_poses"])
        self.block_z = float(g["block_z"])
        self.table_dims = g["table_dims"]
        self.table_pose = g["table_pose"]
# Grasp physics for the tetromino free bodies.  These are the contact-rollout's
# numbers, NOT the planner's: the planner never sees mass or friction.
#   friction  (slide, torsional, rolling) -- cranked well past MuJoCo's defaults
#             [2.0, 0.005, 0.0001]; the 0.005 torsional is the value measured to
#             let an object roll straight through a closed grip.
#   mass      0.05 kg made the carry's inertial load comparable to the grip
#             force the Panda's tendon actuator can supply (gain 0.0157 N/unit).
#   condim    6 so torsional AND rolling friction are actually integrated;
#             condim 4 ignores the rolling term entirely.
# Measured on scene 0 (block settle vs its skeleton slot), cube handles:
#   torsional/rolling  1.0/0.05 -> 25.9 deg final tilt, 17.0 cm error
#                      5.0/0.50 ->  3.9 deg,            10.2 cm
#                     10.0/2.00 ->  5.4 deg,            10.2 cm  (no further gain)
# Mass 0.02 vs 0.008 was within noise, so mass stays at the lighter-but-sane
# 0.02 kg rather than being tuned to whatever flatters one scene.
BLOCK_FRICTION = (4.0, 5.0, 0.5)
BLOCK_MASS = 0.02
BLOCK_CONDIM = 6

WALL_PAINT_T = 0.001    # [m] thickness of the painted goal outline

BLOCK_RGBA = [(0.90, 0.35, 0.35, 1.0), (1.0, 0.72, 0.35, 1.0),
              (0.45, 0.55, 0.90, 1.0), (0.55, 0.80, 0.45, 1.0),
              (0.75, 0.55, 0.85, 1.0)]


def _yaw_quat(yaw):
    """Body quaternion for a SPaSM block pose whose yaw is `yaw`.

    SPaSM's pose convention is 180 deg about x, THEN the yaw about z -- but the
    x-flip is already baked into the geom offsets below (they are written as
    `(x, -y, -z)`), so the BODY only carries the yaw.  Applying the full
    `yaw_to_quat_xyz` here would flip the block twice and stand it on its head.
    """
    return np.array([np.cos(float(yaw) / 2.0), 0.0, 0.0, np.sin(float(yaw) / 2.0)])


def build_scene(num_blocks=3, extra_cells=None, robot="menagerie",
                preplace=None, walls="solid", spawn_poses=None):
    """-> (world, sim, block_names). Geometry entirely from SPaSM.

    `preplace` is a list of slot indices to START already packed at their
    skeleton poses.  A single-block demonstration is scene `k` of the packing
    SEQUENCE, so the blocks below it in that sequence are already in the box
    and are obstacles the arm has to work around -- which is exactly what makes
    slot 2 harder than slot 0.
    """
    from iosp.model import spasm_tasks as ST

    geom = ST.tetris_geometry(num_blocks, extra_cells)
    sim = _GeomView(geom)
    b = M.WorldBuilder(n_scenes=1, robot=robot)

    b.box("table", tuple(np.asarray(geom["table_pose"], float)),
          tuple(np.asarray(geom["table_dims"], float)), M.TABLE_RGBA)
    gd = np.asarray(geom["goal_dims"], float)
    gp = np.asarray(geom["goal_position"], float)
    b.box("goal_floor", tuple(gp), (gd[0], gd[1], gd[2]), (1.0, 1.0, 1.0, 1.0))
    # The goal region can be SOLID walls or a painted OUTLINE on the floor.
    #
    # The walls are not in the trajopt problem -- nothing optimises the carried
    # block for clearance against them -- so with the cube collision geometry
    # they mostly serve to catch a block that would otherwise have dropped into
    # its slot, and the executed result then measures wall collisions rather
    # than packing.  "outline" paints the same footprint flat on the goal floor
    # with collision off, so a block that arrives over its slot simply lands in
    # it.  "solid" restores SPaSM's physical walls.
    if walls not in ("outline", "solid"):
        raise ValueError(f"walls must be 'outline' or 'solid', got {walls!r}")
    wall_z_top = float(gp[2]) + float(gd[2]) / 2.0
    for j, w in enumerate(np.asarray(geom["goal_walls"], float)):
        x1, y1, z1, x2, y2, z2 = [float(v) for v in w]
        if walls == "solid":
            b.box(f"wall{j}", ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2),
                  (x2 - x1, y2 - y1, z2 - z1), (0.86, 0.855, 0.82, 1.0))
        else:
            # Same x/y footprint, flattened onto the goal floor and made
            # non-colliding: a visual guide, not an obstacle.
            b.box(f"wall{j}_outline",
                  ((x1 + x2) / 2, (y1 + y2) / 2, wall_z_top + WALL_PAINT_T / 2),
                  (x2 - x1, y2 - y1, WALL_PAINT_T), (0.55, 0.52, 0.48, 1.0),
                  collide=False)

    # Each block is a free body whose geoms carry SPaSM's 180-degree x flip, so
    # the body's own quaternion is the yaw alone -- the same convention
    # `MjWorld.body_xyzyaw` reads back.
    names = []
    spheres = np.asarray(geom["block_spheres"])               # (n, 6, 4)
    for k in range(num_blocks):
        # CUBES, not spheres -- every geom, body and handle alike.
        #
        # A sphere between two flat finger pads is a two-point contact, and the
        # tetromino's mass hangs ~6 cm below the handle, so it escapes by
        # ROTATING out of the grip no matter how high the friction goes.  A cube
        # gives the pads a flat face, which resists that rotation directly.
        #
        # Cubing only the HANDLE and leaving the body as spheres looks like the
        # tidier fix -- it would keep this model identical to the planner's --
        # but it was measured and fails outright (35-67 cm settle error, vs
        # 0.6-1.1 cm here): a tetromino resting on four spheres rolls, at the
        # pick and again on landing.  The body cubes are load-bearing for
        # stability, not just for the grasp.
        #
        # The cost is a real mismatch against the planner, whose collision model
        # is `block_spheres`: a cube of side 2r matches the sphere across its
        # faces but sweeps r*sqrt(2) at an edge (+12.4 mm) and r*sqrt(3) at a
        # corner (+22.0 mm), which at the skeleton's yaws is +2.8 to +10.1 mm of
        # in-plane half-width.  The executed packing is that much tighter than
        # the one that was solved for, which is why the passing wall margin is
        # only a few mm.
        geoms = [("box", (2 * float(r),) * 3, (float(x), float(-y), float(-z)),
                  BLOCK_RGBA[k % len(BLOCK_RGBA)])
                 for x, y, z, r in spheres[k]]
        pose = np.asarray(sim.block_poses_original[k], float)
        nm = f"block{k}"
        b.free_body(nm, geoms, pos=pose[:3], collide=True,
                    friction=BLOCK_FRICTION, mass=BLOCK_MASS, condim=BLOCK_CONDIM)
        # The planner's arm has no hand; excluding gripper-vs-target contact is
        # what every pick-and-place stack does. Blocks still collide with each
        # other, the walls and the table.
        for g in GRIPPER_BODIES:
            b.exclude(f"s0_{g}", nm)
        names.append(nm)

    world = b.compile()
    # The grasp here is FRICTIONAL (`run_events`, no weld), so the fingers have
    # to hold the handle by contact alone.  `free_body`'s default torsional
    # friction is 0.005, the value measured to let an object roll straight
    # through a closed grip; `build_pickplace_scene` already corrects it.
    # The blocks already carry BLOCK_FRICTION; this raises the FINGER PADS to
    # match, so the contact pair is high-friction on both sides.
    world.set_grasp_friction(torsional=BLOCK_FRICTION[1])
    skel = ST.tetris_skeleton(num_blocks, extra_cells)
    # `spawn_poses` overrides SPaSM's fixed spawns with the ones the SAMPLER
    # drew (see `tetris.sample_tetris_scenes`).  Without it a randomised scene
    # sends the arm to a pick pose no block occupies.
    spawns = (np.asarray(geom["block_poses"], float) if spawn_poses is None
              else np.asarray(spawn_poses, float))
    if spawns.shape[0] < num_blocks:
        raise ValueError(f"spawn_poses has {spawns.shape[0]} rows, "
                         f"need {num_blocks}")
    for k, nm in enumerate(names):
        pose = (np.asarray(skel[k], float) if preplace and k in preplace
                else spawns[k])
        world.set_free(nm, pose[:3], _yaw_quat(pose[3]))
    world.forward()
    return world, sim, names


def rollout_and_score(q, events, num_blocks=3, extra_cells=None, skeleton=None,
                      settle=1.5, dwell=0.6, save=None, robot="menagerie",
                      progress=True, walls="solid", spawn_poses=None):
    """Execute `q` with `events` = [(grasp_row, release_row, block_idx)]."""
    from iosp.model import spasm_tasks as ST

    world, sim, names = build_scene(num_blocks, extra_cells, robot, walls=walls,
                                    spawn_poses=spawn_poses)
    if skeleton is None:
        skeleton = ST.tetris_skeleton(num_blocks, extra_cells)

    ro = R.run_events(world, q, [(g, r, names[k]) for g, r, k in events],
                      settle=settle, dwell=dwell, progress=progress)

    world.data.qpos[:] = ro.qpos[-1]
    world.forward()
    achieved = np.stack([world.body_xyzyaw(nm) for nm in names])

    tilt = np.array([world.body_tilt_deg(nm) for nm in names])
    ok, verdict = ST.packing_success(achieved, num_blocks, extra_cells, tilt)
    planned_cost = ST.tetris_cost(skeleton, num_blocks, extra_cells)
    achieved_cost = verdict["spasm_cost"]
    thresh = ST.TETRIS_COST_THRESH.get(num_blocks)
    worst, mean = ro.tracking_mm_deg(0)

    print(f"\n===== SPaSM trajopt, EXECUTED ({num_blocks} blocks) =====")
    print(f"  TASK {'SUCCESS' if ok else 'FAIL'}: "
          f"in the walls {'yes' if verdict['inside_walls'] else 'NO'} "
          f"(margin {verdict['min_wall_margin_mm']:+.1f} mm), "
          f"no overlap {'yes' if verdict['no_overlap'] else 'NO'} "
          f"(min clearance {verdict['min_block_clearance_mm']:.1f} mm), "
          f"upright {'yes' if verdict['upright'] else 'NO'} "
          f"(max tilt {verdict['max_tilt_deg']:.1f} deg)")
    print(f"  arm tracking error: max {worst:.1f} deg (mean {mean:.1f})")
    for k in range(num_blocks):
        d = np.linalg.norm(achieved[k, :2] - skeleton[k, :2]) * 100
        dy = abs(((achieved[k, 3] - skeleton[k, 3] + np.pi) % (2 * np.pi)) - np.pi)
        print(f"    block {k}: {d:5.2f} cm and {np.degrees(dy):5.1f} deg from plan"
              f"   z {achieved[k, 2]:.3f}   tilt {tilt[k]:4.1f} deg")
    # Reported, not decisive -- see `spasm_tasks.packing_success` for why.
    _c = ("unavailable" if achieved_cost is None else f"{achieved_cost:.4f}")
    print(f"  [diagnostic] SPaSM packing cost {_c} "
          f"(planned {planned_cost:.4f}, its threshold {thresh}) -- "
          f"a planning objective, not the task test")

    if save:
        np.savez(save, q=q, events=np.asarray(events), skeleton=skeleton,
                 achieved=achieved, qpos=ro.qpos, frame_row=ro.frame_row,
                 dt=ro.dt, num_blocks=num_blocks, tilt_deg=tilt, ok=ok)
        print(f"  wrote {save}")
    return dict(world=world, rollout=ro, achieved=achieved, ok=ok,
                verdict=verdict, achieved_cost=achieved_cost,
                planned_cost=planned_cost, thresh=thresh, names=names,
                tilt_deg=tilt)


# ---------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------

TOWER_RGBA = [(0.99,0.25,0.32,1),(1.0,0.42,0.42,1),(0.99,0.49,0.01,1),
              (1.0,0.74,0.09,1),(0.66,0.90,0.03,1),(0.40,0.84,0.24,1),
              (0.22,0.76,0.53,1),(0.05,0.83,0.68,1),(0.01,0.80,0.82,1),
              (0.19,0.71,0.91,1)]


def build_tower_scene(num_blocks=10, num_obs=10, robot="menagerie",
                      collide_obstacles=False, preplace=None):
    """The SPaSM tower scene in MuJoCo: table, ten cubes, and the obstacles.

    `collide_obstacles` is off by default.  SPaSM's obstacles are CLEARANCE
    spheres for its planning cost, not physical bodies -- one has radius 0.3 at
    [0.25, -0.4, 0.0], which swallows part of the table and the robot's own
    workspace.  Making them solid would have the arm fighting a soft penalty as
    if it were a wall.  They are still drawn, and still shape the plan.

    `preplace` is a list of block indices to START already stacked at their
    skeleton poses.  A single-level demonstration is stack level k, so levels
    0..k-1 are already placed and are obstacles the arm has to work around.
    """
    from iosp.model import spasm_tasks as ST

    from iosp.model import spasm_costs as SC
    sim = _TowerGeomView()
    b = M.WorldBuilder(n_scenes=1, robot=robot)
    b.box("table", tuple(SC.TOWER_TABLE_POSE), tuple(SC.TOWER_TABLE_DIMS),
          M.TABLE_RGBA)

    for j, (c, r) in enumerate(zip(SC.TOWER_OBSTACLE_POSES,
                                   SC.TOWER_OBSTACLE_RADII)):
        if abs(c[1]) > 10:            # SPaSM parks one obstacle at y = -100.5
            continue
        b.sphere(f"obs{j}", tuple(c), float(r), M.OBSTACLE_RGBA,
                 collide=collide_obstacles)

    dims = tuple(np.asarray(SC.BLOCK_DIMS, float))
    init = ST.tower_init_state(num_blocks)
    names = []
    for k in range(num_blocks):
        nm = f"block{k}"
        b.free_body(nm, [("box", dims, (0, 0, 0), TOWER_RGBA[k % len(TOWER_RGBA)])],
                    pos=tuple(init[k, :3]), collide=True,
                    friction=BLOCK_FRICTION, mass=BLOCK_MASS, condim=BLOCK_CONDIM)
        for g in GRIPPER_BODIES:
            b.exclude(f"s0_{g}", nm)
        names.append(nm)

    world = b.compile()
    world.set_grasp_friction(torsional=BLOCK_FRICTION[1])
    skeleton = ST.tower_skeleton(num_blocks) if preplace else None
    for k, nm in enumerate(names):
        if preplace and k in preplace:
            pose = np.asarray(skeleton[k], float)
        else:
            pose = np.asarray(init[k], float)
        world.set_free(nm, pose[:3], _yaw_quat(pose[3]))
    world.forward()
    return world, sim, names


def rollout_and_score_tower(q, events, num_blocks=10, skeleton=None,
                            settle=2.0, dwell=0.6, save=None,
                            robot="menagerie", progress=True, gravcomp=True,
                            safety=None):
    from iosp.model import spasm_tasks as ST
    from iosp.model import spasm_costs as SC

    world, sim, names = build_tower_scene(num_blocks, robot=robot)
    if skeleton is None:
        skeleton = ST.tower_skeleton(num_blocks)

    ro = R.run_events(world, q, [(g, r, names[k]) for g, r, k in events],
                      settle=settle, dwell=dwell, progress=progress,
                      gravcomp=gravcomp,
                      grasp_half_width=float(SC.BLOCK_DIMS[0]) / 2 - 0.01,
                      **({} if safety is None else {'safety': safety}))

    world.data.qpos[:] = ro.qpos[-1]
    world.forward()
    achieved = np.stack([world.body_xyzyaw(nm) for nm in names])
    tilt = np.array([world.body_tilt_deg(nm) for nm in names])
    ok, v = ST.tower_success(achieved, tilt, num_blocks)
    planned_cost = ST.tower_cost(skeleton, ST.tower_init_state(num_blocks),
                                 num_blocks)
    worst, mean = ro.tracking_mm_deg(0)

    print(f"\n===== SPaSM tower trajopt, EXECUTED ({num_blocks} blocks) =====")
    print(f"  TASK {'SUCCESS' if ok else 'FAIL'}: "
          f"at height {'yes' if v['at_height'] else 'NO'} "
          f"(worst {v['max_z_err_mm']:.0f} mm), "
          f"supported {'yes' if v['supported'] else 'NO'} "
          f"(worst offset {v['max_stack_offset_mm']:.0f} mm), "
          f"upright {'yes' if v['upright'] else 'NO'} "
          f"(max tilt {v['max_tilt_deg']:.1f} deg)")
    print(f"  arm tracking error: max {worst:.1f} deg (mean {mean:.1f})")
    h = float(sim.block_height)
    for k in range(num_blocks):
        print(f"    level {k}: z {achieved[k,2]:.3f} (target {k*h+h/2:.3f}), "
              f"xy {np.linalg.norm(achieved[k,:2]-skeleton[k,:2])*100:5.2f} cm from plan, "
              f"tilt {tilt[k]:5.1f} deg")
    print(f"  [diagnostic] SPaSM stacking cost {v['spasm_cost']:.4f} "
          f"(planned {planned_cost:.4f}) -- a planning objective, not the task test")

    if save:
        np.savez(save, q=q, events=np.asarray(events), skeleton=skeleton,
                 achieved=achieved, qpos=ro.qpos, frame_row=ro.frame_row,
                 dt=ro.dt, num_blocks=num_blocks, tilt_deg=tilt, ok=ok)
        print(f"  wrote {save}")
    return dict(world=world, rollout=ro, achieved=achieved, ok=ok, verdict=v,
                planned_cost=planned_cost, names=names, tilt_deg=tilt)


# ---------------------------------------------------------------------------
# Pick-and-place
# ---------------------------------------------------------------------------

PP_TABLE_CENTER_XY = (0.55, 0.0)
PP_TABLE_HALF_XY = (0.35, 0.45)
PP_TABLE_HEIGHT = 0.30
PP_TABLE_THICKNESS = 0.02
PP_OBJ_SIZE = 0.05
PP_OBJ_RGBA = (0.95, 0.55, 0.15, 1.0)
PP_REACH_TOL_M = 0.05
PP_TILT_TOL_DEG = 25.0

PP_BUCKET_CENTER_XY = (0.62, 0.20)
PP_BUCKET_INNER_RADIUS = 0.075
PP_BUCKET_WALL_THICKNESS = 0.010
PP_BUCKET_WALL_HEIGHT = 0.12
PP_BUCKET_FLOOR_THICKNESS = 0.010
PP_BUCKET_N_WALLS = 8
PP_BUCKET_RGBA = (0.45, 0.35, 0.25, 1.0)


def _add_bucket(b, center_xy, table_top_z,
                inner_radius=PP_BUCKET_INNER_RADIUS,
                wall_thickness=PP_BUCKET_WALL_THICKNESS,
                wall_height=PP_BUCKET_WALL_HEIGHT,
                floor_thickness=PP_BUCKET_FLOOR_THICKNESS,
                n_walls=PP_BUCKET_N_WALLS):
    """Add an n-gon bucket (floor + wall panels) to the world builder."""
    import math
    cx, cy = float(center_xy[0]), float(center_xy[1])
    tz = float(table_top_z)

    r_out = (inner_radius + wall_thickness) / math.cos(math.pi / n_walls)
    floor_pos = (cx, cy, tz + 0.5 * floor_thickness)
    floor_dims = (2 * r_out, 2 * r_out, floor_thickness)
    b.box("bucket_floor", floor_pos, floor_dims, PP_BUCKET_RGBA)

    r_mid = inner_radius + 0.5 * wall_thickness
    half_width = r_mid * math.tan(math.pi / n_walls)
    for i in range(n_walls):
        theta = 2.0 * math.pi * i / n_walls
        pos = (cx + r_mid * math.cos(theta),
               cy + r_mid * math.sin(theta),
               tz + floor_thickness + 0.5 * wall_height)
        quat = (math.cos(0.5 * theta), 0.0, 0.0, math.sin(0.5 * theta))
        dims = (wall_thickness, 2 * half_width, wall_height)
        b.box(f"bucket_wall_{i}", pos, dims, PP_BUCKET_RGBA, quat=quat)


def build_pickplace_scene(pick_pos, place_pos,
                          bucket_center_xy=PP_BUCKET_CENTER_XY,
                          bucket_inner_radius=PP_BUCKET_INNER_RADIUS,
                          bucket_wall_height=PP_BUCKET_WALL_HEIGHT,
                          bucket_wall_thickness=PP_BUCKET_WALL_THICKNESS,
                          bucket_floor_thickness=PP_BUCKET_FLOOR_THICKNESS,
                          table_top_z=PP_TABLE_HEIGHT,
                          robot="menagerie"):
    """A single pick-and-place scene in MuJoCo: table, bucket, carried box.

    `pick_pos`/`place_pos` are the OBJECT poses (the convention from
    `iosp.model.pickplace`); the EE sits a standoff above them.
    The cube spawns on the table at `pick_pos` xy; the bucket sits at
    `place_pos` xy.  Both are placed on the table surface regardless of the
    z coordinate in the planning scene.
    """
    b = M.WorldBuilder(n_scenes=1, robot=robot)
    table_pos = (PP_TABLE_CENTER_XY[0], PP_TABLE_CENTER_XY[1],
                 table_top_z - 0.5 * PP_TABLE_THICKNESS)
    table_dims = (2 * PP_TABLE_HALF_XY[0], 2 * PP_TABLE_HALF_XY[1],
                  PP_TABLE_THICKNESS)
    b.box("table", table_pos, table_dims, M.TABLE_RGBA)

    place = np.asarray(place_pos, float)
    _add_bucket(b, tuple(place[:2]), table_top_z,
                inner_radius=bucket_inner_radius,
                wall_thickness=bucket_wall_thickness,
                wall_height=bucket_wall_height,
                floor_thickness=bucket_floor_thickness)

    pick = np.asarray(pick_pos, float)
    cube_z = table_top_z + 0.5 * PP_OBJ_SIZE
    spawn = np.array([pick[0], pick[1], cube_z])
    nm = "block0"
    b.free_body(nm, [("box", (PP_OBJ_SIZE,) * 3, (0, 0, 0), PP_OBJ_RGBA)],
                pos=tuple(spawn), collide=True)
    for g in GRIPPER_BODIES:
        b.exclude(f"s0_{g}", nm)

    world = b.compile()
    world.set_grasp_friction()
    world.set_free(nm, spawn)
    world.forward()
    return world, [nm]


def pickplace_success(achieved_pos, target_pos, tilt_deg=None):
    """Did the executed pick-and-place succeed?  -> (ok, dict).

    Success: the object ended up close to the target and was not knocked over.
    For scenes whose place target is in free space (z=0.3), the object always
    falls, so `scored_on` says whether the release or the settled position was
    meaningful.
    """
    achieved = np.asarray(achieved_pos, float)
    target = np.asarray(target_pos, float)
    err = float(np.linalg.norm(achieved[:3] - target[:3]))
    err_xy = float(np.linalg.norm(achieved[:2] - target[:2]))
    upright = True if tilt_deg is None else bool(float(tilt_deg) <= PP_TILT_TOL_DEG)
    ok = err < PP_REACH_TOL_M and upright
    return ok, dict(ok=ok, err_mm=err * 1000, err_xy_mm=err_xy * 1000,
                    upright=upright,
                    tilt_deg=None if tilt_deg is None else float(tilt_deg))


def rollout_and_score_pickplace(q, grasp_row, release_row, pick_pos, place_pos,
                                bucket_center_xy=PP_BUCKET_CENTER_XY,
                                bucket_inner_radius=PP_BUCKET_INNER_RADIUS,
                                bucket_wall_height=PP_BUCKET_WALL_HEIGHT,
                                bucket_wall_thickness=PP_BUCKET_WALL_THICKNESS,
                                bucket_floor_thickness=PP_BUCKET_FLOOR_THICKNESS,
                                table_top_z=PP_TABLE_HEIGHT,
                                settle=1.5, dwell=0.6, save=None,
                                robot="menagerie", progress=True,
                                gravcomp=True):
    """Execute a pick-and-place plan through actuated dynamics."""
    world, names = build_pickplace_scene(
        pick_pos, place_pos,
        bucket_center_xy=bucket_center_xy,
        bucket_inner_radius=bucket_inner_radius,
        bucket_wall_height=bucket_wall_height,
        bucket_wall_thickness=bucket_wall_thickness,
        bucket_floor_thickness=bucket_floor_thickness,
        table_top_z=table_top_z, robot=robot)
    nm = names[0]
    events = [(grasp_row, release_row, nm)]

    ro = R.run_events(world, q, events, settle=settle, dwell=dwell,
                      progress=progress, gravcomp=gravcomp,
                      grasp_half_width=PP_OBJ_SIZE / 2 - 0.005)

    world.data.qpos[:] = ro.qpos[-1]
    world.forward()
    achieved = world.body_pos(nm)
    tilt = world.body_tilt_deg(nm)

    rf = min(ro.release_frame, ro.n_frames - 1)
    world.data.qpos[:] = ro.qpos[rf]
    world.forward()
    release_pos = world.body_pos(nm)

    # Score against where the cube SHOULD rest in the physics scene's bucket,
    # not the planning model's abstract place_pos (whose z is a planning
    # abstraction that doesn't match any physical surface).
    place = np.asarray(place_pos, float)
    target = np.array([place[0], place[1],
                       table_top_z + bucket_floor_thickness + PP_OBJ_SIZE / 2])
    ok, v = pickplace_success(achieved, target, tilt)

    worst, mean = ro.tracking_mm_deg(0)

    print(f"\n===== pick-and-place, EXECUTED =====")
    print(f"  TASK {'SUCCESS' if ok else 'FAIL'}: "
          f"error {v['err_mm']:.1f} mm"
          f" (xy {v['err_xy_mm']:.1f} mm)")
    print(f"  upright {'yes' if v['upright'] else 'NO'} "
          f"(tilt {v['tilt_deg']:.1f} deg)")
    print(f"  release error {pickplace_success(release_pos, target)[1]['err_mm']:.1f} mm"
          f"   settled error {v['err_mm']:.1f} mm"
          f"   (settled z {achieved[2]:.3f} m)")
    print(f"  arm tracking error: max {worst:.1f} deg (mean {mean:.1f})")

    if save:
        np.savez(save, q=q, grasp_row=grasp_row, release_row=release_row,
                 pick_pos=pick_pos, place_pos=place_pos,
                 achieved=achieved, release_pos=release_pos,
                 qpos=ro.qpos, frame_row=ro.frame_row, dt=ro.dt, ok=ok)
        print(f"  wrote {save}")
    return dict(world=world, rollout=ro, achieved=achieved,
                release_pos=release_pos, ok=ok, verdict=v,
                scored_on="settled", names=names, tilt_deg=tilt)
