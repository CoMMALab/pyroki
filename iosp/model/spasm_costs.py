"""SPaSM's tetris and tower task definitions, reimplemented in iosp.

Previously `iosp.model.spasm_tasks` imported the `../spasm` checkout to get
these, which kept a single source of truth but made iosp depend on a sibling
repo AND on its whole stack -- meshcat, xmltodict, its own kinematics module --
so `iosp.viz.tetris_viser`, which renders through mjviser and never touches any
of that, would not start in an environment missing meshcat.

Everything here is a direct port, kept deliberately literal so it can be read
against the original side by side:

    tetris   `spasm/solve.py`      sphere_sphere_penetration,
                                   sphere_wall_penetration, cost
    tower    `spasm/tower_solve.py` rotate, corners_rotated,
                                   block_block_penetration,
                                   spheres_blocks_collision, cost
    scenes   `spasm/tetris_env.py`, `spasm/tower_env.py`

The port is not trusted on inspection.  `iosp/tests/test_spasm_costs.py` pins
it against two numbers the original produced: the saved 3-block packing scores
0.269522 and the saved 10-block stack scores 0.153742, the latter matching the
`opt_errors` stored alongside it in `saved/tower.npz` to seven digits.  When a
SPaSM checkout is present the test also diffs the two implementations directly
on random poses.

Two quirks are faithfully preserved, because changing them would change what
the numbers mean:

  * the penetration terms have DEAD ZONES -- sphere-sphere charges until 10 mm
    of separation, walls until 20 mm -- so a cost above zero does not imply
    overlap, and for a goal only 0.15 m wide against a 0.12 m block the wall
    term can never reach zero at all;
  * `matrix_to_xyzyaw` (and hence every pose these costs see) keeps only
    x, y, z and yaw, DISCARDING roll and pitch, so a tilted placement is
    invisible to them.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Tetris geometry (`spasm/tetris_env.py`)
# ---------------------------------------------------------------------------

SPH_RADIUS = 0.03
WALL_HEIGHT = 0.045
WALL_THICKNESS = 0.015

TETRIS_SHAPE_COORDS = {
    "L": np.array([(0, 0, 0), (0, 1, 0), (0, -1, 0), (1, -1, 0)], float),
    "O": np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)], float),
}
TETRIS_SHAPES = ("O", "L", "O", "O", "O", "L", "L", "L")
TETRIS_SPAWN_XY = [(0.50, 0.35), (0.15, -0.6), (0.00, 0.6), (0.15, 0.6),
                   (0.00, -0.6), (0.50, -0.3), (0.50, -0.1), (0.50, 0.1)]

# SPaSM's own eight spawn slots.  Four of them (x = 0.00 and 0.15, y = +-0.6)
# sit BESIDE the goal, which is fine while the goal is short: at three or five
# blocks it reaches y = +-0.375 at most.  At EIGHT it is 16 cells long, i.e.
# y = +-0.555, and those four spawns land against the outer face of an end wall
# -- the block spawns interpenetrating the wall and is shoved ~5-11 cm in -x
# before the arm ever arrives, so the plan, which was solved for the nominal
# spawn, closes the gripper on empty air.  Measured drift at settle with nothing
# else in the scene: block1 -75.3 mm, block2 -26.8, block3 -56.2, block4 -44.9.
TETRIS_SPAWN_XY = [(0.50, 0.35), (0.15, -0.6), (0.00, 0.6), (0.15, 0.6),
                   (0.00, -0.6), (0.50, -0.3), (0.50, -0.1), (0.50, 0.1)]

# Eight blocks therefore spawn in ONE column clear of the longest goal: x = 0.50
# puts the nearest block face at 0.47, outside the goal's 0.375 plus its wall.
# That is the only free strip -- the goal reaches y = +-0.57 with its walls, and
# the band on its far side (x <= 0.21) is occupied by the arm's own base.  The
# column is spaced along y, which is an L's LONG axis (0.18 m against 0.12), so
# the spawns are yawed 90 deg to lie across the column; at yaw 0 the same
# spacing has neighbouring blocks overlapping and shoving each other on settle.
TETRIS_SPAWN_YAW_8 = np.pi / 2
TETRIS_SPAWN_XY_8 = [(0.50, y) for y in
                     (0.49, 0.35, 0.21, 0.07, -0.07, -0.21, -0.35, -0.49)]


def tetris_spawn_poses(num_blocks=3):
    """(num_blocks, 3) [x, y, yaw] spawn slots clear of this goal's walls."""
    if num_blocks > 5:
        return np.array([(x, y, TETRIS_SPAWN_YAW_8)
                         for x, y in TETRIS_SPAWN_XY_8[:num_blocks]], float)
    return np.array([(x, y, 0.0)
                     for x, y in TETRIS_SPAWN_XY[:num_blocks]], float)


GOAL_POSITION = np.array([0.3, 0.0, -0.005])
# (wide, tall) in cells; `Simulation.__init__`'s match statement.
GOAL_CELLS = {1: (2, 2), 3: (6, 2), 5: (10, 2), 8: (16, 2)}

TABLE_DIMS = np.array([0.8, 1.5, 0.02])
TABLE_POSE = np.array([0.30, 0.0, -0.011])

# `SpasmParams.cost_thresh` per block count, from `solve.py.__main__`.
TETRIS_COST_THRESH = {3: 0.44, 5: 0.42, 8: 0.66}


def create_tetris_spheres(shape, sph_radius=SPH_RADIUS):
    """(6, 4) [x, y, z, r] -- four body spheres plus a two-sphere grasp peg."""
    coords = TETRIS_SHAPE_COORDS[shape]
    n = coords.shape[0]
    sph = np.zeros((n + 2, 4))
    sph[:n, :3] = coords * sph_radius * 2
    sph[:n, 3] = sph_radius
    sph[n] = [0.0, 0.0, -sph_radius * 1.25, sph_radius / 2]
    sph[n + 1] = [0.0, 0.0, -sph_radius * 2, sph_radius / 2]
    sph[:, 2] += -sph[-1, 2]
    return sph


def block_z(sph_radius=SPH_RADIUS):
    """The resting height of a tetromino's grasp frame."""
    s = create_tetris_spheres("L", sph_radius)
    return float((s[:, 2] + s[:, 3]).max() - (s[:, 2] - s[:, 3]).min() - 1e-2)


def goal_dims(num_blocks=3, extra_cells=0, sph_radius=SPH_RADIUS):
    """(dx, dy, dz) of the walled goal, widened by `extra_cells` along y."""
    gw, gt = GOAL_CELLS.get(num_blocks, (10, 2))
    d, buf = sph_radius * 2, sph_radius
    return np.array([gt * d + buf, (gw + extra_cells) * d + buf, 0.01])


def create_walls(goal_center, gdims, wall_height=WALL_HEIGHT,
                 wall_thickness=WALL_THICKNESS):
    """(4, 6) AABBs [x1,y1,z1,x2,y2,z2] around the goal."""
    cx, cy, cz = goal_center
    dx, dy = gdims[0], gdims[1]
    t, h = wall_thickness, wall_height
    return np.array([
        [cx - dx / 2, cy + dy / 2, cz, cx + dx / 2, cy + dy / 2 + t, cz + h],
        [cx - dx / 2, cy - dy / 2 - t, cz, cx + dx / 2, cy - dy / 2, cz + h],
        [cx - dx / 2 - t, cy - dy / 2, cz, cx - dx / 2, cy + dy / 2, cz + h],
        [cx + dx / 2, cy - dy / 2, cz, cx + dx / 2 + t, cy + dy / 2, cz + h],
    ])


def block_pose_to_spheres(spheres, pose):
    """`tetris_env._block_pose_to_spheres`: 180 deg about x, then the yaw."""
    x, y, z, yaw = [float(v) for v in pose]
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.asarray(spheres, float)[:, :3] * np.array([1.0, -1.0, -1.0])
    out = np.empty((p.shape[0], 4))
    out[:, 0] = c * p[:, 0] - s * p[:, 1] + x
    out[:, 1] = s * p[:, 0] + c * p[:, 1] + y
    out[:, 2] = p[:, 2] + z
    out[:, 3] = np.asarray(spheres, float)[:, 3]
    return out


# ---------------------------------------------------------------------------
# Tetris cost (`spasm/solve.py`)
# ---------------------------------------------------------------------------

def sphere_sphere_penetration(s1, s2, margin=0.010):
    """(N, M) penetration with a `margin` dead zone; separated pairs cost 0."""
    a, b = np.asarray(s1, float)[:, None, :], np.asarray(s2, float)[None, :, :]
    dist = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)
    pen = np.maximum(-margin, (a[..., 3] + b[..., 3]) - dist) + margin
    return np.abs(pen)


def sphere_wall_penetration(spheres, goal_center, gdims, blk_z,
                            rizz_aura=0.010):
    """(N,) penetration of the goal walls and floor."""
    sp = np.asarray(spheres, float)
    c, r = sp[..., :3], sp[..., 3]
    x_max, x_min = goal_center[0] + gdims[0] / 2, goal_center[0] - gdims[0] / 2
    y_max, y_min = goal_center[1] + gdims[1] / 2, goal_center[1] - gdims[1] / 2
    aura = rizz_aura * 2
    xy = np.max(np.stack([
        np.maximum(-aura, c[..., 0] - x_max + r) + aura,
        np.maximum(-aura, x_min - c[..., 0] + r) + aura,
        np.maximum(-aura, c[..., 1] - y_max + r) + aura,
        np.maximum(-aura, y_min - c[..., 1] + r) + aura]), axis=0)
    wall = np.maximum(0.0, blk_z + 0.07 - c[..., 2] + r)
    floor = np.maximum(0.0, 0.0 - c[..., 2] + r)
    z = np.where(xy > rizz_aura + 1e-6, wall * 0.6, floor)
    return np.abs(xy) + np.abs(z)


def tetris_cost(block_poses, num_blocks=None, extra_cells=0,
                shapes=None, sph_radius=SPH_RADIUS):
    """`solve.cost`: wall penetration * 3 + sphere penetration * 0.5."""
    poses = np.asarray(block_poses, float)
    n = poses.shape[0] if num_blocks is None else num_blocks
    shapes = shapes or TETRIS_SHAPES[:n]
    gd = goal_dims(n, extra_cells, sph_radius)
    bz = block_z(sph_radius)

    sp = np.stack([block_pose_to_spheres(
        create_tetris_spheres(shapes[k], sph_radius), poses[k])
        for k in range(n)])

    ssp = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ssp += sphere_sphere_penetration(sp[i], sp[j]).sum()
    wall = sphere_wall_penetration(sp.reshape(-1, 4), GOAL_POSITION, gd, bz).sum()
    return float(wall * 3.0 + ssp * 0.5)


# ---------------------------------------------------------------------------
# Tower geometry and cost (`spasm/tower_env.py`, `spasm/tower_solve.py`)
# ---------------------------------------------------------------------------

BLOCK_DIMS = np.array([0.06, 0.06, 0.06])
TOWER_TABLE_DIMS = np.array([1.1, 1.5, 0.02])
TOWER_TABLE_POSE = np.array([0.15, 0.0, -0.011])
Z_ERROR_MUL = 1.0

# SPaSM's obstacle spheres, less two that made the task unsolvable rather than
# hard:
#   [0.15, 0.05, 0.48] r=0.1 -- sat under the arm's own elbow.
#   [0.00, 0.20, 0.20] r=0.1 -- immediately LEFT of the base and only 101 mm
#       from block 3's spawn cell, five times closer to a cube than any other
#       obstacle (next nearest 287 mm).  It walled off the left-hand spawn row,
#       and the solved plan drove the arm 13.2 mm THROUGH it rather than around
#       -- the clearance term is soft, so an obstacle the arm cannot avoid is
#       not an obstacle, it is just a constant penalty on every plan.
TOWER_OBSTACLE_POSES = np.array([
    [0.20, 0.5, 0.6], [0.5, 0.55, 0.4], [0.0, -0.4, 0.3],
    [0.0, -0.1, 0.9], [0.1, -0.2, 0.2], [0.2, -0.5, 0.1],
    [0.3, -0.5 - 100, 0.5], [0.25, -0.4, 0.0]])
TOWER_OBSTACLE_RADII = np.array([0.1, 0.1, 0.2,
                                 0.1, 0.1, 0.1, 0.1, 0.3])


def tower_spawn_poses(num_blocks=10, block_height=0.06):
    """`TowerSimulation.__init__`: two rows on the left of the table."""
    h = block_height / 2.0
    return np.array(
        [[0.4 - (i - 5) * 0.12, 0.30, h, 0.0]
         for i in range(num_blocks // 2, num_blocks)]
        + [[0.4 - i * 0.12, 0.5, h, 0.0] for i in range(num_blocks // 2)])


def _rotate(xy, yaw):
    x, y = xy[0], xy[1]
    return np.array([x * np.cos(yaw) - y * np.sin(yaw),
                     x * np.sin(yaw) + y * np.cos(yaw)])


def _corners_rotated(yaw, block_dims=BLOCK_DIMS):
    rc = _rotate(block_dims[:2] / 2.0, yaw)
    return np.array([[rc[0], rc[1]], [-rc[0], -rc[1]],
                     [rc[1], -rc[0]], [-rc[1], rc[0]]])


def block_block_penetration(pose_a, pose_b, block_dims=BLOCK_DIMS, inflate=1.0):
    half = block_dims[:2] / 2.0 * inflate
    rel = np.asarray(pose_b, float)[:2] - np.asarray(pose_a, float)[:2]
    b_in_a = _rotate(rel, -pose_a[3])
    pen_b_in_a = np.maximum(0.0, half - np.abs(
        _corners_rotated(pose_b[3], block_dims) + b_in_a)).sum()
    a_in_b = _rotate(-rel, -pose_b[3])
    pen_a_in_b = np.maximum(0.0, half - np.abs(
        _corners_rotated(pose_a[3], block_dims) + a_in_b)).sum()
    hz = block_dims[2] / 2.0
    z_touching = ((pose_b[2] - hz * inflate < pose_a[2] + hz)
                  & (pose_b[2] + hz * inflate > pose_a[2] - hz))
    return float(z_touching) * float(pen_b_in_a + pen_a_in_b)


def spheres_blocks_collision(spheres_xyz, spheres_radii, block_states,
                             block_dims=BLOCK_DIMS):
    diag = np.linalg.norm(np.asarray(block_dims, float) / 2)
    d = np.linalg.norm(np.asarray(spheres_xyz, float)[:, None, :]
                       - np.asarray(block_states, float)[None, :, :3], axis=-1)
    return np.maximum(0.0, (np.asarray(spheres_radii, float)[:, None] + diag)
                      - d).sum(axis=0)


def tower_cost(block_poses, init_poses=None, num_blocks=None,
               block_dims=BLOCK_DIMS, z_error_mul=Z_ERROR_MUL,
               obstacle_poses=None, obstacle_radii=None):
    """`tower_solve.cost`: z error * 11 + stability * 7 + pen * 0.01 + obs * 0.1.

    `init_poses` is accepted for signature parity with the original and, as
    there, does not enter the returned value: its `transport_error` term is
    computed and never used.
    """
    poses = np.asarray(block_poses, float)
    n = poses.shape[0] if num_blocks is None else num_blocks
    h = float(block_dims[2])
    if obstacle_poses is None:
        obstacle_poses, obstacle_radii = TOWER_OBSTACLE_POSES, TOWER_OBSTACLE_RADII

    target_z = np.arange(n) * h + h / 2.0
    z_error = float(np.sum((poses[:, 2] - target_z) ** 2) * z_error_mul)

    csum = np.cumsum(poses[:, :2], axis=0)
    strict = np.asarray(block_dims, float)[:2] / 2 * 0.1

    def stability_for(i, height):
        com = (csum[height - 1] - csum[i]) / (height - i - 1)
        off = _rotate(com - poses[i, :2], -poses[i, 3])
        return float(np.maximum(0.0, np.abs(off) - strict).sum())

    idx, hts = [], []
    for ht in range(2, n + 1):
        for i in range(ht - 1):
            idx.append(i)
            hts.append(ht)
    idx = np.asarray(idx)
    stab = np.array([stability_for(i, ht) for i, ht in zip(idx, hts)])
    ops_in = np.exp(idx / n) / np.e / 10
    stability = float((stab * ops_in).sum())

    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pen += block_block_penetration(poses[i], poses[j], block_dims) ** 2

    obs = float(np.abs(spheres_blocks_collision(
        obstacle_poses, obstacle_radii, poses, block_dims)).sum())

    return float(z_error * 11.0 + stability * 7.0 + pen * 0.01 + obs * 0.1)
