"""Forward rollout of the tetris-packing SPaSM model, visualised in viser.

Builds the three-stage composed forward map (IK -> per-segment trajopt ->
refine) at the ground-truth Z_STAR from e8, then plays back the resulting
joint-space trajectory on the Panda URDF with the full tetris environment
faithfully matching SPaSM's Simulation.render(): table, goal floor, goal walls,
and colored tetromino sphere clusters (4 body + 2 stick spheres each).

Multiple scenes are shown side-by-side (offset along y).

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    XLA_FLAGS="--xla_disable_hlo_passes=fusion" \\
        python -m iosp.viz.tetris_viser [--seed 0] [--num-blocks 1] [--n-scenes 4]
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import viser

# ---------------------------------------------------------------------------
# Forward rollout (returns joint + EE paths)
# ---------------------------------------------------------------------------

def _make_fast_solver(n_iters, robot):
    """Stock early-stopping L-BFGS — compiles and converges faster than the
    differentiable fixed-length scan the bilevel fit needs."""
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt
    cfg = DynamicsTrajOptConfig(n_iters=n_iters, early_stop=True,
                                soft_line_search=False, soft_curvature_gate=False)
    return lambda x0, cost_fn: dynamics_trajopt(x0, cost_fn, cfg)


def forward_rollout(seed=0, n_iters=60, n_scenes=4, num_blocks=1):
    """-> (q, ee, scenes) where q is (B, N_FULL, 7), ee is (B, N_FULL, 3)."""
    import jax
    import jax.numpy as jnp
    from ioc.inner import make_inner_solver
    from iosp.model import tetris as tt
    from iosp.config import URDF_PATH, SRDF_PATH, MESH_DIR

    Z_STAR = jnp.array([0.5, 1.5, 1.0, 0.8, 1.0, 2.0], dtype=jnp.float32)

    prob = tt.TetrisProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    fs = _make_fast_solver(n_iters, prob.base.robot)

    rng = np.random.default_rng(seed)
    scenes = tt.sample_tetris_scenes(rng, n_scenes, num_blocks=num_blocks)

    key = jax.random.PRNGKey(seed)
    x0, seg_scenes, q_pick, q_place = prob.seeds(scenes)

    inner_by_phase = {}
    for p in tt.PHASES:
        rf = prob.segment_residual_fn(p)
        scales = prob.calibrate_segment(p, rf, seg_scenes[p], key)
        inner_by_phase[p] = make_inner_solver(rf, scales, forward_solver=fs)

    full_rf = prob.full_residual_fn()
    full_sc = tt.TetrisFullScene(
        scenes.q_start, scenes.q_start,
        scenes.obs_center, scenes.obs_radius,
        q_pick, q_place)
    full_scales = prob.calibrate_full(full_rf, full_sc, key)
    refine = make_inner_solver(full_rf, full_scales, forward_solver=fs)

    theta = jax.nn.softmax(Z_STAR)
    theta_seg = theta[:tt.K_SEG]
    theta_full = theta

    def _solve_forward(scenes_b):
        x0b, seg_scenes_b, q_pick_b, q_place_b = prob.seeds(scenes_b)
        xs = {}
        for phase in tt.PHASES:
            xs[phase] = jax.vmap(
                inner_by_phase[phase].solve,
                in_axes=(0, None, 0))(x0b[phase], theta_seg, seg_scenes_b[phase])
        full_sc_b = tt.TetrisFullScene(
            scenes_b.q_start, scenes_b.q_start,
            scenes_b.obs_center, scenes_b.obs_radius,
            q_pick_b, q_place_b)
        rows = []
        for i, ph in enumerate(tt.PHASES):
            q = jax.vmap(prob.seg[ph].unpack)(xs[ph], seg_scenes_b[ph])
            rows.append(q[:, 1:] if i > 0 else q)
        q_cat = jnp.concatenate(rows, axis=1)
        x0_full = q_cat[:, 1:-1, :].reshape(q_cat.shape[0], -1)
        xs["full"] = jax.vmap(
            refine.solve,
            in_axes=(0, None, 0))(x0_full, theta_full, full_sc_b)
        return xs, full_sc_b

    t0 = time.perf_counter()
    _jit_forward = jax.jit(_solve_forward)
    xs, full_sc2 = _jit_forward(scenes)
    q = jax.vmap(prob.seg["full"].unpack)(xs["full"], full_sc2)
    ee = jax.vmap(prob.ee_positions)(q)
    jax.block_until_ready((q, ee))
    t_cold = time.perf_counter() - t0

    t1 = time.perf_counter()
    xs2, _ = _jit_forward(scenes)
    jax.block_until_ready(xs2)
    t_warm = time.perf_counter() - t1

    q, ee = np.asarray(q), np.asarray(ee)
    print(f"[tetris_viser] rolled out {n_scenes} scenes, q shape {q.shape}")
    print(f"  cold (incl. compile): {t_cold:.1f}s  |  warm solve: {t_warm*1e3:.0f}ms")
    return q, ee, scenes, prob


# ---------------------------------------------------------------------------
# SPaSM tetris environment geometry (faithfully matches Simulation.render())
# ---------------------------------------------------------------------------

SPH_RADIUS = 0.03

# Tetromino sphere definitions (from spasm/tetris/env.py create_tetris_spheres)
def _create_tetris_spheres(shape: str, sph_radius: float = SPH_RADIUS):
    _shape_coords = {
        "L": np.array([(0, 0, 0), (0, 1, 0), (0, -1, 0), (1, -1, 0)], dtype=np.float32),
        "O": np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)], dtype=np.float32),
    }
    coords = _shape_coords[shape]
    n = coords.shape[0]
    spheres = np.zeros((n + 2, 4), dtype=np.float32)
    spheres[:n, :3] = coords * sph_radius * 2
    spheres[:n, 3] = sph_radius
    spheres[n] = [0.0, 0.0, -sph_radius * 1.25, sph_radius * 0.7]
    spheres[n + 1] = [0.0, 0.0, -sph_radius * 2, sph_radius * 0.7]
    z_offset = -spheres[-1, 2]
    spheres[:, 2] += z_offset
    return spheres


def _transform_spheres(spheres, pose_xyzyaw):
    """Transform (6,4) spheres by (x,y,z,yaw) pose."""
    from scipy.spatial.transform import Rotation
    x, y, z, yaw = pose_xyzyaw
    R = Rotation.from_euler('z', yaw).as_matrix()
    pos = spheres[:, :3].copy()
    radii = spheres[:, 3:4].copy()
    pos = pos @ R.T + np.array([x, y, z])
    return np.hstack([pos, radii])


# 8 block definitions: shape assignment and initial poses from Simulation.__init__
_BLOCK_SHAPES = ["O", "L", "O", "O", "O", "L", "L", "L"]
_BLOCK_POSES = [
    [0.50, 0.35, None, 0],
    [0.15, -0.6, None, 0],
    [0.00, 0.6, None, 0],
    [0.15, 0.6, None, 0],
    [0.00, -0.6, None, 0],
    [0.50, -0.3, None, 0],
    [0.50, -0.1, None, 0],
    [0.50, 0.1, None, 0],
]

# Pastel colors (from Simulation.__init__)
_RAW_COLORS = [0xe81416, 0xffa500, 0xfaeb36, 0x79c314, 0x487de7, 0x87369d, 0x5eb40d, 0xffa500]
_PASTEL_FACTOR = 0.6

def _pastel_rgb(hex_color):
    r = (hex_color >> 16) & 0xFF
    g = (hex_color >> 8) & 0xFF
    b = hex_color & 0xFF
    pf = _PASTEL_FACTOR
    return (int(r * pf + 255 * (1 - pf)),
            int(g * pf + 255 * (1 - pf)),
            int(b * pf + 255 * (1 - pf)))

BLOCK_COLORS = [_pastel_rgb(c) for c in _RAW_COLORS]

WALL_COLOR = (220, 218, 210)
TABLE_COLOR = (255, 255, 255)
GOAL_FLOOR_COLOR = (255, 255, 255)


def _get_block_spheres_and_poses(num_blocks):
    """Return (list of (6,4) sphere arrays, list of (4,) poses, list of RGB tuples)."""
    indices = list(range(num_blocks))
    if num_blocks == 2:
        indices = [1, 2]

    spheres_list = []
    poses_list = []
    colors_list = []
    for idx in indices:
        shape = _BLOCK_SHAPES[idx]
        sphs = _create_tetris_spheres(shape)
        spheres_list.append(sphs)

        L_sphs = _create_tetris_spheres("L")
        block_z = float(np.max(L_sphs[:, 2] + L_sphs[:, 3]) - np.min(L_sphs[:, 2] - L_sphs[:, 3]) - 1e-2)
        pose = list(_BLOCK_POSES[idx])
        pose[2] = block_z
        poses_list.append(np.array(pose, dtype=np.float32))
        colors_list.append(BLOCK_COLORS[idx])

    return spheres_list, poses_list, colors_list


def _goal_dims(num_blocks):
    """Return goal_dims and goal_position matching Simulation.__init__."""
    diameter = SPH_RADIUS * 2
    match num_blocks:
        case 1: gw, gt = 2, 2
        case 3: gw, gt = 6, 2
        case 5: gw, gt = 10, 2
        case 8: gw, gt = 16, 2
        case _: gw, gt = 10, 2
    buffer = SPH_RADIUS * 1.0
    gw = gw * diameter + buffer
    gt = gt * diameter + buffer
    return np.array([gt, gw, 0.01]), np.array([0.3, 0.0, -0.005])


def _create_walls(goal_pos, goal_dims, wall_height=0.045, wall_thickness=0.015):
    cx, cy, cz = goal_pos
    cdx, cdy, _ = goal_dims
    walls = []
    walls.append((cx - cdx/2, cy + cdy/2, cz,
                  cx + cdx/2, cy + cdy/2 + wall_thickness, cz + wall_height))
    walls.append((cx - cdx/2, cy - cdy/2 - wall_thickness, cz,
                  cx + cdx/2, cy - cdy/2, cz + wall_height))
    walls.append((cx - cdx/2 - wall_thickness, cy - cdy/2, cz,
                  cx - cdx/2, cy + cdy/2, cz + wall_height))
    walls.append((cx + cdx/2, cy - cdy/2, cz,
                  cx + cdx/2 + wall_thickness, cy + cdy/2, cz + wall_height))
    return walls


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _load_urdf():
    import yourdfpy
    root = pathlib.Path(__file__).resolve().parents[2] / "resources" / "panda"
    return yourdfpy.URDF.load(str(root / "panda_spherized.urdf"), load_meshes=True,
                              build_scene_graph=True, mesh_dir=str(root / "meshes"))


def _polyline(server, name, pts, color, width):
    pts = np.asarray(pts, np.float32)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    return server.scene.add_line_segments(
        name, points=segs,
        colors=np.tile(np.asarray(color, np.uint8), (len(segs), 2, 1)),
        line_width=width)


SCENE_COLORS = [
    (0x3b, 0x7d, 0xd8),
    (0x2a, 0xb0, 0x5e),
    (0xd4, 0x6a, 0x20),
    (0x9b, 0x3d, 0xb8),
    (0xd4, 0x3d, 0x3d),
    (0x20, 0xad, 0xad),
]


SPASM_SOLVED_NPY = pathlib.Path(
    "/home/sadmin/Work/spasm-pyroffi/saved/tetris.npy")


def _load_solved_poses(num_blocks, path=SPASM_SOLVED_NPY):
    """SPaSM placement solution (num_blocks, 4), or None if unavailable."""
    if not pathlib.Path(path).exists():
        return None
    poses = np.load(path)
    if poses.shape != (num_blocks, 4):
        return None
    return poses.astype(np.float32)


def _add_block_group(server, name, group, offset, spheres_list, poses, colors,
                     visible=True):
    """Render one set of tetromino sphere clusters under /{name}/{group}."""
    off = np.asarray(offset)
    handles = []
    for i, (sphs, pose, color) in enumerate(zip(spheres_list, poses, colors)):
        for j, row in enumerate(_transform_spheres(sphs, pose)):
            x, y, z, r = row
            handles.append(server.scene.add_icosphere(
                f"/{name}/{group}/block{i}_sph{j}", radius=float(r),
                position=tuple(off + np.array([x, y, z])),
                color=color, visible=visible))
    return handles


def _add_tetris_env(server, name, offset, num_blocks, scenes=None, scene_idx=None,
                    solved_poses=None):
    """Add the tetris environment: table, goal floor/walls, colored tetromino blocks.

    Blocks are rendered twice — at their spawn poses and (if `solved_poses` is
    given) at the SPaSM placement solution — as two visibility-toggled groups.
    Returns (initial_handles, solved_handles); solved_handles is [] when no
    solution was supplied.
    """
    off = np.asarray(offset)

    # Table
    server.scene.add_box(
        f"/{name}/table",
        dimensions=(0.8, 1.5, 0.02),
        position=tuple(off + np.array([0.30, 0.0, -0.011])),
        color=TABLE_COLOR)

    # Goal floor
    gd, gp = _goal_dims(num_blocks)
    server.scene.add_box(
        f"/{name}/goal_floor",
        dimensions=(float(gd[0]), float(gd[1]), float(gd[2])),
        position=tuple(off + gp),
        color=GOAL_FLOOR_COLOR)

    # Goal walls
    walls = _create_walls(gp, gd)
    for j, (x1, y1, z1, x2, y2, z2) in enumerate(walls):
        dims = (x2 - x1, y2 - y1, z2 - z1)
        pos = tuple(off + np.array([(x1+x2)/2, (y1+y2)/2, (z1+z2)/2]))
        server.scene.add_box(f"/{name}/wall{j}", dimensions=dims,
                             position=pos, color=WALL_COLOR)

    # Tetromino blocks as sphere clusters: spawn poses, and (optionally) the
    # SPaSM placement solution, as two visibility-toggled groups.
    block_spheres, block_poses, block_colors = _get_block_spheres_and_poses(num_blocks)
    initial = _add_block_group(server, name, "blocks_initial", off,
                               block_spheres, block_poses, block_colors,
                               visible=True)
    solved = []
    if solved_poses is not None:
        solved = _add_block_group(server, name, "blocks_solved", off,
                                  block_spheres, solved_poses, block_colors,
                                  visible=False)

    # Pick/place markers if scenes provided
    if scenes is not None and scene_idx is not None:
        pick = np.asarray(scenes.pick_pos[scene_idx])
        server.scene.add_icosphere(
            f"/{name}/pick_marker", radius=0.02,
            position=tuple(off + pick),
            color=(255, 90, 90), opacity=0.7)

        place = np.asarray(scenes.place_pos[scene_idx])
        server.scene.add_icosphere(
            f"/{name}/place_marker", radius=0.02,
            position=tuple(off + place),
            color=(90, 255, 90), opacity=0.7)

    return initial, solved


# SPaSM's neutral arm pose (Simulation.get_neutral_pose, arm joints only).
NEUTRAL_Q = np.array([0.0, -np.pi / 4, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4])


class SceneRollout:
    """One arm + its tetris environment, at a lateral offset."""

    def __init__(self, server, name, offset, q, ee, scenes, scene_idx,
                 color, num_blocks, solved_poses=None):
        from viser.extras import ViserUrdf
        self.server, self.name = server, name
        self.offset = np.asarray(offset, float)
        self.q, self.ee = q, ee

        server.scene.add_frame(f"/{name}", show_axes=False, position=self.offset)
        self.urdf = ViserUrdf(server, _load_urdf(), root_node_name=f"/{name}/robot")

        self.blocks_initial, self.blocks_solved = _add_tetris_env(
            server, name, self.offset, num_blocks, scenes, scene_idx,
            solved_poses=solved_poses)
        self.path = _polyline(server, f"/{name}/path", ee + self.offset, color, 4.0)

        # Start in the default (unsolved) configuration: blocks at their spawn
        # poses, arm at the neutral pose, no trajectory drawn.
        self.show_solved(False)

    def show_solved(self, solved: bool):
        for h in self.blocks_initial:
            h.visible = not solved
        for h in self.blocks_solved:
            h.visible = solved
        self.path.visible = solved
        if not solved:
            self.urdf.update_cfg(NEUTRAL_Q)

    def set_row(self, t):
        self.urdf.update_cfg(np.asarray(self.q[t]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-blocks", type=int, default=1)
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--spread", type=float, default=1.8,
                    help="lateral offset between scenes, metres")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-solve", action="store_true",
                    help="Show only the environment geometry (no forward solve)")
    ap.add_argument("--solved-npy", type=str, default=str(SPASM_SOLVED_NPY),
                    help="SPaSM placement solution to show under 'Solved placements'")
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=4.0, height=4.0,
                          cell_size=0.25, plane="xy")

    solved_poses = _load_solved_poses(args.num_blocks, args.solved_npy)
    if solved_poses is None:
        print(f"[tetris_viser] no {args.num_blocks}-block solution at "
              f"{args.solved_npy}; showing spawn poses only")

    if args.no_solve:
        from viser.extras import ViserUrdf
        initial, solved = _add_tetris_env(
            server, "env", (0, 0, 0), args.num_blocks, solved_poses=solved_poses)
        urdf = ViserUrdf(server, _load_urdf(), root_node_name="/robot")
        urdf.update_cfg(NEUTRAL_Q)

        show = server.gui.add_checkbox("Solved placements", False,
                                       disabled=not solved)

        def on_show(_=None):
            for h in initial:
                h.visible = not show.value
            for h in solved:
                h.visible = show.value

        show.on_update(on_show)
        on_show()

        print(f"[tetris_viser] environment-only mode, num_blocks={args.num_blocks}")
        print(f"Viser server: http://0.0.0.0:{server.get_port()}")
        while True:
            time.sleep(1.0)
    else:
        print("Building forward model and rolling out...", flush=True)
        q, ee, scenes, prob = forward_rollout(
            seed=args.seed, n_iters=args.n_iters,
            n_scenes=args.n_scenes, num_blocks=args.num_blocks)

        from iosp.model import tetris as tt
        B, T, dof = q.shape

        rollouts = []
        y0 = -0.5 * args.spread * (B - 1)
        for i in range(B):
            color = SCENE_COLORS[i % len(SCENE_COLORS)]
            offset = (0.0, y0 + i * args.spread, 0.0)
            r = SceneRollout(
                server, f"scene{i}", offset,
                q[i], ee[i], scenes, i,
                color, args.num_blocks, solved_poses=solved_poses)
            rollouts.append(r)

        show = server.gui.add_checkbox("Solved configuration", False)

        with server.gui.add_folder("Playback"):
            row = server.gui.add_slider("Path row", 0, T - 1, 1, 0)
            play = server.gui.add_checkbox("Play", True)
            speed = server.gui.add_slider("Steps / sec", 1.0, 30.0, 1.0, args.fps)
            info = server.gui.add_markdown("")

        def phase_of(t):
            for p in tt.PHASES:
                s, e = tt.PHASE_SPAN[p]
                if s <= t < e:
                    return p
            return tt.PHASES[-1]

        def set_row_all(_=None):
            if not show.value:
                return
            t = int(row.value)
            for r in rollouts:
                r.set_row(t)
            info.content = (f"row **{t}**/{T - 1} &nbsp; phase **{phase_of(t)}**")

        def on_show(_=None):
            for r in rollouts:
                r.show_solved(show.value)
            if show.value:
                set_row_all()
            else:
                info.content = "default pose — enable *Solved configuration*"

        row.on_update(set_row_all)
        show.on_update(on_show)
        on_show()

        print(f"[tetris_viser] {B} scenes, {T} rows, {dof} DOF")
        print(f"Viser server: http://0.0.0.0:{server.get_port()}")
        while True:
            if show.value and play.value:
                row.value = (int(row.value) + 1) % T
            time.sleep(1.0 / float(speed.value))


if __name__ == "__main__":
    main()
