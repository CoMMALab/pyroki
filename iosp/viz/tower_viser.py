"""Forward rollout of the tower-stacking SPaSM model, visualised in viser.

Builds the three-stage composed forward map (IK -> per-segment trajopt ->
refine) at the ground-truth Z_STAR from e9, then plays back the resulting
joint-space trajectory on the Panda URDF with the tower environment:
table, stacking base, already-placed blocks (as cubes), pick/place markers,
and EE path traces.

Multiple scenes are shown side-by-side (offset along y).

Usage:
    CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    XLA_FLAGS="--xla_disable_hlo_passes=fusion" \\
        python -m iosp.viz.tower_viser [--seed 0] [--stack-level 0] [--n-scenes 4]
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import viser

# ---------------------------------------------------------------------------
# Forward rollout
# ---------------------------------------------------------------------------

def _make_fast_solver(n_iters, robot):
    """Stock early-stopping L-BFGS — compiles and converges faster than the
    differentiable fixed-length scan the bilevel fit needs."""
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt
    cfg = DynamicsTrajOptConfig(n_iters=n_iters, early_stop=True,
                                soft_line_search=False, soft_curvature_gate=False)
    return lambda x0, cost_fn: dynamics_trajopt(x0, cost_fn, cfg)


def forward_rollout(seed=0, n_iters=60, n_scenes=4, stack_level=0):
    """-> (q, ee, scenes) where q is (B, N_FULL, 7), ee is (B, N_FULL, 3).

    Uses the stock early-stopping solver (not the differentiable one) so the
    cold compile is seconds, not minutes.
    """
    import jax
    import jax.numpy as jnp
    from ioc.inner import make_inner_solver
    from iosp.model import tower as tw
    from iosp.config import URDF_PATH, SRDF_PATH, MESH_DIR

    Z_STAR = jnp.array([0.5, 1.5, 1.0, 0.8, 1.2, 1.0, 2.0], dtype=jnp.float32)

    prob = tw.TowerProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR))
    fs = _make_fast_solver(n_iters, prob.base.robot)

    rng = np.random.default_rng(seed)
    scenes = tw.sample_tower_scenes(rng, n_scenes, stack_level=stack_level)

    key = jax.random.PRNGKey(seed)
    x0, seg_scenes, q_pick, q_place = prob.seeds(scenes)

    inner_by_phase = {}
    for p in tw.PHASES:
        rf = prob.segment_residual_fn(p)
        scales = prob.calibrate_segment(p, rf, seg_scenes[p], key)
        inner_by_phase[p] = make_inner_solver(rf, scales, forward_solver=fs)

    full_rf = prob.full_residual_fn()
    full_sc = tw.TowerFullScene(
        scenes.q_start, scenes.q_start,
        scenes.obs_center, scenes.obs_radius,
        q_pick, q_place, scenes.target_z)
    full_scales = prob.calibrate_full(full_rf, full_sc, key)
    refine = make_inner_solver(full_rf, full_scales, forward_solver=fs)

    theta = jax.nn.softmax(Z_STAR)
    theta_seg = theta[:tw.K_SEG]
    theta_full = theta

    def _solve_forward(scenes_b):
        x0b, seg_scenes_b, q_pick_b, q_place_b = prob.seeds(scenes_b)
        xs = {}
        for phase in tw.PHASES:
            xs[phase] = jax.vmap(
                inner_by_phase[phase].solve,
                in_axes=(0, None, 0))(x0b[phase], theta_seg, seg_scenes_b[phase])
        full_sc_b = tw.TowerFullScene(
            scenes_b.q_start, scenes_b.q_start,
            scenes_b.obs_center, scenes_b.obs_radius,
            q_pick_b, q_place_b, scenes_b.target_z)
        rows = []
        for i, ph in enumerate(tw.PHASES):
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
    print(f"[tower_viser] rolled out {n_scenes} scenes, q shape {q.shape}")
    print(f"  cold (incl. compile): {t_cold:.1f}s  |  warm solve: {t_warm*1e3:.0f}ms")
    return q, ee, scenes, prob


# ---------------------------------------------------------------------------
# Environment rendering helpers
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

TABLE_COLOR = (255, 255, 255)
OBSTACLE_COLOR = (255, 255, 255)
BASE_MARKER_COLOR = (230, 204, 50)

# SPaSM tower block colors (from TowerSimulation.__init__)
import random as _random
_TOWER_BLOCK_HEX = [0xfd3f52, 0xff6b6b, 0xfd7e03, 0xffbc16, 0xa9e507,
                     0x65d73d, 0x38c188, 0x0cd4ae, 0x02ccd0, 0x31b5e7]
_random.seed(42)
_random.shuffle(_TOWER_BLOCK_HEX)

def _hex_rgb(h):
    return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)


def _add_tower_env(server, name, offset, scenes, scene_idx, stack_level, num_blocks=10):
    """Add the tower environment matching SPaSM's TowerSimulation.render()."""
    from iosp.model.tower import BASE_XY, BLOCK_DIM, BLOCK_HALF
    off = np.asarray(offset)
    bxy = np.asarray(BASE_XY)
    block_dim = float(BLOCK_DIM)

    # Table (SPaSM tower: 1.1 x 1.5 x 0.02 at [0.15, 0, -0.011])
    server.scene.add_box(
        f"/{name}/table",
        dimensions=(1.1, 1.5, 0.02),
        position=tuple(off + np.array([0.15, 0.0, -0.011])),
        color=TABLE_COLOR)

    # Stacking base marker
    server.scene.add_box(
        f"/{name}/base_marker",
        dimensions=(0.08, 0.08, 0.003),
        position=tuple(off + np.array([float(bxy[0]), float(bxy[1]), 0.0])),
        color=BASE_MARKER_COLOR, opacity=0.6)

    # Block spawn positions (from TowerSimulation.__init__)
    spawn_poses = (
        [[0.4 - (i - num_blocks // 2) * 0.12, 0.30, block_dim / 2, 0.0]
         for i in range(num_blocks // 2, num_blocks)] +
        [[0.4 - i * 0.12, 0.5, block_dim / 2, 0.0]
         for i in range(num_blocks // 2)]
    )
    block_colors = [_hex_rgb(c) for c in _TOWER_BLOCK_HEX[:num_blocks]]

    # Blocks at spawn positions as colored cubes
    for i, (pose, color) in enumerate(zip(spawn_poses, block_colors)):
        server.scene.add_box(
            f"/{name}/block{i}",
            dimensions=(block_dim, block_dim, block_dim),
            position=tuple(off + np.array(pose[:3])),
            color=color)

    # Already-stacked blocks (below the current level)
    for lvl in range(stack_level):
        bz = block_dim * (lvl + 0.5)
        server.scene.add_box(
            f"/{name}/stacked_block{lvl}",
            dimensions=(block_dim, block_dim, block_dim),
            position=tuple(off + np.array([float(bxy[0]), float(bxy[1]), bz])),
            color=(165, 140, 100), opacity=0.7)

    # Pick/place markers
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


class SceneRollout:
    def __init__(self, server, name, offset, q, ee, scenes, scene_idx,
                 color, stack_level):
        from viser.extras import ViserUrdf
        self.server, self.name = server, name
        self.offset = np.asarray(offset, float)
        self.q, self.ee = q, ee

        server.scene.add_frame(f"/{name}", show_axes=False, position=self.offset)
        self.urdf = ViserUrdf(server, _load_urdf(), root_node_name=f"/{name}/robot")

        _add_tower_env(server, name, self.offset, scenes, scene_idx, stack_level)
        _polyline(server, f"/{name}/path", ee + self.offset, color, 4.0)

    def set_row(self, t):
        self.urdf.update_cfg(np.asarray(self.q[t]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stack-level", type=int, default=0)
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--n-iters", type=int, default=60)
    ap.add_argument("--spread", type=float, default=1.8)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    print("Building forward model and rolling out...", flush=True)
    q, ee, scenes, prob = forward_rollout(
        seed=args.seed, n_iters=args.n_iters,
        n_scenes=args.n_scenes, stack_level=args.stack_level)

    from iosp.model import tower as tw
    B, T, dof = q.shape

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=4.0, height=4.0,
                          cell_size=0.25, plane="xy")

    rollouts = []
    y0 = -0.5 * args.spread * (B - 1)
    for i in range(B):
        color = SCENE_COLORS[i % len(SCENE_COLORS)]
        offset = (0.0, y0 + i * args.spread, 0.0)
        r = SceneRollout(
            server, f"scene{i}", offset,
            q[i], ee[i], scenes, i,
            color, args.stack_level)
        rollouts.append(r)

    with server.gui.add_folder("Playback"):
        row = server.gui.add_slider("Path row", 0, T - 1, 1, 0)
        play = server.gui.add_checkbox("Play", True)
        speed = server.gui.add_slider("Steps / sec", 1.0, 30.0, 1.0, args.fps)
        info = server.gui.add_markdown("")

    def phase_of(t):
        for p in tw.PHASES:
            s, e = tw.PHASE_SPAN[p]
            if s <= t < e:
                return p
        return tw.PHASES[-1]

    def set_row_all(_=None):
        t = int(row.value)
        for r in rollouts:
            r.set_row(t)
        info.content = (f"row **{t}**/{T - 1} &nbsp; phase **{phase_of(t)}**")

    row.on_update(set_row_all)
    set_row_all()

    print(f"[tower_viser] {B} scenes, {T} rows, {dof} DOF, "
          f"stack_level={args.stack_level}")
    print(f"Viser server: http://0.0.0.0:{server.get_port()}")
    while True:
        if play.value:
            row.value = (int(row.value) + 1) % T
        time.sleep(1.0 / float(speed.value))


if __name__ == "__main__":
    main()
