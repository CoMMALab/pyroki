"""The MuJoCo viewers must agree with the models they visualise.

Two classes of assertion:

1. **Duplicated constants.**  `tetris_viser`, `tower_viser` and
   `pickplace_viser` each restate a handful of scene constants (standoffs, block
   dimensions) so that `--from-npz` and `--no-solve` can render without pulling
   in the solve-side model and its JAX import.  A duplicate that silently drifts
   would put the manipulated object in the wrong place and make the viewer lie,
   so the model stays the authority and these tests pin the copies to it.

2. **The scene actually builds.**  Compiling a multi-scene MuJoCo model exercises
   the attach-with-prefix path and the joint-name lookups, which are the parts
   that break when a robot description changes underneath.
"""
import numpy as np
import pytest

from iosp.viz import mj_scene as M


# ---------------------------------------------------------------------------
# Duplicated constants
# ---------------------------------------------------------------------------

def test_tetris_standoff_matches_model():
    from iosp.model import tetris as tt
    from iosp.viz import tetris_viser as V
    assert V.STANDOFF == pytest.approx(float(tt.STANDOFF))


def test_tower_constants_match_model():
    from iosp.model import tower as tw
    from iosp.viz import tower_viser as V
    assert V.BLOCK_DIM == pytest.approx(float(tw.BLOCK_DIM))
    assert V.BLOCK_HALF == pytest.approx(float(tw.BLOCK_HALF))
    assert V.STANDOFF == pytest.approx(float(tw.STANDOFF))
    assert np.allclose(V.BASE_XY, np.asarray(tw.BASE_XY))


def test_pickplace_scene_defaults_match_teleop():
    """Bucket geometry defaults in spasm_rollout match sim_teleop's PickPlaceScene."""
    from iosp.checks import spasm_rollout as SR
    assert SR.PP_TABLE_HEIGHT == pytest.approx(0.30)
    assert SR.PP_BUCKET_INNER_RADIUS == pytest.approx(0.075)
    assert SR.PP_BUCKET_N_WALLS == 8


def test_tetris_viewer_spheres_match_the_model():
    """The viewer's tetromino must be the one the planner plans against.

    This used to import `spasm.tetris.env` from a `tamp/` directory -- neither
    of which exists -- so it silently skipped and asserted nothing.  The
    authority is now `iosp.model.spasm_costs`, which is in this repo and always
    importable, so the check runs.
    """
    from iosp.model import spasm_costs as C
    from iosp.viz import tetris_viser as V
    for shape in ("L", "O"):
        ours = V._create_tetris_spheres(shape, V.SPH_RADIUS)
        theirs = C.create_tetris_spheres(shape, V.SPH_RADIUS)
        assert np.allclose(ours, theirs, atol=1e-9), shape


def test_tetris_block_pose_flip_puts_the_block_on_the_table():
    """A posed block rests ON the table with its peg up, flip included.

    Matching the block-frame spheres is not enough: the pose is the GRASP pose
    and the gripper comes from above, so the transform rotates 180 degrees
    about x before applying the yaw.  Dropping the flip renders the tetromino
    upside down -- body in the air, peg underneath -- while every block-frame
    assertion still passes, so the check has to be on world output.
    """
    from iosp.model import spasm_costs as C
    from iosp.viz import tetris_viser as V

    for shape in ("L", "O"):
        for pose in ([0.5, 0.35, 0.095, 0.0], [0.3, -0.1, 0.095, 0.7]):
            ours = V._transform_spheres(
                V._create_tetris_spheres(shape, V.SPH_RADIUS), np.array(pose))
            theirs = C.block_pose_to_spheres(
                C.create_tetris_spheres(shape, V.SPH_RADIUS), np.array(pose))
            assert np.allclose(ours, theirs, atol=1e-9), (shape, pose)

    world = V._transform_spheres(V._create_tetris_spheres("L", V.SPH_RADIUS),
                                 np.array([0.5, 0.35, 0.095, 0.0]))
    assert (world[:, 2] - world[:, 3]).min() == pytest.approx(0.005, abs=1e-4)
    assert world[:4, 2].mean() < world[4:, 2].mean()      # body below the peg


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("robot", M.ROBOTS)
def test_multi_scene_world_compiles(robot):
    b = M.WorldBuilder(n_scenes=3, spread=1.5, robot=robot)
    for i in range(3):
        b.box(f"s{i}_table", b.offset(i) + np.array([0.3, 0.0, -0.011]),
              (0.8, 1.5, 0.02), M.TABLE_RGBA)
        b.free_body(f"s{i}_obj",
                    [("box", (0.06,) * 3, (0, 0, 0), (0.9, 0.5, 0.15, 1.0))],
                    pos=b.offset(i) + np.array([0.45, 0.0, 0.03]))
    world = b.compile()

    assert world.n_scenes == 3
    assert len(world.arm_adr) == 3 and all(len(a) == 7 for a in world.arm_adr)
    # Each arm must address a distinct block of qpos.
    flat = [a for arm in world.arm_adr for a in arm]
    assert len(set(flat)) == len(flat)

    q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8])
    world.set_arm(0, q)
    world.forward()
    assert np.allclose(world.data.qpos[world.arm_adr[0]], q)


def test_ee_frame_matches_pyroffi_fk():
    """The EE the viewer draws is the point the planner optimised.

    `mj_scene` reads it off the last arm link plus a fixed offset rather than
    calling pyroffi, so that a trajectory loaded from an `.npz` renders without
    a JAX round-trip; this is the check that the shortcut is exact.
    """
    jax = pytest.importorskip("jax")
    from ioc.robot.problem import RobotProblem
    from iosp.config import URDF_PATH, SRDF_PATH, MESH_DIR

    q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8])
    prob = RobotProblem.load(str(URDF_PATH), str(SRDF_PATH), str(MESH_DIR),
                             n_timesteps=2)
    want = np.asarray(prob.ee_positions(q))

    for robot in M.ROBOTS:
        world = M.WorldBuilder(n_scenes=1, robot=robot).compile()
        world.set_arm(0, q)
        world.forward()
        assert np.allclose(world.ee_position(0), want, atol=1e-4), robot


@pytest.mark.parametrize("mod,dom", [
    ("iosp.viz.tetris_viser", "tetris"),
    ("iosp.viz.tower_viser", "tower"),
    ("iosp.viz.pickplace_viser", "pickplace"),
])
def test_loader_rejects_the_wrong_domain(tmp_path, mod, dom):
    import importlib
    V = importlib.import_module(mod)
    bad = tmp_path / "bad.npz"
    np.savez(bad, q=np.zeros((1, 4, 7)), domain="not_" + dom,
             idx_pick=1, idx_place=2,
             pick_pos=np.zeros((1, 3)), place_pos=np.zeros((1, 3)),
             obs_center=np.zeros((1, 1, 3)), obs_radius=np.zeros((1, 1)))
    with pytest.raises(SystemExit):
        V._load_npz(str(bad))
