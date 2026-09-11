"""`iosp.model.spasm_costs` is a port; these pin it to what SPaSM produced.

Two of the checks are absolute: SPaSM's saved 3-block packing scores 0.269522
and its saved 10-block stack 0.153742, the latter matching the `opt_errors`
stored beside it in `saved/tower.npz` to seven digits.  Those hold with no
SPaSM checkout present.  When one IS present the last test diffs the two
implementations directly on random poses, which is the check that would catch
a port that happens to reproduce one number by luck.
"""
import numpy as np
import pytest

from iosp.model import spasm_costs as C
from iosp.model import spasm_tasks as ST

# SPaSM's own saved 3-block packing (`spasm/saved/tetris.npy`), inlined so the
# reference does not depend on that checkout being present.
SPASM_SAVED_TETRIS = np.array([
    [0.27000815, -0.09082547, 0.095, 0.00684285],
    [0.30554113, 0.02769212, 0.095, -2.7527552],
    [0.3265914, 0.0898453, 0.095, 3.0794137]], np.float32)
SPASM_SAVED_TETRIS_COST = 0.26952229
SPASM_SAVED_TOWER_COST = 0.15374175      # == saved/tower.npz opt_errors[0]


def test_tetris_cost_matches_spasm_reference():
    got = C.tetris_cost(SPASM_SAVED_TETRIS, 3, extra_cells=0)
    assert got == pytest.approx(SPASM_SAVED_TETRIS_COST, abs=1e-5)


def test_tower_cost_matches_spasm_reference():
    # RAW: this compares against SPaSM's own reported number, so it has to use
    # SPaSM's own stack, not the one translated into the arm's workspace.
    got = C.tower_cost(ST.tower_skeleton_raw(10), ST.tower_init_state(10), 10)
    assert got == pytest.approx(SPASM_SAVED_TOWER_COST, abs=1e-5)


def test_tetris_geometry_self_consistent():
    assert C.block_z() == pytest.approx(0.095, abs=1e-6)
    # SPaSM's 3-block goal is 6x2 cells; ours widens it by two along y.
    assert C.goal_dims(3, 0) == pytest.approx([0.15, 0.39, 0.01], abs=1e-6)
    assert C.goal_dims(3, 2) == pytest.approx([0.15, 0.51, 0.01], abs=1e-6)


def test_penetration_dead_zones():
    """Separated spheres cost nothing until they are inside the 10 mm margin."""
    far = C.sphere_sphere_penetration(np.array([[0.0, 0.0, 0.0, 0.03]]),
                                      np.array([[1.0, 0.0, 0.0, 0.03]]))
    assert far.max() == pytest.approx(0.0, abs=1e-9)
    touching = C.sphere_sphere_penetration(np.array([[0.0, 0.0, 0.0, 0.03]]),
                                           np.array([[0.06, 0.0, 0.0, 0.03]]))
    assert touching.max() == pytest.approx(0.010, abs=1e-9)


def _spasm_available():
    """Importable, not merely present -- the checkout has its own dependencies
    (meshcat, xmltodict, its kinematics module) that iosp deliberately does not
    require, so a directory on disk is no guarantee the cross-check can run."""
    try:
        from iosp.checks import spasm_trajopt as SJ
        if not (SJ.SPASM_ROOT / "spasm").is_dir():
            return False
        with SJ._spasm_cwd():
            import spasm.solve  # noqa: F401
            import spasm.tower_solve  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _spasm_available(), reason="no SPaSM checkout")
def test_port_matches_spasm_implementation_on_random_poses():
    import jax.numpy as jnp
    from iosp.checks import spasm_trajopt as SJ

    with SJ._spasm_cwd():
        from spasm.solve import cost as tcost, SpasmParams
        from spasm.tower_solve import cost as wcost
        params = SpasmParams()
        sim_t = SJ._spasm_tetris_sim(3, extra_cells=0)
        sim_w = SJ._spasm_tower_sim(10)

        rng = np.random.default_rng(0)
        for _ in range(10):
            p = np.stack([[rng.uniform(0.22, 0.38), rng.uniform(-0.2, 0.2),
                           0.095, rng.uniform(-np.pi, np.pi)]
                          for _ in range(3)]).astype(np.float32)
            assert C.tetris_cost(p, 3, extra_cells=0) == pytest.approx(
                float(tcost(params, sim_t, jnp.asarray(p))), abs=1e-5)

        base, init = ST.tower_skeleton(10), ST.tower_init_state(10)
        for _ in range(5):
            p = base.copy()
            p[:, :2] += rng.normal(scale=0.02, size=(10, 2))
            p[:, 2] += rng.normal(scale=0.01, size=10)
            p[:, 3] += rng.normal(scale=0.3, size=10)
            p = p.astype(np.float32)
            assert C.tower_cost(p, init, 10) == pytest.approx(
                float(wcost(sim_w, jnp.asarray(p), jnp.asarray(init))), abs=1e-5)
