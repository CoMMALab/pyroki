"""Every path, task constant and solver default in one place.

Before this module these lived wherever they were first needed, and the
consequences were the kind that waste an afternoon: `study3` imported the URDF
paths from `study0_segment_ablation`, the ground-truth weights from
`recovery_bench`, and the held-out scene offsets from `generalization_check` --
so three "experiments" had to be importable, and stay importable, for any
fourth one to run at all.  Nothing here imports anything from `iosp`.

`THETA_IK_STAR` / `Z_TRAJOPT_STAR` are the DEMONSTRATOR's cost: every synthetic
demonstration in this package is a rollout of the composed model at these
values, which is what makes "did recovery work" a question with an exact
answer rather than a judgement call.
"""

import os
import pathlib

# Set BEFORE `import jax`: XLA reads the host-allocator flag when the backend
# first initialises, and importing this module pulls in jax below.  Doing it
# here means every experiment gets the shared-GPU-friendly default just by
# importing `config`, instead of repeating the line by hand (and getting the
# order wrong).  `setdefault`, so an explicit launch-time value still wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp

# -- resources --------------------------------------------------------------
RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
URDF_PATH = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"
SRDF_PATH = RESOURCE_ROOT / "panda" / "panda.srdf"
MESH_DIR = RESOURCE_ROOT / "panda" / "meshes"

# XLA compile on this composed chain is MEASURED at ~1486s for a single module,
# and every experiment here shares the approach/grasp/place subgraphs, so the
# persistent cache is the difference between a 25-minute rerun and a 2-minute
# one.  `enable_compilation_cache()` must be called before the first trace.
CACHE_DIR = pathlib.Path(__file__).resolve().parent / "data" / "jax_cache"


def setup():
    """One-call experiment startup -- iosp's analogue of `spasm.util.jax_cache_on`.

    Call this ONCE, at the very top of an experiment module, BEFORE importing any
    jax-heavy code: it sets the XLA host-allocator flag (which must be in the
    environment before JAX initialises the backend) and turns on the shared
    persistent compile cache.  Replaces the old scattered pair
    `os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")` +
    `enable_compilation_cache()` that every experiment repeated by hand.

    GPU selection stays on the command line (`CUDA_VISIBLE_DEVICES=<idx>`), the
    same convention SPaSM uses -- never hard-coded here."""
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    enable_compilation_cache()


def enable_compilation_cache():
    """Point JAX at the shared on-disk compile cache.  Call before any jit.

    Settings mirror `spasm.util.jax_cache_on`: cache EVERY module, no size or
    compile-time floor.  The floors matter here because the composed-solver
    graphs (pickplace/tetris/tower, E10's methods) compile as many small XLA
    modules -- at JAX's defaults (5s min compile time) none cleared the bar and
    NOTHING persisted, so every run recompiled from scratch.  `min_entry_size
    = -1` disables the size check outright (SPaSM's setting); `min_compile_time
    = 0` caches regardless of how fast a module compiled.  Autotuning is cached
    separately under `xla_gpu_per_fusion_autotune_cache_dir`."""
    import jax
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# -- the demonstrator's cost (ground truth) ---------------------------------
# grasp standoff, place standoff, place radial, place tangential -- all metres.
# The two in-plane offsets are ZERO here on purpose: the synthetic demonstrator
# releases on the bucket's axis, so every recorded synthetic result is
# reproduced byte-for-byte by the widened parameter vector.
THETA_IK_STAR = jnp.array([0.06, 0.04, 0.0, 0.0], dtype=jnp.float32)
# The tied STANDARD_FEATURES basis: time, path, accel, jerk, effort, clearance.
# Was 7 per-phase entries (approach/grasp/transport/place x {smooth, clearance,
# upright}); the basis is now one shared 6-vector -- see
# `iosp.model.pickplace.STANDARD_FEATURES`.
Z_TRAJOPT_STAR = jnp.array([0.5, 1.0, 2.0, 0.5, 1.0, 2.0], dtype=jnp.float32)
# refine: time, path, accel, jerk, effort, clearance, skeleton
Z_FULL_STAR = jnp.array([0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 1.5], dtype=jnp.float32)

# -- the canonical task ------------------------------------------------------
Q_START = jnp.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8], dtype=jnp.float32)
PICK_POS = jnp.array([0.4, 0.2, 0.3], dtype=jnp.float32)
PLACE_POS = jnp.array([0.4, -0.2, 0.3], dtype=jnp.float32)
OBS_CENTER = jnp.array([0.3, 0.0, 0.4], dtype=jnp.float32)
OBS_RADIUS = jnp.array([0.05], dtype=jnp.float32)

# Held-out scene B: displacements from scene A, scaled by `scene_b(scale)`.
# Chosen large enough that B is a genuine generalization probe and not a
# near-duplicate of A; see `iosp.model.scenes.scene_b`.
SCENE_B_Q_START_OFFSET = jnp.array([0.15, -0.1, 0.0, 0.1, 0.0, -0.1, 0.0], dtype=jnp.float32)
SCENE_B_PICK_OFFSET = jnp.array([0.05, 0.08, -0.03], dtype=jnp.float32)
SCENE_B_PLACE_OFFSET = jnp.array([-0.05, -0.06, 0.04], dtype=jnp.float32)

# -- outer-loop defaults -----------------------------------------------------
# 40 steps, not 12: at 12 neither the wide fit nor the refit is near a minimum
# (path A: init 0.0500 -> wide 0.0371), so a 12-step comparison measures step
# efficiency inside a tiny budget rather than the quality of the identifiable
# subspace, which is the claim these experiments are written to make.
N_STEPS = 40
LR = 0.05
N_ITERS = 60          # inner trajopt iterations
TRACE_FRAC = 0.95     # rank rule for `ioc.identifiability.select_rank(rule="trace")`

# `THETA_IK_STAR` is [0.06, 0.04] m while the trajopt logits are O(1), so the
# raw parameter vector spans two orders of magnitude and a single Adam step
# size cannot serve both blocks.  See `iosp.fit.params.z_scale`.
STANDOFF_SCALE = 0.05  # metres
