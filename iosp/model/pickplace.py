"""Composed pick-and-place planner: differentiable IK chained into differentiable
trajopt, approach -> grasp -> transport -> place.

Architecture
------------
Each IK stage (`grasp_ik`/`place_ik`) is a genuinely separate differentiable
module (`sqp_ik_solve_cuda_batch`) whose output is the boundary condition of
the downstream trajopt segments.  Each segment has its own `solve_implicit`
custom_vjp.  Because `q_pick`/`q_place` flow into the next segment undetached
(no `stop_gradient`), JAX reverse mode composes the IK custom_jvp with each
segment's implicit-adjoint custom_vjp automatically.

Correctness precondition: canonical IK
---------------------------------------
The Panda is 7-DOF against a 6-DOF pose task, so the IK subproblem is
redundant.  pyroffi's default implicit-diff rule assumes minimum-norm
self-motion (pinv), which is wrong on redundant arms.  The canonical
reformulation (`q* = argmin ||q-q_ref||^2 s.t. r(q,t)=0`) gives exact KKT
sensitivity and is forced on for every IK call via
`_implicit_diff.CANONICAL_BY_DEFAULT`.

theta_ik parameterization
-------------------------
`theta_ik` is a standoff offset along a fixed approach axis:

    target_pos = scene.pick_pos (or place_pos) + theta_ik_k * UP_AXIS

This is the only differentiable IK input available -- `cfg_ref` and
`pos_weight`/`ori_weight` are stop-gradiented or baked into the kernel.
`theta_ik` is NOT softmaxed: it is a signed geometric offset, not a cost
weight.  The full outer parameter is `(theta_ik, z_trajopt)`, optimized
jointly; only `z_trajopt` is softmaxed.

Note: the composed chain (IK -> trajopt) is not FD-verifiable at usable step
sizes due to the early-stopping solver's stopping-time discontinuity.  The
adjoint is valid at converged stationary points per `ioc.inner`'s precondition.

theta_trajopt (K=7): `approach.smooth, approach.clearance, grasp.smooth,
transport.smooth, transport.clearance, transport.upright, place.smooth`.

Segments
--------
Each segment reuses `ioc.robot.problem.RobotProblem`/`Scene` unchanged.
`approach`: q_start -> q_pick.  `grasp`: q_pick -> q_pick.
`transport`: q_pick -> q_place.  `place`: q_place -> q_place.
"""

import dataclasses
import os

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

# Persistent XLA compilation cache: this module's forward solver (IK stage +
# 4 chained trajopt segments, implicit-adjoint differentiated) compiles for
# 30-90+ min per distinct shape signature. Without this, EVERY fresh process
# (i.e. every debug/iteration run of any iosp script) pays that cost from
# scratch even when nothing about the traced function changed -- measured
# directly in this investigation. This caches compiled XLA executables to
# disk keyed by the HLO's hash, so a second run with the same shapes/config
# loads from disk in seconds instead of recompiling. Override the directory
# via IOSP_JAX_CACHE_DIR; safe to share across users/processes (read-mostly
# after first population), no correctness effect either way.
jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("IOSP_JAX_CACHE_DIR", os.path.expanduser("~/.cache/jax_pyroffi_iosp")),
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from ioc.robot.problem import RobotProblem, Scene

# Read at CALL time inside `differentiable_ik_solution_batch`, so setting this
# after `pyroffi` is imported still takes effect -- see module docstring.
from pyroffi.optimization_engines import _implicit_diff
_implicit_diff.CANONICAL_BY_DEFAULT = True
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda_batch

# -- the `upright` feature -----------------------------------------------------
# MEASURED on the ten FR3 teleop episodes (scratch/e10_upright_diag.py): the old
# residual `quat[1:3]` is ~0.985 on the demonstrations, i.e. essentially its
# MAXIMUM.  Its zero is not the down orientation -- a straight-down gripper is
# `DOWN_WXYZ=[0,1,0,0]`, whose `quat[1:3]=[1,0]` -- so minimizing it drives the
# wrist toward pointing UP, and any fit on human demos can only respond by
# starving the weight.  `iosp/checks/identifiability.py` measured the same thing
# from the other side on synthetic data: cos(transport.smooth, transport.upright)
# = -0.9999, and `e1_minimal_identifiable` swapped the feature out for that
# reason.  It is replaced here rather than worked around per-experiment.
#
# The correct, yaw-free measure is the world-frame xy of the EE approach axis
# R(quat) @ [0,0,1] = [2(xz+wy), 2(yz-wx)]: zero for any down(+yaw) grasp and
# nonzero exactly when the tool tips off vertical.  This is the same quantity
# `upright_constraint_fn` already uses, so the soft feature and the hard
# constraint can no longer disagree about what "upright" means.
#
# DEADBAND: the operators are not gripper-down.  Their tilt off vertical is 17.7
# deg on average over the whole path and 20.8 deg (max 49.9) through transport,
# so a term whose zero is exact verticality penalizes motion the demonstrations
# actually contain and is misspecified even in its corrected form.  The feature
# is therefore a hinge outside a deadband: inert on demonstrated tilt, and a
# grasp-retention ("do not tip the object out") penalty only past it.
# `UPRIGHT_TILT_DEADBAND` is sin(tilt), so 0.5 = 30 deg, above the demonstrated
# mean and below the spill-grade excursions.  Softplus-smoothed on the same
# reasoning as `RobotProblem.clearance_residual`: a hard hinge is
# nondifferentiable exactly on the ridge the solver rides, which stalls the
# inner solve and invalidates the implicit adjoint.
UPRIGHT_TILT_DEADBAND = 0.5
UPRIGHT_SOFTNESS = 20.0


def tool_axis_tilt_residual(quat, deadband=UPRIGHT_TILT_DEADBAND):
    """(T, 4) wxyz -> (T,) hinge on how far the tool axis tips off vertical.

    Zero (to softplus precision) while the tilt is within `deadband`; grows
    linearly beyond it.  Yaw about world z is free by construction.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    axis_xy = jnp.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x)], axis=-1)
    # sqrt is smoothed away from 0: the gradient of ||a|| is undefined at a = 0,
    # which is precisely where a well-behaved (vertical) waypoint sits.
    tilt = jnp.sqrt(jnp.sum(axis_xy ** 2, axis=-1) + 1e-12)
    return jax.nn.softplus(UPRIGHT_SOFTNESS * (tilt - deadband)) / UPRIGHT_SOFTNESS

N_APPROACH = 8
N_GRASP = 4
N_TRANSPORT = 10
N_PLACE = 4
PHASES = ("approach", "grasp", "transport", "place")
SEGMENT_LEN = {"approach": N_APPROACH, "grasp": N_GRASP,
               "transport": N_TRANSPORT, "place": N_PLACE}

# -- the cost basis ----------------------------------------------------------
# The standard trajopt basis, TIED across phases: one weight vector, applied to
# approach/grasp/transport/place alike.
#
# Tied, not per-phase, for a measured reason.  Each segment minimizes
# `sum_i w_i ||r_i||^2` with FIXED endpoints, so its solution depends only on the
# RATIOS of that phase's own weights -- the per-phase scale is gauge.  Under the
# old per-phase basis that left `grasp` and `place` (single-feature DWELL
# segments, `Scene(q_pick, q_pick)`) with weights that could not change the
# answer at all, and the identifiable count was approach 1 + transport 2 + 0 + 0
# = 3, which is exactly the Gram rank E10 measured.  A tied vector has ONE global
# scale gauge instead of four, so 6 features give 5 identifiable directions.
#
# The features, and how each scales with the segment duration T -- the spread of
# exponents is what makes the speed/smoothness trade-off well posed:
#
#   time       T                  the duration itself
#   path       T^0                EE arc length; purely geometric
#   accel      T^-2               joint acceleration (the old `smooth`)
#   jerk       T^-3               joint jerk
#   effort     mixed              RNEA torque: inertial ~T^-2, gravity ~T^0
#   clearance  T^0                hinge on distance to the world
#
# `path` is measured in EE space, not joint space, deliberately: a joint-space
# arc length is a first difference and would sit in the same family as `accel`
# and `jerk`, which is how `upright` came to be -0.9999 collinear with `smooth`
# (see `iosp/checks/identifiability.py` and `e1_minimal_identifiable`).  A
# Cartesian path length is structurally unrelated to the joint-space derivatives.
#
# `upright` is GONE from the basis.  Keeping the gripper level during the carry
# is grasp retention -- a task requirement, not a preference -- and belongs in
# `upright_constraint_fn`'s hard constraint, which is where `build_teleop`'s own
# "task events -> constraints, free space -> cost" philosophy puts it.
STANDARD_FEATURES = ("time", "path", "accel", "jerk", "effort", "clearance")

# Kept as a dict keyed by phase so every existing caller that iterates
# `SEGMENT_FEATURES[phase]` is unchanged; every phase now carries the same basis.
SEGMENT_FEATURES = {p: STANDARD_FEATURES for p in PHASES}

# Nominal segment durations, seconds -- MEASURED as the median over the ten FR3
# teleop episodes (scratch/e10_timing.py; the raw `stamp_ns` at the gripper
# grasp/release cuts).  These set `T = duration_nominal * exp(s)`, so `s = 0`
# (the seed) is a typical demonstration rather than an arbitrary scale.
# `grasp` is NOT measured: `to_waypoints` cuts it as `(grasp_i, grasp_i + 1)`, a
# SINGLE sample, so its "duration" is one sampling period (0.036 s at 27.8 Hz) --
# an artefact of the skeleton, not a demonstrated quantity.  Feeding that through
# `dt = T / (n - 1)` amplifies this dwell's `accel` by ~5.6e3 and its `jerk` by
# ~4.4e5 against every other phase, and the inner solve stops converging
# (measured: grasp stationarity 2.1e-1 against a 1e-3 tolerance, the worst of the
# four phases).  It gets a plain nominal second instead, and is excluded from the
# duration term of the loss -- there is no measurement there to fit.
DURATION_NOMINAL = {"approach": 7.45, "grasp": 1.0, "transport": 7.03, "place": 4.67}
# Phases whose demonstrated duration is real, and therefore fittable.
TIMED_PHASES = tuple(p for p in PHASES if p != "grasp")


def unpack_z(z):
    """Flat outer vector -> (theta_ik, z_trajopt).  Layout lives with the model."""
    return z[:K_IK], z[K_IK:]

def split_trajopt(theta_trajopt):
    """Flat (K_TRAJOPT,) weight vector -> {phase: weights}.

    The basis is TIED, so this hands every phase the same vector.  The signature
    is unchanged from the per-phase version, which is what keeps every caller of
    the composed forward map working.
    """
    return {p: theta_trajopt for p in PHASES}


def split_trajopt_perseg(theta_trajopt):
    """Flat (K_TRAJOPT_PERSEG,) weight vector -> {phase: weights}.

    Per-segment variant: the first N_FEAT entries go to approach, the next N_FEAT
    to grasp, etc.  Each phase gets its own independent weight block.
    """
    nf = len(STANDARD_FEATURES)
    return {p: theta_trajopt[i * nf:(i + 1) * nf] for i, p in enumerate(PHASES)}


THETA_TRAJOPT_NAMES = STANDARD_FEATURES
K_TRAJOPT = len(THETA_TRAJOPT_NAMES)

N_FEAT_PER_SEG = len(STANDARD_FEATURES)
K_TRAJOPT_PERSEG = N_FEAT_PER_SEG * len(PHASES)
THETA_TRAJOPT_PERSEG_NAMES = tuple(
    f"{p}.{f}" for p in PHASES for f in STANDARD_FEATURES)
# The release point is NOT the bucket's axis.  MEASURED on the teleop set: every
# one of ten operators let go 6.3 cm SHORT of the bucket centre, toward the arm
# base (std 1.8 cm), plus 1.6 cm tangentially -- i.e. over the NEAR RIM, the
# bucket's inner radius being 6.5 cm.  That is a preference, not noise, and a
# +z standoff cannot express it: it is perpendicular to the only direction
# `place.standoff` can move.  So the release target carries two more fitted
# coordinates, in the frame spanned by the base->bucket direction:
#
#   place.radial      + is AWAY from the arm base   (fits ~ -0.063 m)
#   place.tangential  + is the left-hand normal      (fits ~ +0.016 m)
#
# Scene-derived, so they transfer to a bucket somewhere else -- which is what
# keeps the release point PREDICTED rather than read off the demonstration.
# Fitting a single global pair cuts release position error 0.068 -> 0.022 m.
THETA_IK_NAMES = ("grasp.standoff", "place.standoff",
                  "place.radial", "place.tangential")
K_IK = len(THETA_IK_NAMES)

# -- stage 3: the global refine pass -----------------------------------------
# spasm's forward pass is three stages, not two: seed the endpoints with IK,
# trajopt BETWEEN the endpoints, then trajopt the ENTIRE trajectory.  Until now
# this module implemented only the first two, so every iosp experiment was
# inverting a forward model the demonstrator never ran.  The third stage is not
# cosmetic: the per-segment solves are independent given their shared boundary
# conditions, so nothing in stages 1-2 can trade curvature ACROSS a phase
# boundary -- the concatenated path is C0 but kinked at every junction.  The
# refine pass is exactly the stage that removes those kinks, and it is where a
# demonstrator's global preferences (how much smoothness is worth giving up
# skeleton fidelity for) actually live.
#
# Concatenating the four segments and dropping the duplicated boundary row
# leaves 8 + 3 + 9 + 3 = 23 waypoints.  The phase spans below index INTO that
# concatenated path (half-open, and overlapping by one row at each junction --
# row 7 is both approach's last and grasp's first, which is the point).
N_FULL = N_APPROACH + (N_GRASP - 1) + (N_TRANSPORT - 1) + (N_PLACE - 1)
PHASE_SPAN = {}
_start = 0
for _p in PHASES:
    _n = SEGMENT_LEN[_p]
    PHASE_SPAN[_p] = (_start, _start + _n)
    _start += _n - 1
del _start, _p, _n
assert PHASE_SPAN["place"][1] == N_FULL, PHASE_SPAN

# The task skeleton, as indices into the concatenated path: where the discrete
# task structure says the gripper must BE at a pick/place pose.  `unpack`
# already clamps row 0 to q_start and row N_FULL-1 to q_place, so those two are
# enforced by construction; these three are the ones the refine pass could
# otherwise walk away from, and the `skeleton` feature is what prices that.
SKELETON_PICK = (PHASE_SPAN["approach"][1] - 1, PHASE_SPAN["grasp"][1] - 1)   # (7, 10)
SKELETON_PLACE = (PHASE_SPAN["transport"][1] - 1,)                            # (19,)

# -- the TIED cost model -----------------------------------------------------
# One preference vector, shared by every stage, instead of one weight block per
# segment plus another for the refine pass.  Two reasons this is the better
# model and not just a smaller one:
#
# 1. It is what makes the composition differentiable.  With per-stage weights,
#    a segment block theta_s reaches the loss ONLY through the warm start x_bar
#    -- and a converged argmin has zero seed-sensitivity (x0 never enters the
#    stationarity condition), so d(loss)/d(theta_s) = 0 EXACTLY and the segment
#    weights are dead parameters.  Sharing theta puts it directly in stage 3's
#    own stationarity condition, where the implicit adjoint can see it.
# 2. It is the more defensible claim.  "The demonstrator has one set of
#    preferences, applied to whatever trajectory it is currently optimizing"
#    is a hypothesis about an agent; 11 independent per-stage weights is a
#    hypothesis about a pipeline.  It also takes K from 13 to 4, which matters
#    on a problem whose Gram spectrum is already near-rank-deficient.
#
# theta_ik is NOT fitted under this model: IK is a fixed seeding step, not a
# preference.  Holding it constant also makes every stage's context constant
# w.r.t. the fitted parameters.
#
# `skeleton` is refine-only, and does not need masking at the segment level: a
# segment's pick/place rows ARE its clamped boundary conditions, so its
# skeleton residual is identically zero for every feasible x.  Including it
# would only rescale that segment's cost by a constant, which leaves its argmin
# unchanged -- so segments simply carry the first three features, and the
# whitening scales stay per-stage (sigma is units, theta is preference).
SHARED_FEATURES = STANDARD_FEATURES
THETA_SHARED_NAMES = SHARED_FEATURES + ("skeleton",)
K_SHARED = len(THETA_SHARED_NAMES)

FULL_FEATURES = STANDARD_FEATURES + ("skeleton",)
THETA_FULL_NAMES = tuple(f"refine.{f}" for f in FULL_FEATURES)
K_FULL = len(THETA_FULL_NAMES)

UP_AXIS = jnp.array([0.0, 0.0, 1.0])
DOWN_WXYZ = jnp.array([0.0, 1.0, 0.0, 0.0])  # gripper facing down at the target
IK_RNG_KEY = jax.random.PRNGKey(0)  # fixed: gradients don't flow through it

# Winner-selection continuity for the redundant IK.  MEASURED root cause of the
# behavioural-loss spikes in the path-A reconstruction: `sqp_ik_solve_cuda_batch`
# runs `num_seeds=32` seeds in parallel and picks a winner, and at the library
# default `continuity_weight=0.0` that choice is made on POSE ERROR ALONE.  The
# Panda is 7-DOF against a 6-DOF pose task, so the seeds land all over a 1-D
# self-motion manifold whose members all hit the target equally well; the argmax
# among them then flips arbitrarily as the target slides with theta_ik.
#
# Measured on the 40-step path-A fit (`scratch/ik_branch_check.py`): at outer
# steps 14/17/36/37 the chosen `q_pick` jumped 2.5-4.4 rad against an off-spike
# median of 0.007 (up to 712x) while the achieved EE position moved 0.0002-0.002 m,
# exactly on its smooth trend -- joints teleport, pose does not move.  Downstream,
# every segment bounded by that configuration (approach/grasp/transport) swings
# through a different region of Cartesian space, spiking held-out EE RMSE 20-40x
# for one outer step.  It is NOT a trajopt basin flip: `n_restarts=3` on every
# segment left the spikes bit-identical, because the discontinuity is upstream.
#
# A positive weight makes the winner selection prefer the branch nearest
# `previous_cfgs`, which is what makes q*(theta_ik) continuous.  It only breaks
# ties: converged pose residuals are ~1e-8 while branch separations are O(1) rad,
# so this changes WHICH equally-valid solution is returned, not how well the
# target is hit.
#
# CHANGES RECORDED NUMBERS.  Every iosp result predating this used 0.0.
IK_CONTINUITY_WEIGHT = 1.0


def make_composed_forward_solver(n_iters=60, *, soft_line_search=True,
                                 soft_curvature_gate=True, robot=None, method=None):
    """The forward solve for the 4 chained trajopt segments -- deliberately
    NOT `ioc.robot.e1_identifiability.make_dynamics_forward_solver`'s default
    (early-stopping `while_loop`, `DynamicsTrajOptConfig(early_stop=True)`).

    MEASURED root cause of the composed chain's FD check failing (cos=-0.71
    against the implicit adjoint, FD estimate doubling every eps-halving
    across 6 octaves, never converging): early stopping's `grad_tol`-gated
    `while_loop` has a data-dependent trip count, so q*(theta_ik) is not just
    "hard to differentiate" but genuinely DISCONTINUOUS in theta_ik -- an
    infinitesimal shift in the (smooth) boundary condition `q_pick`/`q_place`
    can flip which iteration crosses `grad_tol` and land the solver on a
    different final iterate.  That breaks the one precondition `solve_
    implicit`'s implicit-function-theorem adjoint actually needs (see `ioc.
    inner`'s module docstring: "provided the solve actually converged" to a
    point that varies smoothly with the inputs) -- not a bug in the adjoint
    itself, a violated precondition upstream of it.

    Fix: `DynamicsTrajOptConfig(early_stop=False, unroll_tail=0)` runs a
    FIXED-length `jax.lax.scan` for exactly `n_iters` steps -- no data-
    dependent trip count, so q*(theta_ik) is smooth by construction (the same
    L-BFGS step function, unrolled a fixed number of times, is a composition
    of smooth maps).  `unroll_tail=0` is deliberate: `solve_implicit`'s
    forward pass already wraps this call in `jax.lax.stop_gradient` (its
    gradient comes entirely from the analytic adjoint in `ioc.inner`'s `_bwd`,
    never from differentiating this solve), so the `early_stop=False` branch's
    own truncated-unroll differentiability is irrelevant here -- only its
    fixed-trip-count SHAPE matters.

    Does not touch `ioc.inner.make_inner_solver`'s semantics or any other
    caller's forward solver: this is a different value passed into the same
    pluggable `forward_solver` slot every caller already uses (`e1_
    identifiability.py`, `bench2d`, ...), which are unaffected.
    """
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt

    # `early_stop=False` alone did NOT fix the composed chain's FD check
    # (measured: cos(implicit, FD) = -0.707, unchanged from the early-stopping
    # baseline's -0.7066) -- the outer while_loop's data-dependent trip count
    # was not the only discontinuity.  `soft_line_search=True` (the per-step
    # line search's hard argmax over 5 trial alphas) ALSO measured no change
    # on its own (-0.7066, and re-confirmed non-degenerate via jaxpr diffing
    # after finding and fixing a temperature-scaling bug in it).
    # `soft_curvature_gate=True` targets the third hard branch in this step
    # function: the L-BFGS curvature-pair admit/reject gate and the
    # `direction = where(m_used > 0, dir_lbfgs, dir_gd)` switch it feeds -- see
    # `DynamicsTrajOptConfig.soft_curvature_gate`'s docstring.
    #
    # CAUTION on stacking the two soft flags (default kept for continuity with
    # every result recorded before 2026-08-27, but they are now parameters so
    # the choice can be measured rather than assumed).  On the SINGLE-segment
    # problem, `ioc.diagnostics`'s roughness/FD probe measured:
    #
    #   early_stop=False only            rough/|g| = 72.2   FD cos = 0.906
    #   + both soft flags (this default) rough/|g| = 17.7   FD cos = 0.324
    #
    # i.e. softening the line search and curvature gate buys a smoother
    # landscape at the cost of AGREEMENT with finite differences -- consistent
    # with softening moving the fixed point that the implicit adjoint
    # linearizes about, so the adjoint answers a question about a slightly
    # different solve than the one that ran.  That is the opposite of what
    # stacking them was intended to achieve.  Whether the composed chain
    # behaves like the single segment here is NOT yet measured; until it is,
    # treat `soft_line_search=soft_curvature_gate=False` as the candidate
    # configuration for any run whose gradient fidelity matters.
    method = method or os.environ.get("IOSP_TRAJOPT", "lbfgs")
    if method == "projected_gd" and robot is not None:
        lo = tuple(float(v) for v in np.asarray(robot.joints.lower_limits))
        hi = tuple(float(v) for v in np.asarray(robot.joints.upper_limits))
        opt_cfg = DynamicsTrajOptConfig(n_iters=n_iters, method="projected_gd",
                                        gd_lr=0.1, q_lo=lo, q_hi=hi, dof=len(lo))
    else:
        opt_cfg = DynamicsTrajOptConfig(n_iters=n_iters, early_stop=False, unroll_tail=0,
                                        soft_line_search=soft_line_search,
                                        soft_curvature_gate=soft_curvature_gate)

    def forward_solver(x0, cost_fn):
        return dynamics_trajopt(x0, cost_fn, opt_cfg)

    return forward_solver


def make_stock_forward_solver(n_iters=60):
    """Forward-only solver using the STOCK trajopt config (early_stop=True, hard
    line search + hard curvature gate) -- the original SPaSM solver.

    Compiles FAR faster than `make_composed_forward_solver`: it drops the two
    soft surrogates (`soft_line_search`/`soft_curvature_gate`) and the
    fixed-length unroll (`early_stop=False`) that exist ONLY to make q*(theta)
    smooth for truncated-unrolling gradients, and those are what blow up the
    cold compile.

    SAFE for the implicit adjoint and for every derivative-free method.
    `ioc.inner.solve_implicit` runs the forward under `stop_gradient` and
    rebuilds curvature analytically at x* (H = jax.hessian(cost), one linear
    solve), so -- per that module's docstring -- it "only needs x* to be a
    stationary point, not to know how it was found".  Early stopping converges
    to `grad_tol`, a TIGHTER stationary point than a fixed-length soft run, so
    the adjoint's convergence precondition is if anything better met.

    NOT for `solve_unrolled`, which differentiates THROUGH the solver and must
    keep the fixed-length soft forward (`make_composed_forward_solver`)."""
    from pyroffi.optimization_engines import DynamicsTrajOptConfig, dynamics_trajopt

    cfg = DynamicsTrajOptConfig(n_iters=n_iters)

    def forward_solver(x0, cost_fn):
        return dynamics_trajopt(x0, cost_fn, cfg)

    return forward_solver


# Bucket wall discretisation.  The recorder builds the bucket as `n_walls`
# boxes on a circle; iosp's clearance is sphere-based, so the wall becomes a
# ring of spheres of the wall's own half-thickness, stacked up its height.
# 12 x 3 = 36 spheres is enough to close the ring at this radius (the angular
# gap between sphere centres is ~0.036 m at r = 0.0655, well under a sphere
# diameter) without making the soft-min reduction expensive.
BUCKET_RING_AZIMUTHS = 12
BUCKET_RING_LEVELS = 3


def bucket_spheres(center, inner_radius, wall_height, wall_thickness):
    """Bucket wall -> `(M, 4)` xyz+radius collision spheres, batched over scenes.

    `center` is `(..., 3)` with z at the table top; the returned array carries
    the same leading batch dimensions.  Returns None if the geometry is absent,
    which is what every synthetic scene does.
    """
    if center is None or inner_radius is None:
        return None
    A, L = BUCKET_RING_AZIMUTHS, BUCKET_RING_LEVELS
    th = jnp.linspace(0.0, 2.0 * jnp.pi, A, endpoint=False)          # (A,)
    frac = jnp.arange(1, L + 1) / L                                   # (L,)

    r_w = 0.5 * (wall_thickness[..., 0] if wall_thickness is not None
                 else jnp.full(inner_radius[..., 0].shape, 0.01))
    r_ring = inner_radius[..., 0] + r_w                               # (...,)
    h = wall_height[..., 0] if wall_height is not None else jnp.full(r_ring.shape, 0.12)

    # Everything to (..., L, A) explicitly: the level and azimuth axes broadcast
    # against different trailing dims, so relying on implicit rules here silently
    # produced (..., 1, A) for x/y against (..., L, A) for z.
    cos_t, sin_t = jnp.cos(th)[None, :], jnp.sin(th)[None, :]         # (1, A)
    x = center[..., None, None, 0] + r_ring[..., None, None] * cos_t
    y = center[..., None, None, 1] + r_ring[..., None, None] * sin_t
    z = center[..., None, None, 2] + h[..., None, None] * frac[:, None]
    shape = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
    x, y, z = (jnp.broadcast_to(v, shape) for v in (x, y, z))
    rad = jnp.broadcast_to(r_w[..., None, None], shape)

    out = jnp.stack([x, y, z, rad], axis=-1)
    return out.reshape(*out.shape[:-3], L * A, 4)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PickPlaceScene:
    """Context for one pick-and-place demonstration (one row per batch element
    when used under `jax.vmap`/a leading batch dim, matching `ioc.robot.problem
    .Scene`'s convention)."""

    q_start: jnp.ndarray  # (dof,)
    pick_pos: jnp.ndarray  # (3,)
    place_pos: jnp.ndarray  # (3,)
    obs_center: jnp.ndarray  # (3,)
    obs_radius: jnp.ndarray  # (1,)
    # Yaw of the object to be grasped, about world +z, in radians -- (1,).
    #
    # MEASURED on the teleop demonstrations, and the reason this field exists:
    # the IK stage targeted a FIXED `DOWN_WXYZ`, i.e. a straight-down, zero-yaw
    # grasp, while the human operator rotated the wrist to line the fingers up
    # with the cube's faces -- +-40 deg of tool yaw, tracking the `cube_yaw` the
    # scene was randomised with.  `theta_ik` cannot absorb that: it is a scalar
    # standoff along +z with no orientation degree of freedom at all.  So
    # `q_pick` was the solution to a grasp the demonstrator never used, in every
    # episode, before the trajopt stage even started.
    #
    # `None` (the default) means "no yaw information", which reproduces the old
    # fixed-DOWN behaviour exactly and keeps all 14 synthetic construction sites
    # working unchanged.  None is a valid pytree leaf-less node, so `jax.vmap`
    # and `jax.tree.map` over a scene simply skip it.
    pick_yaw: jnp.ndarray = None  # (1,) or None

    # -- the real world, from the recorded scene ----------------------------
    # All None on every synthetic scene, which reproduces the old
    # single-placeholder-sphere world exactly.  See
    # `iosp_scene_fields` in sim_teleop for why these had to start being
    # emitted: the placeholder sphere sits in the transport corridor and the
    # demos penetrate it, while the table and bucket -- the only things the
    # operator actually avoided -- were not in the world at all.
    table_box: jnp.ndarray = None  # (6,) [c_xyz, half_xyz] of the table
    # (4,) demonstrated seconds per phase, in PHASES order.  The waypoint export
    # resamples each phase to a fixed row count, so this is the ONLY place the
    # demonstration's timing survives -- and it is what makes the `time` feature
    # identifiable rather than gauge.  See `iosp.fit.teleop`'s duration term.
    phase_durations: jnp.ndarray = None
    bucket_center: jnp.ndarray = None  # (3,) xy of the bucket axis, z at the table
    bucket_inner_radius: jnp.ndarray = None  # (1,)
    bucket_wall_height: jnp.ndarray = None  # (1,)
    bucket_wall_thickness: jnp.ndarray = None  # (1,)

    # The ANCHORED grasp: orientation and seed taken from the demonstration
    # rather than predicted from the scene.  Both optional, both None on every
    # synthetic scene, which is the old behaviour exactly.
    #
    # MEASURED on the teleop set, as ||q_pick - q_demo_grasp|| against a 0.938
    # rad approach motion -- i.e. what fraction of the whole reach the seed is
    # already wrong by before trajopt starts:
    #
    #   DOWN target,      seed q_start     0.618   (every run before this)
    #   DOWN*yaw target,  seed q_start     0.530
    #   DOWN*yaw target,  seed demo cfg    0.504   <- seeding ALONE is ~noise
    #   pick_wxyz target, seed q_start     0.146
    #   pick_wxyz target, seed grasp_ref   0.084   <- both, 9% of the motion
    #
    # The middle row is the one to keep in mind: `previous_cfg` selects among
    # solutions to the target POSE, so it can only slide q_pick along the
    # self-motion manifold and cannot repair a wrong orientation.  Anchoring
    # means the orientation, and the seed is a second-order refinement on top.
    #
    # SCOPE, stated because it changes the claim and not just the number: with
    # `pick_wxyz` set, the pipeline no longer predicts HOW to grasp -- the grasp
    # pose is an input.  On held-out episodes that is demonstration information
    # entering at test time, so the honest reading of a generalization result
    # becomes "given this episode's grasp, the fitted cost reproduces the
    # motion".  That is a normal skeleton-given formulation and it isolates the
    # trajectory cost from grasp selection, which is what iosp is about -- but
    # it is a different claim from the one the synthetic experiments make.
    pick_wxyz: jnp.ndarray = None   # (4,) target grasp orientation, or None
    grasp_ref: jnp.ndarray = None   # (dof,) IK seed for the grasp, or None

    # The release, anchored the same way.  There is no `place_yaw` middle tier
    # to fall back on: the bucket is an n-gon approximating a cylinder, so it
    # has no orientation to predict from, and these are either demonstrated or
    # absent.  MEASURED on the teleop set, the fixed straight-down target is
    # even further from the demonstration here than at the grasp -- 19-50 deg
    # of tilt, because the operator TIPS the gripper to dump the cube in rather
    # than lowering it flat.
    place_wxyz: jnp.ndarray = None  # (4,) target release orientation, or None
    place_ref: jnp.ndarray = None   # (dof,) IK seed for the release, or None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FullScene:
    """Context for the stage-3 refine solve.

    Field names `q_start`/`q_goal`/`obs_center`/`obs_radius` deliberately match
    `ioc.robot.problem.Scene` so `RobotProblem.unpack`, `.seed`,
    `.clearance_residual` and `make_world` all accept this unchanged -- the
    refine stage reuses the SAME problem machinery as the segments, at a
    different `n_timesteps`.  `q_pick` is the extra field: it is not a boundary
    of the full path (the gripper passes THROUGH it mid-trajectory), so it
    cannot be carried as `q_start`/`q_goal` and has to ride along in the
    context for the `skeleton` feature to reference.

    Both `q_pick` and `q_goal` (= q_place) come from the IK stage undetached,
    so the refine solve's implicit adjoint composes back through stage 1 the
    same way the segment solves already do.
    """

    q_start: jnp.ndarray  # (dof,)  == scenes.q_start
    q_goal: jnp.ndarray  # (dof,)  == q_place
    obs_center: jnp.ndarray  # (3,)
    obs_radius: jnp.ndarray  # (1,)
    q_pick: jnp.ndarray  # (dof,)
    obs_spheres: jnp.ndarray = None  # (M, 4) xyz+radius, or None
    table_box: jnp.ndarray = None  # (6,) center+half-extents, or None


# A square cube's parallel-jaw grasp repeats every 90 deg, so a target yaw and
# that yaw +- 90 deg are the SAME grasp.  Wrapping into [-45, 45] deg picks the
# representative nearest the un-yawed pose, which keeps the wrist away from its
# limits and stops a 41 deg cube reading as a 49 deg rotation the other way.
GRASP_YAW_PERIOD = jnp.pi / 2


def _wrap_grasp_yaw(yaw):
    return yaw - GRASP_YAW_PERIOD * jnp.round(yaw / GRASP_YAW_PERIOD)


def _down_yaw_wxyz(yaw):
    """(N, 4): `DOWN_WXYZ` rotated by `yaw` about WORLD +z.

    Pre-multiplied, not post-: the cube's yaw is a rotation of the world-frame
    approach, and composing on the other side would rotate about the tool's own
    axis after it has already been flipped upside down, which is the same angle
    the wrong way round.
    """
    yaw = _wrap_grasp_yaw(jnp.reshape(yaw, (-1,)))
    R = jaxlie.SO3.from_z_radians(yaw) @ jaxlie.SO3(
        wxyz=jnp.broadcast_to(DOWN_WXYZ, (yaw.shape[0], 4)))
    return R.wxyz


def _place_frame(place_pos):
    """(radial, tangential) unit vectors in the xy plane, per batch row.

    Radial points from the arm base (world origin, where the FR3 stands) out to
    the bucket; tangential is its left-hand normal.  Defined from the SCENE, so
    a fitted offset means the same thing when the bucket moves -- expressing it
    in world xy instead would make `place.radial` a different preference for
    every episode.
    """
    xy = place_pos[..., :2]
    r = xy / (jnp.linalg.norm(xy, axis=-1, keepdims=True) + 1e-9)
    radial = jnp.concatenate([r, jnp.zeros_like(r[..., :1])], axis=-1)
    tangential = jnp.stack([-r[..., 1], r[..., 0],
                            jnp.zeros_like(r[..., 0])], axis=-1)
    return radial, tangential


def _place_target(scenes, theta_ik):
    """The release target: bucket, lifted by the standoff, shifted in-plane."""
    radial, tangential = _place_frame(scenes.place_pos)
    return (scenes.place_pos + theta_ik[1] * UP_AXIS
            + theta_ik[2] * radial + theta_ik[3] * tangential)


def _target_pose_batch(pos_batch, wxyz=DOWN_WXYZ):
    """Cast to float32 at the IK call boundary: the canonical-IK CUDA kernel's
    custom_jvp is float32-only (same boundary as GRiD's FFI -- see
    `ioc.robot.bases.dynamic`'s docstring for the analogous case), so under
    `JAX_ENABLE_X64=1` a float64 target silently mismatches the tangent dtype
    the custom rule expects and `jax.grad` raises rather than truncating
    quietly.  Casting here keeps the rest of the composed graph (trajopt) at
    its own working dtype."""
    n = pos_batch.shape[0]
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=jnp.broadcast_to(wxyz, (n, 4)).astype(jnp.float32)),
        translation=pos_batch.astype(jnp.float32),
    )


def _ik_batch(problem, target_pos, refs, wxyz=DOWN_WXYZ):
    """Canonical IK over a flat batch of (target, ref) rows.

    The batch axis is whatever the caller folded into it: scenes, IK branches,
    bilevel candidates, or all three.  `sqp_ik_solve_cuda_batch` now carries its
    own `custom_vmap` rule, so `jax.vmap` around this works natively and folds
    the mapped axis into the kernel's problem axis (one launch); no wrapper is
    needed here any more.

    Hand-flattening and vmapping are numerically IDENTICAL (measured 0.0), but
    neither is row-for-row identical to N separate calls: the kernel seeds from
    `rng_key` AND each problem's position in the batch, so a row can land on a
    different IK branch depending on how it was batched.  `IK_CONTINUITY_WEIGHT`
    is what makes that harmless -- winner selection prefers the branch nearest
    `refs`, so the returned branch follows the reference, not the seeding.

    `wxyz` defaults to a fixed straight-down grasp.  Callers that have an
    anchored or object-yaw orientation MUST pass it: this is the entry point
    `multistart`'s flattened forward map uses instead of `grasp_ik`/`place_ik`,
    and while it defaulted silently the whole anchoring path was inert inside
    the fit -- an anchored run and an unanchored one computed the same thing and
    differed only by solver nondeterminism.
    """
    return sqp_ik_solve_cuda_batch(
        problem.base.robot, problem.ee_index, _target_pose_batch(target_pos, wxyz),
        IK_RNG_KEY, refs.astype(jnp.float32),
        continuity_weight=IK_CONTINUITY_WEIGHT,
    ).astype(refs.dtype)


@dataclasses.dataclass(frozen=True)
class PickPlaceProblem:
    """The composed planner: an IK stage plus four chained trajopt segments,
    all sharing one robot/collision model."""

    base: RobotProblem
    seg: dict  # phase name -> RobotProblem (same robot, different n_timesteps)

    @property
    def dof(self):
        return self.base.dof

    @property
    def ee_index(self):
        return self.base.ee_index

    @staticmethod
    def load(urdf_path, srdf_path, mesh_dir, ee_link=None):
        """`ee_link` is passed straight through to `RobotProblem.load`; None
        keeps its Panda default, so every existing caller is unchanged.  The
        teleop fit passes the FR3's hand frame -- see `iosp.model.fr3`."""
        kw = {} if ee_link is None else {"ee_link": ee_link}
        base = RobotProblem.load(urdf_path, srdf_path, mesh_dir, n_timesteps=2, **kw)
        # `n_aux=1` is the segment's log-duration; see `RobotProblem.n_aux`.
        seg = {p: dataclasses.replace(base, n_timesteps=SEGMENT_LEN[p], n_aux=1,
                                      duration_nominal=DURATION_NOMINAL[p])
               for p in PHASES}
        # The stage-3 refine problem shares the same robot/collision model and
        # differs only in n_timesteps, exactly as the segments do.  Keyed
        # "full" in the same dict rather than a separate field so nothing that
        # iterates `seg` over PHASES sees a change.
        seg["full"] = dataclasses.replace(
            base, n_timesteps=N_FULL, n_aux=1,
            duration_nominal=sum(DURATION_NOMINAL.values()),
            pinned_rows=(tuple((i, "q_pick") for i in SKELETON_PICK)
                         + tuple((i, "q_goal") for i in SKELETON_PLACE)))
        return PickPlaceProblem(base=base, seg=seg)

    # -- IK stage ----------------------------------------------------------

    def grasp_ik(self, theta_ik, scenes: PickPlaceScene):
        """(M, dof) previous-cfg batch -> (M, dof) canonical IK solution at
        `pick_pos + theta_ik[0] * UP_AXIS`.  See module docstring for why the
        standoff offset (not weight balance) is `theta_ik`'s meaning.

        `previous_cfg` is cast to float32 for the same reason `_target_pose_
        batch` casts the target: the canonical-IK custom_jvp is float32-only
        throughout (q, cfg_ref, AND target all have to agree), and mixing in a
        float64 `previous_cfg` under x64 promotes the kernel's own output to
        float64 by ordinary JAX promotion rules, which then mismatches the
        rule's declared float32 tangent -- not a target-only issue.  Cast the
        *output* back up so the trajopt stage downstream keeps its own dtype.
        """
        dtype = scenes.q_start.dtype
        target = scenes.pick_pos + theta_ik[0] * UP_AXIS
        # Orientation follows the object's yaw when the scene records one; see
        # `PickPlaceScene.pick_yaw`.  `place_ik` deliberately does NOT do this:
        # the bucket is an n-gon approximating a cylinder, so a place pose has
        # no preferred yaw to track.
        wxyz = self._grasp_wxyz(scenes)
        ref = scenes.q_start if scenes.grasp_ref is None else scenes.grasp_ref
        q = sqp_ik_solve_cuda_batch(
            self.base.robot, self.ee_index, _target_pose_batch(target, wxyz),
            IK_RNG_KEY, ref.astype(jnp.float32),
            continuity_weight=IK_CONTINUITY_WEIGHT,
        )
        return q.astype(dtype)

    def _place_wxyz(self, scenes: PickPlaceScene):
        """Release target orientation: anchored, else straight down.

        Two tiers, not three -- see `PickPlaceScene.place_wxyz` for why there is
        no object-yaw middle tier at the bucket.
        """
        return DOWN_WXYZ if scenes.place_wxyz is None else scenes.place_wxyz

    def _grasp_wxyz(self, scenes: PickPlaceScene):
        """Grasp target orientation: anchored, else object-yaw, else straight down.

        Three tiers, most specific first.  `pick_wxyz` is the demonstrated
        orientation; `pick_yaw` rotates `DOWN_WXYZ` by the object's yaw; the
        fallback is the fixed `DOWN_WXYZ` every synthetic scene uses.
        """
        if scenes.pick_wxyz is not None:
            return scenes.pick_wxyz
        if scenes.pick_yaw is not None:
            return _down_yaw_wxyz(scenes.pick_yaw)
        return DOWN_WXYZ

    def branch_refs(self, scenes: PickPlaceScene, n_branches, key, spread=0.8):
        """(B, dof) reference configurations spanning the redundant IK's
        self-motion manifold.

        `continuity_weight` (see IK_CONTINUITY_WEIGHT) makes
        `sqp_ik_solve_cuda_batch` return the branch nearest `previous_cfgs`.
        So B well-separated references select B different branches -- and each
        one stays continuous in theta_ik, because within a branch the winner
        selection no longer flips.  That is what lets many basins be covered
        WITHOUT putting a hard argmin back inside the differentiated forward
        map: the branch is fixed per candidate for the whole fit, and the
        selection happens once, at the end, over converged results.

        Reference 0 is `q_start` itself, so candidate 0 reproduces exactly what
        the single-branch model does and the batch is a strict superset.
        """
        q0 = scenes.q_start[0]
        jitter = spread * jax.random.normal(key, (n_branches - 1, q0.shape[0]),
                                            dtype=q0.dtype)
        return jnp.concatenate([q0[None, :], q0[None, :] + jitter], axis=0)

    def grasp_ik_branched(self, theta_ik, scenes: PickPlaceScene, refs):
        """(B, M, dof): `grasp_ik` under each of B branch references.

        The branch axis is folded into the solver's existing problem batch
        (n_problems = B * M) rather than looped or vmapped over, so all
        branches for all scenes are ONE kernel launch.
        """
        dtype = scenes.q_start.dtype
        B, M = refs.shape[0], scenes.q_start.shape[0]
        target = scenes.pick_pos + theta_ik[0] * UP_AXIS          # (M, 3)
        target_b = jnp.broadcast_to(target, (B, M, 3)).reshape(B * M, 3)
        refs_b = jnp.broadcast_to(refs[:, None, :], (B, M, refs.shape[-1]))
        # Same grasp orientation as `grasp_ik`, tiled over the branch axis.
        # The branch refs deliberately still come from `refs`, not `grasp_ref`:
        # covering several IK branches is the whole point of this entry point,
        # and seeding every branch from the same anchor would collapse them.
        w = self._grasp_wxyz(scenes)
        wxyz = (DOWN_WXYZ if w is DOWN_WXYZ else
                jnp.broadcast_to(w[None], (B, M, 4)).reshape(B * M, 4))
        q = sqp_ik_solve_cuda_batch(
            self.base.robot, self.ee_index, _target_pose_batch(target_b, wxyz),
            IK_RNG_KEY, refs_b.reshape(B * M, -1).astype(jnp.float32),
            continuity_weight=IK_CONTINUITY_WEIGHT,
        )
        return q.reshape(B, M, -1).astype(dtype)

    def place_ik_branched(self, theta_ik, scenes: PickPlaceScene, q_pick_b):
        """(B, M, dof).  `previous_cfgs` is that branch's own `q_pick`, so the
        place pose stays on the branch the grasp settled into."""
        dtype = q_pick_b.dtype
        B, M = q_pick_b.shape[0], q_pick_b.shape[1]
        target = _place_target(scenes, theta_ik)
        target_b = jnp.broadcast_to(target, (B, M, 3)).reshape(B * M, 3)
        # Orientation anchored as in `place_ik`; the SEED stays each branch's
        # own `q_pick`, for the same reason `grasp_ik_branched` keeps `refs` --
        # a shared anchor would collapse the branches this entry point exists
        # to spread.
        w = self._place_wxyz(scenes)
        wxyz = (DOWN_WXYZ if w is DOWN_WXYZ else
                jnp.broadcast_to(w[None], (B, M, 4)).reshape(B * M, 4))
        q = sqp_ik_solve_cuda_batch(
            self.base.robot, self.ee_index, _target_pose_batch(target_b, wxyz),
            IK_RNG_KEY, q_pick_b.reshape(B * M, -1).astype(jnp.float32),
            continuity_weight=IK_CONTINUITY_WEIGHT,
        )
        return q.reshape(B, M, -1).astype(dtype)

    def place_ik(self, theta_ik, scenes: PickPlaceScene, q_pick):
        """Continuity from the grasp pose: `previous_cfg=q_pick`, so the place
        IK's own null-space choice (and its canonical gradient) is relative to
        where the grasp phase actually left the arm, not the scene's home cfg."""
        dtype = q_pick.dtype
        target = _place_target(scenes, theta_ik)
        # `place_ref` overrides the q_pick continuity seed when the release is
        # anchored: the demonstrated release configuration is a strictly better
        # reference than "wherever the grasp left the arm", and it is what makes
        # the null-space choice match the demonstration rather than merely be
        # continuous with the previous stage.
        wxyz = self._place_wxyz(scenes)
        ref = q_pick if scenes.place_ref is None else scenes.place_ref
        q = sqp_ik_solve_cuda_batch(
            self.base.robot, self.ee_index, _target_pose_batch(target, wxyz),
            IK_RNG_KEY, ref.astype(jnp.float32),
            continuity_weight=IK_CONTINUITY_WEIGHT,
        )
        return q.astype(dtype)

    # -- per-segment trajopt (reuses ioc.robot.problem.RobotProblem/Scene) -

    def segment_residual_fn(self, phase):
        """The tied `STANDARD_FEATURES` basis, evaluated on one segment."""
        problem = self.seg[phase]
        return self._standard_residual_fn(problem)

    def _standard_residual_fn(self, problem):
        """`(x_flat, scene) -> tuple` of the six standard-basis residuals.

        Every derivative is taken with respect to the segment's OWN duration,
        read out of `x_flat`'s auxiliary tail, so shortening the motion really
        does raise the acceleration/jerk/effort terms.  With a fixed `dt` the
        `time` feature would be a constant and its weight pure gauge.
        """

        def residual_fn(x_flat, scene: Scene):
            q = problem.unpack(x_flat, scene)
            T = problem.duration(x_flat)
            dt = problem.dt(x_flat)

            # time: the duration itself.  A residual, so the cost is w * T^2 --
            # monotone in T, which is all a min-time term has to be.
            r_time = jnp.atleast_1d(T)

            # path: EE arc length, as per-step Cartesian increments.  Geometric,
            # independent of T, and structurally unrelated to the joint-space
            # derivatives below (see the basis comment on collinearity).
            p_ee = problem.ee_positions(q)  # (T, 3)
            r_path = (p_ee[1:] - p_ee[:-1]).reshape(-1)

            # accel / jerk: finite differences scaled by the real dt.
            qdd = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt ** 2)
            r_accel = qdd.reshape(-1)
            qddd = (q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]) / (dt ** 3)
            r_jerk = qddd.reshape(-1)

            # effort: RNEA torque at the interior knots.  Gravity makes this
            # posture-dependent and therefore NOT a rescaled `accel` -- that
            # gravity term is the part that carries information a pure
            # smoothness basis cannot express.
            qd = (q[2:] - q[:-2]) / (2.0 * dt)
            tau = problem.robot.inverse_dynamics(q[1:-1], qd, qdd, gravity=-9.81,
                                                 use_cuda=True)
            r_effort = tau.reshape(-1)

            r_clear = problem.clearance_residual(q, scene)
            return (r_time, r_path, r_accel, r_jerk, r_effort, r_clear)

        return residual_fn

    def make_segment_inner(self, phase, forward_solver):
        from ioc.inner import make_inner_solver

        residual_fn = self.segment_residual_fn(phase)
        # Calibrated once per phase on that phase's own straight-line-plus-
        # jitter seed -- same reasoning as `RobotProblem.calibrate`'s docstring
        # (must not calibrate on the exactly-zero-acceleration straight line).
        return residual_fn, self.seg[phase]

    def upright_constraint_fn(self, phase):
        """A HARD grasp-maintenance constraint for `phase`, as a theta-independent
        AL equality term for `ioc.inner.make_inner_solver`'s `constraints_fn`.

        The residual forces the gripper's TOOL AXIS to stay vertical (yaw about
        world z is free, so a yawed grasp is fine; only tipping the approach axis
        off vertical -- which spills the object -- is forbidden).  As a hard
        constraint this keeps the carried object seated REGARDLESS of the fitted
        cost weights, which the soft floor could not.

        NOTE the residual is NOT the soft `upright` feature's `quat[1:3]`: that
        term's zero is not the down orientation (a straight-down gripper is
        `DOWN_WXYZ=[0,1,0,0]`, whose `quat[1:3]=[1,0]`), so as a weak cost it is a
        harmless regularizer dominated by the fixed down endpoints, but as a HARD
        equality it forces the gripper AWAY from vertical and the arm contorts (a
        run doing exactly that flung the cube off the table, 0/10).  The correct,
        yaw-free measure is the world-frame xy-components of the EE approach axis
        R(quat)@[0,0,1] = [2(xz+wy), 2(yz-wx)], which is zero for every down(+yaw)
        grasp and nonzero exactly when the tool tips off vertical.

        Interior waypoints only (`q[1:-1]`): the endpoints are the fixed IK
        boundary conditions, so an equality on them would be a constant residual
        the AL dual could only chase.  Theta-independent by construction, so the
        implicit adjoint linearizes the augmented stationarity with frozen
        multipliers -- see `ioc.inner`."""
        from pyroffi.optimization_engines._trajopt_core import AugmentedLagrangianTerm

        problem = self.seg[phase]

        def constraints_fn(scene):
            def residual_fn(x_flat):
                q = problem.unpack(x_flat, scene)
                quat = problem.robot.forward_kinematics(q)[..., problem.ee_index, 0:4]
                w, x, y, z = quat[1:-1, 0], quat[1:-1, 1], quat[1:-1, 2], quat[1:-1, 3]
                # world xy of the EE approach axis R(quat)@[0,0,1]; zero <=> vertical
                axis_xy = jnp.stack([2.0 * (x * z + w * y),
                                     2.0 * (y * z - w * x)], axis=-1)
                return axis_xy.reshape(-1)

            return (AugmentedLagrangianTerm(residual_fn=residual_fn, kind="eq",
                                            name=f"{phase}.upright_hold"),)

        return constraints_fn

    def calibrate_segment(self, phase, residual_fn, scenes: Scene, key, n_probe=16, jitter=0.15):
        problem = self.seg[phase]

        def raw(scene, k):
            x0 = problem.seed(scene)
            x = x0 + jitter * jax.random.normal(k, x0.shape)
            rs = residual_fn(x, scene)
            return jnp.stack([jnp.sum(r**2) for r in rs])

        keys = jax.random.split(key, n_probe)
        vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)), in_axes=(0, None))(scenes, keys)
        scales = jnp.mean(jnp.abs(vals.reshape(-1, vals.shape[-1])), axis=0)
        assert bool(jnp.all(scales > 1e-8)), f"{phase}: degenerate feature scale {scales}"
        return scales

    def ee_positions(self, q):
        return self.base.ee_positions(q)

    # -- the tied cost model -------------------------------------------------

    def _feature_residuals(self, problem, q, scene, dt, T):
        """`STANDARD_FEATURES`, evaluated on any stage's path, in basis order.

        One implementation for every stage, so "the same preference applied at
        both levels" is literally the same code and not two definitions that
        could drift apart.  `dt`/`T` come from the caller's own decision vector
        (`problem.dt`/`problem.duration`) rather than a module constant, which is
        what lets `time` mean anything -- see `RobotProblem.n_aux`.
        """
        p_ee = problem.ee_positions(q)
        qdd = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt ** 2)
        qd = (q[2:] - q[:-2]) / (2.0 * dt)
        return [
            jnp.atleast_1d(T),
            (p_ee[1:] - p_ee[:-1]).reshape(-1),
            qdd.reshape(-1),
            ((q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]) / (dt ** 3)).reshape(-1),
            problem.robot.inverse_dynamics(q[1:-1], qd, qdd, gravity=-9.81,
                                           use_cuda=True).reshape(-1),
            problem.clearance_residual(q, scene),
        ]

    def shared_segment_residual_fn(self, phase):
        """Stage-2 residuals under the tied model: every segment carries the
        SAME three features, rather than the hand-assigned per-phase subsets in
        `SEGMENT_FEATURES`.

        This does change the model: `upright` now applies to approach and place
        as well as transport.  That is the tied hypothesis taken seriously -- a
        demonstrator who prefers a level gripper prefers it throughout, and
        letting each phase switch features on and off is exactly the
        per-pipeline-stage freedom this model gives up on purpose.
        """
        problem = self.seg[phase]

        def residual_fn(x_flat, scene: Scene):
            q = problem.unpack(x_flat, scene)
            return tuple(self._feature_residuals(problem, q, scene,
                                                 problem.dt(x_flat),
                                                 problem.duration(x_flat)))

        return residual_fn

    def shared_full_residual_fn(self):
        """Stage-3 residuals: the same three features on the full path, plus
        `skeleton`.  `upright` stays restricted to the transport span here --
        that span is a property of the FULL path's indexing, not a per-phase
        feature choice, so it survives the tie."""
        problem = self.seg["full"]
        span = PHASE_SPAN["transport"]

        def residual_fn(x_flat, scene: FullScene):
            q = problem.unpack(x_flat, scene)
            parts = self._feature_residuals(problem, q, scene,
                                            problem.dt(x_flat),
                                            problem.duration(x_flat))
            skel = [q[i] - scene.q_pick for i in SKELETON_PICK]
            skel += [q[i] - scene.q_goal for i in SKELETON_PLACE]
            parts.append(jnp.concatenate(skel))
            return tuple(parts)

        return residual_fn

    @staticmethod
    def split_shared(theta):
        """(theta_segment, theta_full) from one tied theta in R^K_SHARED.

        Segments get the first three components UNRENORMALIZED: dropping
        `skeleton` scales a segment's cost by a positive constant, and a
        positive rescaling leaves its argmin unchanged, so renormalizing here
        would be a no-op that only obscured the tie."""
        return theta[:len(SHARED_FEATURES)], theta

    # -- stage 3: global refine over the whole trajectory --------------------

    def full_residual_fn(self):
        """Residuals for the refine solve, on the full N_FULL-waypoint path.

        `smooth` and `clearance` are the same features the segments already
        carry, but evaluated GLOBALLY -- and that is the whole point of the
        stage.  A second difference straddling row 7 exists only here: no
        per-segment solve has both q[6] and q[8] in its decision vector, so the
        kink at every phase junction is invisible to stages 1-2 and is exactly
        what stage 3 can price.  `upright` is restricted to the transport span
        because that is the only phase where carrying the object constrains the
        gripper's tilt; applying it to approach/place would be a different
        model, not a global version of the same one.

        `skeleton` is the new feature and the one the task structure lives in:
        the squared joint-space deviation of the path from the IK-supplied pick
        pose at rows 7 and 10 and the place pose at row 19.  It is a SOFT
        anchor on purpose.  Hard-clamping those rows (the obvious alternative)
        would make stage 3 a no-op on the skeleton and leave the refine pass
        with nothing to trade -- the demonstrator's global preference we are
        trying to recover IS the exchange rate between "stay on the skeleton"
        and "be smooth", and a hard constraint sets that rate to infinity by
        fiat instead of letting the fit measure it.
        """
        problem = self.seg["full"]
        t0, t1 = PHASE_SPAN["transport"]

        def residual_fn(x_flat, scene: FullScene):
            q = problem.unpack(x_flat, scene)
            parts = self._feature_residuals(problem, q, scene,
                                            problem.dt(x_flat),
                                            problem.duration(x_flat))
            skel = [q[i] - scene.q_pick for i in SKELETON_PICK]
            skel += [q[i] - scene.q_goal for i in SKELETON_PLACE]
            parts.append(jnp.concatenate(skel))
            return tuple(parts)

        return residual_fn

    def full_scenes(self, scenes: PickPlaceScene, q_pick, q_place):
        """Batched `FullScene`; q_pick/q_place stay undetached from the IK stage."""
        return FullScene(scenes.q_start, q_place, scenes.obs_center,
                         scenes.obs_radius, q_pick)

    def concat_segments(self, xs, phase_scenes):
        """(B, N_FULL, dof): the four segment solutions joined into one path,
        dropping the duplicated boundary row between consecutive phases -- the
        joint-space analogue of `full_ee_path`, kept batched and differentiable
        so it can seed stage 3 inside the traced forward pass."""
        rows = []
        for i, phase in enumerate(PHASES):
            problem = self.seg[phase]
            q = jax.vmap(problem.unpack)(xs[phase], phase_scenes[phase])
            rows.append(q[:, 1:] if i > 0 else q)
        return jnp.concatenate(rows, axis=1)

    def full_seed_from_segments(self, xs, phase_scenes):
        """Stage-3 decision vector seeded from stage 2, endpoints stripped.

        Seeding the refine solve from the concatenated segments (rather than a
        fresh straight line) is what makes this a REFINEMENT stage and not a
        fifth independent trajopt: stage 2's output is the only thing that
        carries the skeleton into stage 3's basin."""
        q = self.concat_segments(xs, phase_scenes)
        return q[:, 1:-1, :].reshape(q.shape[0], -1)

    def calibrate_full(self, residual_fn, full_scenes: FullScene, key,
                       n_probe=16, jitter=0.15):
        """Feature scales for the refine stage.

        Probes perturbed straight lines, for the same reason
        `RobotProblem.calibrate` does (a straight-line seed has exactly zero
        acceleration, so calibrating `smooth` on it collapses to the numerical
        floor).  Note the straight line from q_start to q_place passes nowhere
        near q_pick, so `skeleton` is comfortably non-degenerate here -- the
        assert in the shared calibration path would catch it if that changed."""
        problem = self.seg["full"]

        def raw(scene, k):
            x0 = problem.seed(scene)
            x = x0 + jitter * jax.random.normal(k, x0.shape)
            rs = residual_fn(x, scene)
            return jnp.stack([jnp.sum(r**2) for r in rs])

        keys = jax.random.split(key, n_probe)
        vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)), in_axes=(0, None))(
            full_scenes, keys)
        scales = jnp.mean(jnp.abs(vals.reshape(-1, vals.shape[-1])), axis=0)
        scales = jnp.where(scales > 1e-8, scales, 1.0)  # pinned feature -> benign 0 scale
        return scales

    # -- full composed forward solve ----------------------------------------

    def solve(self, theta_ik, theta_trajopt_by_phase, scenes: PickPlaceScene,
              inner_by_phase, x0_by_phase, *, refine=None, theta_full=None):
        """One call per segment's `solve_implicit`, chained through literal
        (undetached) data flow -- the thing this module exists to demonstrate.

        Returns (q_pick, q_place, {phase: x_flat}, {phase: Scene}).

        `refine` (an `ioc.inner` solver for `full_residual_fn`) and
        `theta_full` add spasm's third stage: a single trajopt over the ENTIRE
        concatenated trajectory, seeded from the segment solutions.  Both must
        be given together or both omitted.  When given, two extra entries
        appear in the returned dicts under the key `"full"` -- the refined
        x_flat and its `FullScene` -- while the four per-segment entries stay
        exactly as they were, so a caller can compare stage-2 and stage-3
        output from one solve.

        Keyword-only and defaulting to None: every existing caller
        (`recovery_bench`, `identifiability_check`, `generalization_check`,
        `study0_*`, `study3_*`) is a two-stage caller and is unaffected.
        """
        if (refine is None) != (theta_full is None):
            raise ValueError("pass both `refine` and `theta_full`, or neither")
        q_pick = self.grasp_ik(theta_ik, scenes)
        q_place = self.place_ik(theta_ik, scenes, q_pick)

        _world = self.world_of(scenes)
        phase_scenes = {
            "approach": Scene(scenes.q_start, q_pick, scenes.obs_center,
                              scenes.obs_radius, **_world),
            "grasp": Scene(q_pick, q_pick, scenes.obs_center,
                           scenes.obs_radius, **_world),
            "transport": Scene(q_pick, q_place, scenes.obs_center,
                               scenes.obs_radius, **_world),
            "place": Scene(q_place, q_place, scenes.obs_center,
                           scenes.obs_radius, **_world),
        }
        xs = {}
        for phase in PHASES:
            inner = inner_by_phase[phase]
            sc = phase_scenes[phase]
            xs[phase] = jax.vmap(inner.solve_implicit, in_axes=(0, None, 0))(
                x0_by_phase[phase], theta_trajopt_by_phase[phase], sc)
        if refine is not None:
            # Stage 3.  The seed is stage 2's output undetached, so the refine
            # solve's implicit adjoint composes back through all four segment
            # adjoints AND the IK stage's custom_jvp -- the same composition
            # the two-stage path already relies on, one link longer.
            full_sc = self.full_scenes(scenes, q_pick, q_place)
            x0_full = self.full_seed_from_segments(xs, phase_scenes)
            xs["full"] = jax.vmap(refine.solve_implicit, in_axes=(0, None, 0))(
                x0_full, theta_full, full_sc)
            phase_scenes["full"] = full_sc
        return q_pick, q_place, xs, phase_scenes

    def solve_batched_theta(self, theta_ik, theta_trajopt_by_phase,
                            scenes: PickPlaceScene, inner_by_phase, x0_by_phase):
        """`solve`, but with a PER-ROW cost -- the flattening, for two-stage callers.

        `solve` maps the batch axis over scenes and holds one `theta` for all of
        them (`in_axes=(0, None, 0)`).  That is right when the batch axis IS the
        scene axis, and wrong when a caller wants several independent COSTS
        evaluated at once: multistart alpha candidates, a finite-difference
        probe stack, a CMA-ES population.  Those callers previously ran one
        `solve` per candidate in a Python loop.

        Here every argument is mapped (`in_axes=(0, 0, 0)`), so the caller folds
        its candidate axis into the batch it already has -- replicating the
        scenes and repeating each candidate's theta across them -- and the whole
        population is one batched program.  `iosp.fit.multistart.build` does the
        same thing inline for the three-stage path; this is the two-stage twin.

        Two-stage only (no `refine`/`theta_full`): the callers that need a
        per-row cost are segment-level, and adding an unused third stage here
        would mean a second untested code path.

        `theta_ik` is still shared across rows -- these callers vary the trajopt
        cost, not the IK targets, and a per-row IK would change which kernel
        batch the IK stage sees.
        """
        q_pick = self.grasp_ik(theta_ik, scenes)
        q_place = self.place_ik(theta_ik, scenes, q_pick)
        _world = self.world_of(scenes)
        phase_scenes = {
            "approach": Scene(scenes.q_start, q_pick, scenes.obs_center,
                              scenes.obs_radius, **_world),
            "grasp": Scene(q_pick, q_pick, scenes.obs_center,
                           scenes.obs_radius, **_world),
            "transport": Scene(q_pick, q_place, scenes.obs_center,
                               scenes.obs_radius, **_world),
            "place": Scene(q_place, q_place, scenes.obs_center,
                           scenes.obs_radius, **_world),
        }
        xs = {}
        for phase in PHASES:
            xs[phase] = jax.vmap(inner_by_phase[phase].solve_implicit,
                                 in_axes=(0, 0, 0))(
                x0_by_phase[phase], theta_trajopt_by_phase[phase], phase_scenes[phase])
        return q_pick, q_place, xs, phase_scenes

    def full_ee_path(self, scenes: PickPlaceScene, xs, phase_scenes, batch_index=0):
        """One batch element's EE path.

        If `xs` carries a `"full"` entry (i.e. the solve ran spasm's stage 3),
        this returns the REFINED path -- that is the trajectory the planner
        actually emits, and so the one any behavioral metric must score
        against.  Otherwise it falls back to concatenating the four segments
        and dropping the duplicated boundary row, which is what every
        two-stage caller gets, unchanged.
        """
        if "full" in xs:
            problem = self.seg["full"]
            sc = jax.tree.map(lambda a: a[batch_index], phase_scenes["full"])
            return self.ee_positions(problem.unpack(xs["full"][batch_index], sc))
        rows = []
        for i, phase in enumerate(PHASES):
            problem = self.seg[phase]
            sc = jax.tree.map(lambda a: a[batch_index], phase_scenes[phase])
            q = problem.unpack(xs[phase][batch_index], sc)
            p = self.ee_positions(q)
            rows.append(p[1:] if i > 0 else p)
        return jnp.concatenate(rows, axis=0)

    def full_joint_path(self, scenes: PickPlaceScene, xs, phase_scenes, batch_index=0):
        """One batch element's JOINT path -- `full_ee_path`'s configuration-space
        twin, (T, dof) instead of (T, 3).

        Same concatenation and same duplicated-boundary drop, so row t of this
        and row t of `full_ee_path` are the same waypoint; only the coordinates
        differ.  Exists so an outer loss can be scored in joint space, where the
        7-DOF arm's self-motion manifold IS observable -- `full_ee_path` is blind
        to it by construction (measured: q jumping 2.5-4.4 rad while the EE moves
        0.0002 m; see IK_CONTINUITY_WEIGHT above).
        """
        if "full" in xs:
            problem = self.seg["full"]
            sc = jax.tree.map(lambda a: a[batch_index], phase_scenes["full"])
            return problem.unpack(xs["full"][batch_index], sc)
        rows = []
        for i, phase in enumerate(PHASES):
            problem = self.seg[phase]
            sc = jax.tree.map(lambda a: a[batch_index], phase_scenes[phase])
            q = problem.unpack(xs[phase][batch_index], sc)
            rows.append(q[1:] if i > 0 else q)
        return jnp.concatenate(rows, axis=0)

    def full_joint_paths(self, scenes: PickPlaceScene, xs, phase_scenes):
        """(B, T, dof) -- `full_joint_path` for the WHOLE batch, vmapped.

        `full_joint_path` takes a scalar `batch_index` and gathers one row, so
        the callers that actually want every row wrote

            jnp.stack([full_joint_path(..., batch_index=i) for i in range(B)])

        which unrolls B copies of the unpack (and, for the EE twin, of the
        forward kinematics) into the graph.  The solves upstream of it are
        already vmapped over the batch; only this readout was not, so the
        unrolling bought nothing and cost compile time linear in B.  This is the
        same map with the batch as a `vmap` axis.

        The `PHASES` loop stays a Python loop on purpose: those four segments
        are concatenated along TIME, not batched over -- it is a genuine
        4-element concat, not a hidden batch axis.
        """
        if "full" in xs:
            return jax.vmap(self.seg["full"].unpack)(xs["full"], phase_scenes["full"])
        rows = []
        for i, phase in enumerate(PHASES):
            q = jax.vmap(self.seg[phase].unpack)(xs[phase], phase_scenes[phase])
            rows.append(q[:, 1:] if i > 0 else q)   # drop the duplicated boundary
        return jnp.concatenate(rows, axis=1)

    def full_ee_paths(self, scenes: PickPlaceScene, xs, phase_scenes):
        """(B, T, 3) -- `full_ee_path` for the whole batch; see `full_joint_paths`.

        Row t here and row t of `full_joint_paths` are the same waypoint, and
        row b matches `full_ee_path(..., batch_index=b)`, so this is a drop-in
        for the stack-comprehension idiom it replaces -- bit-identical to it
        when both are evaluated in the same jit (asserted in
        `tests/test_batched_paths_gpu.py`).  Comparing across a jit boundary is
        NOT a meaningful equivalence check here: this FK is float32 and moves by
        ~3e-4 m between compilation/precision contexts, the looped form
        included, so a mismatch there says something about the FK's precision
        and nothing about the batching.

        `ee_positions` already indexes with an ellipsis, so it broadcasts over
        leading axes on its own -- no `vmap` wrapper needed or wanted.
        """
        return self.ee_positions(self.full_joint_paths(scenes, xs, phase_scenes))

    @staticmethod
    def world_of(scenes: PickPlaceScene):
        """`Scene`'s optional world fields for these scenes, as kwargs.

        Empty on a synthetic scene (no table/bucket recorded), which leaves the
        collision world exactly the single placeholder sphere it has always been.
        """
        return dict(
            obs_spheres=bucket_spheres(scenes.bucket_center,
                                       scenes.bucket_inner_radius,
                                       scenes.bucket_wall_height,
                                       scenes.bucket_wall_thickness),
            table_box=scenes.table_box,
        )

    def seeds(self, scenes: PickPlaceScene, theta_ik, forward_solver_free=None):
        """Interior-waypoint seeds per phase, from the SAME IK call used by the
        differentiable forward path (not a separate disconnected precompute --
        see module docstring / the old design's `seed_ik` this replaces)."""
        q_pick = self.grasp_ik(theta_ik, scenes)
        q_place = self.place_ik(theta_ik, scenes, q_pick)
        world = self.world_of(scenes)
        phase_scenes = {
            "approach": Scene(scenes.q_start, q_pick, scenes.obs_center,
                              scenes.obs_radius, **world),
            "grasp": Scene(q_pick, q_pick, scenes.obs_center,
                           scenes.obs_radius, **world),
            "transport": Scene(q_pick, q_place, scenes.obs_center,
                               scenes.obs_radius, **world),
            "place": Scene(q_place, q_place, scenes.obs_center,
                           scenes.obs_radius, **world),
        }
        x0 = {p: jax.vmap(self.seg[p].seed)(phase_scenes[p]) for p in PHASES}
        return x0, phase_scenes, q_pick, q_place
