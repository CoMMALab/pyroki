"""The robot-side inverse-optimal-control problem: scenes, features, demos.

A context ("scene") is a pair of fixed joint-space endpoints plus one spherical
obstacle.  The decision variable is the stack of interior waypoints; endpoints
are clamped, so the inner problem is unconstrained and Gauss-Newton solvable.

Two choices here decide whether the whole experiment measures anything, and both
are enforced rather than assumed:

**Identifiability lives in the scene distribution.**  If the obstacle never
blocks the seed path, the collision feature is inactive, x*(theta) is the
straight line for every theta, and no method -- however exact its gradients --
can recover the weights.  `sample_scenes` therefore anchors each obstacle *on*
the straight-line end-effector path and offsets it by about one radius, so the
collision term is active to differing degrees across contexts.  That variation
is precisely what the weights are identified from; `ioc.analytic.kkt_fit` can
certify it after the fact via the Gram matrix.

**Feature whitening must not be calibrated on the seed.**  The straight-line
seed has exactly zero acceleration, so a seed-based scale for the smoothness
feature collapses to the numerical floor and silently applies an enormous weight
to it, pinning x*(theta) to the straight line for every theta.  `calibrate`
probes randomly perturbed trajectories instead.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pyroffi as pk

EE_LINK = "panda_hand"  # default; `load(ee_link=...)` for a different arm
CLEARANCE_MARGIN = 0.05  # [m] distance at which the collision feature turns on
SOFTMIN_TAU = 0.02  # [m] temperature of the soft-min over collision pairs
SOFTNESS = 60.0  # smoothing of the collision hinge; keeps the Hessian continuous
BOX_SOFTNESS = 200.0  # smoothing of the box-face max in `signed_clearance`

Q_START = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8])
Q_GOAL = np.array([0.9, -0.2, 0.0, -1.8, 0.0, 1.7, 0.8])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Scene:
    """A single demonstration context: fixed endpoints plus one spherical obstacle."""

    q_start: jnp.ndarray  # (dof,)
    q_goal: jnp.ndarray  # (dof,)
    obs_center: jnp.ndarray  # (3,)
    obs_radius: jnp.ndarray  # (1,)

    # -- the rest of the world, optional -------------------------------------
    # `obs_center`/`obs_radius` is ONE sphere, which is all the synthetic
    # studies ever needed.  The teleop scenes need more: MEASURED on the ten FR3
    # episodes, the single sphere they inherit is a stale placeholder from an
    # obstacle the recorder no longer builds, it sits IN the transport corridor
    # (z 0.39-0.49 above a 0.35 table, between the cube and the buckets), and the
    # demonstrations penetrate it on 68 of 230 rows.  Meanwhile the two things
    # the operator actually had to avoid -- the table and the bucket wall -- were
    # not in the world at all, so a fitted path gliding through the bucket rim
    # was the model being correct about a world with no bucket in it.
    #
    # Both default to None, which is a leaf-less pytree node: `jax.vmap` and
    # `jax.tree.map` skip it and every existing caller is byte-identical.
    obs_spheres: jnp.ndarray = None  # (M, 4) xyz+radius, or None
    # (6,) [cx, cy, cz, hx, hy, hz] axis-aligned box, or None.  A BOX, not a
    # `z >= table_top` half-space: the arm is mounted BESIDE the table (base at
    # x ~ 0, table spanning x 0.20-0.90), so a half-space marks the robot's own
    # base links as permanently penetrating and the soft-min collapses to a
    # constant -0.38 m for every configuration.
    table_box: jnp.ndarray = None


def make_world(scene: Scene):
    return pk.collision.Sphere.from_center_and_radius(
        scene.obs_center[None, :], scene.obs_radius
    )


@dataclasses.dataclass(frozen=True)
class RobotProblem:
    """Everything about the robot and the trajectory parameterization.

    Holds no cost basis: a basis is a `residual_fn(x_flat, scene)` built by
    `ioc.robot.bases`, so the same verified solver, screening and metrics are
    reused across E1's K=3 basis, E2's K=9/16 bases and E3's dynamic basis.
    """

    robot: object
    robot_coll: object
    dof: int
    ee_index: int
    n_timesteps: int
    # Interior rows to HARD-clamp in `unpack`, as ((row_idx, scene_attr), ...).
    # Empty for every base/segment problem (endpoints-only clamp, unchanged).
    # The composed FULL trajectory sets this to pin the grasp/place waypoints
    # (q_pick/q_place) exactly, the way SPaSM and cutamp keep the grasp poses as
    # HARD task waypoints -- a soft `skeleton` cost lets the refine drift off the
    # grasp (~100 mm) and, for tetris, wander out of joint limits.
    pinned_rows: tuple = ()
    # Extra decision variables appended to `x_flat` after the interior
    # waypoints.  0 (the default) is byte-identical to every existing caller.
    #
    # The standard trajopt basis needs ONE: the segment's log-duration `s`, with
    # `T = duration_nominal * exp(s)`.  Exponential rather than `T` itself, or a
    # clipped/softplus `T`, because the inner solve is required to stay
    # UNCONSTRAINED for Gauss-Newton (see the module docstring) and `exp` keeps
    # `T > 0` for every real `s` with no barrier and no kink.
    #
    # A duration variable is what makes a min-time term mean anything: with `dt`
    # fixed, "time" is a constant and its weight is pure gauge.  With `dt` free
    # the basis features scale as different powers of T -- path T^0, acceleration
    # T^-2, jerk T^-3, effort mixed, time T^+1 -- and that spread of exponents is
    # exactly what makes the speed/smoothness trade-off identifiable.
    n_aux: int = 0
    duration_nominal: float = 1.0

    @staticmethod
    def load(urdf_path, srdf_path, mesh_dir, n_timesteps, ee_link=EE_LINK):
        """`ee_link` names the frame every EE-space quantity is read at.

        Defaulted to the Panda's hand so every existing caller is unchanged; it
        is a parameter because the teleop demonstrations were recorded on an
        FR3, whose links are `fr3_*`, and scoring a demo against a forward model
        with different kinematics would confound the fit with a robot mismatch.
        """
        import yourdfpy

        urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
        robot = pk.Robot.from_urdf(urdf)
        robot_coll = pk.collision.RobotCollisionSpherized.from_urdf(
            urdf, srdf_path=srdf_path
        )
        return RobotProblem(
            robot=robot,
            robot_coll=robot_coll,
            dof=robot.joints.num_actuated_joints,
            ee_index=robot.links.names.index(ee_link),
            n_timesteps=n_timesteps,
        )

    # -- trajectory parameterization -----------------------------------------

    def n_x(self):
        """Length of `x_flat`: interior waypoints plus any auxiliary variables."""
        return (self.n_timesteps - 2) * self.dof + self.n_aux

    def unpack_aux(self, x_flat):
        """The auxiliary tail of `x_flat`, `(n_aux,)`.  Empty when `n_aux == 0`."""
        return x_flat[x_flat.shape[0] - self.n_aux:] if self.n_aux else x_flat[:0]

    def duration(self, x_flat):
        """Segment duration `T = duration_nominal * exp(s)`, seconds.

        Falls back to `duration_nominal` when the problem carries no auxiliary
        variable, so a basis that asks for `dt` works unchanged on a fixed-time
        problem."""
        if not self.n_aux:
            return jnp.asarray(self.duration_nominal)
        return self.duration_nominal * jnp.exp(self.unpack_aux(x_flat)[0])

    def dt(self, x_flat):
        """Uniform knot spacing implied by the segment duration."""
        return self.duration(x_flat) / (self.n_timesteps - 1)

    def unpack(self, x_flat, scene):
        """Interior waypoints -> full trajectory with the endpoints clamped on."""
        n_q = (self.n_timesteps - 2) * self.dof
        interior = x_flat[:n_q].reshape(self.n_timesteps - 2, self.dof)
        q = jnp.concatenate(
            [scene.q_start[None, :], interior, scene.q_goal[None, :]], axis=0
        )
        for row, attr in self.pinned_rows:
            q = q.at[row].set(getattr(scene, attr))
        return q

    def seed(self, scene):
        """Straight line in joint space between the endpoints.

        Auxiliary variables seed at 0, i.e. the nominal duration."""
        alphas = jnp.linspace(0.0, 1.0, self.n_timesteps)[1:-1, None]
        q = ((1 - alphas) * scene.q_start + alphas * scene.q_goal).reshape(-1)
        if self.n_aux:
            q = jnp.concatenate([q, jnp.zeros(self.n_aux, q.dtype)])
        return q

    def seeds(self, scenes):
        return jax.vmap(self.seed)(scenes)

    def ee_positions(self, q):
        return self.robot.forward_kinematics(q)[..., self.ee_index, 4:7]

    # -- shared feature pieces -------------------------------------------------

    def signed_clearance(self, q, scene):
        """Per-waypoint smooth signed clearance ``d_min`` (T,) to the world:
        the soft-min over all robot spheres of distance to every obstacle.
        Positive = clear, negative = penetrating.  Every reduction is a soft-min
        (closed-form sphere/half-space distance), never a hard max, so ``d_min``
        is smooth and its gradient vanishes at a true optimum -- see
        ``clearance_residual`` for why that matters for the implicit adjoint/FD.

        The world is the primary sphere, plus `scene.obs_spheres` if present,
        plus the `scene.table_z` half-space if present.  All three reduce into
        ONE soft-min, so adding geometry never introduces a hard switch between
        obstacle sets."""
        coll = self.robot_coll.at_config(self.robot, q)  # (T, S, N) spheres
        c = coll.pose.translation()  # (T, S, N, 3)
        r = coll.radius  # (T, S, N) or broadcastable
        T = c.shape[0]

        parts = [(jnp.linalg.norm(c - scene.obs_center, axis=-1)
                  - r - scene.obs_radius[0]).reshape(T, -1)]

        if scene.obs_spheres is not None:
            # (T, S, N, 1, 3) vs (M, 3) -> (T, S, N, M)
            o = scene.obs_spheres
            d = (jnp.linalg.norm(c[..., None, :] - o[..., :3], axis=-1)
                 - r[..., None] - o[..., 3])
            parts.append(d.reshape(T, -1))

        if scene.table_box is not None:
            # Smooth EXTERIOR distance to an axis-aligned box.  `softplus`
            # rather than `max(., 0)` for the same reason `clearance_residual`
            # avoids a hard max: the kink sits exactly on the face the solver
            # rides.  Saturates at ~0 inside the box instead of going negative,
            # which is acceptable here -- the demonstrations DO touch the table,
            # so what matters is that the term switches on at contact, not that
            # it keeps growing once the arm is through the surface.
            box = scene.table_box
            qx = jnp.abs(c - box[:3]) - box[3:]
            sp = jax.nn.softplus(BOX_SOFTNESS * qx) / BOX_SOFTNESS
            parts.append((jnp.linalg.norm(sp, axis=-1) - r).reshape(T, -1))

        d = jnp.concatenate(parts, axis=-1)
        return -SOFTMIN_TAU * jax.scipy.special.logsumexp(-d / SOFTMIN_TAU, axis=-1)

    def clearance_residual(self, q, scene):
        """Smooth sphere-level clearance to the obstacle.

        Computed analytically rather than through
        `compute_world_collision_distance`: that helper reduces over the spheres
        within each link with a hard max, and a hard max/min is
        nondifferentiable exactly where the nearest sphere switches identity --
        which is the ridge the optimizer drives the solution onto.  The result is
        a cost that converges while its gradient never vanishes: the solve stalls
        at a non-stationary point (measured: only ~53% of scenes reached
        ||grad|| < 1e-5 even at 400 iterations), and both the implicit function
        theorem and finite differences become invalid there.  Sphere-vs-sphere
        distance is smooth in closed form, so every reduction here is a soft-min.
        """
        d_min = self.signed_clearance(q, scene)
        return jax.nn.softplus(SOFTNESS * (CLEARANCE_MARGIN - d_min)) / SOFTNESS

    def collision_constraints_fn(self, margin=CLEARANCE_MARGIN):
        """A theta-INDEPENDENT collision inequality for `ioc.inner`'s constrained
        path: `constraints_fn(scene) -> (AugmentedLagrangianTerm,)` enforcing
        ``margin - d_min(q) <= 0`` (keep every waypoint at least ``margin`` clear
        of the obstacle), using the smooth soft-min clearance so the augmented
        stationarity is well-conditioned -- unlike the hard torque hinge, see
        `torque-constraint-deferred`.  Well-suited to the IOSP domains, whose
        forward solve routes through this same base problem."""
        from pyroffi.optimization_engines._trajopt_core import AugmentedLagrangianTerm

        def constraints_fn(scene):
            def residual(x_flat):
                q = self.unpack(x_flat, scene)
                return (margin - self.signed_clearance(q, scene)).reshape(-1)

            return (AugmentedLagrangianTerm(
                residual_fn=residual, kind="ineq",
                rho0=1.0, rho_max=1e3, penalty_scale=3.0, name="collision"),)

        return constraints_fn

    # -- scenes ----------------------------------------------------------------

    def sample_scenes(self, rng, n, q_start=None, q_goal=None):
        """Sample `n` contexts with obstacles anchored on the seed path.

        See the module docstring: the anchoring is what makes the weights
        identifiable at all.
        """
        q_start = Q_START[: self.dof] if q_start is None else q_start
        q_goal = Q_GOAL[: self.dof] if q_goal is None else q_goal

        starts, goals = [], []
        for _ in range(n):
            jitter = rng.normal(scale=0.10, size=self.dof)
            starts.append(np.asarray(q_start) + jitter)
            goals.append(np.asarray(q_goal) - jitter)
        starts, goals = np.stack(starts), np.stack(goals)

        centers, radii = [], []
        for i in range(n):
            scene_i = Scene(
                q_start=jnp.asarray(starts[i]),
                q_goal=jnp.asarray(goals[i]),
                obs_center=jnp.zeros(3),
                obs_radius=jnp.ones(1),
            )
            q_seed = self.unpack(self.seed(scene_i), scene_i)
            p = np.asarray(self.ee_positions(q_seed))
            t = rng.integers(self.n_timesteps // 3, 2 * self.n_timesteps // 3)
            r = rng.uniform(0.08, 0.14)
            # Offset the obstacle just about one radius off the path: a little
            # less and it blocks, a little more and it is clear.  Anchoring it
            # exactly *on* the path instead leaves the seed deeply penetrating in
            # every context, which saturates the collision feature and destroys
            # the tradeoff.
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            centers.append(p[t] + direction * (r + rng.uniform(-0.05, 0.10)))
            radii.append(np.array([r]))

        return Scene(
            q_start=jnp.asarray(starts),
            q_goal=jnp.asarray(goals),
            obs_center=jnp.asarray(np.stack(centers)),
            obs_radius=jnp.asarray(np.stack(radii)),
        )

    # -- feature whitening -----------------------------------------------------

    def calibrate(self, residual_fn, scenes, key, n_probe=16, jitter=0.15):
        """Nominal feature magnitudes, probed on *perturbed* trajectories.

        Calibrating on the straight-line seed is a silent failure; see the module
        docstring.
        """

        def raw(scene, k):
            x0 = self.seed(scene)
            x = x0 + jitter * jax.random.normal(k, x0.shape)
            rs = residual_fn(x, scene)
            return jnp.stack([jnp.sum(r**2) for r in rs])

        keys = jax.random.split(key, n_probe)
        vals = jax.vmap(jax.vmap(raw, in_axes=(None, 0)), in_axes=(0, None))(
            scenes, keys
        )
        scales = jnp.mean(jnp.abs(vals.reshape(-1, vals.shape[-1])), axis=0)
        assert bool(jnp.all(scales > 1e-8)), f"degenerate feature scale: {scales}"
        return scales


def screen_scenes(problem, pool, stationarity, theta_star, conv_tol, n_keep, chunk=5):
    """Keep only contexts whose inner solve actually reaches stationarity at theta*.

    Both the implicit function theorem and finite differences assume x*(theta) is
    a converged minimizer.  On contexts where the solver plateaus, the returned x
    still depends on the solver path and FD picks up sensitivity the adjoint
    cannot see (measured: agreement falls from cos = 0.9999 to cos = 0.59).
    Discarding them is a stated precondition of every robot experiment, and the
    discard rate is reported alongside the results.

    Chunked because the pool is several times `n_keep` and vmapping all of it at
    once exhausts device memory as M grows (measured OOM at 9.4 GB for M=10).

    Returns (scenes, discard_rate, stationarity_values).
    """
    x0_pool = problem.seeds(pool)
    vals = np.concatenate(
        [
            np.asarray(
                jax.vmap(lambda x, s: stationarity(x, theta_star, s))(
                    x0_pool[i : i + chunk],
                    jax.tree.map(lambda a: a[i : i + chunk], pool),
                )
            )
            for i in range(0, x0_pool.shape[0], chunk)
        ]
    )
    keep = np.flatnonzero(vals < conv_tol)
    if len(keep) < n_keep:
        raise RuntimeError(
            f"only {len(keep)}/{len(vals)} scenes converged below {conv_tol:g}; "
            "loosen conv_tol or raise n_newton"
        )
    # Measure the rate before truncating, otherwise it just reports
    # 1 - n_keep/len(pool) and carries no information about the solver.
    discard_rate = 1.0 - len(keep) / len(vals)
    keep = keep[:n_keep]
    return jax.tree.map(lambda a: a[keep], pool), discard_rate, vals[keep]


def make_demos(problem, solver, scenes, theta_star, rng, demo_noise):
    """Demonstrations: optimal under theta*, plus i.i.d. observation noise.

    The endpoints are treated as exactly observed, so noise is zeroed there --
    they are boundary conditions of the inner problem, not free variables, and
    perturbing them would move the problem rather than the demonstration.
    """
    x0s = problem.seeds(scenes)
    x_star = jax.vmap(lambda x0, s: solver(x0, theta_star, s))(x0s, scenes)
    demos = jax.vmap(problem.unpack)(x_star, scenes)
    if demo_noise > 0:
        noise = jnp.asarray(rng.normal(scale=demo_noise, size=demos.shape))
        demos = demos + noise.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return x0s, x_star, demos


def evaluate(problem, z, solver, cost, scenes, demos, x0s, theta_star):
    """Score a recovered z: weight error, cost regret, and end-effector RMSE.

    Regret is measured under the *true* cost: re-solve with theta_hat, then
    evaluate that trajectory under theta*, minus the optimum's value.  This is
    the behavioural metric, and the honest one for a misspecified basis -- a
    method optimizes what it can express and pays for what it cannot.
    """
    from ioc.metrics import simplex_metrics

    theta = jax.nn.softmax(z)

    def one(scene, demo, x0):
        x_hat = solver(x0, theta, scene)
        x_star = solver(x0, theta_star, scene)
        q_hat = problem.unpack(x_hat, scene)
        regret = cost(x_hat, theta_star, scene) - cost(x_star, theta_star, scene)
        ee_rmse = jnp.sqrt(
            jnp.mean(
                jnp.sum(
                    (problem.ee_positions(q_hat) - problem.ee_positions(demo)) ** 2,
                    axis=-1,
                )
            )
        )
        return regret, ee_rmse

    regret, ee_rmse = jax.vmap(one)(scenes, demos, x0s)
    l1, cos = simplex_metrics(theta, theta_star)
    return {
        "theta_hat": [float(t) for t in theta],
        "theta_l1": l1,
        "theta_cos": cos,
        "regret": float(jnp.mean(regret)),
        "ee_rmse": float(jnp.mean(ee_rmse)),
    }


def make_outer(problem, solver, scenes, demos, x0s):
    """L(z) = mean over contexts of the end-effector path error, theta = softmax(z).

    The loss is compared in *task* space rather than joint space: it is what a
    demonstration actually pins down, and it keeps the outer objective meaningful
    when a redundant arm reaches the same end-effector path by a different
    configuration.
    """

    def per_scene(z, scene, demo, x0):
        theta = jax.nn.softmax(z)
        xs = solver(x0, theta, scene)
        p = problem.ee_positions(problem.unpack(xs, scene))
        p_demo = problem.ee_positions(demo)
        return jnp.mean(jnp.sum((p - p_demo) ** 2, axis=-1))

    def loss(z):
        vals = jax.vmap(per_scene, in_axes=(None, 0, 0, 0))(z, scenes, demos, x0s)
        return jnp.mean(vals)

    return loss
