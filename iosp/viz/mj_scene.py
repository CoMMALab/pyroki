"""MuJoCo scene construction + mjviser playback, shared by the IOSP domain viewers.

Why MuJoCo instead of the viser primitives these viewers used to draw: the arm,
the table, the goal walls and the manipulated object become GEOMS, so
``mj_forward`` computes contacts and the viewer can show them.  That is the
question ``iosp.checks.feasibility_report`` answers numerically, watched frame by
frame -- a curated problem whose demonstration clips a wall is visible here and
invisible in a URDF-plus-boxes rendering.

Playback is KINEMATIC: the planned joint path is written into ``qpos`` and
``mj_forward`` is called.  These trajectories are plans, not torque-tracked
executions, so stepping physics would show the arm collapse under gravity rather
than the demonstration.  Contacts are still computed, which is all the
feasibility question needs.

Two robot models, selected by ``--robot``:

``menagerie``
    The Menagerie Panda -- real meshes, a hand and fingers.  What
    ``iosp.checks.mujoco_feasibility`` builds.  The default: it looks right and
    the gripper schedule reads.
``spherized``
    ``resources/panda/panda_spherized.urdf`` (``iosp.config.URDF_PATH``) -- the
    EXACT collision model the planner optimised against, 7 joints and no hand.
    Use it when a penetration number has to line up with
    ``feasibility_report``, which is deliberately free of a cross-model
    confound: on ``scratch/feas/tetris_aligned.npz`` this viewer reports the
    same 5.10 mm ``panda_link1``/``panda_link5`` contact on scene 5 that the
    check does.  It reaches the path through ``iosp.config``, so unlike
    ``menagerie`` it pulls in JAX.

Multiple scenes are shown side by side: each gets its own copy of the arm,
attached under an ``s<i>_`` prefix at a lateral offset, so one MuJoCo model
holds the whole comparison.  Environment geometry is added to the worldbody, so
callers place it at ``builder.offset(i) + local_position`` -- the same ``off +
...`` idiom the viser versions used.
"""

from __future__ import annotations

import time

import numpy as np

ROBOTS = ("menagerie", "spherized")

# Arm-joint names differ between the two models; the finger joints exist only on
# the Menagerie one.
_ARM_JOINTS = {
    "menagerie": [f"joint{i}" for i in range(1, 8)],
    "spherized": [f"panda_joint{i}" for i in range(1, 8)],
}
_FINGER_JOINTS = {
    "menagerie": ["finger_joint1", "finger_joint2"],
    "spherized": [],
}

FINGER_OPEN = 0.04          # [m] per-finger opening when not holding anything

# pyroffi's end-effector frame, measured against its own FK: the last arm link's
# body origin plus 107 mm along that body's +z.  Both robot models carry that
# link (under different names), so the EE path this module draws is the SAME
# point the planner optimised, not a viewer-specific approximation of it.
_EE_BODY = {"menagerie": "link7", "spherized": "panda_link7"}
EE_OFFSET = np.array([0.0, 0.0, 0.107])

TABLE_RGBA = (0.85, 0.84, 0.80, 1.0)
MARKER_PICK_RGBA = (1.0, 0.35, 0.35, 0.55)
MARKER_PLACE_RGBA = (0.35, 1.0, 0.35, 0.55)
OBSTACLE_RGBA = (0.62, 0.62, 0.68, 0.55)


def _robot_child_spec(robot):
    """A fresh MjSpec for one arm.  A spec can only be attached once, so this is
    called per scene rather than cached."""
    import mujoco

    if robot == "menagerie":
        from robot_descriptions import panda_mj_description
        return mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
    if robot == "spherized":
        from iosp.config import URDF_PATH
        return mujoco.MjSpec.from_file(str(URDF_PATH))
    raise ValueError(f"robot must be one of {ROBOTS}, got {robot!r}")


class WorldBuilder:
    """Assemble one MuJoCo model holding `n_scenes` arms and their environments.

    Sizes are given as FULL dimensions / radii (not MuJoCo's half-extents), so
    the call sites read the same as the `server.scene.add_box(...)` lines they
    replace.
    """

    def __init__(self, n_scenes=1, spread=1.8, robot="menagerie", floor=True,
                 floor_z=-0.05):
        import mujoco

        self.n_scenes = int(n_scenes)
        self.spread = float(spread)
        self.robot = robot
        self.spec = mujoco.MjSpec()
        # The attached children each carry their own integrator; pin the parent's
        # so the attach does not warn about resolving the conflict.
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # MuJoCo's offscreen framebuffer defaults to 640x480, which is the cap on
        # `mujoco.Renderer` size; raise it so the same world can be screenshotted
        # at a useful resolution without rebuilding it.
        self.spec.visual.global_.offwidth = 1920
        self.spec.visual.global_.offheight = 1080

        # Contact stiffness.  MuJoCo's default `solref = (0.02, 1)` is a 20 ms
        # time constant -- ten timesteps -- so a 50 g block released a few
        # millimetres from a 15 mm wall sinks most of the way INTO it before the
        # constraint pushes back.  That transient is what makes a perfectly
        # valid packing look like the blocks clip through the walls.  Tightening
        # it to 2 ms (one timestep) and raising `solimp`'s width keeps the
        # penetration at the sub-millimetre level without making the solver
        # stiff enough to jitter.
        self.spec.option.o_solref = [0.002, 1.0]
        self.spec.option.o_solimp = [0.95, 0.99, 0.0005, 0.5, 2.0]

        light = self.spec.worldbody.add_light()
        light.pos = [0.0, 0.0, 3.0]
        light.dir = [0.0, 0.0, -1.0]
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

        if floor:
            g = self.spec.worldbody.add_geom()
            g.name = "floor"
            g.type = mujoco.mjtGeom.mjGEOM_PLANE
            g.size = [0.0, 0.0, 0.05]
            # Below the tables, which sit just under z=0: a floor AT z=0 hides
            # them, and the table is the surface the scene has to read against.
            g.pos = [0.0, 0.0, float(floor_z)]
            g.rgba = [0.29, 0.29, 0.33, 1.0]
            g.contype, g.conaffinity = 0, 0     # decoration; the table carries contact

        for i in range(self.n_scenes):
            frame = self.spec.worldbody.add_frame(pos=list(self.offset(i)))
            self.spec.attach(_robot_child_spec(robot), prefix=f"s{i}_", frame=frame)

        self._free_bodies = []

    # -- placement ---------------------------------------------------------

    def offset(self, i):
        """Lateral offset of scene `i`, centred on y=0."""
        return np.array([0.0, (i - 0.5 * (self.n_scenes - 1)) * self.spread, 0.0])

    # -- static geometry ---------------------------------------------------

    CONTACT_SOLREF = (0.002, 1.0)
    CONTACT_SOLIMP = (0.95, 0.99, 0.0005, 0.5, 2.0)

    def _stiffen(self, g):
        """Per-geom contact stiffness; see the note in `__init__`."""
        g.solref = list(self.CONTACT_SOLREF)
        g.solimp = list(self.CONTACT_SOLIMP)
        return g

    def box(self, name, pos, dims, rgba, collide=True, group=0, quat=None):
        import mujoco
        g = self.spec.worldbody.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [float(d) / 2.0 for d in dims]
        g.pos = [float(v) for v in pos]
        if quat is not None:
            g.quat = [float(v) for v in quat]
        g.rgba = list(rgba)
        g.group = int(group)
        if not collide:
            g.contype, g.conaffinity = 0, 0
        return self._stiffen(g)

    def sphere(self, name, pos, radius, rgba, collide=True, group=0):
        import mujoco
        g = self.spec.worldbody.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [float(radius), 0.0, 0.0]
        g.pos = [float(v) for v in pos]
        g.rgba = list(rgba)
        g.group = int(group)
        if not collide:
            g.contype, g.conaffinity = 0, 0
        return self._stiffen(g)

    def marker(self, name, pos, radius, rgba, group=0):
        """A non-colliding target marker."""
        return self.sphere(name, pos, radius, rgba, collide=False, group=group)

    # -- the manipulated object -------------------------------------------

    def free_body(self, name, geoms, pos=(0.0, 0.0, 0.0), collide=False,
                  friction=None, mass=0.05, condim=4):
        """A free-floating body whose pose is written directly each frame.

        `geoms` is a list of ``(kind, size, local_pos, rgba)`` where `kind` is
        "box" or "sphere" and `size` is the full dimensions / the radius.  A
        tetromino is one body of six spheres; a cube is one body of one box.

        `collide` is off by default: the carried object's pose is INFERRED from
        the EE path (`carry_schedule`), not planned, so its contacts would be an
        artefact of that interpolation and would swamp the arm-versus-world
        penetration the viewer exists to show.  Turn it on to watch the object
        settle against the goal walls, knowing what the number then means.
        """
        import mujoco

        body = self.spec.worldbody.add_body()
        body.name = name
        body.pos = [float(v) for v in pos]
        body.add_freejoint(name=f"{name}_free")
        for j, (kind, size, local, rgba) in enumerate(geoms):
            g = body.add_geom()
            g.name = f"{name}_g{j}"
            if kind == "box":
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size = [float(s) / 2.0 for s in size]
            elif kind == "sphere":
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size = [float(size), 0.0, 0.0]
            else:
                raise ValueError(f"unknown geom kind {kind!r}")
            g.pos = [float(v) for v in local]
            g.rgba = list(rgba)
            g.mass = float(mass) / max(len(geoms), 1)
            g.friction = list(friction) if friction is not None else [2.0, 0.005, 0.0001]
            g.condim = int(condim)
            self._stiffen(g)
            if not collide:
                g.contype, g.conaffinity = 0, 0
        self._free_bodies.append(name)
        return body

    # -- grasp constraint --------------------------------------------------

    def weld(self, name, body1, body2, active=False):
        """An initially-inactive weld, used as the grasp in a dynamic rollout.

        A friction grasp with the Panda hand on a 1.5 cm peg is its own research
        problem, and failing it would tell you about MuJoCo's contact solver
        rather than about the plan.  Welding the object to the hand for the
        duration of the carry keeps the grasp idealised while everything else --
        the arm's tracking under gravity and its own inertia, the object's fall
        and settle after release, every contact -- stays real.
        """
        import mujoco
        e = self.spec.add_equality()
        e.name = name
        e.type = mujoco.mjtEq.mjEQ_WELD
        e.objtype = mujoco.mjtObj.mjOBJ_BODY
        e.name1 = body1
        e.name2 = body2
        e.active = bool(active)
        # Engage over ~5 ms rather than in one step (which rings), but no
        # softer: a compliant weld lets the object lag behind the hand and, on a
        # fast swing, slip out of the grasp entirely -- a constraint artefact
        # that would read as a dropped block.
        e.solref = [0.005, 1.0]
        e.solimp = [0.95, 0.999, 0.001, 0.5, 2.0]
        return e

    def exclude(self, body1, body2):
        """Turn off contact between one pair of bodies.

        Used for the gripper versus the object it is about to grasp.  The
        planner optimises a 7-DOF arm with NO HAND (`iosp.config.URDF_PATH` is
        the spherized arm alone), so it cannot know the gripper's width and its
        approach sweeps the fingers through the target -- on tetris that shoves
        the block about 3 cm before the grasp closes, and the grasp then carries
        the shove to the release.  Excluding the pair is what a real pick-and-
        place stack does: the object you are grasping is not an obstacle for the
        gripper.  Everything else about the object stays physical, including its
        contacts with the table, the walls and the other blocks after release.
        """
        e = self.spec.add_exclude()
        e.bodyname1 = body1
        e.bodyname2 = body2
        return e

    # -- finish ------------------------------------------------------------

    def compile(self):
        return MjWorld(self.spec.compile(), self.robot, self.n_scenes,
                       self._free_bodies, [self.offset(i) for i in range(self.n_scenes)])


class MjWorld:
    """A compiled multi-scene model.

    Two ways to drive it.  KINEMATIC (`set_arm` + `forward`) writes the planned
    angles straight into ``qpos``: it shows the plan as drawn, and the carried
    object's pose has to be faked by `carry_schedule`.  DYNAMIC (`set_ctrl` +
    `step`) writes the plan into the position servos' targets and integrates,
    so what you watch is the arm's actual response under gravity and its own
    inertia, and the object moves because it is welded to the hand and then
    dropped.  `iosp.viz.mj_rollout` drives the second.
    """

    def __init__(self, model, robot, n_scenes, free_bodies, offsets):
        import mujoco

        self.model = model
        self.data = mujoco.MjData(model)
        self.robot = robot
        self.n_scenes = n_scenes
        self.offsets = offsets
        self._free_bodies = list(free_bodies)

        def jadr(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"joint {name!r} not in the compiled model")
            return int(model.jnt_qposadr[jid])

        self.arm_adr = [[jadr(f"s{i}_{n}") for n in _ARM_JOINTS[robot]]
                        for i in range(n_scenes)]
        self.ee_body = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          f"s{i}_{_EE_BODY[robot]}")
                        for i in range(n_scenes)]
        self.finger_adr = [[jadr(f"s{i}_{n}") for n in _FINGER_JOINTS[robot]]
                           for i in range(n_scenes)]
        self.free_adr = {n: jadr(f"{n}_free") for n in free_bodies}

        def aid(name):
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            return None if a < 0 else int(a)

        # The Menagerie model ships position servos named actuator1..7 plus a
        # tendon-driven gripper (actuator8); the spherized URDF import has none,
        # so a dynamic rollout requires `--robot menagerie`.
        self.arm_act = [[a for a in (aid(f"s{i}_actuator{k}")
                                     for k in range(1, 8)) if a is not None]
                        for i in range(n_scenes)]
        self.grip_act = [aid(f"s{i}_actuator8") for i in range(n_scenes)]
        self.weld_id = {}
        for e in range(model.neq):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e)
            if nm:
                self.weld_id[nm] = e

        # Bodies belonging to each arm, for the adjacent-link contact filter.
        self._arm_bodies = [set() for _ in range(n_scenes)]
        for b in range(model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm is None:
                continue
            for i in range(n_scenes):
                if nm.startswith(f"s{i}_"):
                    self._arm_bodies[i].add(b)
                    break

        # Arm (and finger) DOF addresses, for gravity compensation: everything
        # belonging to an attached robot, and nothing belonging to a free body.
        arm_dofs = []
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm is None or nm.endswith("_free"):
                continue
            n_dof = {mujoco.mjtJoint.mjJNT_FREE: 6,
                     mujoco.mjtJoint.mjJNT_BALL: 3}.get(
                         mujoco.mjtJoint(model.jnt_type[j]), 1)
            arm_dofs.extend(range(int(model.jnt_dofadr[j]),
                                  int(model.jnt_dofadr[j]) + n_dof))
        self._arm_dof_adr = np.asarray(arm_dofs, dtype=int)

        self.set_neutral()

    @classmethod
    def wrap(cls, model, arm_joints, ee_body, ee_offset, free_bodies=(),
             finger_joints=(), grip_actuator=None, arm_actuators=None):
        """Adopt a model built ELSEWHERE, e.g. `sim_teleop.pickplace.model`.

        `WorldBuilder` assumes it assembled the world itself and that every
        robot joint carries an `s<i>_` prefix.  The teleop pick-and-place scene
        is generated by its own module -- FR3 plus Franka Hand plus the table,
        bucket and cube -- and rebuilding it here would be a second copy of a
        geometry that already exists and is already randomised per episode.  So
        this takes the compiled model as given and only records where its arm,
        gripper and free bodies live, after which every method above works
        unchanged.
        """
        import mujoco

        self = cls.__new__(cls)
        self.model = model
        self.data = mujoco.MjData(model)
        self.robot = "external"
        self.n_scenes = 1
        self.offsets = [np.zeros(3)]

        def jadr(name):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise KeyError(f"joint {name!r} not in the model")
            return int(model.jnt_qposadr[j])

        def aid(name):
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            return None if a < 0 else int(a)

        self._free_bodies = list(free_bodies)
        self._arm_joint_names = list(arm_joints)
        self.arm_adr = [[jadr(n) for n in arm_joints]]
        self.finger_adr = [[jadr(n) for n in finger_joints]]
        self.ee_body = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)]
        self._ee_offset = np.asarray(ee_offset, float)
        self.free_adr = {n: jadr(f"{n}_free") for n in free_bodies}
        acts = list(arm_actuators) if arm_actuators else list(arm_joints)
        self.arm_act = [[a for a in (aid(n) for n in acts) if a is not None]]
        self.grip_act = [aid(grip_actuator) if grip_actuator else None]
        self.weld_id = {}

        self._arm_bodies = [set()]
        arm_dofs = []
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm is None or nm.endswith("_free"):
                continue
            n_dof = {mujoco.mjtJoint.mjJNT_FREE: 6,
                     mujoco.mjtJoint.mjJNT_BALL: 3}.get(
                         mujoco.mjtJoint(model.jnt_type[j]), 1)
            arm_dofs.extend(range(int(model.jnt_dofadr[j]),
                                  int(model.jnt_dofadr[j]) + n_dof))
        self._arm_dof_adr = np.asarray(arm_dofs, dtype=int)
        self.forward()
        return self

    # -- writing state -----------------------------------------------------

    def set_arm(self, i, q):
        q = np.asarray(q, float).ravel()
        for k, adr in enumerate(self.arm_adr[i]):
            self.data.qpos[adr] = q[k]

    def set_fingers(self, i, opening):
        for adr in self.finger_adr[i]:
            self.data.qpos[adr] = float(opening)

    def set_free(self, name, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        adr = self.free_adr[name]
        self.data.qpos[adr:adr + 3] = np.asarray(pos, float)
        self.data.qpos[adr + 3:adr + 7] = np.asarray(quat, float)

    def set_neutral(self):
        """A neutral arm pose on every scene, fingers open."""
        for i in range(self.n_scenes):
            self.set_arm(i, NEUTRAL_Q)
            self.set_fingers(i, FINGER_OPEN)
        self.forward()

    def forward(self):
        import mujoco
        mujoco.mj_forward(self.model, self.data)

    def set_grasp_friction(self, torsional=0.1):
        """Raise torsional friction on finger pads and free-body geoms.

        Matches sim_teleop/sandbox.py's calibrated fix: MuJoCo's default
        torsional friction (0.005) lets the cube roll through a closed grip;
        measured slip was 31 mm at 0.005 vs 7 mm at 0.1.
        """
        import mujoco
        n = 0
        for g in range(self.model.ngeom):
            body = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[g]) or ""
            gname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            is_finger = "finger" in body
            is_free = any(gname.startswith(f"{nm}_") for nm in self._free_bodies)
            if is_finger or is_free:
                self.model.geom_friction[g, 1] = torsional
                n += 1
        return n

    # -- reading kinematics ------------------------------------------------

    def ee_position(self, i):
        """Scene `i`'s end-effector position at the current state (world frame)."""
        b = self.ee_body[i]
        R = self.data.xmat[b].reshape(3, 3)
        return self.data.xpos[b] + R @ getattr(self, "_ee_offset", EE_OFFSET)

    def ee_path(self, i, q_rows):
        """FK the whole joint path of scene `i` -> (T, 3), without touching JAX.

        Used both to draw the EE trace and to carry the manipulated object, so a
        trajectory loaded from a `scratch/feas/*.npz` extract (which stores only
        `q`) renders exactly like one just solved.
        """
        q_rows = np.asarray(q_rows, float)
        saved = self.data.qpos.copy()
        out = np.empty((len(q_rows), 3))
        for t, q in enumerate(q_rows):
            self.set_arm(i, q)
            self.forward()
            out[t] = self.ee_position(i)
        self.data.qpos[:] = saved
        self.forward()
        return out

    # -- reading contacts --------------------------------------------------

    def contacts(self, min_depth_mm=0.0):
        """Current-frame contacts as (depth_mm, geom1, geom2), deepest first.

        Contacts between adjacent links of the SAME arm are dropped: the
        spherized model's neighbour links always overlap slightly, which is a
        property of the collision model, not of the trajectory.  This is the
        same filter `iosp.checks.feasibility_report` applies.
        """
        import mujoco

        m, d = self.model, self.data
        out = []
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            same_arm = any(b1 in bodies and b2 in bodies
                           for bodies in self._arm_bodies)
            if same_arm and abs(b1 - b2) <= 1:
                continue
            depth = -float(con.dist) * 1000.0
            if depth < min_depth_mm:
                continue
            out.append((depth, self._geom_label(con.geom1),
                        self._geom_label(con.geom2)))
        out.sort(reverse=True)
        return out

    def _geom_label(self, gid):
        """A readable name for a geom.  URDF-imported collision geoms are
        usually unnamed, so fall back to the body they hang off."""
        import mujoco
        nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
        if nm:
            return nm
        body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(self.model.geom_bodyid[gid]))
        return f"{body or '?'}[geom{int(gid)}]"

    def worst_penetration_mm(self):
        c = self.contacts()
        return c[0][0] if c else 0.0

    # -- dynamics ----------------------------------------------------------

    def set_ctrl(self, i, q_target, gripper=None):
        """Write scene `i`'s position-servo targets (and the gripper command).

        Targets outside an actuator's ``ctrlrange`` are CLAMPED by MuJoCo, which
        is deliberate: a plan that commands a joint past its limit physically
        cannot be followed, and the resulting tracking error is the honest
        rendering of that violation rather than a number in a report.
        """
        q_target = np.asarray(q_target, float).ravel()
        for k, a in enumerate(self.arm_act[i]):
            self.data.ctrl[a] = q_target[k]
        if gripper is not None and self.grip_act[i] is not None:
            self.data.ctrl[self.grip_act[i]] = float(gripper)

    def gravity_compensate(self):
        """Cancel gravity and Coriolis on the arm joints, as a real Panda does.

        Menagerie's position actuators are plain PD servos with no feedforward,
        so an extended arm SAGS: the spring has to be stretched by a constant
        error to hold the limb up, and that steady-state droop does not shrink
        however long you wait at a waypoint (verified: 4x the dwell leaves the
        worst tracking error unchanged at 15.7 deg).  Franka's own controller
        compensates gravity, so leaving it out measures the absence of a
        feedforward term rather than anything about the plan.

        `qfrc_bias` is exactly the gravity + Coriolis + centrifugal term, so
        applying it as an external force cancels it -- but ONLY on the arm's own
        degrees of freedom.  Applying it across the whole vector also cancels
        gravity on the manipulated objects, which then float away instead of
        falling; the point of the rollout is that they fall.
        """
        self.data.qfrc_applied[:] = 0.0
        adr = self._arm_dof_adr
        self.data.qfrc_applied[adr] = self.data.qfrc_bias[adr]

    def step(self, n=1, gravcomp=False):
        import mujoco
        for _ in range(int(n)):
            if gravcomp:
                self.gravity_compensate()
            mujoco.mj_step(self.model, self.data)

    def attach_pose(self, body_name, free_name):
        """Relative pose of a free body w.r.t. `body_name`, right now.

        -> (rel_pos, rel_quat) in `body_name`'s frame, which `carry_free` below
        replays.  This is the grasp: whatever offset the object happens to sit
        at when the gripper closes is what gets carried, so a plan whose grasp
        row misses the object carries that miss through to the release.
        """
        import mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, free_name)
        R1 = self.data.xmat[bid].reshape(3, 3)
        rel_pos = R1.T @ (self.data.xpos[fid] - self.data.xpos[bid])
        q1, q2 = np.empty(4), np.empty(4)
        mujoco.mju_mat2Quat(q1, self.data.xmat[bid].ravel())
        mujoco.mju_mat2Quat(q2, self.data.xmat[fid].ravel())
        q1inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        rel_quat = np.empty(4)
        mujoco.mju_mulQuat(rel_quat, q1inv, q2)
        return rel_pos, rel_quat

    def carry_free_compliant(self, body_name, free_name, rel_pos, rel_quat,
                             omega=15.0, max_accel=25.0):
        """Drive a carried body toward the gripper with a FORCE, not a teleport.

        `carry_free` writes the object's pose directly, so contacts are computed
        but never resolved: the block passes through a goal wall with nothing
        pushing back, and a plan that sweeps it through one looks clean.  This
        applies a critically damped spring toward the same target pose through
        `xfrc_applied`, so the wall can push back and a bad plan's penetration
        actually shows up.

        The gains are derived from the body's own MASS and INERTIA, not fixed
        numbers: a tetromino is 50 g with an inertia around 1e-5 kg m^2, so a
        hand-picked 4 N m torque is a ~4e5 rad/s^2 angular acceleration and the
        integrator diverges immediately.  Specifying the tracker by its natural
        frequency `omega` instead makes it mass-independent and stable at the
        2 ms timestep (omega * dt = 0.05).

        `max_accel` caps the tracker at a couple of g.  Uncapped, it happily
        drives the block into a wall at 20 g; the contact then ejects it at
        25 m/s the moment the carry ends.  A real gripper cannot push a 50 g
        block through a wall either, so the cap is the physical statement: when
        the plan commands the block somewhere it cannot go, the grasp SLIPS
        rather than the wall giving way, and the resulting placement error is
        the honest consequence of the plan.
        """
        import mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, free_name)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{free_name}_free")
        adr = int(self.model.jnt_dofadr[jid])

        m = float(self.model.body_mass[fid])
        inertia = np.asarray(self.model.body_inertia[fid], float)

        R1 = self.data.xmat[bid].reshape(3, 3)
        target = self.data.xpos[bid] + R1 @ rel_pos
        # Track the body's CoM, which is what `xfrc_applied` accelerates.
        com = np.asarray(self.data.xipos[fid], float)
        com_off = com - np.asarray(self.data.xpos[fid], float)
        err = (target + com_off) - com
        v = np.asarray(self.data.qvel[adr:adr + 3], float)

        acc = omega ** 2 * err - 2.0 * omega * v
        n = np.linalg.norm(acc)
        if n > max_accel:
            acc *= max_accel / n
        # Cancel gravity on the carried body too: the gripper is holding it.
        f = m * (acc - np.asarray(self.model.opt.gravity, float))

        q1, q2, want, e = (np.empty(4) for _ in range(4))
        mujoco.mju_mat2Quat(q1, self.data.xmat[bid].ravel())
        mujoco.mju_mat2Quat(q2, self.data.xmat[fid].ravel())
        mujoco.mju_mulQuat(want, q1, rel_quat)
        mujoco.mju_mulQuat(e, want, np.array([q2[0], -q2[1], -q2[2], -q2[3]]))
        axis = e[1:] * (2.0 if e[0] >= 0 else -2.0)
        w = np.asarray(self.data.qvel[adr + 3:adr + 6], float)
        alpha = omega ** 2 * axis - 2.0 * omega * w
        tau = float(np.mean(inertia)) * alpha

        self.data.xfrc_applied[fid, :3] = f
        self.data.xfrc_applied[fid, 3:] = tau
        return target

    def clear_body_force(self, free_name):
        import mujoco
        fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, free_name)
        self.data.xfrc_applied[fid, :] = 0.0

    def carry_free(self, body_name, free_name, rel_pos, rel_quat):
        """Write a free body's pose from `body_name`'s, at the given offset.

        The grasp is KINEMATIC rather than an `mjEQ_WELD`.  A weld puts the
        grasp inside the constraint solver, where it competes with the position
        servos and the contacts for the same solver iterations and holds the
        object at whatever pose its `eq_data` was authored with -- three
        separate ways for the carried object to sit somewhere other than in the
        gripper.  Writing the pose directly is exact and has none of them, and
        the physics that the rollout is actually asking about -- the arm's
        response under gravity, the release, the fall, the settle, every
        contact -- is untouched, because it all happens outside this window.
        """
        import mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        R1 = self.data.xmat[bid].reshape(3, 3)
        pos = self.data.xpos[bid] + R1 @ rel_pos
        q1, quat = np.empty(4), np.empty(4)
        mujoco.mju_mat2Quat(q1, self.data.xmat[bid].ravel())
        mujoco.mju_mulQuat(quat, q1, rel_quat)
        adr = self.free_adr[free_name]
        self.data.qpos[adr:adr + 3] = pos
        self.data.qpos[adr + 3:adr + 7] = quat
        return pos

    def set_free_vel(self, free_name, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0)):
        """Velocity of a free body, so a released object keeps its momentum."""
        jid = self.free_adr[free_name]
        # A free joint's qvel block starts at its dof address, not its qpos one.
        import mujoco
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                              f"{free_name}_free")
        adr = int(self.model.jnt_dofadr[j])
        self.data.qvel[adr:adr + 3] = np.asarray(lin, float)
        self.data.qvel[adr + 3:adr + 6] = np.asarray(ang, float)

    def reset(self):
        import mujoco
        mujoco.mj_resetData(self.model, self.data)

    def arm_q(self, i):
        """Scene `i`'s ACTUAL arm angles at the current state."""
        return np.array([self.data.qpos[a] for a in self.arm_adr[i]])

    def body_xyzyaw(self, name):
        """A body's pose as SPaSM writes block poses: [x, y, z, yaw].

        Inverts `conversions.yaw_to_quat_xyz` (180 deg about x, then yaw about
        z), so the pose a rollout ACHIEVES can be fed straight back into SPaSM's
        own packing / stacking cost.
        """
        import mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        R = self.data.xmat[bid].reshape(3, 3)
        # Undo the x-flip, then read the remaining rotation about z.  Column 0
        # of R is the block frame's +x in world, which the flip leaves in plane.
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        p = self.data.xpos[bid]
        return np.array([p[0], p[1], p[2], yaw], float)

    def body_tilt_deg(self, name):
        """How far a body's own +z has tipped away from world +z, in degrees.

        For a SPaSM block the 180-degree x flip lives in the GEOM offsets, not
        in the body frame, so an upright block has its body +z pointing up and
        reads 0 here; one knocked onto its side reads near 90.  A toppled block
        is a task failure that comparing positions alone cannot catch.
        """
        import mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        z = self.data.xmat[bid].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))

    def body_pos(self, name):
        import mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return np.array(self.data.xpos[bid])

    @property
    def has_actuators(self):
        n = len(getattr(self, "_arm_joint_names", None)
                or _ARM_JOINTS[self.robot])
        return all(len(a) == n for a in self.arm_act)

    def joint_limits(self):
        """(lo, hi) arm-joint limits, (7,) each, read off scene 0's copy."""
        import mujoco
        names = getattr(self, "_arm_joint_names", None)
        if names is None:
            names = [f"s0_{n}" for n in _ARM_JOINTS[self.robot]]
        lo = np.empty(len(names))
        hi = np.empty_like(lo)
        for k, name in enumerate(names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo[k], hi[k] = self.model.jnt_range[jid]
        return lo, hi


# SPaSM's neutral arm pose (Simulation.get_neutral_pose, arm joints only).
NEUTRAL_Q = np.array([0.0, -np.pi / 4, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4])


# One colour per scene, for the EE traces.
SCENE_COLORS = [
    (0x3b, 0x7d, 0xd8), (0x2a, 0xb0, 0x5e), (0xd4, 0x6a, 0x20),
    (0x9b, 0x3d, 0xb8), (0xd4, 0x3d, 0x3d), (0x20, 0xad, 0xad),
]


def polyline(server, name, pts, color, width=4.0):
    """EE trace, drawn as viser line segments over the same server mjviser uses.

    MuJoCo has no line primitive, so the trace stays a viser scene node; it
    shares the server, so it lands in the same view as the model.
    """
    pts = np.asarray(pts, np.float32)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    return server.scene.add_line_segments(
        name, points=segs,
        colors=np.tile(np.asarray(color, np.uint8), (len(segs), 2, 1)),
        line_width=width)


def play(world, n_rows, set_row, port=8080, fps=6.0, banner="",
         info_fn=None, extra_gui=None, camera_azimuth=140.0,
         keyframe_every=1, verdict_md=None):
    """Serve `world` in mjviser with a frame slider, play toggle and verdict.

    `set_row(t)` writes frame `t` into `world` (it should NOT call `forward`;
    this does).  `info_fn(t)` returns an extra markdown line, or None.
    `extra_gui(server, refresh)` adds domain-specific controls, where `refresh`
    re-renders the current frame.

    `keyframe_every` is the upsampling factor `resample_rows` was called with:
    frames that are multiples of it are the planner's own rows, and the panel
    says which of the two you are looking at.  `verdict_md` is a markdown block
    pinned above the playback controls -- the per-scene success readout, which
    is a property of the whole trajectory and so must not scroll away with the
    frame.
    """
    import mjviser
    import viser

    server = viser.ViserServer(port=port)
    scene = mjviser.ViserMujocoScene(server, world.model, num_envs=1)
    tabs = scene.create_visualization_gui(camera_azimuth=camera_azimuth)

    state = {"row": 0}

    with tabs.add_tab("Playback", icon=viser.Icon.PLAYER_PLAY):
        if verdict_md:
            server.gui.add_markdown(verdict_md)
        row = server.gui.add_slider("Frame", 0, max(n_rows - 1, 1), 1, 0)
        playing = server.gui.add_checkbox("Play", True)
        speed = server.gui.add_slider("Frames / sec", 1.0, 120.0, 1.0, float(fps))
        loop = server.gui.add_checkbox("Loop", True)
        show_con = server.gui.add_checkbox("Show contacts", False)
        info = server.gui.add_markdown("")

    def refresh(_=None):
        t = int(row.value)
        state["row"] = t
        set_row(t)
        world.forward()
        scene.update_from_mjdata(world.data)
        if keyframe_every > 1:
            k, r = divmod(t, keyframe_every)
            kind = f"keyframe **{k}**" if r == 0 else f"interp after keyframe {k}"
            lines = [f"frame **{t}**/{n_rows - 1} ({kind})"]
        else:
            lines = [f"row **{t}**/{n_rows - 1}"]
        extra = info_fn(t) if info_fn is not None else None
        if extra:
            lines.append(extra)
        con = world.contacts(min_depth_mm=0.05)
        if con:
            depth, g1, g2 = con[0]
            lines.append(f"deepest contact **{depth:.1f} mm** &nbsp; `{g1}` / `{g2}`")
        else:
            lines.append("no contact")
        info.content = " &nbsp;|&nbsp; ".join(lines)

    row.on_update(refresh)
    show_con.on_update(lambda _=None: setattr(scene, "show_contact_points",
                                              show_con.value))
    if extra_gui is not None:
        extra_gui(server, refresh)
    refresh()

    if banner:
        print(banner)
    print(f"Viser server: http://0.0.0.0:{server.get_port()}")
    # Advance on a monotonic clock rather than sleeping a frame period per
    # iteration: at 60 fps the per-frame `mj_forward` + scene push is a
    # significant fraction of the 16 ms budget, and sleeping the full period on
    # top of it halves the effective rate -- which is most of what "choppy"
    # meant at the old 6 rows/sec default.
    try:
        next_t = time.perf_counter()
        while True:
            if playing.value and n_rows > 1:
                nxt = int(row.value) + 1
                if nxt >= n_rows:
                    if not loop.value:
                        playing.value = False
                        continue
                    nxt = 0
                row.value = nxt
                next_t += 1.0 / float(speed.value)
                dt = next_t - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    next_t = time.perf_counter()     # we are behind; don't burst
            else:
                time.sleep(0.02)
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def play_multi(entries, port=8080, fps=60.0, camera_azimuth=140.0):
    """Serve multiple independent scenes with a dropdown to switch between them.

    `entries` is a list of dicts, each with:
        world, n_frames, set_row, label, verdict_md (optional), info_fn (optional)
    """
    import threading

    import mjviser
    import viser

    server = viser.ViserServer(port=port)
    # Reentrant: viser fires `on_update` callbacks synchronously, so assigning
    # `row.value` while holding the lock re-enters `refresh` on the same thread.
    lock = threading.RLock()
    state = {"idx": 0, "mj_scene": None, "switching": False}

    labels = [e["label"] for e in entries]

    def _activate(idx):
        """Tear down the old 3D scene and build a new one for entry `idx`."""
        state["switching"] = True
        try:
            if state["mj_scene"] is not None:
                state["mj_scene"].rebuild_visual_handles = lambda: None
                state["mj_scene"] = None
            server.scene.reset()
            w = entries[idx]["world"]
            sc = mjviser.ViserMujocoScene(server, w.model, num_envs=1)
            state["mj_scene"] = sc
            state["idx"] = idx
        finally:
            state["switching"] = False

    _activate(0)

    gui_tabs = server.gui.add_tab_group()
    with gui_tabs.add_tab("Playback", icon=viser.Icon.PLAYER_PLAY):
            scene_select = server.gui.add_dropdown(
                "Scene", labels, initial_value=labels[0])
            verdict_label = server.gui.add_markdown(
                entries[0].get("verdict_md", ""))
            row = server.gui.add_slider(
                "Frame", 0, max(entries[0]["n_frames"] - 1, 1), 1, 0)
            playing = server.gui.add_checkbox("Play", True)
            speed = server.gui.add_slider(
                "Frames / sec", 1.0, 120.0, 1.0, float(fps))
            loop = server.gui.add_checkbox("Loop", True)
            info = server.gui.add_markdown("")

    def refresh(_=None):
        with lock:
            if state["switching"] or state["mj_scene"] is None:
                return
            idx = state["idx"]
            e = entries[idx]
            t = min(int(row.value), e["n_frames"] - 1)
            e["set_row"](t)
            e["world"].forward()
            state["mj_scene"].update_from_mjdata(e["world"].data)
            lines = [f"**{e['label']}** &nbsp;|&nbsp; "
                     f"frame **{t}**/{e['n_frames'] - 1}"]
            ifn = e.get("info_fn")
            if ifn:
                extra = ifn(t)
                if extra:
                    lines.append(extra)
            info.content = " &nbsp;|&nbsp; ".join(lines)

    def _on_scene_change(_=None):
        with lock:
            idx = labels.index(scene_select.value)
            if idx == state["idx"]:
                return
            _activate(idx)
            e = entries[idx]
            row.max = max(e["n_frames"] - 1, 1)
            row.value = 0
            verdict_label.content = e.get("verdict_md", "")
        refresh()

    scene_select.on_update(_on_scene_change)
    row.on_update(refresh)
    refresh()

    print(f"Viser server: http://0.0.0.0:{server.get_port()}")
    try:
        next_t = time.perf_counter()
        while True:
            if state["switching"]:
                time.sleep(0.02)
                next_t = time.perf_counter()
                continue
            e = entries[state["idx"]]
            if playing.value and e["n_frames"] > 1:
                nxt = int(row.value) + 1
                if nxt >= e["n_frames"]:
                    if not loop.value:
                        playing.value = False
                        continue
                    nxt = 0
                row.value = nxt
                next_t += 1.0 / float(speed.value)
                dt = next_t - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    next_t = time.perf_counter()
            else:
                time.sleep(0.02)
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Temporal upsampling
# ---------------------------------------------------------------------------

def resample_rows(q_rows, factor, kind="cubic"):
    """Upsample a joint path (T, D) -> (T', D) with T' = (T-1)*factor + 1.

    The planner's output is a coarse skeleton -- tetris is 22 rows -- so playing
    it one row per frame looks like a slideshow no matter the frame rate.  The
    original rows are preserved exactly (indices ``t * factor``), so the
    keyframes the solver actually optimised are still visited and still the ones
    the verdict is computed on; the frames between them are interpolation, and
    the playback panel says so.

    ``cubic`` is a natural cubic spline through the rows (C2, no velocity
    discontinuity at the waypoints); ``linear`` keeps the exact piecewise-linear
    path, which is jerky at the corners but introduces no overshoot.
    """
    q_rows = np.asarray(q_rows, float)
    factor = max(int(factor), 1)
    T = len(q_rows)
    if factor == 1 or T < 2:
        return q_rows.copy()
    src = np.arange(T, dtype=float)
    dst = np.linspace(0.0, T - 1.0, (T - 1) * factor + 1)
    if kind == "cubic" and T >= 4:
        from scipy.interpolate import CubicSpline
        out = CubicSpline(src, q_rows, axis=0, bc_type="natural")(dst)
    else:
        out = np.stack([np.interp(dst, src, q_rows[:, d])
                        for d in range(q_rows.shape[1])], axis=1)
    out[::factor] = q_rows          # pin the keyframes against spline drift
    return out


# ---------------------------------------------------------------------------
# Success verdict
# ---------------------------------------------------------------------------

REACH_TOL_MM = 50.0        # `iosp.checks.feasibility_report`'s thresholds,
PENETRATION_TOL_MM = 5.0   # so the viewer's verdict and the check's agree.


class Verdict:
    """Per-scene pass/fail on the same four criteria `feasibility_report` uses.

    Computed here from MuJoCo FK on the ORIGINAL (non-interpolated) rows, so no
    JAX import is needed and `--from-npz` gets a verdict too.  The EE frame is
    `EE_OFFSET` off the last arm link, which is the point the planner optimised.
    """

    __slots__ = ("pick_mm", "place_mm", "penetration_mm", "limit_deg",
                 "worst_contact")

    def __init__(self, pick_mm, place_mm, penetration_mm, limit_deg,
                 worst_contact):
        self.pick_mm = pick_mm
        self.place_mm = place_mm
        self.penetration_mm = penetration_mm
        self.limit_deg = limit_deg
        self.worst_contact = worst_contact

    @property
    def pick_ok(self):
        return self.pick_mm < REACH_TOL_MM

    @property
    def place_ok(self):
        return self.place_mm < REACH_TOL_MM

    @property
    def collision_ok(self):
        return self.penetration_mm < PENETRATION_TOL_MM

    @property
    def limits_ok(self):
        return self.limit_deg < 1e-3

    @property
    def ok(self):
        return (self.pick_ok and self.place_ok and self.collision_ok
                and self.limits_ok)

    def _flags(self):
        return [("pick", self.pick_ok, f"{self.pick_mm:.0f} mm"),
                ("place", self.place_ok, f"{self.place_mm:.0f} mm"),
                ("collision", self.collision_ok,
                 f"{self.penetration_mm:.1f} mm"),
                ("limits", self.limits_ok, f"{self.limit_deg:.1f} deg")]

    def line(self):
        """One-line console summary."""
        head = "SUCCESS" if self.ok else "FAIL"
        body = "  ".join(f"{'ok' if good else 'X'} {n} {val}"
                         for n, good, val in self._flags())
        return f"{head:>7}   {body}"

    def markdown(self):
        head = ("<span style='color:#2ab05e'>**SUCCESS**</span>" if self.ok
                else "<span style='color:#d43d3d'>**FAIL**</span>")
        body = " &nbsp; ".join(f"{'✓' if good else '✗'} {n} {val}"
                               for n, good, val in self._flags())
        return f"{head} &nbsp;|&nbsp; {body}"


def evaluate(world, i, q_rows, pick, place, grasp_row, release_row):
    """-> `Verdict` for scene `i`'s path (positions in the scene's WORLD frame).

    `pick`/`place` are the demonstration's EE targets in scene-local
    coordinates, matching what the domain viewers hold.
    """
    q_rows = np.asarray(q_rows, float)
    off = world.offsets[i]
    ee = world.ee_path(i, q_rows) - off

    saved = world.data.qpos.copy()
    worst, worst_con = 0.0, None
    for t in range(len(q_rows)):
        world.set_arm(i, q_rows[t])
        world.forward()
        for depth, g1, g2 in world.contacts():
            if not (_scene_of(g1) == i or _scene_of(g2) == i):
                continue
            if depth > worst:
                worst, worst_con = depth, (g1, g2)
    world.data.qpos[:] = saved
    world.forward()

    lo, hi = world.joint_limits()
    over = np.maximum(0.0, np.maximum(lo - q_rows, q_rows - hi)).max()

    return Verdict(
        pick_mm=float(np.linalg.norm(ee[grasp_row] - np.asarray(pick, float))) * 1000,
        place_mm=float(np.linalg.norm(ee[release_row] - np.asarray(place, float))) * 1000,
        penetration_mm=worst,
        limit_deg=float(over) * 180.0 / np.pi,
        worst_contact=worst_con,
    )


def _scene_of(label):
    """Scene index encoded in a geom label's `s<i>_` prefix, or None."""
    if not label.startswith("s"):
        return None
    head = label[1:].split("_", 1)[0]
    return int(head) if head.isdigit() else None
