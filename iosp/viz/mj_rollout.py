"""Dynamic (actuated) rollout of a planned joint path, recorded for playback.

The planned rows become position-servo targets, MuJoCo integrates under gravity,
and:

  * the arm's ACTUAL angles diverge from the plan wherever the plan is beyond
    what the servos and the joint ``ctrlrange`` can deliver -- a commanded angle
    past a joint limit is clamped, so a limit-violating plan shows up as
    tracking error instead of as a line in a report;
  * every manipulated object is a real free body with mass and collision.  The
    gripper closes on it under MuJoCo's contact physics and holds it by friction
    -- no kinematic attach, no weld, no pose teleporting.  Whether the grasp
    holds is determined entirely by the physics;
  * success is read off the object's resting pose at the end, not off a
    schedule that moved it.

The rollout is run once, headless, and every scene's ``qpos`` is recorded.  The
viewer then replays recorded states, so playback stays smooth and scrubbable and
the physics is not re-integrated when you drag the slider.
"""
from __future__ import annotations

import numpy as np

# MuJoCo's Panda hand tendon actuator: ctrl is 0..255 for 0..0.04 m per finger.
GRIP_OPEN = 255.0
FINGER_CTRL_PER_M = 255.0 / 0.04


def grip_ctrl(half_width_m):
    """Gripper command that closes the fingers to `half_width_m` per finger.

    Commanding 0 (fully closed) makes the fingers CRUSH whatever is between
    them: on a 1.5 cm tetromino peg they shove the block about a centimetre
    before the grasp is established, and the object then carries that
    displacement to the release.  Closing to the object's own half-width lets
    the fingers touch without driving through.
    """
    return float(np.clip(half_width_m * FINGER_CTRL_PER_M, 0.0, GRIP_OPEN))


DEFAULT_GRASP_HALF_WIDTH = 0.015    # tetromino peg radius; a 3 cm cube -> 0.03

DEFAULT_ROW_TIME = None     # None = time-parameterize against velocity limits
DEFAULT_SETTLE = 1.5        # [s] after release, for the object to come to rest
GRASP_LAG = 0.05            # [s] the fingers get to close before the weld snaps
DEFAULT_DWELL = 0.6         # [s] held stationary at the grasp and release rows

# The fingers must not start closing while the ARM is still moving.  A position
# servo lags its reference, so at the instant the dwell begins the hand is still
# several degrees (centimetres at the pads) short of the grasp pose and is
# travelling into it; a pad that arrives sideways sweeps the object out of the
# grip before it can close on it -- measured on the tower's stack levels 2-4,
# where the cube was shoved 38 mm and the fingers then closed on nothing.
# So the close is GATED on the arm having converged, not on a fixed delay:
# once every joint is within `GRASP_SETTLE_TOL` of its command the fingers move.
# The gate is only ever a wait INSIDE the dwell -- if the arm is already settled
# (the common case, and what the tetris numbers were measured with) it fires at
# `GRASP_LAG` exactly as before, and it always fires by the end of the dwell so
# a never-converging joint cannot stall the grasp entirely.
GRASP_SETTLE_TOL = np.radians(1.0)      # [rad] per-joint command-tracking gate
MIN_ROW_TIME = 0.15         # [s] floor, so near-identical rows still get a frame

# Panda joint velocity limits [rad/s] (Franka datasheet).  A plan is a sequence
# of waypoints with no timing attached; executing it means choosing that timing,
# and choosing it badly is a property of the executor, not of the plan.  The
# tetris skeleton steps up to 50 deg between adjacent rows, so a flat 0.25 s per
# row commands ~200 deg/s -- past every one of these -- and what you then watch
# is the servos saturating rather than the plan failing.
VEL_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
# Fraction of the limit to actually command.  0.5 was measured to leave the
# position servo a STEADY 7.4 deg behind its reference through every travel row
# -- centimetres at the pads -- so the hand was still visibly moving into place
# when the grasp dwell began and swept the block sideways before the fingers
# could close on it.  0.25 halves that to 3.7 deg and is what turns the tower's
# level-1 place from "picked up, then dropped" into a correct placement (z
# 0.087 against a 0.090 target, tilt 2.5 deg).  Going further to 0.12 buys
# another 2 deg of tracking and no measurable improvement in where blocks end
# up, at twice the wall-clock, so the speed stops here.
VEL_SAFETY = 0.25


def row_durations(q_rows, safety=VEL_SAFETY, floor=MIN_ROW_TIME):
    """Per-row execution time honouring the joint velocity limits -> (T-1,).

    Row `t` gets however long its largest joint displacement needs at
    `safety` x the limit, so the commanded reference never asks for a speed the
    real arm could not produce.
    """
    dq = np.abs(np.diff(np.asarray(q_rows, float), axis=0))       # (T-1, 7)
    need = (dq / (VEL_LIMIT * safety)).max(axis=-1)               # per row
    return np.maximum(need, floor)


class Rollout:
    """Recorded result of a dynamic rollout.

    `qpos` is (F, nq) -- full recorded state, replayable by writing it back.
    `q_cmd` / `q_act` are (n_scenes, F, 7): the commanded and achieved arm
    angles, whose difference is the tracking error.  `frame_row` maps each
    recorded frame to the planner row it was commanding (fractional), so the
    viewer can still name the phase.
    """

    def __init__(self, qpos, q_cmd, q_act, frame_row, dt, grasp_frame,
                 release_frame):
        self.qpos = qpos
        self.q_cmd = q_cmd
        self.q_act = q_act
        self.frame_row = frame_row
        self.dt = dt
        self.grasp_frame = grasp_frame
        self.release_frame = release_frame

    @property
    def n_frames(self):
        return self.qpos.shape[0]

    def tracking_mm_deg(self, i):
        """Per-scene worst and mean tracking error, in degrees."""
        err = np.abs(self.q_act[i] - self.q_cmd[i])
        return float(np.degrees(err.max())), float(np.degrees(err.mean()))


def run_events(world, q_rows, events, row_time=DEFAULT_ROW_TIME,
               settle=DEFAULT_SETTLE, fps=60.0, progress=True,
               safety=VEL_SAFETY, dwell=DEFAULT_DWELL,
               grasp_half_width=DEFAULT_GRASP_HALF_WIDTH, hand_body="s0_hand",
               gravcomp=True):
    """Single-scene rollout with SEVERAL pick-and-place events.

    The gripper opens and closes via the position actuators, and MuJoCo's
    contact physics determines whether the object is held.  No kinematic carry,
    no weld -- this is the honest test of whether the planned grasp pose
    actually encloses the object.

    `events` is a list of ``(grasp_row, release_row, body_name)``.  A multi-block
    task is one continuous joint path with the gripper opening and closing
    several times along it, so each block has to be picked up where the path
    reaches it and left where the path leaves it.

    Every grasp and release row gets its own dwell: a position servo lags a
    moving reference, so acting on the gripper while the arm is still travelling
    grasps several centimetres short.
    """
    if not world.has_actuators:
        raise SystemExit("a dynamic rollout needs position actuators: "
                         "use robot='menagerie'")

    q_rows = np.asarray(q_rows, float)
    T = q_rows.shape[0]
    dt = world.model.opt.timestep
    durs = (row_durations(q_rows, safety) if row_time is None
            else np.full(T - 1, float(row_time)))
    row_steps = np.maximum((durs / dt).round().astype(int), 1)
    dwell_steps = max(int(round(dwell / dt)), 0)
    grasp_lag_steps = int(round(GRASP_LAG / dt))
    closed_ctrl = grip_ctrl(grasp_half_width)
    # NOTE: the fingers travel FULLY OPEN and only close during the grasp
    # dwell.  Fully open the pads sit 40 mm off the hand axis, inside the 60 mm
    # footprint of a tetromino's body cubes, so descending onto a block shoves
    # it 6-8 mm sideways before the grip closes.  Pre-closing the fingers for
    # the approach is the obvious fix and is measured to be WORSE (8-block
    # placement error, 6 scenes: 1.3/2.6/31.0/31.5/24.0/0.5 cm fully open vs
    # 6.3/16.4/33.7/23.3/18.9/25.1 at a 30 mm approach half-width and worse
    # still below that) -- narrowed pads stop clearing the block at all and
    # strike it head-on.  Left fully open deliberately.
    open_ctrl = GRIP_OPEN

    stop_rows = sorted({r for g, rl, _ in events for r in (g, rl)})
    u_of_step, stop_step = [], {}
    for t in range(T - 1):
        if t in stop_rows:
            stop_step[t] = len(u_of_step)
            u_of_step.extend([float(t)] * dwell_steps)
        u_of_step.extend(t + np.arange(row_steps[t]) / row_steps[t])
    if (T - 1) in stop_rows:
        stop_step[T - 1] = len(u_of_step)
        u_of_step.extend([float(T - 1)] * dwell_steps)
    settle_steps = max(int(round(settle / dt)), 0)
    u_of_step.extend([float(T - 1)] * (settle_steps + 1))
    u_of_step = np.asarray(u_of_step)
    total = len(u_of_step) - 1
    record_every = max(int(round(1.0 / (fps * dt))), 1)

    # Per-event step windows, measured from the START of each dwell -- the
    # dwell exists so the gripper can act while the arm is PARKED, so the
    # gripper command has to move inside it, not after it.
    #
    # These windows used to start at `stop_step[g] + dwell_steps`, the end of
    # the grasp dwell, with the close then delayed a further `GRASP_LAG`.  The
    # fingers therefore spent the entire 0.6 s stop commanded OPEN and only
    # began closing 0.05 s into the LIFT, so they swept up past the object
    # while still travelling -- a near-miss on every scene whose cube sat even
    # slightly off the finger centreline.  Starting at `stop_step[g]` gives the
    # close `dwell - GRASP_LAG` of stationary time to complete.
    #
    # The release end moves for the same reason: at `stop_step[rl] +
    # dwell_steps` the fingers opened only as the arm was already leaving the
    # target, dragging the object with them.  Opening `GRASP_LAG` into the
    # release dwell drops it while parked over the bucket.
    windows = [(stop_step[g], stop_step[rl] + grasp_lag_steps, name)
               for g, rl, name in events]

    if progress:
        print(f"  [rollout] {T} rows, {len(events)} pick-place events -> "
              f"{total * dt:.1f}s ({dwell:.1f}s dwell at each of "
              f"{len(stop_rows)} stop rows, {settle:.1f}s settle)")

    # Reset clears any leftover state, but it also restores every free body to
    # the pose baked into the model -- which would silently undo a scene the
    # caller set up, e.g. tetris slots already packed into the goal before this
    # demonstration starts.  Snapshot those poses and put them back.
    free_snapshot = {nm: world.data.qpos[a:a + 7].copy()
                     for nm, a in world.free_adr.items()}
    world.reset()
    for nm, a in world.free_adr.items():
        world.data.qpos[a:a + 7] = free_snapshot[nm]
    world.set_arm(0, q_rows[0])
    world.set_ctrl(0, q_rows[0], open_ctrl)
    world.forward()

    qpos_rec, cmd_rec, act_rec, row_rec = [], [], [], []
    closed_at = [None] * len(windows)
    for step in range(total + 1):
        u = u_of_step[step]
        base = int(np.floor(u))
        hi = min(base + 1, T - 1)
        cmd = q_rows[base] * (1.0 - (u - base)) + q_rows[hi] * (u - base)

        # Latch each window closed once the arm has converged onto the grasp
        # command (or the dwell has run out); a window stays closed to its
        # release even though tracking degrades again during the lift.
        for w, (a, b, _) in enumerate(windows):
            if closed_at[w] is None and a + grasp_lag_steps <= step <= b:
                settled = np.max(np.abs(world.arm_q(0) - cmd)) <= GRASP_SETTLE_TOL
                if settled or step >= a + dwell_steps:
                    closed_at[w] = step
        world.set_ctrl(0, cmd,
                       closed_ctrl if any(closed_at[w] is not None
                                          and closed_at[w] <= step <= windows[w][1]
                                          for w in range(len(windows))) else open_ctrl)

        world.step(gravcomp=gravcomp)
        if step % record_every == 0 or step == total:
            qpos_rec.append(world.data.qpos.copy())
            cmd_rec.append(cmd.copy())
            act_rec.append(world.arm_q(0))
            row_rec.append(u)
        if progress and total and step % max(total // 10, 1) == 0:
            print(f"  [rollout] {100 * step // max(total, 1):3d}%", flush=True)

    return Rollout(qpos=np.stack(qpos_rec),
                   q_cmd=np.stack(cmd_rec)[None], q_act=np.stack(act_rec)[None],
                   frame_row=np.asarray(row_rec), dt=dt * record_every,
                   grasp_frame=(windows[0][0] + grasp_lag_steps) // record_every,
                   release_frame=(windows[-1][1]) // record_every)


def run(world, q_rows, grasp_row, release_row, carried_names,
        row_time=DEFAULT_ROW_TIME, settle=DEFAULT_SETTLE, fps=60.0,
        weld_names=None, progress=True, safety=VEL_SAFETY,
        dwell=DEFAULT_DWELL, grasp_half_width=DEFAULT_GRASP_HALF_WIDTH,
        hand_bodies=None):
    """Integrate the plan on every scene at once.  -> `Rollout`.

    `q_rows` is (n_scenes, T, 7) planned angles and `carried_names` the
    per-scene free-body names; `hand_bodies` defaults to ``s<i>_hand``.
    `row_time` of None (the default) times each row against the joint velocity
    limits via `row_durations`; pass a float to force a flat time per row
    instead.  `weld_names` is accepted and ignored -- the grasp is kinematic.
    """
    if not world.has_actuators:
        raise SystemExit(
            "a dynamic rollout needs position actuators: use --robot menagerie "
            "(the spherized URDF import has none)")

    q_rows = np.asarray(q_rows, float)
    n_scenes, T, _ = q_rows.shape
    dt = world.model.opt.timestep

    # One shared timeline across scenes, so the side-by-side comparison stays
    # frame-aligned: each row takes as long as the slowest scene needs.
    if row_time is None:
        durs = np.stack([row_durations(q_rows[i], safety) for i in range(n_scenes)]
                        ).max(axis=0)
    else:
        durs = np.full(T - 1, float(row_time))
    row_steps = np.maximum((durs / dt).round().astype(int), 1)     # (T-1,)

    # DWELL at the grasp and release rows.  A position servo lags a moving
    # reference, so sweeping the commanded path straight through the grasp row
    # closes the gripper while the arm is still ~9 cm short of it -- and the
    # object then gets welded at that offset and carries the error all the way
    # to the release.  The plan's pinned rows are waypoints to ARRIVE at, so the
    # executor holds each one still long enough for the arm to converge before
    # the gripper acts, exactly as a real one would.
    dwell_steps = max(int(round(dwell / dt)), 0)

    # Step timeline as an explicit fractional-row reference, so dwells are just
    # repeated entries and every index below reads off the same array.
    u_of_step = []
    grasp_step = release_step = None
    for t in range(T - 1):
        if t == grasp_row:
            grasp_step = len(u_of_step)
            u_of_step.extend([float(t)] * dwell_steps)
        if t == release_row:
            release_step = len(u_of_step)
            u_of_step.extend([float(t)] * dwell_steps)
        u_of_step.extend(t + np.arange(row_steps[t]) / row_steps[t])
    if grasp_row >= T - 1:
        grasp_step = len(u_of_step)
        u_of_step.extend([float(T - 1)] * dwell_steps)
    if release_row >= T - 1:
        release_step = len(u_of_step)
        u_of_step.extend([float(T - 1)] * dwell_steps)
    settle_steps = max(int(round(settle / dt)), 0)
    u_of_step.extend([float(T - 1)] * (settle_steps + 1))
    u_of_step = np.asarray(u_of_step)
    total = len(u_of_step) - 1
    record_every = max(int(round(1.0 / (fps * dt))), 1)
    grasp_lag_steps = int(round(GRASP_LAG / dt))
    closed_ctrl = grip_ctrl(grasp_half_width)
    hands = hand_bodies or [f"s{i}_hand" for i in range(n_scenes)]
    carried = list(carried_names or [])

    if progress:
        print(f"  [rollout] {T} rows -> {total * dt:.1f}s total "
              f"({dwell:.1f}s dwell at rows {grasp_row}/{release_row}, "
              f"{settle:.1f}s settle, slowest row {durs.max():.2f}s)")

    world.reset()
    # Start ON the plan's first row rather than at MuJoCo's zero pose: the
    # servos cannot teleport, and a rollout that spends its first half-second
    # flinging the arm from the model's default pose to row 0 would be measuring
    # that transient, not the plan.
    for i in range(n_scenes):
        world.set_arm(i, q_rows[i, 0])
        world.set_ctrl(i, q_rows[i, 0], GRIP_OPEN)
    world.forward()

    qpos_rec, cmd_rec, act_rec, row_rec = [], [], [], []
    carrying = False
    grasp_rel, carry_prev, carry_now = None, None, None

    for step in range(total + 1):
        # ---- commanded reference -----------------------------------------
        u = u_of_step[step]
        base = int(np.floor(u))
        hi = min(base + 1, T - 1)
        frac = u - base
        cmd = q_rows[:, base] * (1.0 - frac) + q_rows[:, hi] * frac

        # The dwell exists so the arm can ARRIVE before the gripper acts, so
        # the gripper acts at the END of the grasp dwell, not its start.  The
        # fingers follow the WELD by `GRASP_LAG` rather than leading it: closing
        # first would let them nudge the object out of the pose the weld is
        # about to capture.
        grip = (closed_ctrl
                if (grasp_step + dwell_steps + grasp_lag_steps) <= step
                <= (release_step + dwell_steps) else GRIP_OPEN)
        for i in range(n_scenes):
            world.set_ctrl(i, cmd[i], grip)

        # ---- grasp / release ---------------------------------------------
        want_carry = ((grasp_step + dwell_steps) <= step
                      <= (release_step + dwell_steps))
        if want_carry and not carrying:
            # Capture at the END of the grasp dwell, when the arm has actually
            # converged on the grasp row -- capturing mid-approach would bake
            # the servo's lag into the carried offset.
            grasp_rel = [world.attach_pose(h, c) for h, c in zip(hands, carried)]
            carrying = True
        elif carrying and not want_carry:
            # Hand the object to free dynamics with the velocity it had, so a
            # release from a moving gripper throws it rather than dropping it.
            for c, prev, now in zip(carried, carry_prev, carry_now):
                world.set_free_vel(c, lin=(now - prev) / dt)
            carrying = False
        if carrying:
            carry_prev = list(carry_now) if carry_now else None
            carry_now = [world.carry_free(h, c, rp, rq)
                         for h, c, (rp, rq) in zip(hands, carried, grasp_rel)]
            if carry_prev is None:
                carry_prev = list(carry_now)
            for c in carried:
                world.set_free_vel(c)

        world.step()

        if step % record_every == 0 or step == total:
            qpos_rec.append(world.data.qpos.copy())
            cmd_rec.append(cmd.copy())
            act_rec.append(np.stack([world.arm_q(i) for i in range(n_scenes)]))
            row_rec.append(u)
        if progress and total and step % max(total // 10, 1) == 0:
            print(f"  [rollout] {100 * step // max(total, 1):3d}%", flush=True)

    return Rollout(
        qpos=np.stack(qpos_rec),
        q_cmd=np.stack(cmd_rec, axis=1),
        q_act=np.stack(act_rec, axis=1),
        frame_row=np.asarray(row_rec),
        dt=dt * record_every,
        grasp_frame=(grasp_step + dwell_steps + grasp_lag_steps) // record_every,
        release_frame=(release_step + dwell_steps) // record_every,
    )


def replayer(world, rollout):
    """-> `set_row(t)` writing recorded frame `t` back into the world."""
    def set_row(t):
        world.data.qpos[:] = rollout.qpos[int(t)]
        world.data.qvel[:] = 0.0
    return set_row
