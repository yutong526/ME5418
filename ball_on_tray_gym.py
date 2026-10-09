"""Ball-on-Tray reinforcement learning environment (ME5418 project, stage 1).

A solid ball rolls on a square tray that can tilt about two axes (pitch and
roll). The agent commands tilt increments and must keep the ball at the tray
centre while the (unobserved) mobile base occasionally accelerates.

The manipulator is not simulated: by default the tray is an ideal actuator,
i.e. the commanded tilt is reached instantly, and the sensors are ideal. All
quantities are expressed in the tray frame, whose origin is the tray centre.
The goal is the origin.

Optional realism parameters (observation noise, first-order actuator lag and
action delay, all switched off by default) relax these idealisations; see the
``BallOnTrayEnv`` docstring.

Sign convention:
    positive pitch -> the ball accelerates towards +x
    positive roll  -> the ball accelerates towards +y

This file contains the environment ``BallOnTrayEnv`` and, below it, the 3D
renderer ``TrayRenderer3D`` that draws it.

Written for gym==0.26.2 (``import gym``, not gymnasium) and Python 3.7.
The only other dependency is numpy; matplotlib is imported lazily and only
when the environment is rendered.
"""

import time
from collections import deque

import gym
import numpy as np
from gym import spaces


class BallOnTrayEnv(gym.Env):
    """Ball balancing on a two-axis tilting tray.

    Action space:
        ``Box(-1, 1, shape=(2,))`` = (d_pitch, d_roll). The action is clipped
        to [-1, 1], multiplied by ``TILT_STEP_DEG`` and added to the current
        tilt, which is then clipped to +/- ``MAX_TILT_DEG``.

    Observation space:
        ``Box(-inf, inf, shape=(18,))``. One frame is the normalised vector
        ``[x, y, vx, vy, pitch, roll]`` (position / tray half size,
        velocity / ``VEL_SCALE``, tilt / max tilt). The last ``N_STACK`` frames
        are concatenated, ordered from oldest to newest. Values are normalised
        but NOT clipped, so they may leave [-1, 1] (e.g. speed above 1 m/s, or
        the position on the step where the ball leaves the tray).
        The rolling resistance coefficient and the base acceleration are not
        part of the observation.

    Reward (per step):
        r = (1 - d) + GOAL_BONUS * [dist < GOAL_RADIUS]
            - ACTION_RATE_WEIGHT * ||a_t - a_{t-1}||^2
            - ACTION_WEIGHT * ||a_t||^2
        with d = ||(x, y)|| / d_max in [0, 1], d_max the tray half diagonal and
        a_t the clipped normalised action (a_{t-1} = 0 after reset).
        If the ball leaves the tray, the reward of that step is exactly
        ``OUT_OF_BOUNDS_REWARD``.

    Episode end:
        terminated = True  when |x| or |y| exceeds the tray half size.
        truncated  = True  when ``max_steps`` steps were taken without the
                           ball leaving the tray.

    Public attributes (physical units, not normalised) for visualisation:
        ball_pos (2,) [m], ball_vel (2,) [m/s], tilt (2,) [rad] = (pitch, roll),
        the actual tilt of the tray, tilt_cmd (2,) [rad], the commanded tilt
        (equal to ``tilt`` unless ``actuator_tau`` > 0),
        a_base (2,) [m/s^2], c_rr [-], tray_half_size [m], step_count [-],
        disturbances (list of pulses of the current episode),
        last_reward and episode_return (bookkeeping for the on-screen text).

    Args:
        friction_range: (low, high) of the rolling resistance coefficient
            c_rr [-], sampled uniformly at every reset and constant during the
            episode. Default is a fixed value; the full curriculum range is
            (0.005, 0.05).
        disturbance_range: (low, high) magnitude of the base acceleration
            pulses [m/s^2], or None for no disturbance at all. The full
            curriculum range is (0.5, 2.0).
        max_steps: number of control steps before the episode is truncated.
        render_mode: None (no rendering), "human" (interactive 3D matplotlib
            window, updated automatically by ``reset`` and ``step``) or
            "rgb_array" (``render`` returns an image). See ``render``.

    Optional realism parameters. All default to "off", in which case the
    environment behaves exactly as without them (same trajectories,
    observations and rewards for the same seed and actions):

        pos_noise_std: standard deviation [m] of zero-mean Gaussian noise
            added to the measured ball position.
        vel_noise_std: standard deviation [m/s] of zero-mean Gaussian noise
            added to the measured ball velocity.
            Physical meaning: measurement error of the camera or pressure
            sensor that tracks the ball. Relaxes the "ideal sensor"
            assumption of the proposal.
            The noise is added to the measurement in physical units, which is
            then normalised and pushed onto the frame stack. Every new frame
            is disturbed once; frames already in the stack keep their noise.
            The initial frame of ``reset`` is noisy too. Only the observation
            is affected: the physical state, the reward, the out-of-bounds
            check and the state in ``info`` always use the true values.
            The noise comes from its own random generator, derived from the
            ``reset`` seed, so changing the noise settings never changes the
            initial position, c_rr or the disturbance plan of a seed.
        actuator_tau: time constant [s] of a first-order lag between the
            commanded tilt and the actual tilt; 0 means the commanded tilt is
            reached instantly.
            Physical meaning: the motors and the arm cannot reach a commanded
            angle instantly. Relaxes the "ideal actuator" assumption of the
            proposal.
            Actions accumulate into the commanded tilt ``tilt_cmd`` (same
            increment rule, clipped to +/- ``MAX_TILT_DEG``). In every
            integration sub-step the actual tilt follows it:
                tilt += alpha * (tilt_cmd - tilt),
                alpha = 1 - exp(-sub_dt / actuator_tau)
            The physics and the observation use the actual tilt ``tilt``;
            ``info`` contains both.
        action_delay_steps: number of control steps (non-negative integer) by
            which an action is delayed: the action given at step t is added
            to the commanded tilt at step t + k. Until then zero actions are
            applied (the commanded tilt does not change).
            Physical meaning: latency of sensing, computation and
            communication. Relaxes the assumption that an action takes effect
            within the same control step.
            The action penalties of the reward use the action given by the
            agent, not the delayed one.
        Processing order in ``step``: delay queue -> commanded tilt ->
        first-order lag -> actual tilt -> ball dynamics.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ----------------------------------------------------------------- physics
    GRAVITY = 9.81                  # g, gravitational acceleration [m/s^2]
    ROLLING_FACTOR = 5.0 / 7.0      # 1 / (1 + I/(m r^2)) for a solid sphere [-]
    DT = 0.02                       # control period [s] (50 Hz)
    N_SUBSTEPS = 4                  # integration sub-steps per control step [-]
    TRAY_HALF_SIZE = 0.15           # half side length of the 0.3 m tray [m]
    REST_SPEED_EPS = 1e-6           # below this speed the ball is "at rest" [m/s]

    # ----------------------------------------------------------------- control
    MAX_TILT_DEG = 15.0             # tilt limit per axis [deg]
    TILT_STEP_DEG = 3.0             # tilt increment for |action| = 1 [deg/step]

    # ------------------------------------------------------------- observation
    VEL_SCALE = 1.0                 # velocity normalisation [m/s]
    N_STACK = 3                     # number of stacked frames [-]
    FRAME_DIM = 6                   # [x, y, vx, vy, pitch, roll]

    # ------------------------------------------------------------------ reward
    GOAL_RADIUS = 0.02              # radius of the bonus zone around the goal [m]
    GOAL_BONUS = 0.5                # bonus while inside the goal radius [-]
    ACTION_RATE_WEIGHT = 0.05       # beta, weight of ||a_t - a_{t-1}||^2 [-]
    ACTION_WEIGHT = 0.01            # c, weight of ||a_t||^2 [-]
    OUT_OF_BOUNDS_REWARD = -10.0    # reward of the step where the ball falls [-]

    # ------------------------------------------------------------------- reset
    RESET_POS_FRACTION = 0.8        # initial |x|, |y| <= fraction * half size [-]

    # ------------------------------------------------------------- disturbance
    MIN_PULSES = 1                  # minimum number of pulses per episode [-]
    MAX_PULSES = 3                  # maximum number of pulses per episode [-]
    PULSE_DURATION_RANGE = (0.2, 1.0)   # pulse duration [s]
    MAX_PULSE_RESAMPLE = 20         # placement attempts before a pulse is dropped

    # ----------------------------------------------------------------- realism
    # Mixed with the reset seed to seed the observation-noise generator, so
    # that its stream is independent of self.np_random.
    NOISE_SEED_STREAM = 5418

    # --------------------------------------------------------------- rendering
    BALL_RADIUS = 0.01              # drawn ball radius [m] (not used by the physics)

    def __init__(self, friction_range=(0.02, 0.02), disturbance_range=None,
                 max_steps=500, render_mode=None, pos_noise_std=0.0,
                 vel_noise_std=0.0, actuator_tau=0.0, action_delay_steps=0):
        super().__init__()

        if len(friction_range) != 2 or not 0.0 <= friction_range[0] <= friction_range[1]:
            raise ValueError("friction_range must be (low, high) with 0 <= low <= high")
        if disturbance_range is not None and (
                len(disturbance_range) != 2
                or not 0.0 <= disturbance_range[0] <= disturbance_range[1]):
            raise ValueError("disturbance_range must be None or (low, high) with 0 <= low <= high")
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be a positive integer")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError("unsupported render_mode: {}".format(render_mode))
        if not (pos_noise_std >= 0.0 and vel_noise_std >= 0.0):
            raise ValueError("pos_noise_std and vel_noise_std must be >= 0")
        if not actuator_tau >= 0.0:
            raise ValueError("actuator_tau must be >= 0")
        if (isinstance(action_delay_steps, bool)
                or not isinstance(action_delay_steps, (int, np.integer))
                or action_delay_steps < 0):
            raise ValueError("action_delay_steps must be a non-negative integer")

        self.friction_range = (float(friction_range[0]), float(friction_range[1]))
        self.disturbance_range = (
            None if disturbance_range is None
            else (float(disturbance_range[0]), float(disturbance_range[1])))
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.pos_noise_std = float(pos_noise_std)               # [m]
        self.vel_noise_std = float(vel_noise_std)               # [m/s]
        self.actuator_tau = float(actuator_tau)                 # [s]
        self.action_delay_steps = int(action_delay_steps)       # [control steps]

        # Derived constants.
        self.tray_half_size = self.TRAY_HALF_SIZE                 # [m]
        self.max_tilt = np.deg2rad(self.MAX_TILT_DEG)             # [rad]
        self.tilt_step = np.deg2rad(self.TILT_STEP_DEG)           # [rad/step]
        self.sub_dt = self.DT / self.N_SUBSTEPS                   # [s]
        self.max_dist = np.sqrt(2.0) * self.tray_half_size        # half diagonal [m]
        # Fraction of the tilt error removed per sub-step by the actuator lag:
        # alpha = 1 - exp(-sub_dt / tau) [-] (exact for a constant command).
        self._lag_alpha = (1.0 - np.exp(-self.sub_dt / self.actuator_tau)
                           if self.actuator_tau > 0.0 else 1.0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.N_STACK * self.FRAME_DIM,), dtype=np.float32)

        # Public physical state (physical units, not normalised).
        self.ball_pos = np.zeros(2, dtype=np.float64)   # (x, y) [m]
        self.ball_vel = np.zeros(2, dtype=np.float64)   # (vx, vy) [m/s]
        self.tilt = np.zeros(2, dtype=np.float64)       # actual (pitch, roll) [rad]
        self.tilt_cmd = np.zeros(2, dtype=np.float64)   # commanded (pitch, roll) [rad]
        self.a_base = np.zeros(2, dtype=np.float64)     # base acceleration [m/s^2]
        self.c_rr = self.friction_range[0]              # rolling resistance coeff. [-]
        self.step_count = 0                             # control steps since reset [-]
        # Disturbance schedule of the episode: list of dicts with keys
        # "start" [s], "duration" [s] and "accel" (2,) [m/s^2].
        self.disturbances = []
        # Bookkeeping shown by the renderer (not used by the dynamics).
        self.last_reward = 0.0                          # reward of the last step [-]
        self.episode_return = 0.0                       # sum of rewards since reset [-]

        self._prev_action = np.zeros(2, dtype=np.float64)
        self._frames = deque(maxlen=self.N_STACK)
        # Actions waiting to take effect (only used if action_delay_steps > 0).
        self._action_queue = deque()
        # Generator of the observation noise (only created if noise is enabled).
        self._noise_rng = None

        # 3D renderer (TrayRenderer3D), created lazily on the first render() call.
        self._renderer = None

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        """Start a new episode.

        The ball is placed uniformly inside the central 80 % of the tray with
        zero velocity, the tray is level, and the rolling resistance and the
        disturbance schedule of the episode are sampled.

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)  # seeds self.np_random

        self.c_rr = self._sample_friction()
        self.disturbances = self._sample_disturbances()
        self.ball_pos = self._sample_initial_position()
        self.ball_vel = np.zeros(2, dtype=np.float64)
        self.tilt = np.zeros(2, dtype=np.float64)
        self.tilt_cmd = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.a_base = self._disturbance_at(0.0)
        self._prev_action = np.zeros(2, dtype=np.float64)
        self.last_reward = 0.0
        self.episode_return = 0.0
        self._reset_action_queue()
        self._reset_noise_rng(seed)

        # Fill the whole frame stack with the initial frame.
        frame = self._get_frame()
        self._frames.clear()
        for _ in range(self.N_STACK):
            self._frames.append(frame)

        if self.render_mode == "human":
            self.render()

        info = self._get_info(self._zero_reward_terms(), out_of_bounds=False)
        return self._get_obs(), info

    def step(self, action):
        """Advance the environment by one control period (DT).

        Args:
            action: (d_pitch, d_roll), clipped to [-1, 1].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)

        # Delay queue -> commanded tilt (-> lag -> actual tilt in _integrate).
        self._update_tilt_command(self._delay_action(action))

        # The base acceleration is held constant during the control step and
        # evaluated at the start time of the step, t = step_count * DT [s].
        self.a_base = self._disturbance_at(self.step_count * self.DT)

        out_of_bounds = self._integrate()
        self.step_count += 1

        reward, reward_terms = self._compute_reward(action, out_of_bounds)
        self._prev_action = action
        self._frames.append(self._get_frame())
        self.last_reward = reward
        self.episode_return += reward

        terminated = bool(out_of_bounds)
        # A fall on the very last step is reported as termination only.
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        if self.render_mode == "human":
            self.render()

        info = self._get_info(reward_terms, out_of_bounds)
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """Draw the current state in 3D (a tilting tray with the ball on it).

        The mode is chosen in the constructor (gym 0.26 API):

            "human"      update an interactive matplotlib window in real time
                         and return None. ``reset`` and ``step`` already call
                         this, so user code normally does not have to.
            "rgb_array"  return the frame as an (H, W, 3) uint8 array.
            None         do nothing.

        The picture is drawn by ``TrayRenderer3D`` (defined below in this
        file), which is created on the first call and reused afterwards; see
        that class for the elements of the picture. Rendering only reads the public state
        (``ball_pos``, ``ball_vel``, ``tilt``, ``a_base``, ``c_rr``,
        ``tray_half_size``, ``step_count``, ``last_reward``,
        ``episode_return``) and never changes the physical state.
        """
        if self.render_mode is None or self.window_closed:
            return None
        if self._renderer is None:
            self._renderer = TrayRenderer3D(self, mode=self.render_mode)
        return self._renderer.render(self.last_reward, self.episode_return)

    @property
    def window_closed(self):
        """True once the user has closed the "human" window; render() is then a no-op."""
        return self._renderer is not None and self._renderer.window_closed

    def close(self):
        """Close the render window (if any) and free the figure."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------- reset sub-routines
    def _sample_friction(self):
        """Sample the rolling resistance coefficient c_rr [-] of the episode."""
        low, high = self.friction_range
        return float(self.np_random.uniform(low, high))

    def _sample_initial_position(self):
        """Sample the initial ball position [m], uniform in the central 80 % square."""
        limit = self.RESET_POS_FRACTION * self.tray_half_size
        return self.np_random.uniform(-limit, limit, size=2).astype(np.float64)

    def _sample_disturbances(self):
        """Sample the base-acceleration pulse schedule of the episode.

        1-3 pulses are drawn. Each has a magnitude uniform in
        ``disturbance_range`` [m/s^2], a direction uniform in [0, 2*pi) [rad],
        a duration uniform in ``PULSE_DURATION_RANGE`` [s] and a start time
        uniform in [0, T - duration] with T = max_steps * DT [s], so that every
        pulse lies completely inside the episode.

        Pulses never overlap: a pulse that overlaps an already placed one (or
        that does not fit in the episode) gets a new start time and duration,
        at most ``MAX_PULSE_RESAMPLE`` times. After that the pulse is dropped,
        so an episode may contain fewer pulses than drawn (possibly none when
        max_steps is very small).

        Returns:
            list of dicts {"start": [s], "duration": [s], "accel": (2,) [m/s^2]},
            sorted by start time. Empty when ``disturbance_range`` is None.
        """
        if self.disturbance_range is None:
            return []

        episode_time = self.max_steps * self.DT                     # T [s]
        n_pulses = int(self.np_random.integers(self.MIN_PULSES, self.MAX_PULSES + 1))

        pulses = []
        for _ in range(n_pulses):
            magnitude = self.np_random.uniform(*self.disturbance_range)   # [m/s^2]
            direction = self.np_random.uniform(0.0, 2.0 * np.pi)          # [rad]
            accel = magnitude * np.array([np.cos(direction), np.sin(direction)])

            for _ in range(self.MAX_PULSE_RESAMPLE):
                duration = float(self.np_random.uniform(*self.PULSE_DURATION_RANGE))  # [s]
                if duration > episode_time:
                    continue
                start = float(self.np_random.uniform(0.0, episode_time - duration))   # [s]
                end = start + duration
                overlaps = any(
                    start < p["start"] + p["duration"] and p["start"] < end
                    for p in pulses)
                if not overlaps:
                    pulses.append({"start": start, "duration": duration, "accel": accel})
                    break

        pulses.sort(key=lambda p: p["start"])
        return pulses

    def _reset_action_queue(self):
        """Fill the delay queue with zero actions (the tilt command stays put)."""
        self._action_queue.clear()
        for _ in range(self.action_delay_steps):
            self._action_queue.append(np.zeros(2, dtype=np.float64))

    def _reset_noise_rng(self, seed):
        """(Re)seed the observation-noise generator at reset.

        The generator is separate from ``self.np_random`` and is derived from
        the reset seed, so the noise is reproducible but never shifts the
        random stream that draws c_rr, the disturbances and the start position.
        A reset without a seed continues the existing noise stream. Nothing is
        done while the noise is switched off.
        """
        if self.pos_noise_std <= 0.0 and self.vel_noise_std <= 0.0:
            return
        if seed is not None:
            self._noise_rng = np.random.default_rng([int(seed), self.NOISE_SEED_STREAM])
        elif self._noise_rng is None:
            self._noise_rng = np.random.default_rng()

    # ----------------------------------------------------------------- actuator
    def _delay_action(self, action):
        """Return the action that takes effect in this step.

        With ``action_delay_steps`` = k > 0 the given action is queued and the
        action given k steps ago is returned (zero during the first k steps).
        Without delay the given action is returned unchanged.
        """
        if self.action_delay_steps == 0:
            return action
        self._action_queue.append(action)
        return self._action_queue.popleft()

    def _update_tilt_command(self, action):
        """Add the (possibly delayed) action to the commanded tilt.

            tilt_cmd <- clip(tilt_cmd + action * tilt_step, +/- max_tilt)   [rad]

        Ideal actuator (``actuator_tau`` == 0): the actual tilt is the
        commanded tilt, tilt <- clip(tilt + action * tilt_step, +/- max_tilt).
        With a lag the actual tilt follows in ``_integrate``.
        """
        if self.actuator_tau > 0.0:
            self.tilt_cmd = np.clip(self.tilt_cmd + action * self.tilt_step,
                                    -self.max_tilt, self.max_tilt)
            return
        self.tilt = np.clip(self.tilt + action * self.tilt_step, -self.max_tilt, self.max_tilt)
        self.tilt_cmd = self.tilt.copy()

    # ----------------------------------------------------------------- dynamics
    def _disturbance_at(self, t):
        """Return the base acceleration a_base (2,) [m/s^2] at episode time t [s].

        a_base equals the pulse active at t (start <= t < start + duration) and
        zero outside all pulses.
        """
        for pulse in self.disturbances:
            if pulse["start"] <= t < pulse["start"] + pulse["duration"]:
                return pulse["accel"].copy()
        return np.zeros(2, dtype=np.float64)

    def _drive_acceleration(self):
        """Return the driving acceleration of the ball (2,) [m/s^2].

        Per axis, for a solid sphere rolling without slipping on a tray tilted
        by ``tilt`` whose base accelerates horizontally with ``a_base``:

            a_drive = (5/7) * (g * sin(tilt) - a_base)      [m/s^2]
        """
        return self.ROLLING_FACTOR * (self.GRAVITY * np.sin(self.tilt) - self.a_base)

    def _integrate(self):
        """Integrate the ball dynamics over one control step.

        The control period DT is split into ``N_SUBSTEPS`` semi-implicit Euler
        sub-steps (tilt and a_base are constant during the control step).
        Integration stops as soon as the ball leaves the tray.

        With an actuator lag (``actuator_tau`` > 0) the actual tilt moves
        towards the commanded tilt at the start of every sub-step, and the
        driving acceleration is re-evaluated with it:

            tilt <- tilt + alpha * (tilt_cmd - tilt),  alpha = 1 - exp(-h / tau)

        Returns:
            True if the ball left the tray (|x| or |y| > tray half size).
        """
        if self.actuator_tau > 0.0:
            return self._integrate_with_lag()

        a_drive = self._drive_acceleration()
        for _ in range(self.N_SUBSTEPS):
            self._substep(a_drive, self.sub_dt)
            if self._is_out_of_bounds():
                return True
        return False

    def _integrate_with_lag(self):
        """Same as ``_integrate``, with the first-order actuator lag.

        The lag is advanced in all sub-steps, also after the ball has left the
        tray, so the tilt always covers a whole control period.
        """
        out_of_bounds = False
        for _ in range(self.N_SUBSTEPS):
            self.tilt = self.tilt + self._lag_alpha * (self.tilt_cmd - self.tilt)   # [rad]
            if not out_of_bounds:
                self._substep(self._drive_acceleration(), self.sub_dt)
                out_of_bounds = self._is_out_of_bounds()
        return out_of_bounds

    def _substep(self, a_drive, h):
        """One semi-implicit Euler sub-step of length h [s].

        Rolling resistance decelerates the ball with constant magnitude

            a_rr = (5/7) * c_rr * g                         [m/s^2]

        opposite to the velocity (the same 5/7 rolling factor as the drive,
        so that the ball starts to move when g*sin(tilt) > c_rr*g).

        Dead zone: a ball at rest (speed < REST_SPEED_EPS) stays at rest while
        ||a_drive|| <= a_rr.

        Otherwise:
            v <- v + a_drive * h                            [m/s]
            v <- v * max(0, 1 - a_rr * h / ||v||)           resistance; it can
                 only reduce the speed to zero, never reverse the velocity
            p <- p + v * h                                  [m]  (uses the new v)
        """
        a_rr = self.ROLLING_FACTOR * self.c_rr * self.GRAVITY       # [m/s^2]

        if (np.linalg.norm(self.ball_vel) < self.REST_SPEED_EPS
                and np.linalg.norm(a_drive) <= a_rr):
            self.ball_vel = np.zeros(2, dtype=np.float64)
            return

        vel = self.ball_vel + a_drive * h
        speed = np.linalg.norm(vel)
        speed_loss = a_rr * h                                       # [m/s]
        if speed <= speed_loss:
            vel = np.zeros(2, dtype=np.float64)
        else:
            vel = vel * (1.0 - speed_loss / speed)

        self.ball_vel = vel
        self.ball_pos = self.ball_pos + vel * h

    def _is_out_of_bounds(self):
        """True if the ball centre is outside the tray (|x| or |y| > half size)."""
        return bool(np.any(np.abs(self.ball_pos) > self.tray_half_size))

    # ------------------------------------------------------------------- reward
    def _zero_reward_terms(self):
        """Reward breakdown with all terms set to zero (used at reset)."""
        return {"distance": 0.0, "goal_bonus": 0.0, "action_rate": 0.0,
                "action_magnitude": 0.0, "out_of_bounds": 0.0}

    def _compute_reward(self, action, out_of_bounds):
        """Compute the step reward and its breakdown.

            d = ||(x, y)|| / d_max,  d_max = sqrt(2) * tray_half_size
            r = (1 - d)
                + GOAL_BONUS * [||(x, y)|| < GOAL_RADIUS]
                - ACTION_RATE_WEIGHT * ||a_t - a_{t-1}||^2
                - ACTION_WEIGHT * ||a_t||^2

        If the ball left the tray, r = OUT_OF_BOUNDS_REWARD and all other terms
        are reported as zero, so the terms always sum to the reward.

        Args:
            action: clipped normalised action a_t.
            out_of_bounds: whether the ball left the tray in this step.

        Returns:
            (reward, terms) where terms is a dict of the individual terms.
        """
        terms = self._zero_reward_terms()
        if out_of_bounds:
            terms["out_of_bounds"] = self.OUT_OF_BOUNDS_REWARD
            return float(self.OUT_OF_BOUNDS_REWARD), terms

        dist = float(np.linalg.norm(self.ball_pos))                 # [m]
        terms["distance"] = 1.0 - dist / self.max_dist
        terms["goal_bonus"] = self.GOAL_BONUS if dist < self.GOAL_RADIUS else 0.0
        terms["action_rate"] = -self.ACTION_RATE_WEIGHT * float(
            np.sum((action - self._prev_action) ** 2))
        terms["action_magnitude"] = -self.ACTION_WEIGHT * float(np.sum(action ** 2))
        return float(sum(terms.values())), terms

    # -------------------------------------------------------------- observation
    def _get_frame(self):
        """Build one normalised frame [x, y, vx, vy, pitch, roll] (not clipped).

            position / tray_half_size, velocity / VEL_SCALE, tilt / max_tilt

        Position and velocity are the measured values, i.e. they include the
        observation noise if it is enabled. Each call draws new noise, so this
        is called exactly once per new frame.
        """
        pos, vel = self._measure()
        return np.concatenate([
            pos / self.tray_half_size,
            vel / self.VEL_SCALE,
            self.tilt / self.max_tilt,
        ]).astype(np.float32)

    def _measure(self):
        """Return the measured ball position [m] and velocity [m/s].

        Without noise these are the true values. With noise, independent
        zero-mean Gaussian errors with standard deviations ``pos_noise_std``
        and ``vel_noise_std`` are added; the true state is not modified. No
        random number is drawn for a quantity whose standard deviation is 0.
        """
        pos, vel = self.ball_pos, self.ball_vel
        if self.pos_noise_std > 0.0:
            pos = pos + self._noise_rng.normal(0.0, self.pos_noise_std, size=2)
        if self.vel_noise_std > 0.0:
            vel = vel + self._noise_rng.normal(0.0, self.vel_noise_std, size=2)
        return pos, vel

    def _get_obs(self):
        """Concatenate the stacked frames, oldest first, into an (18,) vector."""
        return np.concatenate(list(self._frames)).astype(np.float32)

    def _get_info(self, reward_terms, out_of_bounds):
        """Build the info dict (hidden parameters, raw state, reward breakdown)."""
        return {
            "c_rr": self.c_rr,                      # [-]
            "a_base": self.a_base.copy(),           # [m/s^2]
            "ball_pos": self.ball_pos.copy(),       # [m]
            "ball_vel": self.ball_vel.copy(),       # [m/s]
            "tilt": self.tilt.copy(),               # actual tilt [rad]
            "tilt_cmd": self.tilt_cmd.copy(),       # commanded tilt [rad]
            "step_count": self.step_count,
            "reward_terms": dict(reward_terms),
            "out_of_bounds": bool(out_of_bounds),
        }


# ==============================================================================
# 3D rendering
# ==============================================================================
class TrayRenderer3D(object):
    """Draws a BallOnTrayEnv as a tilting tray with a ball in 3D (mplot3d only).

    The renderer only reads the public attributes of the environment and never
    changes its state. ``BallOnTrayEnv.render`` uses it when the environment is
    created with a render mode; it can also be used directly on an environment
    created with ``render_mode=None``, as ``demo.py`` does.

    Read-only inputs taken from the environment: ``ball_pos``, ``ball_vel``,
    ``tilt``, ``a_base``, ``c_rr``, ``tray_half_size``, ``step_count`` (plus
    the constants ``DT``, ``GOAL_RADIUS``, ``max_steps``, ``BALL_RADIUS``).

    Args:
        env: the BallOnTrayEnv to draw (create it with ``render_mode=None``).
        mode: "human" opens a window that plays in real time;
            "rgb_array" draws off-screen and ``render`` returns an image.
        ball_radius: drawn ball radius [m]; defaults to ``env.BALL_RADIUS``.

    Public attributes:
        window_closed: True once the user has closed the "human" window.
        frames_drawn / frames_skipped: counters of ``render`` calls that were
            drawn / skipped to keep real-time pace ("human" mode).
        label: caption shown in the top-left corner of the picture, e.g. the
            name of the policy that is running; empty (nothing shown) by
            default. Change it with ``set_label``.

    Geometry (derived from the sign convention of the environment)
    --------------------------------------------------------------
    World frame W: origin at the pivot (centre of the tray surface), z up.
    Tray frame  T: x_t, y_t in the tray surface, n = surface normal.
    R is the rotation matrix whose columns are x_t, y_t, n expressed in W, so a
    tray-frame point p_T is drawn at p_W = R @ p_T.

    Gravity is g_W = (0, 0, -g). Its component along a tray axis is what
    accelerates the ball along that axis:

        a_x  ~  g_W . x_t = -g * (x_t)_z ,      a_y  ~  g_W . y_t = -g * (y_t)_z

    The environment uses a_x ~ g * sin(pitch) and a_y ~ g * sin(roll) (positive
    pitch -> ball accelerates towards +x, positive roll -> towards +y). Both hold
    exactly if and only if the third row of R is

        R[2, :] = (-sin(pitch), -sin(roll), c),   c = sqrt(1 - sin^2(pitch) - sin^2(roll))

    i.e. the +x side of the tray is lower by sin(pitch) per metre and the +y side
    by sin(roll) per metre. This fixes R up to a rotation about the vertical axis;
    the tray has no yaw, so the tilt without twist (rotation about a horizontal
    axis that takes e_z to n) is used. With s_x = sin(pitch), s_y = sin(roll):

            | 1 - s_x^2/(1+c)    -s_x*s_y/(1+c)     s_x |
        R = | -s_x*s_y/(1+c)     1 - s_y^2/(1+c)    s_y |
            | -s_x               -s_y               c   |

    The normal n = (s_x, s_y, c) leans towards the downhill direction. For a single
    axis this reduces to a rotation by +pitch about the world y axis, or by -roll
    about the world x axis. Angles are drawn to scale (no exaggeration).
    """

    MODES = ("human", "rgb_array")

    # ------------------------------------------------------------------ figure
    FIGSIZE = (9.0, 6.0)            # figure size [inch]
    DPI = 100                       # -> 900 x 600 pixel frames
    FONT_SIZE = 14                  # font size of the text panel [pt]
    VIEW_ELEV = 25.0                # fixed camera elevation [deg]
    VIEW_AZIM = -60.0               # fixed camera azimuth [deg]

    # ------------------------------------------------------------------- scene
    TRAY_THICKNESS = 0.006          # thickness of the drawn tray slab [m]
    PIVOT_HEIGHT = 0.12             # height of the pivot above the ground [m]
    VIEW_HALF_WIDTH = 0.19          # x and y axis limits are +/- this [m]
    VIEW_TOP = 0.10                 # upper z axis limit [m]
    VIEW_BOTTOM_MARGIN = 0.02       # visible space below the ground [m]
    LINE_LIFT = 0.0005              # lines are drawn this far above the surface [m]
    TRAIL_LENGTH = 75               # number of past positions in the trail (1.5 s)
    DISTURBANCE_ARROW_SCALE = 0.05  # arrow length per unit |a_base| [m / (m/s^2)]
    ARROW_HEAD_LENGTH = 0.015       # length of the arrow head [m]
    BALL_N_LAT = 8                  # ball mesh resolution (latitude bands)
    BALL_N_LON = 14                 # ball mesh resolution (longitude bands)

    # ------------------------------------------------------------------ timing
    MAX_SKIPPED_FRAMES = 9          # never skip more consecutive frames than this

    def __init__(self, env, mode="human", ball_radius=None):
        if mode not in self.MODES:
            raise ValueError("mode must be one of {}".format(self.MODES))
        self.env = env
        self.mode = mode
        self.ball_radius = float(
            getattr(env, "BALL_RADIUS", 0.01) if ball_radius is None else ball_radius)

        self.window_closed = False
        self.frames_drawn = 0
        self.frames_skipped = 0
        self.label = ""

        self._fig = None                # created lazily on the first render()
        self._artists = {}              # name -> artist updated every frame
        self._unit_sphere = None        # (F, 4, 3) facets of a unit sphere
        self._trail = deque(maxlen=self.TRAIL_LENGTH)   # tray-frame (x, y) points
        self._last_step = 0             # env.step_count at the last render() call
        self._wall_start = None         # wall-clock time of step 0 of the episode [s]
        self._skipped_in_a_row = 0
        self._last_values = (0.0, 0.0)  # (reward, episode_return) of the last call

    # --------------------------------------------------------------- geometry
    @staticmethod
    def tray_rotation(pitch, roll):
        """Rotation matrix R (tray frame -> world frame) for the given tilt [rad].

        See the class docstring for the derivation. The third row is
        (-sin(pitch), -sin(roll), c): the +x side of the tray is lower for
        pitch > 0 and the +y side is lower for roll > 0, which is exactly where
        the environment accelerates the ball.
        """
        s_x = np.sin(pitch)
        s_y = np.sin(roll)
        c = np.sqrt(1.0 - s_x * s_x - s_y * s_y)    # cosine of the total tilt angle
        k = 1.0 / (1.0 + c)
        return np.array([
            [1.0 - k * s_x * s_x, -k * s_x * s_y, s_x],
            [-k * s_x * s_y, 1.0 - k * s_y * s_y, s_y],
            [-s_x, -s_y, c],
        ])

    def rotation(self):
        """Rotation matrix of the tray for the current ``env.tilt``."""
        return self.tray_rotation(self.env.tilt[0], self.env.tilt[1])

    def tray_to_world(self, points):
        """Transform tray-frame points (..., 3) [m] to world coordinates [m]."""
        return np.asarray(points, dtype=np.float64).dot(self.rotation().T)

    def ball_center_world(self):
        """World position of the ball centre [m].

        The ball touches the tray surface at the tray-frame point (x, y, 0), so
        its centre is one radius away along the surface normal:
        p_T = (x, y, ball_radius).
        """
        x, y = self.env.ball_pos
        return self.tray_to_world([x, y, self.ball_radius])

    # ------------------------------------------------------------- public API
    def render(self, reward=0.0, episode_return=0.0, force=False):
        """Draw the current state of the environment.

        Call this once after ``env.reset`` and once after every ``env.step``.

        Args:
            reward: reward of the last step (only displayed, supplied by the caller).
            episode_return: return of the episode so far (only displayed).
            force: draw even if the frame would be skipped ("human" mode).

        Returns:
            "rgb_array" mode: the frame as an (H, W, 3) uint8 array.
            "human" mode: None. The window is paced to real time: the call
            sleeps when it is ahead of the simulation clock and skips drawing
            when it is more than one control period behind, so playback speed
            stays close to real time on slow machines.
        """
        if self.window_closed:
            return None
        if self._fig is None:
            self._init_figure()
        self._last_values = (float(reward), float(episode_return))
        self._track_episode()

        if self.mode == "rgb_array":
            self._update_artists()
            canvas = self._fig.canvas
            canvas.draw()
            self.frames_drawn += 1
            # buffer_rgba() is (H, W, 4) uint8; drop the alpha channel.
            return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

        period = self.env.DT                                        # [s]
        target = self._wall_start + self.env.step_count * period    # [s]
        behind = time.perf_counter() - target
        if behind > period and not force:
            if self._skipped_in_a_row < self.MAX_SKIPPED_FRAMES:
                self._skipped_in_a_row += 1
                self.frames_skipped += 1
                return None
            # Too slow even with skipping: accept the delay instead of freezing.
            self._wall_start += behind

        self._update_artists()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self.frames_drawn += 1
        self._skipped_in_a_row = 0

        if self.env.step_count == 0:
            # The first frame of an episode (window creation, first draw) is
            # slow; start the real-time clock only after it is on screen.
            self._wall_start = time.perf_counter()
            return None
        remaining = target - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        return None

    def set_label(self, text):
        """Set the caption shown in the top-left corner (from the next frame on).

        Used by the demo to say which policy is running. An empty string
        removes the caption.
        """
        self.label = str(text)

    def hold(self, seconds):
        """Show the current state and keep the window responsive for a while."""
        if self.mode != "human" or self._fig is None:
            return
        self.render(self._last_values[0], self._last_values[1], force=True)
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and not self.window_closed:
            self._fig.canvas.flush_events()
            time.sleep(0.02)
        # The pause must not count as lag of the simulation clock.
        if self._wall_start is not None:
            self._wall_start += seconds

    def close(self):
        """Close the window (if any) and free the figure."""
        if self._fig is not None:
            if self.mode == "human":
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            self._fig = None
        self._artists = {}
        self._trail.clear()
        self._wall_start = None
        self.window_closed = False

    # ---------------------------------------------------------------- set-up
    def _init_figure(self):
        """Create the figure and every artist exactly once."""
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if self.mode == "human":
            import matplotlib.pyplot as plt
            self._fig = plt.figure(figsize=self.FIGSIZE, dpi=self.DPI)
            self._fig.canvas.mpl_connect("close_event", self._on_window_close)
        else:
            # Off-screen figure: no window and no GUI backend required.
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            self._fig = Figure(figsize=self.FIGSIZE, dpi=self.DPI)
            FigureCanvasAgg(self._fig)

        half = self.env.tray_half_size
        width = self.VIEW_HALF_WIDTH
        ground = -self.PIVOT_HEIGHT
        z_low = ground - self.VIEW_BOTTOM_MARGIN

        ax = self._fig.add_axes([-0.035, -0.08, 0.72, 1.16], projection="3d")
        # Fixed camera, fixed limits and equal scale on all three axes, so the
        # picture never rescales or jitters between frames.
        ax.set_proj_type("ortho")
        ax.view_init(elev=self.VIEW_ELEV, azim=self.VIEW_AZIM)
        ax.set_xlim(-width, width)
        ax.set_ylim(-width, width)
        ax.set_zlim(z_low, self.VIEW_TOP)
        ax.set_box_aspect((2.0 * width, 2.0 * width, self.VIEW_TOP - z_low))
        ax.set_axis_off()
        # Draw order is given by the explicit zorder below, not by matplotlib's
        # depth heuristic. This is valid because the camera elevation (25 deg)
        # is larger than the largest total tilt (about 21 deg), so the camera
        # always looks at the top face of the tray and everything lying on the
        # tray is in front of it.
        ax.computed_zorder = False

        # ---- static artists: ground, base, pillar, level reference, axes ----
        ground_plate = np.array([[[-width, -width, ground], [width, -width, ground],
                                  [width, width, ground], [-width, width, ground]]])
        ax.add_collection3d(Poly3DCollection(
            ground_plate, facecolors="#eeeeee", edgecolors="#cccccc", zorder=1))
        base = 0.04
        base_plate = np.array([[[-base, -base, ground], [base, -base, ground],
                                [base, base, ground], [-base, base, ground]]])
        ax.add_collection3d(Poly3DCollection(
            base_plate, facecolors="#888888", edgecolors="#555555", zorder=2))
        # Pillar from the ground up to the pivot, about which the tray rotates.
        ax.plot([0.0, 0.0], [0.0, 0.0], [ground, -self.TRAY_THICKNESS],
                color="#555555", linewidth=5.0, solid_capstyle="butt", zorder=3)
        # Outline of the level (zero tilt) tray as a reference for the tilt.
        ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half],
                [0.0] * 5, color="#999999", linewidth=0.8, linestyle=":", zorder=3)
        # World axes on the ground, to show where +x and +y are.
        corner = np.array([-width + 0.02, -width + 0.02, ground])
        for direction, label in (((0.07, 0.0, 0.0), "x"), ((0.0, 0.07, 0.0), "y")):
            tip = corner + np.array(direction)
            ax.plot([corner[0], tip[0]], [corner[1], tip[1]], [corner[2], tip[2]],
                    color="#333333", linewidth=1.5, zorder=3)
            ax.text(tip[0], tip[1], tip[2], " " + label, fontsize=self.FONT_SIZE - 1, zorder=3)

        # ---- dynamic artists (their vertices are updated every frame) ----
        tray = Poly3DCollection(self._tray_faces(np.eye(3)), zorder=4,
                                facecolors=["#f3ecdc"] + ["#b9a98a"] * 5,
                                edgecolors="#333333", linewidths=1.0)
        ax.add_collection3d(tray)
        goal_circle, = ax.plot([], [], [], color="#2ca02c", linestyle="--",
                               linewidth=1.5, zorder=5)
        goal_point, = ax.plot([], [], [], color="#2ca02c", linestyle="none",
                              marker="+", markersize=9, zorder=5)
        trail, = ax.plot([], [], [], color="#ff7f0e", linewidth=1.5, alpha=0.7, zorder=6)
        self._unit_sphere, ball_colors = self._make_unit_sphere()
        ball = Poly3DCollection(self._unit_sphere * self.ball_radius, zorder=7,
                                facecolors=ball_colors, edgecolors="none")
        ax.add_collection3d(ball)
        arrow, = ax.plot([], [], [], color="#d62728", linewidth=2.0, zorder=8)
        text = self._fig.text(0.68, 0.93, "", va="top", ha="left",
                              family="monospace", fontsize=self.FONT_SIZE)
        label = self._fig.text(0.03, 0.955, "", va="top", ha="left",
                               fontsize=self.FONT_SIZE + 4, fontweight="bold")

        self._fig.legend(
            handles=[
                Line2D([], [], linestyle="none", marker="o", markersize=8,
                       markerfacecolor="#ff7f0e", markeredgecolor="none", label="ball"),
                Line2D([], [], color="#ff7f0e", linewidth=1.5, alpha=0.7, label="recent path"),
                Line2D([], [], color="#2ca02c", linestyle="--", label="goal radius"),
                Line2D([], [], color="#999999", linestyle=":", label="level reference"),
                Line2D([], [], color="#d62728", linewidth=2.0, marker=">",
                       label="disturbance push"),
            ],
            loc="lower left", bbox_to_anchor=(0.665, 0.06),
            fontsize=self.FONT_SIZE - 3, frameon=False)

        self._artists = {"tray": tray, "goal_circle": goal_circle,
                         "goal_point": goal_point, "trail": trail, "ball": ball,
                         "arrow": arrow, "text": text, "label": label}
        if self.mode == "human":
            plt.show(block=False)

    def _tray_faces(self, rot):
        """World vertices (6, 4, 3) of the tray slab; face 0 is the top surface.

        The top surface is the tray-frame plane z = 0, the slab extends to
        z = -TRAY_THICKNESS.
        """
        h = self.env.tray_half_size
        t = self.TRAY_THICKNESS
        corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        top = np.column_stack([corners, np.zeros(4)])
        bottom = np.column_stack([corners, -t * np.ones(4)])
        faces = [top, bottom[::-1]]
        for i in range(4):
            j = (i + 1) % 4
            faces.append(np.array([top[i], top[j], bottom[j], bottom[i]]))
        return np.array(faces).dot(rot.T)

    def _make_unit_sphere(self):
        """Facets (F, 4, 3) of a unit sphere and one shaded colour per facet.

        The ball only translates, so the facet colours (simple diffuse shading
        from a fixed light) are computed once.
        """
        lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, self.BALL_N_LAT + 1)
        lon = np.linspace(0.0, 2.0 * np.pi, self.BALL_N_LON + 1)

        def point(la, lo):
            return [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]

        facets = []
        for i in range(self.BALL_N_LAT):
            for j in range(self.BALL_N_LON):
                facets.append([point(lat[i], lon[j]), point(lat[i], lon[j + 1]),
                               point(lat[i + 1], lon[j + 1]), point(lat[i + 1], lon[j])])
        facets = np.array(facets)

        normals = facets.mean(axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        light = np.array([-0.3, -0.5, 0.8])
        light /= np.linalg.norm(light)
        shade = 0.55 + 0.45 * np.clip(normals.dot(light), 0.0, 1.0)
        colors = np.outer(shade, np.array([1.0, 0.5, 0.05]))        # orange
        return facets, np.clip(colors, 0.0, 1.0)

    # ---------------------------------------------------------------- updates
    def _track_episode(self):
        """Restart trail and clock on a new episode; record the ball position."""
        step = self.env.step_count
        if self._wall_start is None or step == 0 or step < self._last_step:
            self._trail.clear()
            self._wall_start = time.perf_counter() - step * self.env.DT
            self._skipped_in_a_row = 0
        if not self._trail or step != self._last_step:
            self._trail.append(np.array(self.env.ball_pos, dtype=np.float64))
        self._last_step = step

    def _update_artists(self):
        """Move the existing artists to the current state (nothing is re-created)."""
        env = self.env
        rot = self.rotation()
        lift = self.LINE_LIFT

        self._artists["tray"].set_verts(self._tray_faces(rot))

        # Goal marker and goal-radius circle, drawn on the tilted surface.
        angle = np.linspace(0.0, 2.0 * np.pi, 41)
        circle = np.column_stack([env.GOAL_RADIUS * np.cos(angle),
                                  env.GOAL_RADIUS * np.sin(angle),
                                  lift * np.ones_like(angle)]).dot(rot.T)
        self._artists["goal_circle"].set_data_3d(circle[:, 0], circle[:, 1], circle[:, 2])
        centre = rot.dot([0.0, 0.0, lift])
        self._artists["goal_point"].set_data_3d([centre[0]], [centre[1]], [centre[2]])

        # Trail: stored in the tray frame, so it tilts together with the tray.
        trail_t = np.array(self._trail)
        trail = np.column_stack([trail_t, lift * np.ones(len(trail_t))]).dot(rot.T)
        self._artists["trail"].set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])

        # Ball: centre one radius above the surface along the tray normal.
        ball_t = np.array([env.ball_pos[0], env.ball_pos[1], self.ball_radius])
        ball_w = rot.dot(ball_t)
        self._artists["ball"].set_verts(self._unit_sphere * self.ball_radius + ball_w)

        # Inertial force on the ball: opposite to a_base, in the tray plane.
        push = -np.asarray(env.a_base, dtype=np.float64) * self.DISTURBANCE_ARROW_SCALE
        length = np.linalg.norm(push)
        arrow = self._artists["arrow"]
        if length < 1e-4:
            arrow.set_visible(False)
        else:
            along = push / length
            across = np.array([-along[1], along[0]])
            tip = ball_t[:2] + push
            back = tip - self.ARROW_HEAD_LENGTH * along
            head = 0.4 * self.ARROW_HEAD_LENGTH * across
            points_t = np.array([ball_t[:2], tip, back + head, tip, back - head])
            points = np.column_stack(
                [points_t, self.ball_radius * np.ones(len(points_t))]).dot(rot.T)
            arrow.set_data_3d(points[:, 0], points[:, 1], points[:, 2])
            arrow.set_visible(True)

        reward, episode_return = self._last_values
        a_base_norm = float(np.linalg.norm(env.a_base))
        pitch_deg, roll_deg = np.rad2deg(env.tilt)
        lines = [
            "step   {:d} / {:d}".format(env.step_count, env.max_steps),
            "time   {:.2f} s".format(env.step_count * env.DT),
            "",
            "pitch  {:+6.2f} deg".format(pitch_deg),
            "roll   {:+6.2f} deg".format(roll_deg),
            "c_rr   {:.4f}".format(env.c_rr),
            "",
            "reward {:+.3f}".format(reward),
            "return {:+.2f}".format(episode_return),
            "",
            "disturbance",
            "ACTIVE {:.2f} m/s^2".format(a_base_norm) if a_base_norm > 0.0 else "none",
        ]
        if np.any(np.abs(env.ball_pos) > env.tray_half_size):
            lines += ["", "BALL OFF TRAY"]
        self._artists["text"].set_text("\n".join(lines))
        self._artists["label"].set_text(self.label)

    def _on_window_close(self, event):
        """Matplotlib callback: the user closed the window."""
        self.window_closed = True
