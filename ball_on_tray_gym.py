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
        render_mode: None (no rendering), "human" (interactive matplotlib
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
    # These constants only affect the picture, never the dynamics.
    RENDER_FIGSIZE = (9.0, 6.0)     # figure size [inch]
    RENDER_DPI = 100                # -> 900 x 600 pixel frames
    RENDER_FONT_SIZE = 14           # font size of the text panel [pt]
    BALL_RADIUS = 0.01              # drawn ball radius [m] (not used by the physics)
    TRAIL_LENGTH = 75               # number of past positions in the trail (1.5 s)
    VIEW_MARGIN = 0.06              # visible border around the tray [m]
    DISTURBANCE_ARROW_SCALE = 0.05  # arrow length per unit |a_base| [m / (m/s^2)]
    TILT_ARROW_LENGTH = 0.12        # arrow length per axis at maximum tilt [m]

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
        # True once the user has closed the "human" window; render() is then a no-op.
        self.window_closed = False

        self._prev_action = np.zeros(2, dtype=np.float64)
        self._frames = deque(maxlen=self.N_STACK)
        # Actions waiting to take effect (only used if action_delay_steps > 0).
        self._action_queue = deque()
        # Generator of the observation noise (only created if noise is enabled).
        self._noise_rng = None

        # Rendering state, created lazily on the first render() call.
        self._fig = None                # matplotlib figure
        self._artists = {}              # name -> artist updated every frame
        self._use_blit = False          # redraw only the moving artists ("human")
        self._background = None         # cached static background for blitting
        self._trail = deque(maxlen=self.TRAIL_LENGTH)
        self._trail_step = 0            # step_count of the newest trail point
        self._last_frame_time = None    # wall-clock time of the last "human" frame [s]

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
        """Draw the current state as a top view of the tray.

        The mode is chosen in the constructor (gym 0.26 API):

            "human"      update an interactive matplotlib window at
                         ``metadata["render_fps"]`` and return None. ``reset``
                         and ``step`` already call this, so user code normally
                         does not have to.
            "rgb_array"  return the frame as an (H, W, 3) uint8 array.
            None         do nothing.

        Elements of the picture:
            * tray outline, goal marker at the centre, dashed goal-radius circle
            * the ball and the trail of its last ``TRAIL_LENGTH`` positions
            * red arrow at the ball: inertial force caused by the base
              acceleration (direction -a_base, length proportional to ||a_base||),
              shown only while a disturbance pulse is active
            * blue arrow at the centre: downhill direction of the tray, i.e.
              the direction in which the tilt accelerates the ball
              (components proportional to pitch and roll)
            * text panel: step and time, pitch and roll [deg], c_rr, reward of
              the last step, return of the episode, disturbance status

        The figure and all artists are created once; later calls only update
        their data. Rendering only reads the public state (``ball_pos``,
        ``ball_vel``, ``tilt``, ``a_base``, ``c_rr``, ``tray_half_size``,
        ``step_count``, ``last_reward``, ``episode_return``) and never changes
        the physical state.
        """
        if self.render_mode is None or self.window_closed:
            return None
        if self._fig is None:
            self._init_render()
        self._update_artists()

        if self.render_mode == "rgb_array":
            canvas = self._fig.canvas
            canvas.draw()
            # buffer_rgba() is (H, W, 4) uint8; drop the alpha channel.
            return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

        self._draw_human_frame()
        return None

    def close(self):
        """Close the render window (if any) and free the figure."""
        if self._fig is not None:
            if self.render_mode == "human":
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            self._fig = None
        self._artists = {}
        self._background = None
        self._trail.clear()
        self._last_frame_time = None
        self.window_closed = False

    # ---------------------------------------------------------------- rendering
    def _init_render(self):
        """Create the figure and every artist exactly once."""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            self._fig = plt.figure(figsize=self.RENDER_FIGSIZE, dpi=self.RENDER_DPI)
            canvas = self._fig.canvas
            canvas.mpl_connect("close_event", self._on_window_close)
            canvas.mpl_connect("draw_event", self._on_draw)
            self._use_blit = bool(getattr(canvas, "supports_blit", False))
        else:
            # Off-screen figure: no window and no GUI backend required.
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            self._fig = Figure(figsize=self.RENDER_FIGSIZE, dpi=self.RENDER_DPI)
            FigureCanvasAgg(self._fig)
            self._use_blit = False

        half = self.tray_half_size
        limit = half + self.VIEW_MARGIN
        # Square axes on the left (492 x 492 px), text panel on the right.
        ax = self._fig.add_axes([0.105, 0.11, 0.5467, 0.82])
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", fontsize=self.RENDER_FONT_SIZE - 1)
        ax.set_ylabel("y [m]", fontsize=self.RENDER_FONT_SIZE - 1, labelpad=2)
        ax.tick_params(labelsize=self.RENDER_FONT_SIZE - 3)
        ax.set_title("Ball on Tray (top view, tray frame)", fontsize=self.RENDER_FONT_SIZE)

        # Static artists.
        ax.add_patch(Rectangle((-half, -half), 2.0 * half, 2.0 * half,
                               facecolor="#f3ecdc", edgecolor="#333333", linewidth=2.0))
        ax.add_patch(Circle((0.0, 0.0), self.GOAL_RADIUS, fill=False,
                            edgecolor="#2ca02c", linestyle="--", linewidth=1.5))
        ax.plot([0.0], [0.0], marker="+", markersize=10, color="#2ca02c")

        # Dynamic artists (their data is updated every frame).
        trail, = ax.plot([], [], color="#ff7f0e", linewidth=1.5, alpha=0.7, zorder=3)
        tilt_arrow = FancyArrowPatch((0.0, 0.0), (0.01, 0.0), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=2.0,
                                     color="#1f77b4", alpha=0.85, zorder=4)
        ball = Circle((0.0, 0.0), self.BALL_RADIUS, facecolor="#ff7f0e",
                      edgecolor="black", linewidth=1.0, zorder=5)
        disturbance_arrow = FancyArrowPatch((0.0, 0.0), (0.01, 0.0), arrowstyle="-|>",
                                            mutation_scale=14, linewidth=2.0,
                                            color="#d62728", zorder=6)
        ax.add_patch(tilt_arrow)
        ax.add_patch(ball)
        ax.add_patch(disturbance_arrow)
        text = self._fig.text(0.68, 0.93, "", va="top", ha="left",
                              family="monospace", fontsize=self.RENDER_FONT_SIZE)

        self._fig.legend(
            handles=[
                Line2D([], [], linestyle="none", marker="o", markersize=8,
                       markerfacecolor="#ff7f0e", markeredgecolor="black", label="ball"),
                Line2D([], [], color="#ff7f0e", linewidth=1.5, alpha=0.7, label="recent path"),
                Line2D([], [], color="#2ca02c", linestyle="--", label="goal radius"),
                Line2D([], [], color="#1f77b4", linewidth=2.0, marker=">",
                       label="downhill (tilt)"),
                Line2D([], [], color="#d62728", linewidth=2.0, marker=">",
                       label="disturbance push"),
            ],
            loc="lower left", bbox_to_anchor=(0.665, 0.09),
            fontsize=self.RENDER_FONT_SIZE - 3, frameon=False)

        self._artists = {"trail": trail, "tilt_arrow": tilt_arrow, "ball": ball,
                         "disturbance_arrow": disturbance_arrow, "text": text}
        # Animated artists are skipped by a normal draw, which lets the static
        # background be cached and only these artists be redrawn (blitting).
        for artist in self._artists.values():
            artist.set_animated(self._use_blit)

        self._background = None
        self._trail.clear()
        if self.render_mode == "human":
            plt.show(block=False)

    def _update_artists(self):
        """Copy the current public state into the existing artists."""
        # Trail: restart it on a new episode, add at most one point per step.
        if self.step_count == 0 or self.step_count < self._trail_step:
            self._trail.clear()
        if not self._trail or self.step_count != self._trail_step:
            self._trail.append(self.ball_pos.copy())
        self._trail_step = self.step_count
        trail = np.array(self._trail)
        self._artists["trail"].set_data(trail[:, 0], trail[:, 1])

        self._artists["ball"].center = (self.ball_pos[0], self.ball_pos[1])

        # Downhill arrow from the tray centre: +pitch -> +x, +roll -> +y.
        downhill = self.tilt / self.max_tilt * self.TILT_ARROW_LENGTH      # [m]
        self._set_arrow(self._artists["tilt_arrow"], np.zeros(2), downhill)

        # Inertial force felt by the ball in the tray frame: opposite to a_base.
        push = -self.a_base * self.DISTURBANCE_ARROW_SCALE                 # [m]
        self._set_arrow(self._artists["disturbance_arrow"], self.ball_pos, push)

        a_base_norm = float(np.linalg.norm(self.a_base))
        pitch_deg, roll_deg = np.rad2deg(self.tilt)
        lines = [
            "step   {:d} / {:d}".format(self.step_count, self.max_steps),
            "time   {:.2f} s".format(self.step_count * self.DT),
            "",
            "pitch  {:+6.2f} deg".format(pitch_deg),
            "roll   {:+6.2f} deg".format(roll_deg),
            "c_rr   {:.4f}".format(self.c_rr),
            "",
            "reward {:+.3f}".format(self.last_reward),
            "return {:+.2f}".format(self.episode_return),
            "",
            "disturbance",
            "ACTIVE {:.2f} m/s^2".format(a_base_norm) if a_base_norm > 0.0 else "none",
        ]
        if self._is_out_of_bounds():
            lines += ["", "BALL OFF TRAY"]
        self._artists["text"].set_text("\n".join(lines))

    @staticmethod
    def _set_arrow(arrow, start, vector):
        """Place an arrow at ``start`` pointing along ``vector``; hide it if ~zero."""
        if np.linalg.norm(vector) < 1e-4:
            arrow.set_visible(False)
            return
        arrow.set_positions((start[0], start[1]),
                            (start[0] + vector[0], start[1] + vector[1]))
        arrow.set_visible(True)

    def _draw_human_frame(self):
        """Show the updated artists in the window and keep real-time pace."""
        canvas = self._fig.canvas
        if self._use_blit:
            if self._background is None:
                canvas.draw()       # full draw; _on_draw caches the background
            if self._background is None:
                self._background = canvas.copy_from_bbox(self._fig.bbox)
            canvas.restore_region(self._background)
            for artist in self._artists.values():
                self._fig.draw_artist(artist)
            canvas.blit(self._fig.bbox)
        else:
            canvas.draw_idle()
        canvas.flush_events()

        # Sleep so that frames are shown at render_fps (= 1 / DT, real time).
        period = 1.0 / self.metadata["render_fps"]                  # [s]
        now = time.perf_counter()
        if self._last_frame_time is not None:
            remaining = period - (now - self._last_frame_time)
            if remaining > 0.0:
                time.sleep(remaining)
        self._last_frame_time = time.perf_counter()

    def _on_draw(self, event):
        """Matplotlib callback: cache the static background after a full draw
        (first frame, window resize) for blitting."""
        if self._fig is not None and self._use_blit:
            self._background = self._fig.canvas.copy_from_bbox(self._fig.bbox)

    def _on_window_close(self, event):
        """Matplotlib callback: the user closed the window."""
        self.window_closed = True

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
