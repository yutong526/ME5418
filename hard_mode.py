"""Harder Ball-on-Tray variants, built to separate LQR from a learned policy.

Why this exists
---------------
``demo_lqr.py`` shows that a fixed-gain LQR controller already solves the
environment on 50/50 seeds and beats a hand-tuned PD controller by 1.7 % of
return. There is almost no room left for a learned policy to demonstrate value,
so the environment has to be made harder. The question is *how*.

Pushing the physics (more friction spread, stronger pulses) is the obvious
route and the wrong one: at low ``c_rr`` with a strong pulse the ball passes
the point of no return before any controller could react, because the tilt is
capped at 15 deg and moves only 3 deg per step. Those episodes are
*physically unrecoverable*, so a learned policy fails them too. Difficulty of
that kind adds no discriminating power.

This module adds difficulty along two axes that leave the physics untouched.
Neither wrapper touches ``ball_on_tray_gym.py``, so its 14 tests keep passing.

1. ``ShapedReward`` shrinks the goal radius and raises the goal bonus, turning
   the task from "keep the ball on the tray" into "hold it inside a small
   circle". With the stock 20 mm radius a tuned LQR is already inside ~96 % of
   the time, so raising the bonus alone changes nothing: the *radius* is the
   knob that matters.

2. ``SensorNoiseDelay`` adds measurement noise and latency, so holding that
   circle requires *estimating* the state rather than reading it.

Measured result: neither axis is sufficient alone, and together they are.
With perfect sensors a tuned LQR holds the 5 mm circle 91 % of the time during
quiet periods, so that precision is physically achievable; with the
``realistic`` sensors the best LQR in the grid manages only 53 %, while
survival stays at 98 % in both cases. The 38-point gap is therefore *not* a
physical limit -- it is the cost of imperfect state information, and it is the
part a learned policy can plausibly claim. Run ``--diagnose`` to reproduce
that split.

Why a fixed gain cannot close it: the grid search picks a *lower* gain as the
sensors degrade (K_x falls from 179 to 56), because high gain holds the small
circle but amplifies noise, and low gain is robust but imprecise. A single
linear gain must sit at one point on that trade-off. A policy with the three
stacked frames can be nonlinear and history-dependent instead -- filtering hard
while the signal is quiet, reacting decisively to a genuine excursion, and
extrapolating across the known delay.

LQG (a Kalman filter with a delay-augmented state) is the classical answer to
this and would do better than plain LQR. It is still not optimal here: it
assumes linear dynamics with known parameters and Gaussian noise, whereas
``c_rr`` is an unknown constant and ``a_base`` is a non-Gaussian jump process.
Both are hidden from the observation, so the optimal estimator is nonlinear.

Fairness
--------
Any claim of the form "LQR cannot do this" invites the reply "you did not tune
LQR properly". ``--tune`` answers that in advance: it grid-searches the LQR
cost weights against the *actual shaped reward* and reports the best gain
found, so the comparison is against the best LQR of this class, not against
one arbitrary weighting.

Examples:
    python hard_mode.py --reward precision --compare 50
    python hard_mode.py --reward pinpoint --tune --seeds 20
    python hard_mode.py --sweep                       # reward presets x policies
    python hard_mode.py --reward precision --sensors realistic --compare 50
    python hard_mode.py --reward pinpoint --view 3d   # watch one episode
"""

import argparse
from collections import deque

import gym
import numpy as np

from ball_on_tray_gym import BallOnTrayEnv
from demo import DEFAULT_SEED, DISTURBANCE_RANGE, FRICTION_RANGE, pd_policy, save_gif
from demo_lqr import (SCALE_POS, SCALE_TILT, SCALE_VEL, build_model, make_lqr_policy,
                      run_episode, solve_dare)
from renderer_3d import TrayRenderer3D

# ------------------------------------------------------------- reward presets
# The goal radius is the knob that matters: the stock 20 mm circle is so wide
# that a good controller sits inside it almost always, leaving no headroom. The
# bonus is raised alongside so that the bonus term stays worth roughly as much
# as the distance term, keeping the two reward components comparable in scale.
REWARD_PRESETS = {
    "default":   {"bonus": 0.5, "radius": 0.020},    # as shipped in the environment
    "precision": {"bonus": 2.0, "radius": 0.010},    # 10 mm
    "pinpoint":  {"bonus": 4.0, "radius": 0.005},    # 5 mm
}

# ------------------------------------------------------------- sensor presets
# Noise is one standard deviation in physical units; delays are in control
# steps of DT = 20 ms.
#   pos  [m]     a camera tracking a 2 cm ball on a 30 cm tray: a few mm
#   vel  [m/s]   differentiated from position, so already partly filtered
#   tilt [rad]   a joint encoder, far more accurate than the vision channel
#   obs_delay    camera exposure plus detection: 20-60 ms is typical
#   act_delay    servo lag between command and motion: 20-40 ms
SENSOR_PRESETS = {
    "clean":     {"pos": 0.0,   "vel": 0.0,  "tilt": 0.0,   "obs_delay": 0, "act_delay": 0},
    "camera":    {"pos": 0.002, "vel": 0.03, "tilt": 0.002, "obs_delay": 1, "act_delay": 0},
    "realistic": {"pos": 0.003, "vel": 0.06, "tilt": 0.004, "obs_delay": 2, "act_delay": 1},
    "harsh":     {"pos": 0.005, "vel": 0.12, "tilt": 0.009, "obs_delay": 3, "act_delay": 2},
}

# Offset so the noise stream is reproducible from the episode seed but
# independent of the environment's own random stream.
NOISE_SEED_OFFSET = 10007

# Grid searched by --tune. W_POS is held at 1.0, since only the ratios matter.
TUNE_GRID = {
    "vel": (0.01, 0.05, 0.2),
    "tilt": (0.005, 0.02, 0.08),
    "action": (0.0005, 0.002, 0.008),
}

# Winners of that grid search under the "pinpoint" reward, one per sensor
# preset. Reproduce with, e.g.
#     python hard_mode.py --reward pinpoint --sensors realistic --tune
# Note how the optimal gain *falls* as the sensors degrade: holding a 5 mm
# circle wants high gain, but high gain amplifies measurement noise, and a
# single linear gain has to pick one point on that trade-off.
TUNED_WEIGHTS = {
    "clean":     {"vel": 0.2,  "tilt": 0.005, "action": 0.0005},   # K_x = 178.6
    "camera":    {"vel": 0.2,  "tilt": 0.005, "action": 0.002},    # K_x = 105.6
    "realistic": {"vel": 0.01, "tilt": 0.02,  "action": 0.008},    # K_x =  56.1
    "harsh":     {"vel": 0.2,  "tilt": 0.08,  "action": 0.008},    # K_x =  49.6
}

# Statistics measured over 50 seeds, stamped onto the --demo frames so the
# recording carries its own context. Reproduce with --tune.
DEMO_STATS = {
    "clean":     "tuned LQR: 87% of reward ceiling, 81% of steps in the circle",
    "realistic": "tuned LQR: 57% of reward ceiling, 47% of steps in the circle",
}


# ------------------------------------------------------- the reward wrapper
class ShapedReward(gym.Wrapper):
    """Replace the environment's goal bonus with a narrower, larger one.

    The environment's own reward is

        r = (1 - d) + GOAL_BONUS * [dist < GOAL_RADIUS] - action penalties

    and ``info["reward_terms"]["goal_bonus"]`` reports the bonus term alone, so
    the wrapper can swap that single term out exactly rather than recomputing
    the whole reward:

        r' = r - old_bonus_term + new_bonus * [dist < new_radius]

    On the step the ball leaves the tray the environment returns exactly
    ``OUT_OF_BOUNDS_REWARD`` with every other term zero, and that step is
    passed through untouched.

    Args:
        env: the environment (or wrapper) to wrap.
        bonus: reward added per step while inside the goal radius [-].
        radius: goal radius [m].

    Attributes:
        ceiling: the best achievable per-step reward, ``1 + bonus``, used for
            reporting how much of the available reward a policy captures.
    """

    def __init__(self, env, bonus=0.5, radius=0.02):
        super().__init__(env)
        if bonus < 0.0:
            raise ValueError("bonus must be >= 0")
        if not 0.0 < radius <= env.tray_half_size:
            raise ValueError("radius must be in (0, tray_half_size]")

        self.bonus = float(bonus)
        self.radius = float(radius)
        self.ceiling = 1.0 + self.bonus

    def step(self, action):
        """Step the inner environment and rewrite the goal-bonus term."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        info["in_goal"] = False
        if not info["out_of_bounds"]:
            distance = float(np.linalg.norm(info["ball_pos"]))        # [m]
            new_bonus = self.bonus if distance < self.radius else 0.0
            delta = new_bonus - info["reward_terms"]["goal_bonus"]
            reward += delta
            info["reward_terms"] = dict(info["reward_terms"], goal_bonus=new_bonus)
            info["in_goal"] = distance < self.radius

            # The base environment already added the *unshaped* reward to its
            # own bookkeeping, and that bookkeeping is what the renderers show.
            # Correct it by the same delta. gym.Wrapper forwards attribute
            # reads but not writes, so go through `unwrapped` explicitly.
            base = self.unwrapped
            base.last_reward = reward
            base.episode_return += delta

        return obs, reward, terminated, truncated, info

    def describe(self):
        """One-line summary of the shaped reward."""
        return "goal radius {:.0f} mm, bonus {:.1f} (per-step ceiling {:.1f})".format(
            self.radius * 1e3, self.bonus, self.ceiling)


# ------------------------------------------------------- the sensor wrapper
class SensorNoiseDelay(gym.Wrapper):
    """Add Gaussian measurement noise and latency to a ``BallOnTrayEnv``.

    The physics, the reward and the episode-end conditions are untouched: only
    what the policy observes, and when its action takes effect, change.

    Noise is drawn **once per measurement**, not once per observation. A frame
    taken at time t appears in three consecutive stacked observations and must
    carry the same corrupted value each time; otherwise a policy could average
    three independent samples of one instant and the noise would be far less
    harmful than real sensor noise. The wrapper therefore keeps its own history
    of already-corrupted frames and assembles the stack from that.

    Args:
        env: the environment to wrap.
        pos_noise: std of the position measurement noise [m].
        vel_noise: std of the velocity measurement noise [m/s].
        tilt_noise: std of the tilt measurement noise [rad].
        obs_delay: how many steps stale the observation is [-].
        act_delay: how many steps before an action takes effect [-].
    """

    def __init__(self, env, pos_noise=0.0, vel_noise=0.0, tilt_noise=0.0,
                 obs_delay=0, act_delay=0):
        super().__init__(env)

        for name, value in (("pos_noise", pos_noise), ("vel_noise", vel_noise),
                            ("tilt_noise", tilt_noise)):
            if value < 0.0:
                raise ValueError("{} must be >= 0".format(name))
        if int(obs_delay) < 0 or int(act_delay) < 0:
            raise ValueError("delays must be >= 0 steps")

        self.pos_noise = float(pos_noise)
        self.vel_noise = float(vel_noise)
        self.tilt_noise = float(tilt_noise)
        self.obs_delay = int(obs_delay)
        self.act_delay = int(act_delay)

        self._frame_dim = env.FRAME_DIM
        self._n_stack = env.N_STACK

        # The observation is normalised, so convert the physical standard
        # deviations to that scale (the same scaling as _get_frame uses).
        max_tilt = np.deg2rad(env.MAX_TILT_DEG)                       # [rad]
        self._sigma = np.array([
            self.pos_noise / env.TRAY_HALF_SIZE,
            self.pos_noise / env.TRAY_HALF_SIZE,
            self.vel_noise / env.VEL_SCALE,
            self.vel_noise / env.VEL_SCALE,
            self.tilt_noise / max_tilt,
            self.tilt_noise / max_tilt,
        ], dtype=np.float64)
        self._noisy = bool(np.any(self._sigma > 0.0))

        self._history = deque(maxlen=self._n_stack + self.obs_delay)
        self._pending = deque(maxlen=max(self.act_delay, 1))
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        """Reset the environment and refill the measurement and action queues."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._rng = np.random.default_rng(
            None if seed is None else seed + NOISE_SEED_OFFSET)

        corrupted = self._corrupt(obs[-self._frame_dim:])
        self._history.clear()
        for _ in range(self._history.maxlen):
            self._history.append(corrupted)

        self._pending.clear()
        for _ in range(self.act_delay):
            self._pending.append(np.zeros(2, dtype=np.float32))

        return self._observation(), info

    def step(self, action):
        """Apply a (possibly delayed) action and return a (possibly stale) observation."""
        applied = self._delay_action(action)
        obs, reward, terminated, truncated, info = self.env.step(applied)

        self._history.append(self._corrupt(obs[-self._frame_dim:]))
        info["applied_action"] = np.asarray(applied, dtype=np.float32)
        info["true_frame"] = np.asarray(obs[-self._frame_dim:], dtype=np.float32)

        return self._observation(), reward, terminated, truncated, info

    def _delay_action(self, action):
        """Return the action that actually reaches the tray on this step."""
        if self.act_delay == 0:
            return action
        self._pending.append(np.asarray(action, dtype=np.float32))
        return self._pending.popleft()

    def _corrupt(self, frame):
        """Add one independent noise draw to a single normalised frame."""
        frame = np.asarray(frame, dtype=np.float64)
        if not self._noisy:
            return frame.astype(np.float32)
        return (frame + self._rng.normal(0.0, 1.0, self._frame_dim) * self._sigma
                ).astype(np.float32)

    def _observation(self):
        """Assemble the delayed, noisy frame stack.

        ``_history`` holds ``n_stack + obs_delay`` corrupted frames, oldest
        first. The newest frame the policy may see sits ``obs_delay`` places
        from the end, so the window it needs is the first ``n_stack`` entries.
        """
        return np.concatenate(list(self._history)[:self._n_stack]).astype(np.float32)

    def describe(self):
        """One-line summary of the configured sensors."""
        if not self._noisy and self.obs_delay == 0 and self.act_delay == 0:
            return "ideal (no noise, no delay)"
        return ("pos {:.0f} mm, vel {:.0f} mm/s, tilt {:.2f} deg, "
                "obs delay {}, act delay {}".format(
                    self.pos_noise * 1e3, self.vel_noise * 1e3,
                    np.rad2deg(self.tilt_noise), self.obs_delay, self.act_delay))


# ------------------------------------------------------------- building a run
def make_env(reward_config, sensor_config, render_mode=None):
    """Build ``BallOnTrayEnv`` wrapped for the requested difficulty.

    ``ShapedReward`` goes outermost so it can read the reward breakdown that
    the inner environment puts in ``info``.

    Returns:
        The wrapped environment. Its ``ceiling`` attribute is the per-step
        reward ceiling of the shaped reward.
    """
    env = BallOnTrayEnv(friction_range=FRICTION_RANGE,
                        disturbance_range=DISTURBANCE_RANGE, render_mode=render_mode)
    env = SensorNoiseDelay(env, **_sensor_kwargs(sensor_config))
    return ShapedReward(env, bonus=reward_config["bonus"],
                        radius=reward_config["radius"])


def _sensor_kwargs(config):
    """Translate a sensor preset into ``SensorNoiseDelay`` keyword arguments."""
    return {"pos_noise": config["pos"], "vel_noise": config["vel"],
            "tilt_noise": config["tilt"], "obs_delay": config["obs_delay"],
            "act_delay": config["act_delay"]}


def describe_sensors(config):
    """One-line summary of a sensor preset, without building an environment."""
    if not any((config["pos"], config["vel"], config["tilt"],
                config["obs_delay"], config["act_delay"])):
        return "ideal (no noise, no delay)"
    return ("pos {:.0f} mm, vel {:.0f} mm/s, tilt {:.2f} deg, "
            "obs delay {}, act delay {}".format(
                config["pos"] * 1e3, config["vel"] * 1e3,
                np.rad2deg(config["tilt"]), config["obs_delay"], config["act_delay"]))


def lqr_policy_for(weights):
    """Design an LQR controller for one set of dimensionless cost weights.

    Args:
        weights: dict with keys "vel", "tilt", "action" (the position weight is
            fixed at 1.0, since only the ratios between weights matter).

    Returns:
        (policy, K) with K of shape (1, 3).
    """
    A, B = build_model()
    Q = np.diag([1.0 / SCALE_POS ** 2,
                 weights["vel"] / SCALE_VEL ** 2,
                 weights["tilt"] / SCALE_TILT ** 2])
    R = np.array([[weights["action"]]])
    K, _, _ = solve_dare(A, B, Q, R)
    return make_lqr_policy(K), K


# ----------------------------------------------------------------- the rollouts
def rollout(env, policy_fn, seed):
    """Run one episode without rendering.

    Returns:
        A dict with the return, step count, whether the ball fell, the fraction
        of steps spent inside the goal radius, the episode's ``c_rr`` and the
        peak disturbance magnitude.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    peak_disturbance = max(
        [float(np.linalg.norm(p["accel"])) for p in env.disturbances] or [0.0])

    total, terminated, truncated, in_goal = 0.0, False, False, 0
    while not (terminated or truncated):
        action = env.action_space.sample() if policy_fn is None else policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        in_goal += int(info["in_goal"])

    steps = max(env.step_count, 1)
    return {"return": total, "steps": env.step_count, "fell": terminated,
            "in_goal": in_goal / float(steps), "c_rr": env.c_rr,
            "peak_disturbance": peak_disturbance}


def evaluate(env, policy_fn, n_seeds):
    """Roll out seeds ``0 .. n_seeds-1``."""
    return [rollout(env, policy_fn, seed) for seed in range(n_seeds)]


def summarise(results, ceiling):
    """Condense a result list into the numbers the tables below print."""
    returns = np.array([r["return"] for r in results])
    steps = sum(r["steps"] for r in results)
    return {
        "mean": returns.mean(),
        "std": returns.std(),
        "survival": sum(0 if r["fell"] else 1 for r in results) / float(len(results)),
        "per_step": returns.sum() / steps,
        "captured": (returns.sum() / steps) / ceiling,
        "in_goal": float(np.mean([r["in_goal"] for r in results])),
    }


# ------------------------------------------------------------------- reporting
def print_table(rows, ceiling):
    """Print one row per policy.

    ``in goal`` is the headroom indicator: it is the fraction of steps a policy
    keeps the ball inside the goal radius, which is exactly what the bonus pays
    for. A controller already near 100 % leaves nothing for a learned policy to
    win; a controller well below it does.
    """
    header = "{:<10}{:>18}{:>11}{:>13}{:>11}{:>11}".format(
        "policy", "return", "survived", "reward/step", "captured", "in goal")
    print(header)
    print("-" * len(header))
    for name, stats in rows:
        print("{:<10}{:>11.1f} +-{:>5.1f}{:>10.0f} %{:>13.3f}{:>10.0f} %{:>10.0f} %".format(
            name, stats["mean"], stats["std"], 100.0 * stats["survival"],
            stats["per_step"], 100.0 * stats["captured"], 100.0 * stats["in_goal"]))
    print("\nper-step ceiling = 1 + bonus = {:.1f}".format(ceiling))


def compare(policies, reward_config, sensor_config, n_seeds):
    """Evaluate every policy under one difficulty configuration."""
    env = make_env(reward_config, sensor_config)
    print("reward       : {}".format(env.describe()))
    print("sensors      : {}".format(env.env.describe()))
    print("{} seeds, c_rr in {}, |a_base| in {} m/s^2\n".format(
        n_seeds, FRICTION_RANGE, DISTURBANCE_RANGE))

    rows = [(name, summarise(evaluate(env, fn, n_seeds), env.ceiling))
            for name, fn in policies]
    print_table(rows, env.ceiling)
    env.close()
    return rows


def sweep(policies, sensor_config, n_seeds):
    """Run every reward preset against every policy."""
    print("sensors      : {}".format(describe_sensors(sensor_config)))
    print("{} seeds per cell. Each cell is captured reward / time in goal.\n".format(
        n_seeds))

    header = "{:<12}{:>12}".format("reward", "radius") + "".join(
        "{:>20}".format(name) for name, _ in policies)
    print(header)
    print("-" * len(header))

    for preset, reward_config in REWARD_PRESETS.items():
        env = make_env(reward_config, sensor_config)
        row = "{:<12}{:>9.0f} mm".format(preset, reward_config["radius"] * 1e3)
        for _, policy_fn in policies:
            stats = summarise(evaluate(env, policy_fn, n_seeds), env.ceiling)
            row += "{:>12.0f} % /{:>4.0f} %".format(
                100.0 * stats["captured"], 100.0 * stats["in_goal"])
        print(row)
        env.close()

    print("\n'captured' is reward per step as a fraction of the 1 + bonus ceiling.")
    print("A large gap between 'captured' and 100 % is headroom a learned policy")
    print("could claim; 'time in goal' says how much of it is the bonus term.")


def stamp_frames(frames, lines):
    """Burn a caption into the top-left corner of every frame.

    The renderers have a fixed text panel, and ``renderer_3d.py`` is a tested
    deliverable we would rather not touch, so the caption is drawn afterwards
    on the recorded images instead. Without it a two-segment recording gives
    the viewer no way to tell which condition is on screen.

    Args:
        frames: list of (H, W, 3) uint8 arrays.
        lines: list of strings; the first is drawn larger.

    Returns:
        A new list of stamped frames.
    """
    from PIL import Image, ImageDraw, ImageFont

    def font_of(size):
        for name in ("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    fonts = [font_of(24)] + [font_of(16)] * (len(lines) - 1)
    stamped = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        draw = ImageDraw.Draw(image)
        y = 8
        heights = []
        for text, font in zip(lines, fonts):
            # textbbox on newer Pillow; getsize is deprecated but is the only
            # option on the oldest versions this project supports.
            if hasattr(draw, "textbbox"):
                heights.append(draw.textbbox((0, 0), text, font=font)[3])
            else:
                heights.append(font.getsize(text)[1])
        box = sum(heights) + 6 * len(lines) + 8
        draw.rectangle([(0, 0), (image.size[0], box)], fill=(26, 26, 30))
        for text, font, height in zip(lines, fonts, heights):
            draw.text((14, y), text, fill=(245, 245, 245), font=font)
            y += height + 6
        stamped.append(np.asarray(image))
    return stamped


def demo(reward_config, conditions, seed, view, gif_path):
    """Record one episode per sensor condition into a single clip.

    Every condition uses the *same seed*, so the start position, the friction
    and the disturbance schedule are identical and only the sensors differ, and
    the LQR weights re-tuned for that condition, so each segment shows the best
    fixed-gain controller of its class rather than one arbitrary weighting.

    Args:
        reward_config: the shaped-reward settings.
        conditions: sensor preset names to show, in order.
        seed: episode seed, shared by every segment.
        view: "2d" or "3d".
        gif_path: where to write the clip, or None to open a window instead.
    """
    frames = [] if gif_path else None
    summaries = []

    for name in conditions:
        sensor_config = SENSOR_PRESETS[name]
        weights = TUNED_WEIGHTS[name]
        policy, K = lqr_policy_for(weights)

        mode = "rgb_array" if gif_path else "human"
        env = make_env(reward_config, sensor_config,
                       render_mode=None if view == "3d" else mode)
        renderer = TrayRenderer3D(env, mode=mode) if view == "3d" else None

        segment = [] if gif_path else None
        summary = run_episode(env, "lqr", policy, seed, segment, renderer)
        if renderer is not None:
            renderer.close()
        env.close()

        if frames is not None:
            frames.extend(stamp_frames(segment, [
                "{} sensors".format(name),
                describe_sensors(sensor_config),
                DEMO_STATS.get(name, ""),
            ]))

        summary["sensors"] = name
        summary["gain"] = K[0, 0]
        summaries.append(summary)

    print("reward       : goal radius {:.0f} mm, bonus {:.1f} "
          "(per-step ceiling {:.1f})".format(
              reward_config["radius"] * 1e3, reward_config["bonus"],
              1.0 + reward_config["bonus"]))
    print("seed         : {} (identical physics in every segment)\n".format(seed))
    header = "{:<12}{:>10}{:>9}{:>26}{:>10}".format(
        "sensors", "return", "steps", "ended because", "K_x")
    print(header)
    print("-" * len(header))
    for item in summaries:
        print("{:<12}{:>10.1f}{:>9d}{:>26}{:>10.1f}".format(
            item["sensors"], item["return"], item["steps"], item["reason"],
            item["gain"]))

    if gif_path:
        save_gif(frames, gif_path)
        print("\nsaved {} frames to {}".format(len(frames), gif_path))


def diagnose(reward_config, n_seeds):
    """Split the LQR shortfall into an information part and a physics part.

    For every sensor preset the LQR weights are re-tuned under that same
    preset, so each row is the best LQR of its class rather than one arbitrary
    weighting. Time inside the goal radius is then reported separately for

        quiet  steps with no disturbance pulse active -- the physics is
               undemanding here, so whatever the controller misses is its own
               doing;
        pulse  steps with a pulse active -- a strong pulse displaces the ball
               by more than the goal radius whatever the controller does.

    The ``clean`` row is the reference: it is what a controller with perfect
    state information achieves, and therefore what is physically possible. The
    gap between it and the noisy rows is attributable to state estimation
    alone, which is the headroom a learned policy can claim.
    """
    print("reward       : goal radius {:.0f} mm, bonus {:.1f}".format(
        reward_config["radius"] * 1e3, reward_config["bonus"]))
    print("{} seeds for tuning and evaluation\n".format(n_seeds))
    header = "{:<12}{:>26}{:>10}{:>10}{:>11}{:>10}".format(
        "sensors", "best weights vel/tilt/act", "quiet", "pulse", "survival", "K_x")
    print(header)
    print("-" * len(header))

    reference = None
    for name in ("clean", "camera", "realistic", "harsh"):
        sensor_config = SENSOR_PRESETS[name]
        env = make_env(reward_config, sensor_config)

        best = None
        for w_vel in TUNE_GRID["vel"]:
            for w_tilt in TUNE_GRID["tilt"]:
                for w_action in TUNE_GRID["action"]:
                    weights = {"vel": w_vel, "tilt": w_tilt, "action": w_action}
                    policy, _ = lqr_policy_for(weights)
                    stats = summarise(evaluate(env, policy, n_seeds), env.ceiling)
                    if best is None or stats["per_step"] > best[1]["per_step"]:
                        best = (weights, stats)
        weights = best[0]

        policy, K = lqr_policy_for(weights)
        quiet_hits = quiet_steps = pulse_hits = pulse_steps = survived = 0
        for seed in range(n_seeds):
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            terminated = truncated = False
            while not (terminated or truncated):
                obs, _, terminated, truncated, info = env.step(policy(obs))
                if np.linalg.norm(info["a_base"]) == 0.0:
                    quiet_steps += 1
                    quiet_hits += int(info["in_goal"])
                else:
                    pulse_steps += 1
                    pulse_hits += int(info["in_goal"])
            survived += 0 if terminated else 1
        env.close()

        quiet = quiet_hits / float(max(quiet_steps, 1))
        if reference is None:
            reference = quiet
        print("{:<12}{:>26}{:>9.0f} %{:>9.0f} %{:>10.0f} %{:>10.1f}".format(
            name,
            "{:.3g} / {:.3g} / {:.4g}".format(
                weights["vel"], weights["tilt"], weights["action"]),
            100.0 * quiet, 100.0 * pulse_hits / float(max(pulse_steps, 1)),
            100.0 * survived / n_seeds, K[0, 0]))

    print("\nWith perfect sensors the best LQR holds the circle {:.0f} % of the quiet".format(
        100.0 * reference))
    print("time, so that precision is physically achievable. Survival stays high in")
    print("every row, so the degraded rows are not failing physically -- they are")
    print("failing to estimate the state. That difference is the headroom.")


def tune(reward_config, sensor_config, n_seeds):
    """Grid-search the LQR cost weights against the shaped reward.

    This exists to pre-empt the objection that LQR was simply not tuned for the
    new reward. If the best LQR in the grid still leaves a large gap to the
    ceiling, that gap is structural: a quadratic cost cannot express a step
    bonus, so no weighting recovers it.
    """
    env = make_env(reward_config, sensor_config)
    print("reward       : {}".format(env.describe()))
    print("sensors      : {}".format(env.env.describe()))
    print("{} seeds per candidate, {} candidates\n".format(
        n_seeds, len(TUNE_GRID["vel"]) * len(TUNE_GRID["tilt"]) * len(TUNE_GRID["action"])))

    header = "{:>8}{:>8}{:>10}{:>12}{:>11}{:>11}".format(
        "w_vel", "w_tilt", "w_action", "reward/step", "captured", "in goal")
    print(header)
    print("-" * len(header))

    best = None
    for w_vel in TUNE_GRID["vel"]:
        for w_tilt in TUNE_GRID["tilt"]:
            for w_action in TUNE_GRID["action"]:
                weights = {"vel": w_vel, "tilt": w_tilt, "action": w_action}
                policy, K = lqr_policy_for(weights)
                stats = summarise(evaluate(env, policy, n_seeds), env.ceiling)
                print("{:>8.3g}{:>8.3g}{:>10.4g}{:>12.3f}{:>10.0f} %{:>10.0f} %".format(
                    w_vel, w_tilt, w_action, stats["per_step"],
                    100.0 * stats["captured"], 100.0 * stats["in_goal"]))
                if best is None or stats["per_step"] > best[0]["per_step"]:
                    best = (stats, weights, K)

    stats, weights, K = best
    print("\nbest LQR     : w_vel {:.3g}, w_tilt {:.3g}, w_action {:.4g}".format(
        weights["vel"], weights["tilt"], weights["action"]))
    print("  gain K     : x {:8.3f}   v {:8.3f}   theta {:8.3f}".format(
        K[0, 0], K[0, 1], K[0, 2]))
    print("  reward/step: {:.3f} of {:.1f} ceiling ({:.0f} % captured)".format(
        stats["per_step"], env.ceiling, 100.0 * stats["captured"]))
    print("  time in goal: {:.0f} %   survival: {:.0f} %".format(
        100.0 * stats["in_goal"], 100.0 * stats["survival"]))
    print("\nHeadroom for a learned policy: {:.0f} % of the per-step ceiling.".format(
        100.0 * (1.0 - stats["captured"])))
    env.close()


# ------------------------------------------------------------------------- main
def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Ball-on-Tray difficulty variants for an LQR / RL comparison.")
    parser.add_argument("--reward", choices=sorted(REWARD_PRESETS), default="precision",
                        help="reward preset; shrinks the goal radius and raises the "
                             "bonus (default: precision)")
    parser.add_argument("--sensors", choices=sorted(SENSOR_PRESETS), default="clean",
                        help="sensor preset; noise and latency (default: clean, so the "
                             "reward shaping is studied on its own)")
    parser.add_argument("--bonus", type=float, default=None,
                        help="override the goal bonus [-]")
    parser.add_argument("--goal-radius", type=float, default=None,
                        help="override the goal radius [mm]")
    parser.add_argument("--policy", choices=["random", "pd", "lqr", "all"], default="all",
                        help="policy to run (default: all)")
    parser.add_argument("--compare", metavar="N", type=int, default=None,
                        help="evaluate over N seeds and print a statistics table")
    parser.add_argument("--sweep", action="store_true",
                        help="evaluate every reward preset against every policy")
    parser.add_argument("--tune", action="store_true",
                        help="grid-search the LQR weights against the shaped reward")
    parser.add_argument("--demo", metavar="PRESETS", default=None,
                        help="record one episode per sensor preset into a single clip, "
                             "all under the same seed and each with its own re-tuned "
                             "LQR gain; comma-separated, e.g. 'clean,realistic'")
    parser.add_argument("--diagnose", action="store_true",
                        help="re-tune the LQR under every sensor preset and split its "
                             "time in goal into quiet and disturbed steps, which "
                             "separates the information limit from the physical one")
    parser.add_argument("--seeds", type=int, default=50,
                        help="seeds per cell for --sweep and --tune (default: 50)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed of the watched episode (default: {})".format(DEFAULT_SEED))
    parser.add_argument("--view", choices=["2d", "3d"], default="2d",
                        help="view used when watching an episode (default: 2d)")
    parser.add_argument("--save-gif", metavar="PATH", default=None,
                        help="render off-screen and save the episode(s) to a gif")
    return parser.parse_args()


def resolve(args):
    """Build the reward and sensor configurations from presets plus overrides."""
    reward = dict(REWARD_PRESETS[args.reward])
    if args.bonus is not None:
        reward["bonus"] = args.bonus
    if args.goal_radius is not None:
        reward["radius"] = args.goal_radius / 1e3        # mm -> m
    return reward, dict(SENSOR_PRESETS[args.sensors])


def main():
    """Evaluate, tune, or watch the harder environment."""
    args = parse_args()
    reward_config, sensor_config = resolve(args)

    if args.demo:
        conditions = [name.strip() for name in args.demo.split(",")]
        unknown = [name for name in conditions if name not in SENSOR_PRESETS]
        if unknown:
            raise SystemExit("unknown sensor preset(s): {}; choose from {}".format(
                ", ".join(unknown), ", ".join(sorted(SENSOR_PRESETS))))
        demo(reward_config, conditions, args.seed, args.view, args.save_gif)
        return
    if args.diagnose:
        diagnose(reward_config, args.seeds)
        return
    if args.tune:
        tune(reward_config, sensor_config, args.seeds)
        return

    lqr, K = lqr_policy_for({"vel": 0.05, "tilt": 0.02, "action": 0.002})
    # None marks the random policy, which samples from the action space.
    policies = [("random", None), ("pd", pd_policy), ("lqr", lqr)]
    if args.policy != "all":
        policies = [item for item in policies if item[0] == args.policy]

    if args.sweep:
        sweep(policies, sensor_config, args.seeds)
        return
    if args.compare is not None:
        compare(policies, reward_config, sensor_config, args.compare)
        return

    # Otherwise watch (or record) one episode per policy, reusing the demo
    # episode loop so the on-screen information matches demo_lqr.py.
    mode = "rgb_array" if args.save_gif else "human"
    # In the 3D view the environment itself must not render; the renderer draws it.
    env = make_env(reward_config, sensor_config,
                   render_mode=None if args.view == "3d" else mode)
    renderer = TrayRenderer3D(env, mode=mode) if args.view == "3d" else None
    frames = [] if args.save_gif else None
    print("reward       : {}".format(env.describe()))
    print("sensors      : {}\n".format(describe_sensors(sensor_config)))

    for name, policy_fn in policies:
        summary = run_episode(env, name, policy_fn, args.seed, frames, renderer)
        print("policy       : {}".format(summary["policy"]))
        print("steps        : {}".format(summary["steps"]))
        print("return       : {:.2f}".format(summary["return"]))
        print("ended because: {}\n".format(summary["reason"]))
    if renderer is not None:
        renderer.close()
    env.close()

    if args.save_gif:
        save_gif(frames, args.save_gif)
        print("saved {} frames to {}".format(len(frames), args.save_gif))


if __name__ == "__main__":
    main()
