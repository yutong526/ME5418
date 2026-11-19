"""Optional extra inputs of the agent (PPO only), switched off by default.

The environment and its 18-number observation are not changed. ``FeatureEnv``
wraps a ``BallOnTrayEnv`` and builds the input of the network from it:

    [ 18-number observation, x/y/vx/vy of every frame multiplied by obs_scale ]
  + [ the last ``action_history`` actions, oldest first ]           2 per action
  + [ disturbance estimate ]                                        2 numbers

The dynamics, the reward and the episode end stay exactly those of the
wrapped environment; only what the policy sees is extended. With the default
options (``FEATURE_DEFAULTS``) nothing is wrapped at all.

Why these inputs
----------------
action_history
    With an action delay the action given at step t only takes effect at
    step t + 1, and with an actuator lag the commanded tilt runs ahead of the
    actual tilt. Neither the action waiting in the delay queue nor the
    commanded tilt is in the observation, so under ``--realistic`` the
    observation is no longer a Markov state: the policy cannot tell what it
    has already asked for and over-corrects. The agent's own last actions
    (the clipped actions it passed to ``step``) restore that information.
    They are known to any deployed controller, so this is not privileged.

disturbance_feature
    The base acceleration a_base is not observed. It shows up as the part of
    the ball's acceleration that the tilt does not explain:

        est = (v_t - v_{t-1}) / DT - (5/7) * g * sin(tilt_t)
            ~ -(5/7) * a_base - rolling resistance                [m/s^2]

    (per axis, from the two newest frames; in the ideal environment the tilt
    of the newest frame is the tilt that acted during the last step). The
    network has both frames, but the signal is small: a 1 m/s^2 push changes
    the normalised velocity by only ~0.014 per step. The estimate is passed
    divided by 2, so that it stays within about [-1, 1] over the full
    disturbance range. With observation noise it is not useful: 0.03 m/s of
    velocity noise over 20 ms is ~2 m/s^2, more than the disturbance itself.

obs_scale
    Multiplies position and velocity in the network input (the tilt is not
    scaled). Near the goal the normalised position is ~0.01-0.1, so a larger
    scale gives the first layer a larger signal. Under observation noise it
    also scales the noise and hurts.

These options and their effect were found in the experiments of 2026-11-06
(PPO, 1M steps, three training seeds each, paired with LQR on new seeds):
``disturbance_feature`` with ``obs_scale`` 5 in the ideal condition, and
``action_history`` 3 with ``--realistic``.
"""

from collections import deque

import numpy as np
from gym import spaces

from ball_on_tray_gym import BallOnTrayEnv

FEATURE_DEFAULTS = {
    "action_history": 0,            # number of past actions in the input [-]
    "disturbance_feature": False,   # append the unexplained-acceleration estimate
    "obs_scale": 1.0,               # factor on position and velocity in the input [-]
}
DISTURBANCE_FEATURE_SCALE = 2.0     # the estimate [m/s^2] is divided by this [m/s^2]
BASE_OBS_DIM = BallOnTrayEnv.N_STACK * BallOnTrayEnv.FRAME_DIM     # 18


def feature_config(config=None):
    """Return the three feature options, taken from ``config`` or the defaults.

    Args:
        config: a dict that may contain any of the keys of ``FEATURE_DEFAULTS``
            (e.g. a training configuration); other keys are ignored. Missing
            keys, and ``None``, give the defaults, so checkpoints written
            before these options existed load as the plain 18-number input.

    Raises:
        ValueError: for a negative ``action_history`` or a non-positive ``obs_scale``.
    """
    config = config or {}
    features = {key: config.get(key, default) for key, default in FEATURE_DEFAULTS.items()}
    features["action_history"] = int(features["action_history"])
    features["disturbance_feature"] = bool(features["disturbance_feature"])
    features["obs_scale"] = float(features["obs_scale"])
    if features["action_history"] < 0:
        raise ValueError("action_history must be >= 0")
    if not features["obs_scale"] > 0.0:
        raise ValueError("obs_scale must be > 0")
    return features


def checkpoint_features(checkpoint):
    """Feature options a checkpoint written by ``train.py`` was trained with."""
    return feature_config(checkpoint.get("train_config"))


def is_default(features):
    """True if the options leave the input as the plain 18-number observation."""
    return feature_config(features) == FEATURE_DEFAULTS


def feature_dim(features=None):
    """Size of the network input for these options (18 with the defaults)."""
    f = feature_config(features)
    return BASE_OBS_DIM + 2 * f["action_history"] + (2 if f["disturbance_feature"] else 0)


def describe(features=None):
    """Short text for logs, e.g. ``"last 3 actions"`` or ``"plain observation"``."""
    f = feature_config(features)
    parts = []
    if f["action_history"]:
        parts.append("last {} actions".format(f["action_history"]))
    if f["disturbance_feature"]:
        parts.append("disturbance estimate")
    if f["obs_scale"] != 1.0:
        parts.append("position/velocity x{:g}".format(f["obs_scale"]))
    return ", ".join(parts) if parts else "plain observation"


class FeatureEnv(object):
    """A ``BallOnTrayEnv`` whose observation is the extended agent input.

    ``reset`` and ``step`` follow the gym 0.26 API of the wrapped environment
    and return the extended observation; every other attribute
    (``step_count``, ``c_rr``, ``ball_pos``, ``action_space``, ...) is read
    from the wrapped environment, so the training loops, the evaluation and
    the 3D renderer work with it unchanged.

    Args:
        env: the ``BallOnTrayEnv`` to wrap.
        action_history, disturbance_feature, obs_scale: see ``FEATURE_DEFAULTS``.
    """

    def __init__(self, env, action_history=0, disturbance_feature=False, obs_scale=1.0):
        self.env = env
        self.features = feature_config({"action_history": action_history,
                                        "disturbance_feature": disturbance_feature,
                                        "obs_scale": obs_scale})
        self.n_actions = self.features["action_history"]
        self.obs_dim = feature_dim(self.features)
        frame_scale = np.array([self.features["obs_scale"]] * 4 + [1.0, 1.0], np.float32)
        self._obs_scale = np.tile(frame_scale, env.N_STACK)
        self._actions = deque(maxlen=max(self.n_actions, 1))

    def __getattr__(self, name):
        # Only called for attributes not found on the wrapper itself.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset the wrapped environment; the action history is cleared to zeros."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._actions.clear()
        for _ in range(self.n_actions):
            self._actions.append(np.zeros(2, np.float32))
        return self._features(obs), info

    def step(self, action):
        """Step the wrapped environment and record the clipped action."""
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.n_actions:
            self._actions.append(action.astype(np.float32))
        return self._features(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def _features(self, obs):
        """Build the network input from the 18-number observation ``obs``."""
        env = self.env
        parts = [obs * self._obs_scale]
        if self.n_actions:
            parts.extend(self._actions)
        if self.features["disturbance_feature"]:
            new = obs[-env.FRAME_DIM:]
            old = obs[-2 * env.FRAME_DIM:-env.FRAME_DIM]
            tilt = new[4:6] * np.deg2rad(env.MAX_TILT_DEG)                         # [rad]
            est = ((new[2:4] - old[2:4]) * env.VEL_SCALE / env.DT
                   - env.ROLLING_FACTOR * env.GRAVITY * np.sin(tilt))              # [m/s^2]
            parts.append((est / DISTURBANCE_FEATURE_SCALE).astype(np.float32))
        return np.concatenate(parts).astype(np.float32)


def make_env(env_kwargs=None, features=None):
    """Create a ``BallOnTrayEnv``, wrapped in ``FeatureEnv`` unless the options are the defaults."""
    env = BallOnTrayEnv(**(env_kwargs or {}))
    if is_default(features):
        return env
    return FeatureEnv(env, **feature_config(features))
