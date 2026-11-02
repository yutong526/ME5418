"""Shared training infrastructure: curriculum, parallel environments, evaluation, logging.

Curriculum (staged domain randomisation)
----------------------------------------
The proposal trains "first with a fixed friction coefficient and no
disturbances, then progressively randomizing the friction coefficient and
disturbance magnitude". ``CURRICULA`` writes that down as stages, each active
for a fraction of the total training steps:

    staged     fixed c_rr, no pulses -> random c_rr -> half-strength pulses
               -> full range (the task of the evaluation)
    full       the full range from the first step (ablation: no curriculum)
    fixed      fixed c_rr and no pulses throughout (sanity check: easiest task)

``--realistic`` in ``train.py`` adds the realism parameters (observation
noise, actuator lag, action delay) to the last stage.

Parallel environments
---------------------
``VecEnv`` steps N independent ``BallOnTrayEnv`` instances in a plain Python
loop, so the networks act on a batch of N observations per call. It is written
here rather than taken from ``gym.vector`` for two reasons: the training loops
need the *true* last observation of an episode (and its ``info``, for the
asymmetric critic) separately from the observation after the automatic reset,
and every episode must get its own seed so that a run can be reproduced
exactly. Episode seeds are drawn from ``TRAIN_SEED_BASE`` upwards, disjoint
from the seeds used for evaluation (0-99 final, 500-519 during training) and
by ``networks/experiments.py`` (1000-2049).
"""

import csv
import os
import time

import numpy as np

from ball_on_tray_gym import BallOnTrayEnv
from demo import REALISTIC_KWARGS      # the same settings as demo.py --realistic

# ------------------------------------------------------------- task ranges
FIXED_FRICTION = (0.02, 0.02)           # mid-range c_rr [-], the environment default
FULL_FRICTION = (0.005, 0.05)           # full c_rr range [-]
HALF_DISTURBANCE = (0.5, 1.25)          # ||a_base|| [m/s^2], first half of the range
FULL_DISTURBANCE = (0.5, 2.0)           # ||a_base|| [m/s^2], full range

# Each stage: (name, fraction of the total steps, environment arguments).
CURRICULA = {
    "staged": (
        ("fixed",       0.10, {"friction_range": FIXED_FRICTION, "disturbance_range": None}),
        ("friction",    0.15, {"friction_range": FULL_FRICTION, "disturbance_range": None}),
        ("half-pulses", 0.15, {"friction_range": FULL_FRICTION,
                               "disturbance_range": HALF_DISTURBANCE}),
        ("full",        0.60, {"friction_range": FULL_FRICTION,
                               "disturbance_range": FULL_DISTURBANCE}),
    ),
    "full": (
        ("full",        1.00, {"friction_range": FULL_FRICTION,
                               "disturbance_range": FULL_DISTURBANCE}),
    ),
    "fixed": (
        ("fixed",       1.00, {"friction_range": FIXED_FRICTION, "disturbance_range": None}),
    ),
}

# ------------------------------------------------------------------- seeds
TRAIN_SEED_BASE = 100000                # training episode seeds start here
EVAL_SEEDS = tuple(range(500, 520))     # evaluation during training (model selection)
TEST_SEEDS = tuple(range(0, 100))       # final evaluation, as evaluate_baselines.py


class Curriculum(object):
    """Maps the training step to the active stage of a curriculum.

    Args:
        name: key of ``CURRICULA``.
        total_steps: total number of environment steps of the run.
        realistic: add ``REALISTIC_KWARGS`` to the last stage.
    """

    def __init__(self, name, total_steps, realistic=False):
        if name not in CURRICULA:
            raise ValueError("unknown curriculum {!r}; choose from {}".format(
                name, sorted(CURRICULA)))
        self.name = name
        self.stages = []
        start = 0
        stages = CURRICULA[name]
        for index, (stage_name, fraction, kwargs) in enumerate(stages):
            kwargs = dict(kwargs)
            if realistic and index == len(stages) - 1:
                kwargs.update(REALISTIC_KWARGS)
                stage_name += "+realistic"
            self.stages.append((start, stage_name, kwargs))
            start += int(round(fraction * total_steps))

    def stage_at(self, step):
        """Return (index, name, env_kwargs) of the stage active at ``step``."""
        index = 0
        for i, (start, _, _) in enumerate(self.stages):
            if step >= start:
                index = i
        _, name, kwargs = self.stages[index]
        return index, name, kwargs

    def final_env_kwargs(self):
        """Environment arguments of the last stage (the task being trained for)."""
        return dict(self.stages[-1][2])


class VecEnv(object):
    """N ``BallOnTrayEnv`` instances stepped together, with automatic reset.

    A finished episode is reset at once and its next episode uses the
    environment arguments of the current curriculum stage (``set_env_kwargs``),
    so stage changes happen at episode boundaries, never inside an episode.

    Args:
        num_envs: number of environments N.
        env_kwargs: arguments of ``BallOnTrayEnv`` for the first episodes.
        seed: seed of the episode-seed stream.
    """

    def __init__(self, num_envs, env_kwargs, seed):
        self.num_envs = int(num_envs)
        self._kwargs = dict(env_kwargs)
        self._seed_rng = np.random.RandomState(seed)
        self.envs = [None] * self.num_envs
        self._built_kwargs = [None] * self.num_envs   # arguments each env was built with
        self.obs = np.zeros((self.num_envs, BallOnTrayEnv.N_STACK * BallOnTrayEnv.FRAME_DIM),
                            dtype=np.float32)
        self.infos = [None] * self.num_envs
        self.episode_returns = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        for i in range(self.num_envs):
            self._reset(i)

    def set_env_kwargs(self, env_kwargs):
        """Use these environment arguments for every episode started from now on."""
        self._kwargs = dict(env_kwargs)

    def _next_seed(self):
        return TRAIN_SEED_BASE + int(self._seed_rng.randint(0, 2 ** 30))

    def _reset(self, i):
        """Start a new episode in environment i with the current arguments."""
        env = self.envs[i]
        if env is None or self._built_kwargs[i] != self._kwargs:
            if env is not None:
                env.close()
            env = BallOnTrayEnv(**self._kwargs)
            self.envs[i] = env
            self._built_kwargs[i] = dict(self._kwargs)
        obs, info = env.reset(seed=self._next_seed())
        self.obs[i] = obs
        self.infos[i] = info
        self.episode_returns[i] = 0.0
        self.episode_lengths[i] = 0

    def step(self, actions):
        """Step every environment and reset the ones whose episode ended.

        Args:
            actions: (N, 2) actions in [-1, 1].

        Returns:
            dict with, per environment i:
              "next_obs" (N, 18)    the true next observation (last observation
                                    of the episode if it ended at this step)
              "next_infos"          the info dicts that belong to ``next_obs``
              "rewards" (N,), "terminated" (N,), "truncated" (N,)
              "finished"            list of (return, length, fell) for every
                                    episode that ended at this step
            After the call, ``self.obs`` and ``self.infos`` hold the
            observation the policy acts on next (after any reset).
        """
        result = {"next_obs": np.zeros_like(self.obs), "next_infos": [None] * self.num_envs,
                  "rewards": np.zeros(self.num_envs), "finished": [],
                  "terminated": np.zeros(self.num_envs, dtype=bool),
                  "truncated": np.zeros(self.num_envs, dtype=bool)}
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            result["next_obs"][i] = obs
            result["next_infos"][i] = info
            result["rewards"][i] = reward
            result["terminated"][i] = terminated
            result["truncated"][i] = truncated
            self.episode_returns[i] += reward
            self.episode_lengths[i] += 1
            if terminated or truncated:
                result["finished"].append(
                    (self.episode_returns[i], int(self.episode_lengths[i]), bool(terminated)))
                self._reset(i)
            else:
                self.obs[i] = obs
                self.infos[i] = info
        return result

    def close(self):
        for env in self.envs:
            if env is not None:
                env.close()


def evaluate_policy(policy, seeds=EVAL_SEEDS, env_kwargs=None):
    """Run a deterministic policy (obs -> action) on fixed seeds.

    Args:
        policy: function mapping one observation (18,) to an action (2,).
        seeds: episode seeds.
        env_kwargs: environment arguments; default: the full task range.

    Returns:
        dict with "returns" (array), "mean", "std", "survival" (fraction of
        episodes that reached the time limit) and "goal" (mean fraction of
        steps with the ball inside the goal radius).
    """
    kwargs = {"friction_range": FULL_FRICTION, "disturbance_range": FULL_DISTURBANCE}
    kwargs.update(env_kwargs or {})
    env = BallOnTrayEnv(**kwargs)
    returns, survived, goal = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        total, in_goal, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            in_goal += int(np.linalg.norm(info["ball_pos"]) < env.GOAL_RADIUS)
        returns.append(total)
        survived.append(bool(truncated and not terminated))
        goal.append(in_goal / float(env.step_count))
    env.close()
    returns = np.asarray(returns)
    return {"returns": returns, "mean": float(returns.mean()), "std": float(returns.std()),
            "survival": float(np.mean(survived)), "goal": float(np.mean(goal))}


class ProgressLog(object):
    """Write one CSV row per log event and print a short line.

    Columns are fixed by the first row; later rows may leave columns empty.
    """

    def __init__(self, path, columns):
        self.path = path
        self.columns = list(columns)
        self.start = time.perf_counter()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", newline="") as handle:
            csv.writer(handle).writerow(self.columns)

    def elapsed(self):
        return time.perf_counter() - self.start

    def write(self, row):
        with open(self.path, "a", newline="") as handle:
            csv.writer(handle).writerow([_fmt(row.get(c, "")) for c in self.columns])


def _fmt(value):
    if isinstance(value, float):
        return "{:.6g}".format(value)
    return value


def recent_episode_stats(finished, window=20):
    """Mean return, mean length and fall rate of the last ``window`` episodes."""
    recent = finished[-window:]
    if not recent:
        return {}
    returns = [r for r, _, _ in recent]
    return {"train_return": float(np.mean(returns)),
            "train_length": float(np.mean([n for _, n, _ in recent])),
            "train_falls": float(np.mean([f for _, _, f in recent]))}
