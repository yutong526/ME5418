"""Record a golden trajectory snapshot of BallOnTrayEnv.

The snapshot pins down the behaviour of the environment before the optional
realism parameters were added. ``test_realism.py`` replays the same seeds and
actions and compares the result with the stored file, so any change to the
default behaviour is detected.

IMPORTANT: run this script only on the code from before the realism changes
(commit 46fcb1f, tag ``pre-realism``). Do not regenerate the snapshot after
that: a file produced by later code would only compare the code with itself
and the regression test would no longer prove anything.

What is recorded:
    * two trajectory configurations:
        "default"  BallOnTrayEnv()
        "full"     friction_range=(0.005, 0.05), disturbance_range=(0.5, 2.0)
      with seeds 0..9. For each seed the environment is reset 3 times in a
      row, the first time with ``reset(seed=seed)`` and then twice without a
      seed, so the continuation of the random stream is covered as well.
      After every reset, 200 steps follow with a fixed action sequence drawn
      from an independent generator (not the environment's). Stepping
      continues after the ball has left the tray, so every episode has exactly
      200 steps.
    * one time-limit configuration:
        "truncation"  BallOnTrayEnv(max_steps=500), all-zero actions
      with seeds 0..4, 3 consecutive resets each and the full 500 steps. The
      ball stays at rest, so ``truncated`` becomes True at step 500, the path
      the two configurations above never reach.
    * per step: obs, reward, terminated, truncated and the numeric info fields;
      per reset: the initial obs, c_rr, ball position and the disturbance plan

Usage (from the project root):
    python tests/make_golden.py             # writes tests/golden_v1.npz
    python tests/make_golden.py --force     # overwrite an existing snapshot

Only regenerate the file on purpose: it must come from the code version whose
behaviour is to be preserved.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ball_on_tray_gym import BallOnTrayEnv  # noqa: E402

GOLDEN_PATH = os.path.join(HERE, "golden_v1.npz")

SEEDS = tuple(range(10))
N_RESETS = 3                    # consecutive resets per seed
N_STEPS = 200                   # steps after every reset
MAX_PULSES = 3                  # the disturbance plan is padded to this length
# Action amplitude per reset index. 1.5 exceeds [-1, 1] and exercises the
# clipping. The last sequence is a bounded walk (see make_actions), which keeps
# the tilt small and the ball on the tray for longer.
ACTION_SCALES = (1.5, 0.5, 0.5)
BOUNDED_RESET = 2               # reset index that uses the bounded-walk sequence
ACTION_SEED_BASE = 20261009     # seed offset of the independent action generator

CONFIGS = (
    ("default", {}),
    ("full", {"friction_range": (0.005, 0.05), "disturbance_range": (0.5, 2.0)}),
)

# Time-limit configuration: zero actions until the episode is truncated.
TRUNCATION_NAME = "truncation"
TRUNCATION_KWARGS = {"max_steps": 500}
TRUNCATION_SEEDS = tuple(range(5))
TRUNCATION_STEPS = 500

REWARD_TERM_KEYS = ("distance", "goal_bonus", "action_rate",
                    "action_magnitude", "out_of_bounds")


def make_actions():
    """Fixed action sequences, shape (seeds, resets, steps, 2), float64.

    Drawn from ``np.random.RandomState`` (a stream that is stable across numpy
    versions), independent of the environment's generator. The same actions
    are used for both configurations. Per reset index: large uniform actions
    (clipped by the environment), medium uniform actions, and a bounded walk
    that keeps the tilt within 3 deg.
    """
    actions = np.zeros((len(SEEDS), N_RESETS, N_STEPS, 2), dtype=np.float64)
    for i, seed in enumerate(SEEDS):
        for j in range(N_RESETS):
            rng = np.random.RandomState(ACTION_SEED_BASE + 100 * seed + j)
            noise = ACTION_SCALES[j] * rng.uniform(-1.0, 1.0, size=(N_STEPS, 2))
            if j == BOUNDED_RESET:
                # w_t = clip(w_{t-1} + noise_t, -1, 1), a_t = w_t - w_{t-1}:
                # the tilt is then w_t times the tilt step, i.e. it stays
                # within one tilt step (3 deg) of level.
                walk = np.zeros(2)
                for t in range(N_STEPS):
                    new_walk = np.clip(walk + noise[t], -1.0, 1.0)
                    noise[t] = new_walk - walk
                    walk = new_walk
            actions[i, j] = noise
    return actions


def plan_array(env):
    """Disturbance plan as a (MAX_PULSES, 4) array [start, duration, ax, ay], NaN padded."""
    plan = np.full((MAX_PULSES, 4), np.nan)
    for k, pulse in enumerate(env.disturbances):
        plan[k] = [pulse["start"], pulse["duration"], pulse["accel"][0], pulse["accel"][1]]
    return plan


def make_truncation_actions():
    """All-zero actions for the time-limit configuration, (seeds, resets, steps, 2)."""
    return np.zeros((len(TRUNCATION_SEEDS), N_RESETS, TRUNCATION_STEPS, 2), dtype=np.float64)


def record(make_env, actions, seeds=SEEDS):
    """Run the snapshot protocol on one configuration.

    Args:
        make_env: function without arguments that builds the environment.
        actions: action sequences of shape (seeds, resets, steps, 2).
        seeds: the seeds, one per first index of ``actions``.

    Returns:
        dict of arrays, each with leading shape (seeds, resets[, steps]).
    """
    n_seeds = len(seeds)
    n_steps = actions.shape[2]
    lead = (n_seeds, N_RESETS, n_steps)
    data = {
        "obs": np.zeros(lead + (18,), dtype=np.float32),
        "reward": np.zeros(lead, dtype=np.float64),
        "terminated": np.zeros(lead, dtype=np.bool_),
        "truncated": np.zeros(lead, dtype=np.bool_),
        "info_c_rr": np.zeros(lead, dtype=np.float64),
        "info_a_base": np.zeros(lead + (2,), dtype=np.float64),
        "info_ball_pos": np.zeros(lead + (2,), dtype=np.float64),
        "info_ball_vel": np.zeros(lead + (2,), dtype=np.float64),
        "info_tilt": np.zeros(lead + (2,), dtype=np.float64),
        "info_step_count": np.zeros(lead, dtype=np.int64),
        "info_reward_terms": np.zeros(lead + (len(REWARD_TERM_KEYS),), dtype=np.float64),
        "info_out_of_bounds": np.zeros(lead, dtype=np.bool_),
        "reset_obs": np.zeros((n_seeds, N_RESETS, 18), dtype=np.float32),
        "reset_c_rr": np.zeros((n_seeds, N_RESETS), dtype=np.float64),
        "reset_ball_pos": np.zeros((n_seeds, N_RESETS, 2), dtype=np.float64),
        "reset_plan": np.zeros((n_seeds, N_RESETS, MAX_PULSES, 4), dtype=np.float64),
        "reset_n_pulses": np.zeros((n_seeds, N_RESETS), dtype=np.int64),
    }

    for i, seed in enumerate(seeds):
        env = make_env()
        for j in range(N_RESETS):
            # Only the first reset is seeded; the next two continue the stream.
            obs, info = env.reset(seed=seed) if j == 0 else env.reset()
            data["reset_obs"][i, j] = obs
            data["reset_c_rr"][i, j] = info["c_rr"]
            data["reset_ball_pos"][i, j] = info["ball_pos"]
            data["reset_plan"][i, j] = plan_array(env)
            data["reset_n_pulses"][i, j] = len(env.disturbances)

            for t in range(n_steps):
                obs, reward, terminated, truncated, info = env.step(actions[i, j, t])
                data["obs"][i, j, t] = obs
                data["reward"][i, j, t] = reward
                data["terminated"][i, j, t] = terminated
                data["truncated"][i, j, t] = truncated
                data["info_c_rr"][i, j, t] = info["c_rr"]
                data["info_a_base"][i, j, t] = info["a_base"]
                data["info_ball_pos"][i, j, t] = info["ball_pos"]
                data["info_ball_vel"][i, j, t] = info["ball_vel"]
                data["info_tilt"][i, j, t] = info["tilt"]
                data["info_step_count"][i, j, t] = info["step_count"]
                data["info_reward_terms"][i, j, t] = [
                    info["reward_terms"][key] for key in REWARD_TERM_KEYS]
                data["info_out_of_bounds"][i, j, t] = info["out_of_bounds"]
        env.close()
    return data


def record_all():
    """Record every configuration; keys are '<config>/<field>' plus 'actions'."""
    actions = make_actions()
    snapshot = {"actions": actions}
    for name, kwargs in CONFIGS:
        data = record(lambda kwargs=kwargs: BallOnTrayEnv(**kwargs), actions)
        for key, value in data.items():
            snapshot["{}/{}".format(name, key)] = value

    data = record(lambda: BallOnTrayEnv(**TRUNCATION_KWARGS),
                  make_truncation_actions(), TRUNCATION_SEEDS)
    for key, value in data.items():
        snapshot["{}/{}".format(TRUNCATION_NAME, key)] = value
    return snapshot


def main():
    """Write the snapshot file and print what it covers."""
    parser = argparse.ArgumentParser(description="Record the golden trajectory snapshot.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing snapshot file")
    args = parser.parse_args()

    if os.path.exists(GOLDEN_PATH) and not args.force:
        print("{} already exists; use --force to overwrite it.".format(GOLDEN_PATH))
        sys.exit(1)

    snapshot = record_all()
    np.savez_compressed(GOLDEN_PATH, **snapshot)

    print("saved {} ({} arrays, {:.0f} kB)".format(
        GOLDEN_PATH, len(snapshot), os.path.getsize(GOLDEN_PATH) / 1024.0))
    print("python {}, numpy {}".format(sys.version.split()[0], np.__version__))
    for name, _ in CONFIGS:
        terminated = snapshot[name + "/terminated"]
        on_tray = ~snapshot[name + "/info_out_of_bounds"]
        active = np.any(snapshot[name + "/info_a_base"] != 0.0, axis=-1)
        print("{:<8}: {} episodes x {} steps, {} steps on the tray, "
              "{} episodes left the tray, {} steps with an active disturbance "
              "({} of them on the tray), c_rr {:.4f}..{:.4f}".format(
                  name, terminated.shape[0] * terminated.shape[1], N_STEPS,
                  int(on_tray.sum()), int(np.any(terminated, axis=-1).sum()),
                  int(active.sum()), int((active & on_tray).sum()),
                  snapshot[name + "/reset_c_rr"].min(), snapshot[name + "/reset_c_rr"].max()))
    truncated = snapshot[TRUNCATION_NAME + "/truncated"]
    print("{:<8}: {} episodes x {} steps, truncated only at the last step in {} episodes, "
          "terminated steps {}".format(
              TRUNCATION_NAME, truncated.shape[0] * truncated.shape[1], truncated.shape[2],
              int((truncated[..., -1] & ~np.any(truncated[..., :-1], axis=-1)).sum()),
              int(snapshot[TRUNCATION_NAME + "/terminated"].sum())))


if __name__ == "__main__":
    main()
