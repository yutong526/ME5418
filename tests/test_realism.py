"""Tests for the optional realism parameters of BallOnTrayEnv.

Covers the observation noise (``pos_noise_std``, ``vel_noise_std``), the
first-order actuator lag (``actuator_tau``) and the action delay
(``action_delay_steps``), and checks against ``golden_v1.npz`` (next to this
file) that the default behaviour is the same as before these parameters existed.

Run with ``python tests/test_realism.py`` (no test framework required). The functions
are also named so that pytest can collect them if it is installed.
"""

import os
import sys
import warnings

import numpy as np
from gym.utils.env_checker import check_env

# Make the project root and this directory importable, wherever this file is
# run from. The snapshot path is resolved relative to make_golden.py itself.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from ball_on_tray_gym import BallOnTrayEnv  # noqa: E402
import make_golden  # noqa: E402  (tests/make_golden.py)

FULL = {"friction_range": (0.005, 0.05), "disturbance_range": (0.5, 2.0)}
GOLDEN_ATOL = 1e-9          # tolerance of the regression test (see its docstring)
ZERO_ACTION = np.zeros(2, dtype=np.float64)

HALF = BallOnTrayEnv.TRAY_HALF_SIZE                     # position scale [m]
VEL_SCALE = BallOnTrayEnv.VEL_SCALE                     # velocity scale [m/s]
MAX_TILT = np.deg2rad(BallOnTrayEnv.MAX_TILT_DEG)       # tilt scale [rad]
TILT_STEP = np.deg2rad(BallOnTrayEnv.TILT_STEP_DEG)     # tilt per unit action [rad]


def _fixed_actions(n_steps, scale, seed=0):
    """Reproducible action sequence of shape (n_steps, 2), independent of the env."""
    return scale * np.random.RandomState(seed).uniform(-1.0, 1.0, size=(n_steps, 2))


# ------------------------------------------------------------- 1. regression
def test_default_behaviour_matches_golden_snapshot():
    """With default realism parameters the environment reproduces the snapshot.

    The snapshot was recorded with the code from before the realism parameters
    were added. Floating-point fields are compared with atol = 1e-9 so that a
    different machine or numpy build cannot cause a false alarm; flags and
    counters must match exactly. (On the development machine all fields were
    also verified to be bit-for-bit identical.)
    """
    golden = np.load(make_golden.GOLDEN_PATH)
    current = make_golden.record_all()
    assert set(golden.files) == set(current.keys())
    for key in golden.files:
        expected, actual = golden[key], current[key]
        assert expected.shape == actual.shape and expected.dtype == actual.dtype, key
        if expected.dtype.kind == "f":
            assert np.allclose(expected, actual, rtol=0.0, atol=GOLDEN_ATOL,
                               equal_nan=True), key
        else:
            assert np.array_equal(expected, actual), key
    # The snapshot contains every kind of step: on the tray, off the tray,
    # under disturbance and at the time limit.
    assert golden["full/terminated"].any() and golden["truncation/truncated"].any()
    print("    compared {} arrays".format(len(golden.files)))


# -------------------------------------------- 2. noise leaves the world alone
def test_noise_does_not_change_the_episode():
    """Noise settings change neither the sampled episode nor the true trajectory."""
    actions = _fixed_actions(120, 0.4)

    def run(seed, **noise):
        env = BallOnTrayEnv(**dict(FULL, **noise))
        log = []
        for k in range(3):                                  # 3 consecutive resets
            obs, info = env.reset(seed=seed) if k == 0 else env.reset()
            plan = [(p["start"], p["duration"], tuple(p["accel"])) for p in env.disturbances]
            episode = {"obs0": obs, "pos0": info["ball_pos"], "c_rr": info["c_rr"],
                       "plan": plan, "obs": [], "state": [], "reward": []}
            for action in actions:
                obs, reward, _, _, info = env.step(action)
                episode["obs"].append(obs)
                episode["state"].append(np.concatenate(
                    [info["ball_pos"], info["ball_vel"], info["tilt"], info["a_base"]]))
                episode["reward"].append(reward)
            log.append(episode)
        return log

    for seed in (0, 7):
        clean = run(seed)
        for noise in ({"pos_noise_std": 0.002, "vel_noise_std": 0.03},
                      {"pos_noise_std": 0.01},
                      {"vel_noise_std": 0.2}):
            noisy = run(seed, **noise)
            for a, b in zip(clean, noisy):
                # Same initial state and disturbance plan, also after resets
                # that continue the random stream.
                assert np.array_equal(a["pos0"], b["pos0"])
                assert a["c_rr"] == b["c_rr"] and a["plan"] == b["plan"]
                # Same true trajectory and reward under the same actions ...
                assert np.array_equal(np.array(a["state"]), np.array(b["state"]))
                assert np.array_equal(np.array(a["reward"]), np.array(b["reward"]))
                # ... but a different observation.
                assert not np.array_equal(a["obs0"], b["obs0"])
                assert not np.array_equal(np.array(a["obs"]), np.array(b["obs"]))

    # The noise itself is reproducible for a given seed.
    first = run(3, pos_noise_std=0.002, vel_noise_std=0.03)
    second = run(3, pos_noise_std=0.002, vel_noise_std=0.03)
    assert all(np.array_equal(np.array(a["obs"]), np.array(b["obs"]))
               for a, b in zip(first, second))


# --------------------------------------------------------- 3. noise statistics
def test_noise_statistics():
    """Noise has the configured standard deviation, zero mean, one draw per frame."""
    pos_std, vel_std, n_steps = 0.002, 0.03, 6000
    env = BallOnTrayEnv(max_steps=n_steps, pos_noise_std=pos_std, vel_noise_std=vel_std)
    obs, info = env.reset(seed=11)
    true_pos = info["ball_pos"].copy()

    # The noisy initial frame fills the whole stack (one draw, three copies).
    assert np.array_equal(obs[0:6], obs[6:12]) and np.array_equal(obs[6:12], obs[12:18])
    assert not np.allclose(obs[12:14] * HALF, true_pos, atol=1e-6)

    pos_err, vel_err = [], []
    for _ in range(n_steps):                # level tray, zero action: the ball rests
        previous = obs
        obs, _, terminated, _, info = env.step(ZERO_ACTION)
        assert not terminated
        assert np.array_equal(info["ball_pos"], true_pos)           # true state unchanged
        assert np.array_equal(info["ball_vel"], np.zeros(2))
        # Older frames are shifted, not disturbed again.
        assert np.array_equal(obs[0:12], previous[6:18])
        pos_err.append(obs[12:14].astype(np.float64) * HALF - true_pos)
        vel_err.append(obs[14:16].astype(np.float64) * VEL_SCALE)
        assert np.array_equal(obs[16:18], np.zeros(2))              # tilt is not noisy
    pos_err, vel_err = np.array(pos_err), np.array(vel_err)

    n = pos_err.size
    assert abs(pos_err.std() / pos_std - 1.0) < 0.05
    assert abs(vel_err.std() / vel_std - 1.0) < 0.05
    assert abs(pos_err.mean()) < 4.0 * pos_std / np.sqrt(n)
    assert abs(vel_err.mean()) < 4.0 * vel_std / np.sqrt(n)
    # The two axes are disturbed independently.
    assert abs(np.corrcoef(pos_err[:, 0], pos_err[:, 1])[0, 1]) < 0.05
    print("    {} samples: pos std {:.5f} m (set {:.5f}), vel std {:.5f} m/s (set {:.5f})".format(
        n, pos_err.std(), pos_std, vel_err.std(), vel_std))

    # Only one of the two noises switched on.
    env = BallOnTrayEnv(vel_noise_std=vel_std)
    obs, info = env.reset(seed=1)
    assert np.allclose(obs[12:14] * HALF, info["ball_pos"], atol=1e-7)
    assert not np.array_equal(obs[14:16], np.zeros(2))

    # A reset without a seed works as well.
    env = BallOnTrayEnv(pos_noise_std=pos_std)
    obs, _ = env.reset()
    assert obs.shape == (18,) and np.all(np.isfinite(obs))


# ------------------------------------------------------------ 4. actuator lag
def test_actuator_lag():
    """Step response of the first-order lag, and the ideal case tau = 0."""
    for tau in (0.1, 0.06):
        env = BallOnTrayEnv(actuator_tau=tau)
        env.reset(seed=0)
        n_steps = int(round(tau / env.DT))              # tau is a multiple of DT here
        # Command step to (+3, -3) deg in the first step, then hold.
        _, _, _, _, info = env.step(np.array([1.0, -1.0]))
        assert np.allclose(np.rad2deg(info["tilt_cmd"]), [3.0, -3.0])
        for _ in range(n_steps - 1):
            obs, _, _, _, info = env.step(ZERO_ACTION)
        ratio = info["tilt"] / info["tilt_cmd"]
        assert np.allclose(ratio, 1.0 - np.exp(-1.0), atol=0.01), ratio      # 63.2 %
        assert np.array_equal(info["tilt"], env.tilt)
        assert np.array_equal(info["tilt_cmd"], env.tilt_cmd)
        # The observation shows the actual tilt, not the command.
        assert np.allclose(obs[16:18], env.tilt / MAX_TILT, atol=1e-6)
        # Later the actual tilt converges to the command.
        for _ in range(int(round(10.0 * tau / env.DT))):
            env.step(ZERO_ACTION)
        assert np.allclose(env.tilt, env.tilt_cmd, rtol=1e-3)
    print("    tilt / command after tau: {:.4f} (expected {:.4f})".format(
        ratio[0], 1.0 - np.exp(-1.0)))

    # tau = 0: the actual tilt equals the command immediately.
    env = BallOnTrayEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([1.0, -1.0]))
    assert np.allclose(np.rad2deg(info["tilt"]), [3.0, -3.0])
    assert np.array_equal(info["tilt"], info["tilt_cmd"])

    # Actual and commanded tilt always stay within +/- 15 deg.
    env = BallOnTrayEnv(actuator_tau=0.05, max_steps=400)
    env.reset(seed=1)
    limit = MAX_TILT + 1e-12
    saturated = False
    for action in _fixed_actions(400, 3.0, seed=5):
        env.step(action)
        assert np.all(np.abs(env.tilt) <= limit) and np.all(np.abs(env.tilt_cmd) <= limit)
        # The lag never overshoots: the actual tilt is behind the command.
        saturated = saturated or bool(np.any(np.abs(env.tilt_cmd) >= MAX_TILT - 1e-12))
    assert saturated                                    # the limit was actually reached


# ------------------------------------------------------------ 5. action delay
def test_action_delay():
    """An action given at step t changes the commanded tilt at step t + k."""
    actions = _fixed_actions(30, 0.3, seed=2)           # small: the tilt limit is not hit
    for delay in (0, 1, 2, 3):
        env = BallOnTrayEnv(action_delay_steps=delay)
        env.reset(seed=0)
        applied_sum = np.zeros(2)
        for t, action in enumerate(actions):            # t = 0 is the first step
            _, _, _, _, info = env.step(action)
            if t >= delay:
                applied_sum = applied_sum + actions[t - delay]
            # Commanded tilt = sum of the actions that have taken effect so far.
            assert np.allclose(info["tilt_cmd"], applied_sum * TILT_STEP, atol=1e-12)
            assert np.array_equal(info["tilt"], info["tilt_cmd"])       # no lag here
            if t < delay:
                assert np.array_equal(info["tilt"], np.zeros(2))       # nothing arrived yet
            elif t == delay:
                assert np.allclose(info["tilt"], actions[0] * TILT_STEP, atol=1e-12)
            # The reward penalises the action the agent gave, not the delayed one.
            expected = -env.ACTION_WEIGHT * float(np.sum(action ** 2))
            assert np.isclose(info["reward_terms"]["action_magnitude"], expected)
            previous = actions[t - 1] if t > 0 else np.zeros(2)
            expected = -env.ACTION_RATE_WEIGHT * float(np.sum((action - previous) ** 2))
            assert np.isclose(info["reward_terms"]["action_rate"], expected)

        # A reset empties the queue: pending actions of the old episode are dropped.
        env.reset(seed=0)
        for _ in range(max(delay, 1)):
            _, _, _, _, info = env.step(ZERO_ACTION)
            assert np.array_equal(info["tilt_cmd"], np.zeros(2))

    # Delay and lag together: queue -> command -> lag -> actual tilt.
    env = BallOnTrayEnv(action_delay_steps=1, actuator_tau=0.1)
    env.reset(seed=0)
    env.step(np.array([1.0, 0.0]))
    assert np.array_equal(env.tilt_cmd, np.zeros(2)) and np.array_equal(env.tilt, np.zeros(2))
    env.step(ZERO_ACTION)
    assert np.isclose(np.rad2deg(env.tilt_cmd[0]), 3.0)
    assert 0.0 < env.tilt[0] < env.tilt_cmd[0]


# ------------------------------------- 6. reward and termination use the truth
def test_reward_and_termination_ignore_noise():
    """Even with huge noise, reward and out-of-bounds check use the true position."""
    fooled = 0
    for seed in range(5):
        clean = BallOnTrayEnv(**FULL)
        noisy = BallOnTrayEnv(pos_noise_std=0.05, vel_noise_std=0.5, **FULL)
        clean.reset(seed=seed)
        noisy.reset(seed=seed)
        for action in _fixed_actions(500, 0.5, seed=seed):
            _, reward_c, term_c, trunc_c, info_c = clean.step(action)
            obs_n, reward_n, term_n, trunc_n, info_n = noisy.step(action)
            assert reward_c == reward_n
            assert term_c == term_n and trunc_c == trunc_n
            assert info_c["out_of_bounds"] == info_n["out_of_bounds"]
            assert np.array_equal(info_c["ball_pos"], info_n["ball_pos"])
            assert info_c["reward_terms"] == info_n["reward_terms"]
            # Steps where the measured position is off the tray but the ball is not.
            if not term_n and np.any(np.abs(obs_n[12:14]) > 1.0):
                fooled += 1
            if term_c or trunc_c:
                break
    assert fooled > 0       # the case that matters actually occurred
    print("    {} steps with a measured position off the tray while the ball was on it".format(
        fooled))


# ----------------------------------------------------------------- 7. gym API
def test_check_env_with_realism_parameters():
    """gym's checker passes for every combination of the new parameters."""
    settings = [
        {"pos_noise_std": 0.002, "vel_noise_std": 0.03},
        {"actuator_tau": 0.05},
        {"actuator_tau": 0.10},
        {"action_delay_steps": 1},
        {"action_delay_steps": 2},
        {"pos_noise_std": 0.002, "vel_noise_std": 0.03, "actuator_tau": 0.05,
         "action_delay_steps": 1},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kwargs in settings:
            check_env(BallOnTrayEnv(**kwargs))
            env = BallOnTrayEnv(**dict(FULL, **kwargs))
            check_env(env)
            assert env.observation_space.shape == (18,)     # the observation size is unchanged


def test_invalid_parameters_are_rejected():
    """Negative or non-integer settings raise ValueError."""
    for kwargs in ({"pos_noise_std": -0.001}, {"vel_noise_std": -1.0}, {"actuator_tau": -0.1},
                   {"action_delay_steps": -1}, {"action_delay_steps": 1.5},
                   {"action_delay_steps": True}):
        try:
            BallOnTrayEnv(**kwargs)
        except ValueError:
            continue
        raise AssertionError("no ValueError for {}".format(kwargs))


if __name__ == "__main__":
    tests = [
        test_default_behaviour_matches_golden_snapshot,
        test_noise_does_not_change_the_episode,
        test_noise_statistics,
        test_actuator_lag,
        test_action_delay,
        test_reward_and_termination_ignore_noise,
        test_check_env_with_realism_parameters,
        test_invalid_parameters_are_rejected,
    ]
    for test in tests:
        print("[RUN ] {}".format(test.__name__))
        test()
        print("[ OK ] {}".format(test.__name__))
    print("\nAll {} tests passed.".format(len(tests)))
