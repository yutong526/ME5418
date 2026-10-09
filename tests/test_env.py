"""Tests for BallOnTrayEnv.

Run with ``python tests/test_env.py`` (no test framework required). The functions
are also named so that pytest can collect them if it is installed.
"""

import os
import sys
import warnings

import numpy as np
from gym.utils.env_checker import check_env

# Make the project root importable, wherever this file is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ball_on_tray_gym import BallOnTrayEnv  # noqa: E402

ZERO_ACTION = np.zeros(2, dtype=np.float32)
FULL_FRICTION = (0.005, 0.05)       # full curriculum range of c_rr [-]
FULL_DISTURBANCE = (0.5, 2.0)       # full curriculum range of |a_base| [m/s^2]


def _make_still_env(c_rr, tilt_deg, pos=(0.0, 0.0)):
    """Return a reset env with fixed c_rr, the given (pitch, roll) and ball position."""
    env = BallOnTrayEnv(friction_range=(c_rr, c_rr), disturbance_range=None)
    env.reset(seed=0)
    env.ball_pos = np.array(pos, dtype=np.float64)
    env.tilt = np.deg2rad(np.array(tilt_deg, dtype=np.float64))
    return env


# ------------------------------------------------------------------ 1. API
def test_check_env():
    """gym's own checker passes, for the default and the full-curriculum env."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(BallOnTrayEnv())
        check_env(BallOnTrayEnv(friction_range=FULL_FRICTION,
                                disturbance_range=FULL_DISTURBANCE))
    for w in caught:
        print("    check_env warning: {}".format(str(w.message).splitlines()[0]))


# -------------------------------------------------------- 2. random policy
def test_random_policy():
    """A random policy runs 5 episodes without errors."""
    env = BallOnTrayEnv(friction_range=FULL_FRICTION, disturbance_range=FULL_DISTURBANCE)
    env.action_space.seed(0)
    for episode in range(5):
        obs, info = env.reset(seed=episode)
        total_reward, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert obs.shape == (18,) and obs.dtype == np.float32
            assert np.all(np.isfinite(obs))
            assert np.isclose(sum(info["reward_terms"].values()), reward)
            total_reward += reward
        assert terminated != truncated
        print("    episode {}: steps = {:3d}, return = {:8.3f}, {}".format(
            episode, env.step_count, total_reward,
            "terminated (out of bounds)" if terminated else "truncated (time limit)"))


# ---------------------------------------------------------------- 3. physics
def test_ball_stays_still_on_level_tray():
    """Zero tilt, no disturbance: the ball does not move."""
    env = BallOnTrayEnv()
    env.reset(seed=1)
    start = env.ball_pos.copy()
    for _ in range(200):
        env.step(ZERO_ACTION)
    assert np.array_equal(env.ball_pos, start)
    assert np.array_equal(env.ball_vel, np.zeros(2))


def test_frictionless_acceleration():
    """c_rr = 0, tilt 5 deg: a = (5/7) * g * sin(5 deg) along the tilted axis."""
    a_expected = (5.0 / 7.0) * 9.81 * np.sin(np.deg2rad(5.0))     # [m/s^2]
    n_steps = 10
    t = n_steps * BallOnTrayEnv.DT                                 # [s]
    n_sub = n_steps * BallOnTrayEnv.N_SUBSTEPS
    h = BallOnTrayEnv.DT / BallOnTrayEnv.N_SUBSTEPS                # [s]

    for axis in (0, 1):     # 0: pitch -> +x, 1: roll -> +y
        tilt = [0.0, 0.0]
        tilt[axis] = 5.0
        env = _make_still_env(c_rr=0.0, tilt_deg=tilt)
        for _ in range(n_steps):
            env.step(ZERO_ACTION)
        a_measured = env.ball_vel[axis] / t
        assert np.isclose(a_measured, a_expected, rtol=1e-9), (a_measured, a_expected)
        assert env.ball_vel[1 - axis] == 0.0 and env.ball_pos[1 - axis] == 0.0
        # Semi-implicit Euler: x_n = a * h^2 * n * (n + 1) / 2
        assert np.isclose(env.ball_pos[axis], a_expected * h * h * n_sub * (n_sub + 1) / 2.0)
    print("    measured a = {:.6f} m/s^2, expected {:.6f} m/s^2".format(a_measured, a_expected))


def test_dead_zone():
    """c_rr = 0.05, tilt 1 deg: the ball stays at rest (inside the dead zone)."""
    env = _make_still_env(c_rr=0.05, tilt_deg=[1.0, 0.0], pos=(0.05, -0.03))
    for _ in range(200):
        env.step(ZERO_ACTION)
    assert np.array_equal(env.ball_pos, np.array([0.05, -0.03]))
    assert np.array_equal(env.ball_vel, np.zeros(2))


def test_rolling_resistance_outside_dead_zone():
    """c_rr = 0.05, tilt 5 deg: a = (5/7) * g * (sin(5 deg) - c_rr)."""
    a_expected = (5.0 / 7.0) * 9.81 * (np.sin(np.deg2rad(5.0)) - 0.05)
    env = _make_still_env(c_rr=0.05, tilt_deg=[5.0, 0.0])
    for _ in range(10):
        env.step(ZERO_ACTION)
    assert np.isclose(env.ball_vel[0] / (10 * env.DT), a_expected, rtol=1e-9)


def test_resistance_stops_ball_without_reversal():
    """On a level tray a moving ball decelerates to exactly zero and stays there."""
    env = _make_still_env(c_rr=0.05, tilt_deg=[0.0, 0.0], pos=(-0.1, 0.0))
    env.ball_vel = np.array([0.2, 0.0])
    for _ in range(100):
        env.step(ZERO_ACTION)
        assert env.ball_vel[0] >= 0.0
    assert np.array_equal(env.ball_vel, np.zeros(2))


def test_seed_reproducibility():
    """The same seed gives the same initial state and disturbance schedule."""
    env = BallOnTrayEnv(friction_range=FULL_FRICTION, disturbance_range=FULL_DISTURBANCE)

    def snapshot(seed):
        obs, _ = env.reset(seed=seed)
        plan = [(p["start"], p["duration"], tuple(p["accel"])) for p in env.disturbances]
        return obs.copy(), env.ball_pos.copy(), env.c_rr, plan

    obs_a, pos_a, c_rr_a, plan_a = snapshot(42)
    env.step(env.action_space.sample())
    obs_b, pos_b, c_rr_b, plan_b = snapshot(42)
    assert np.array_equal(obs_a, obs_b)
    assert np.array_equal(pos_a, pos_b)
    assert c_rr_a == c_rr_b
    assert plan_a == plan_b and len(plan_a) >= 1

    _, pos_c, _, _ = snapshot(43)
    assert not np.array_equal(pos_a, pos_c)


# ------------------------------------------------------------- extra checks
def test_disturbance_schedule():
    """Pulses respect the ranges, lie inside the episode and never overlap."""
    env = BallOnTrayEnv(disturbance_range=FULL_DISTURBANCE)
    episode_time = env.max_steps * env.DT
    for seed in range(200):
        env.reset(seed=seed)
        pulses = env.disturbances
        assert 1 <= len(pulses) <= 3
        for p in pulses:
            assert 0.2 <= p["duration"] <= 1.0
            assert 0.5 <= np.linalg.norm(p["accel"]) <= 2.0 + 1e-12
            assert 0.0 <= p["start"] and p["start"] + p["duration"] <= episode_time
        for first, second in zip(pulses[:-1], pulses[1:]):
            assert first["start"] + first["duration"] <= second["start"]

    # Episode shorter than the shortest pulse: pulses are dropped, no endless loop.
    short = BallOnTrayEnv(disturbance_range=FULL_DISTURBANCE, max_steps=5)
    short.reset(seed=0)
    assert short.disturbances == []

    # No disturbance at all when disturbance_range is None.
    env = BallOnTrayEnv()
    env.reset(seed=0)
    assert env.disturbances == []


def test_disturbance_pushes_ball():
    """a_base > 0 along +x pushes the ball towards -x: a = -(5/7) * a_base."""
    env = _make_still_env(c_rr=0.0, tilt_deg=[0.0, 0.0])
    env.disturbances = [{"start": 0.0, "duration": 1.0, "accel": np.array([1.0, 0.0])}]
    for _ in range(10):
        _, _, _, _, info = env.step(ZERO_ACTION)
    assert np.array_equal(info["a_base"], np.array([1.0, 0.0]))
    assert np.isclose(env.ball_vel[0] / (10 * env.DT), -5.0 / 7.0, rtol=1e-9)


def test_termination_and_truncation():
    """Out of bounds -> terminated with reward -10; time limit -> truncated."""
    env = BallOnTrayEnv()
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
    assert terminated and not truncated
    assert reward == -10.0 and info["out_of_bounds"]
    assert np.any(np.abs(env.ball_pos) > env.tray_half_size)

    env = BallOnTrayEnv(max_steps=20)
    env.reset(seed=0)
    for step in range(20):
        _, _, terminated, truncated, info = env.step(ZERO_ACTION)
        assert not terminated
        assert truncated == (step == 19)
    assert not info["out_of_bounds"]


def test_action_and_observation():
    """Action clipping, tilt limit, normalisation and frame stacking order."""
    env = BallOnTrayEnv()
    obs, _ = env.reset(seed=3)
    frame0 = obs[:6]
    assert np.array_equal(obs, np.tile(frame0, 3))
    assert np.allclose(frame0[:2], env.ball_pos / 0.15) and np.all(np.abs(frame0[:2]) <= 0.8)
    assert np.array_equal(frame0[2:], np.zeros(4))

    # An action outside [-1, 1] is clipped: one step changes the tilt by 3 deg.
    obs1, _, _, _, _ = env.step(np.array([5.0, -5.0]))
    assert np.allclose(np.rad2deg(env.tilt), [3.0, -3.0])
    assert np.array_equal(obs1[:6], frame0) and np.array_equal(obs1[6:12], frame0)
    assert np.allclose(obs1[16:], [0.2, -0.2])

    # Oldest frame first: the previous newest frame moves one slot to the left.
    obs2, _, _, _, _ = env.step(ZERO_ACTION)
    assert np.array_equal(obs2[:6], frame0) and np.array_equal(obs2[6:12], obs1[12:])

    # The accumulated tilt saturates at +/- 15 deg.
    env = _make_still_env(c_rr=0.02, tilt_deg=[0.0, 0.0])
    for _ in range(8):
        env.step(np.array([1.0, -1.0]))
        if np.any(np.abs(env.ball_pos) > env.tray_half_size):
            break
    assert np.all(np.abs(np.rad2deg(env.tilt)) <= 15.0 + 1e-9)


def test_reward_terms():
    """Reward formula, with a_{t-1} = 0 after reset."""
    env = _make_still_env(c_rr=0.05, tilt_deg=[0.0, 0.0], pos=(0.01, 0.0))
    action = np.array([0.1, 0.0])       # 0.3 deg: inside the dead zone, ball stays put
    _, reward, _, _, info = env.step(action)
    expected = (1.0 - 0.01 / (0.15 * np.sqrt(2.0))) + 0.5 - 0.05 * 0.01 - 0.01 * 0.01
    assert np.isclose(reward, expected)
    assert info["reward_terms"]["goal_bonus"] == 0.5

    # Same action again: the action-rate term vanishes.
    _, reward, _, _, info = env.step(action)
    assert info["reward_terms"]["action_rate"] == 0.0
    assert np.isclose(reward, expected + 0.05 * 0.01)


def test_render_rgb_array():
    """rgb_array mode returns an (H, W, 3) uint8 image that follows the state."""
    env = BallOnTrayEnv(disturbance_range=FULL_DISTURBANCE, render_mode="rgb_array")
    env.reset(seed=0)
    first = env.render()
    height = int(round(env.RENDER_FIGSIZE[1] * env.RENDER_DPI))
    width = int(round(env.RENDER_FIGSIZE[0] * env.RENDER_DPI))
    assert isinstance(first, np.ndarray)
    assert first.shape == (height, width, 3) and first.dtype == np.uint8

    figure = env._fig
    state = (env.ball_pos.copy(), env.ball_vel.copy(), env.tilt.copy())
    for _ in range(5):
        env.step(np.array([1.0, 0.5]))
    before = (env.ball_pos.copy(), env.ball_vel.copy(), env.tilt.copy())
    later = env.render()
    after = (env.ball_pos, env.ball_vel, env.tilt)
    assert later.shape == first.shape and later.dtype == np.uint8
    assert not np.array_equal(first, later)             # the picture changed
    assert env._fig is figure                           # the figure is reused
    assert all(np.array_equal(a, b) for a, b in zip(before, after))  # state untouched
    assert not np.array_equal(state[2], env.tilt)
    env.close()
    assert env._fig is None

    # Without a render mode, render() does nothing.
    env = BallOnTrayEnv()
    env.reset(seed=0)
    assert env.render() is None


if __name__ == "__main__":
    tests = [
        test_check_env,
        test_random_policy,
        test_ball_stays_still_on_level_tray,
        test_frictionless_acceleration,
        test_dead_zone,
        test_rolling_resistance_outside_dead_zone,
        test_resistance_stops_ball_without_reversal,
        test_seed_reproducibility,
        test_disturbance_schedule,
        test_disturbance_pushes_ball,
        test_termination_and_truncation,
        test_action_and_observation,
        test_reward_terms,
        test_render_rgb_array,
    ]
    for test in tests:
        print("[RUN ] {}".format(test.__name__))
        test()
        print("[ OK ] {}".format(test.__name__))
    print("\nAll {} tests passed.".format(len(tests)))
