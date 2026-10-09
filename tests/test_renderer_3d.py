"""Tests for TrayRenderer3D.

Run with ``python tests/test_renderer_3d.py`` (no test framework required). The
functions are also named so that pytest can collect them if it is installed.
"""

import os
import sys

import numpy as np

# Make the project root importable, wherever this file is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ball_on_tray_gym import BallOnTrayEnv, TrayRenderer3D  # noqa: E402

FULL_FRICTION = (0.005, 0.05)       # full range of c_rr [-]
FULL_DISTURBANCE = (0.5, 2.0)       # full range of |a_base| [m/s^2]


def _make(seed=0):
    """Return a reset environment (no built-in rendering) and its 3D renderer."""
    env = BallOnTrayEnv(friction_range=FULL_FRICTION, disturbance_range=FULL_DISTURBANCE,
                        render_mode=None)
    env.reset(seed=seed)
    return env, TrayRenderer3D(env, mode="rgb_array")


def test_image_shape_and_type():
    """rgb_array mode returns an (H, W, 3) uint8 image that follows the state."""
    env, renderer = _make()
    height = int(round(renderer.FIGSIZE[1] * renderer.DPI))
    width = int(round(renderer.FIGSIZE[0] * renderer.DPI))

    first = renderer.render(0.0, 0.0)
    assert isinstance(first, np.ndarray)
    assert first.shape == (height, width, 3) and first.dtype == np.uint8

    figure, artists = renderer._fig, dict(renderer._artists)
    total = 0.0
    for _ in range(5):
        _, reward, _, _, _ = env.step(np.array([1.0, 0.5]))
        total += reward
    later = renderer.render(reward, total)
    assert later.shape == first.shape and later.dtype == np.uint8
    assert not np.array_equal(first, later)                     # the picture changed
    # Figure and artists are created once and only updated afterwards.
    assert renderer._fig is figure
    assert all(renderer._artists[name] is artist for name, artist in artists.items())

    renderer.close()
    assert renderer._fig is None


def test_label_is_shown_on_the_picture():
    """The caption is empty by default and changes the picture when set."""
    env, renderer = _make()
    assert renderer.label == ""
    plain = renderer.render(0.0, 0.0)
    assert np.array_equal(plain, renderer.render(0.0, 0.0))    # rendering is repeatable

    artists = dict(renderer._artists)
    renderer.set_label("Policy: PD controller")
    labelled = renderer.render(0.0, 0.0)
    assert labelled.shape == plain.shape
    changed = np.any(labelled != plain, axis=2)
    assert changed.any()
    # The caption sits in the top-left corner and touches nothing else.
    rows, cols = np.nonzero(changed)
    assert rows.max() < 0.15 * plain.shape[0] and cols.max() < 0.6 * plain.shape[1]
    assert all(renderer._artists[name] is artist for name, artist in artists.items())

    renderer.set_label("")
    assert np.array_equal(plain, renderer.render(0.0, 0.0))    # removed again
    renderer.close()


def test_rotation_is_proper():
    """R is a rotation matrix (orthonormal, det +1) and the identity at zero tilt."""
    assert np.allclose(TrayRenderer3D.tray_rotation(0.0, 0.0), np.eye(3))
    for pitch_deg, roll_deg in [(15, 0), (0, -15), (15, 15), (-7, 12), (3, -3)]:
        rot = TrayRenderer3D.tray_rotation(np.deg2rad(pitch_deg), np.deg2rad(roll_deg))
        assert np.allclose(rot.T.dot(rot), np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rot), 1.0)


def test_lower_side_is_where_the_ball_accelerates():
    """Positive pitch lowers the +x side of the tray, positive roll the +y side."""
    env, renderer = _make()
    half = env.tray_half_size
    plus_x, minus_x = [half, 0.0, 0.0], [-half, 0.0, 0.0]
    plus_y, minus_y = [0.0, half, 0.0], [0.0, -half, 0.0]

    def height(point):
        return renderer.tray_to_world(point)[2]

    env.tilt = np.deg2rad([10.0, 0.0])              # pitch > 0
    assert height(plus_x) < 0.0 < height(minus_x)
    assert np.isclose(height(plus_y), 0.0) and np.isclose(height(minus_y), 0.0)

    env.tilt = np.deg2rad([-10.0, 0.0])             # pitch < 0
    assert height(plus_x) > 0.0 > height(minus_x)

    env.tilt = np.deg2rad([0.0, 10.0])              # roll > 0
    assert height(plus_y) < 0.0 < height(minus_y)
    assert np.isclose(height(plus_x), 0.0) and np.isclose(height(minus_x), 0.0)

    env.tilt = np.deg2rad([0.0, -10.0])             # roll < 0
    assert height(plus_y) > 0.0 > height(minus_y)


def test_slope_matches_env_dynamics():
    """Gravity along the drawn tray axes equals g*sin(pitch) and g*sin(roll).

    This is the term that drives the ball in the environment, also when pitch
    and roll are both non-zero, and the angles are drawn to scale.
    """
    env, renderer = _make()
    gravity = np.array([0.0, 0.0, -env.GRAVITY])
    for pitch_deg, roll_deg in [(5, 0), (0, 5), (15, 15), (-12, 4), (8, -15)]:
        env.tilt = np.deg2rad([float(pitch_deg), float(roll_deg)])
        rot = renderer.rotation()
        x_axis, y_axis = rot[:, 0], rot[:, 1]
        assert np.isclose(gravity.dot(x_axis), env.GRAVITY * np.sin(env.tilt[0]))
        assert np.isclose(gravity.dot(y_axis), env.GRAVITY * np.sin(env.tilt[1]))

        # Same sign as the acceleration the environment actually produces.
        env.a_base = np.zeros(2)
        drive = env._drive_acceleration()
        assert np.allclose(drive, env.ROLLING_FACTOR * np.array(
            [gravity.dot(x_axis), gravity.dot(y_axis)]))

    # A single-axis tilt is drawn with the true angle (no exaggeration).
    env.tilt = np.deg2rad([15.0, 0.0])
    normal = renderer.rotation()[:, 2]
    assert np.isclose(np.rad2deg(np.arccos(normal[2])), 15.0)


def test_ball_sits_on_the_tray_surface():
    """The ball centre is always exactly one radius away from the tray plane."""
    env, renderer = _make(seed=4)
    env.action_space.seed(4)
    radius = renderer.ball_radius
    checked = 0
    for episode in range(3):
        env.reset(seed=episode)
        terminated = truncated = False
        while not (terminated or truncated) and env.step_count < 150:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            rot = renderer.rotation()
            normal = rot[:, 2]
            centre = renderer.ball_center_world()
            # The tray surface is the plane through the pivot (origin) with this normal.
            assert np.isclose(normal.dot(centre), radius, atol=1e-12)
            # The contact point is the tray-frame point (x, y, 0) on the surface.
            contact = centre - radius * normal
            assert np.isclose(normal.dot(contact), 0.0, atol=1e-12)
            assert np.allclose(rot.T.dot(contact)[:2], env.ball_pos, atol=1e-12)
            checked += 1
    assert checked > 30

    # The drawn ball mesh agrees: all vertices lie on the sphere around that
    # centre and none of them is below the tray surface.
    env.reset(seed=1)
    env.tilt = np.deg2rad([12.0, -9.0])
    renderer.render(0.0, 0.0)
    normal = renderer.rotation()[:, 2]
    centre = renderer.ball_center_world()
    vertices = renderer._unit_sphere.reshape(-1, 3) * radius + centre
    assert np.allclose(np.linalg.norm(vertices - centre, axis=1), radius)
    assert np.min(vertices.dot(normal)) >= -1e-12
    renderer.close()


def test_renderer_does_not_change_the_environment():
    """Rendering only reads the environment."""
    env, renderer = _make(seed=2)
    for _ in range(10):
        env.step(np.array([0.3, -0.6]))
    before = (env.ball_pos.copy(), env.ball_vel.copy(), env.tilt.copy(), env.a_base.copy(),
              env.c_rr, env.step_count, env.last_reward, env.episode_return)
    obs_before = env._get_obs()
    for _ in range(3):
        renderer.render(1.0, 2.0)
    after = (env.ball_pos, env.ball_vel, env.tilt, env.a_base,
             env.c_rr, env.step_count, env.last_reward, env.episode_return)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert np.array_equal(obs_before, env._get_obs())
    assert env.render_mode is None and env._renderer is None   # the env made no renderer
    renderer.close()


if __name__ == "__main__":
    tests = [
        test_image_shape_and_type,
        test_label_is_shown_on_the_picture,
        test_rotation_is_proper,
        test_lower_side_is_where_the_ball_accelerates,
        test_slope_matches_env_dynamics,
        test_ball_sits_on_the_tray_surface,
        test_renderer_does_not_change_the_environment,
    ]
    for test in tests:
        print("[RUN ] {}".format(test.__name__))
        test()
        print("[ OK ] {}".format(test.__name__))
    print("\nAll {} tests passed.".format(len(tests)))
