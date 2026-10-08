"""LQR demo of the Ball-on-Tray environment.

An infinite-horizon discrete-time LQR controller, designed from a linearised
model of the environment and compared against the random and PD policies of
``demo.py``.

Model
-----
The two axes of the environment are exactly decoupled: the ball's x motion is
driven by the pitch only and its y motion by the roll only, so one single-axis
design is computed and the same gain is applied to both axes.

Because the action is a tilt *increment*, the tilt is itself a state. With
k = (5/7)*g [m/s^2 per rad], c = TILT_STEP [rad per unit action] and the
small-angle approximation sin(theta) ~ theta, the per-axis state is

    z = [x, v, theta]^T     [m, m/s, rad],      u = action in [-1, 1]

The environment updates the tilt *before* integrating the ball, and integrates
one control period as N semi-implicit Euler sub-steps of h = DT / N with the
acceleration held constant:

    v_i = v_0 + i*a*h
    x_N = x_0 + h*sum_i v_i = x_0 + v_0*DT + a*DT^2*(N+1)/(2N)

so the position coefficient is p = (N+1)/(2N) = 0.625 for N = 4, not the 0.5 of
an exact double integrator. This gives

        | 1   DT   p*k*DT^2 |         | p*k*c*DT^2 |
    A = | 0   1    k*DT     |     B = | k*c*DT     |
        | 0   0    1        |         | c          |

which reproduces one environment step to ~1e-7 m; the only residual error is
the small-angle approximation itself.

The base acceleration ``a_base`` is an *unmodelled* disturbance (it is not part
of the observation), so the controller has no feed-forward term for it and
shows a transient excursion during each disturbance pulse. The rolling
resistance ``c_rr`` is likewise unmodelled, which is deliberate: like
``pd_policy`` in ``demo.py``, this controller reads nothing but the newest
observation frame, so the comparison with a future RL agent stays fair.

Note that [x, v, theta] is fully contained in a single observation frame, so no
state estimator is needed. The three stacked frames of the observation only
help infer the hidden ``c_rr`` and ``a_base``, which neither LQR nor PD uses.

Design
------
Q and R are specified as dimensionless weights divided by the square of a
characteristic scale of each state, so the weights are directly comparable:
position by the tray half size, velocity by ``VEL_SCALE``, tilt by the tilt
limit, and the action is already normalised to [-1, 1].

The discrete algebraic Riccati equation is solved by iterating

    P <- A'PA - A'PB (R + B'PB)^-1 B'PA + Q

to convergence. This avoids a scipy dependency, which would otherwise have to
be added to ``environment.yml``; the system is 3x1, so the iteration costs
microseconds.

Examples:
    python demo_lqr.py                                  # random, then PD, then LQR
    python demo_lqr.py --policy lqr --seed 3            # LQR only, another seed
    python demo_lqr.py --view 3d                        # same episodes, 3D view
    python demo_lqr.py --save-gif media/demo_lqr.gif    # no window, write a gif
    python demo_lqr.py --compare 50                     # no window, statistics table
"""

import argparse
import time

import numpy as np

from ball_on_tray_gym import BallOnTrayEnv
# The episode setup, the PD baseline and the reporting are shared with the 2D demo.
from demo import (DEFAULT_SEED, DISTURBANCE_RANGE, END_PAUSE, FRAME_DIM,
                  FRICTION_RANGE, GIF_FRAME_SKIP, MAX_TILT, POS_SCALE,
                  VEL_SCALE, pd_policy, print_summary, save_gif)
# The 3D view reuses the renderer and the timing report of the 3D demo.
from demo_3d import print_render_stats
from renderer_3d import TrayRenderer3D

# ------------------------------------------------------------------ LQR weights
# Dimensionless; each is divided by the square of the matching scale below.
# Chosen by a grid search over the environment's own return (the quadratic cost
# is only a proxy for it: the reward's distance term is linear and its goal
# bonus is a step), then validated on held-out seeds not used for the search.
# The result is insensitive to W_VEL; W_TILT and W_ACTION do the work.
W_POS = 1.0         # weight on the position error [-]
W_VEL = 0.05        # weight on the velocity [-]
W_TILT = 0.02       # weight on the tilt (keeps the tray near level) [-]
W_ACTION = 0.002    # weight on the action (control effort) [-]

# Characteristic scales used to normalise the states.
SCALE_POS = BallOnTrayEnv.TRAY_HALF_SIZE                    # [m]
SCALE_VEL = BallOnTrayEnv.VEL_SCALE                         # [m/s]
SCALE_TILT = np.deg2rad(BallOnTrayEnv.MAX_TILT_DEG)         # [rad]

# Riccati iteration.
DARE_TOL = 1e-12        # convergence threshold on ||P_new - P||_inf
DARE_MAX_ITER = 10000   # iteration cap


# ------------------------------------------------------------------- the design
def build_model():
    """Build the discrete single-axis model (A, B) described in the docstring.

    Returns:
        (A, B) with shapes (3, 3) and (3, 1); state [x, v, theta], input u.
    """
    dt = BallOnTrayEnv.DT                                       # [s]
    n_sub = BallOnTrayEnv.N_SUBSTEPS                            # [-]
    k = BallOnTrayEnv.ROLLING_FACTOR * BallOnTrayEnv.GRAVITY    # [m/s^2 per rad]
    c = np.deg2rad(BallOnTrayEnv.TILT_STEP_DEG)                 # [rad per unit action]
    p = (n_sub + 1.0) / (2.0 * n_sub)                           # position coefficient [-]

    A = np.array([[1.0, dt, p * k * dt ** 2],
                  [0.0, 1.0, k * dt],
                  [0.0, 0.0, 1.0]])
    B = np.array([[p * k * c * dt ** 2],
                  [k * c * dt],
                  [c]])
    return A, B


def build_weights():
    """Build the LQR cost matrices.

    Returns:
        (Q, R) with shapes (3, 3) and (1, 1).
    """
    Q = np.diag([W_POS / SCALE_POS ** 2,
                 W_VEL / SCALE_VEL ** 2,
                 W_TILT / SCALE_TILT ** 2])
    R = np.array([[W_ACTION]])
    return Q, R


def solve_dare(A, B, Q, R, tol=DARE_TOL, max_iter=DARE_MAX_ITER):
    """Solve the discrete algebraic Riccati equation by value iteration.

    Iterates ``P <- A'PA - A'PB (R + B'PB)^-1 B'PA + Q`` from ``P = Q`` until
    ``P`` stops changing, then returns the optimal gain.

    Args:
        A, B: system matrices, shapes (n, n) and (n, m).
        Q, R: cost matrices, shapes (n, n) and (m, m).
        tol: convergence threshold on the largest entry change of P.
        max_iter: iteration cap.

    Returns:
        (K, P, n_iter) with K of shape (m, n). The optimal input is u = -K z.

    Raises:
        RuntimeError: if the iteration did not converge within ``max_iter``.
    """
    P = Q.copy()
    for i in range(1, max_iter + 1):
        BtP = B.T.dot(P)
        # Gain of this iteration: K = (R + B'PB)^-1 B'PA
        K = np.linalg.solve(R + BtP.dot(B), BtP.dot(A))
        P_next = A.T.dot(P).dot(A) - A.T.dot(P).dot(B).dot(K) + Q
        if np.max(np.abs(P_next - P)) < tol:
            return K, P_next, i
        P = P_next
    raise RuntimeError("Riccati iteration did not converge in {} steps".format(max_iter))


def design_lqr():
    """Run the full design and report it.

    Returns:
        (K, info) where K has shape (1, 3) and info holds the model, the cost
        matrices, the iteration count and the closed-loop poles.
    """
    A, B = build_model()
    Q, R = build_weights()
    K, P, n_iter = solve_dare(A, B, Q, R)
    poles = np.linalg.eigvals(A - B.dot(K))
    return K, {"A": A, "B": B, "Q": Q, "R": R, "P": P,
               "n_iter": n_iter, "poles": poles}


def print_design(K, info):
    """Print the gain, the closed-loop poles and the stability check."""
    radii = np.abs(info["poles"])
    print("LQR design (one axis, applied to both)")
    print("  weights      : pos {:.3g}, vel {:.3g}, tilt {:.3g}, action {:.3g}".format(
        W_POS, W_VEL, W_TILT, W_ACTION))
    print("  Riccati      : converged in {} iterations".format(info["n_iter"]))
    print("  gain K       : x {:8.3f} [1/m]   v {:8.3f} [s/m]   theta {:8.3f} [1/rad]".format(
        K[0, 0], K[0, 1], K[0, 2]))
    print("  closed loop  : |z| = {}".format(
        ", ".join("{:.4f}".format(r) for r in sorted(radii, reverse=True))))
    print("  stable       : {} (all poles inside the unit circle)".format(
        "yes" if np.max(radii) < 1.0 else "NO"))
    # Tilt reachable in one step from rest at the tray edge, as a sanity figure.
    print("  settling     : ~{:.2f} s to decay by 1/e (slowest pole)".format(
        -BallOnTrayEnv.DT / np.log(np.max(radii))))
    print("")


# ------------------------------------------------------------------- the policy
def make_lqr_policy(K):
    """Build a policy function ``policy(obs) -> action`` from the gain K.

    The newest observation frame (last 6 entries) is de-normalised to position
    [m], velocity [m/s] and tilt [rad], the per-axis state z = [x, v, theta] is
    assembled for each axis, and the same gain is applied to both:

        u = clip(-K z, -1, 1)

    Args:
        K: gain of shape (1, 3) from ``design_lqr``.

    Returns:
        A function mapping an observation to a (2,) float32 action.
    """
    gain = K.ravel()        # [k_x, k_v, k_theta]

    def policy(obs):
        frame = np.asarray(obs, dtype=np.float64)[-FRAME_DIM:]
        pos = frame[0:2] * POS_SCALE            # [m]
        vel = frame[2:4] * VEL_SCALE            # [m/s]
        tilt = frame[4:6] * MAX_TILT            # [rad]
        # One axis per column: z[:, 0] is the x/pitch axis, z[:, 1] the y/roll axis.
        z = np.vstack([pos, vel, tilt])         # (3, 2)
        return np.clip(-gain.dot(z), -1.0, 1.0).astype(np.float32)

    return policy


# ----------------------------------------------------------------- the episodes
def run_episode(env, policy_name, policy_fn, seed, frames=None, renderer=None):
    """Run one episode under a policy function and return a summary dict.

    Works for both views. In the 2D view the environment draws itself and
    ``renderer`` is None; in the 3D view the environment is created with
    ``render_mode=None`` and ``renderer`` draws it from outside.

    Args:
        env: the environment (2D rendering follows its render_mode).
        policy_name: name used in the summary.
        policy_fn: ``policy_fn(obs) -> action``, or None for the random policy.
        seed: seed of the episode (and of the random policy).
        frames: if a list is given, rgb_array frames are appended to it.
        renderer: a TrayRenderer3D for the 3D view, or None for the 2D view.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    # The 3D renderer needs an explicit first frame; the 2D one is already
    # drawn by reset() in "human" mode and fetched by render() for a gif.
    if renderer is not None:
        first = renderer.render(0.0, 0.0)
    elif frames is not None:
        first = env.render()
    else:
        first = None
    if frames is not None:
        frames.append(first)
    # Timing starts after the first frame, which includes opening the window.
    drawn_before = 0 if renderer is None else renderer.frames_drawn
    wall_start = time.perf_counter()

    total_reward, terminated, truncated = 0.0, False, False
    peak_dist = float(np.linalg.norm(env.ball_pos))             # [m]
    while not (terminated or truncated or _window_closed(env, renderer)):
        action = env.action_space.sample() if policy_fn is None else policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        peak_dist = max(peak_dist, float(np.linalg.norm(env.ball_pos)))

        keep = env.step_count % GIF_FRAME_SKIP == 0 or terminated or truncated
        if renderer is not None:
            if frames is None:
                # On screen the renderer keeps real-time pace and skips frames.
                renderer.render(reward, total_reward)
            elif keep:
                frames.append(renderer.render(reward, total_reward))
        elif frames is not None and keep:
            frames.append(env.render())

    wall_time = time.perf_counter() - wall_start

    # Keep the final frame visible for a moment.
    fps = env.metadata["render_fps"]
    if frames is not None:
        frames.extend([frames[-1]] * int(END_PAUSE * fps / GIF_FRAME_SKIP))
    elif renderer is not None:
        renderer.hold(END_PAUSE)
    elif env.render_mode == "human":
        for _ in range(int(END_PAUSE * fps)):
            env.render()

    if terminated:
        reason = "ball left the tray"
    elif truncated:
        reason = "step limit reached"
    else:
        reason = "window closed by user"
    return {"policy": policy_name, "steps": env.step_count, "return": total_reward,
            "reason": reason, "c_rr": env.c_rr, "disturbances": list(env.disturbances),
            "peak_dist": peak_dist, "terminated": terminated,
            "frames_drawn": 0 if renderer is None else renderer.frames_drawn - drawn_before,
            "wall_time": wall_time}


def _window_closed(env, renderer):
    """True once the user has closed the window of whichever view is active."""
    return env.window_closed if renderer is None else renderer.window_closed


def compare(policies, n_seeds):
    """Run every policy over seeds 0..n_seeds-1 and print a statistics table.

    No rendering is done, so this is the quantitative counterpart of the demo.
    """
    env = BallOnTrayEnv(friction_range=FRICTION_RANGE,
                       disturbance_range=DISTURBANCE_RANGE, render_mode=None)
    print("{} seeds, c_rr in {}, |a_base| in {} m/s^2\n".format(
        n_seeds, FRICTION_RANGE, DISTURBANCE_RANGE))
    header = "{:<8}{:>16}{:>16}{:>12}{:>14}{:>12}".format(
        "policy", "return", "steps", "survived", "peak |pos| m", "reward/step")
    print(header)
    print("-" * len(header))

    for name, policy_fn in policies:
        returns, steps, peaks, fell = [], [], [], 0
        for seed in range(n_seeds):
            summary = run_episode(env, name, policy_fn, seed)
            returns.append(summary["return"])
            steps.append(summary["steps"])
            peaks.append(summary["peak_dist"])
            fell += int(summary["terminated"])
        returns, steps, peaks = np.array(returns), np.array(steps), np.array(peaks)
        print("{:<8}{:>9.1f} +-{:>5.1f}{:>9.1f} +-{:>5.1f}{:>9d}/{:<2d}{:>14.4f}{:>12.3f}".format(
            name, returns.mean(), returns.std(), steps.mean(), steps.std(),
            n_seeds - fell, n_seeds, peaks.mean(), returns.sum() / steps.sum()))

    env.close()
    print("\nper-step reward ceiling = 1 + GOAL_BONUS = {:.1f}".format(
        1.0 + BallOnTrayEnv.GOAL_BONUS))


# ------------------------------------------------------------------------- main
def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Ball-on-Tray LQR demo.")
    parser.add_argument("--policy", choices=["random", "pd", "lqr", "all"], default="all",
                        help="policy to show; 'all' plays random, then PD, then LQR "
                             "(default: all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="random seed of the episode(s) (default: {})".format(DEFAULT_SEED))
    parser.add_argument("--view", choices=["2d", "3d"], default="2d",
                        help="'2d' uses the environment's built-in top view, '3d' draws "
                             "the tilting tray with TrayRenderer3D (default: 2d)")
    parser.add_argument("--save-gif", metavar="PATH", default=None,
                        help="do not open a window; render off-screen and save the "
                             "episode(s) to this gif file")
    parser.add_argument("--compare", metavar="N", type=int, default=None,
                        help="do not render; instead run every policy over N seeds "
                             "and print a statistics table")
    return parser.parse_args()


def main():
    """Design the controller, then either show episodes or print statistics."""
    args = parse_args()

    K, info = design_lqr()
    print_design(K, info)
    lqr = make_lqr_policy(K)

    # None marks the random policy, which samples from the action space instead.
    available = [("random", None), ("pd", pd_policy), ("lqr", lqr)]
    if args.policy != "all":
        available = [item for item in available if item[0] == args.policy]

    if args.compare is not None:
        compare(available, args.compare)
        return

    mode = "rgb_array" if args.save_gif else "human"
    if args.view == "3d":
        # The environment itself does not render; the 3D renderer draws it from outside.
        env = BallOnTrayEnv(friction_range=FRICTION_RANGE,
                           disturbance_range=DISTURBANCE_RANGE, render_mode=None)
        renderer = TrayRenderer3D(env, mode=mode)
    else:
        env = BallOnTrayEnv(friction_range=FRICTION_RANGE,
                           disturbance_range=DISTURBANCE_RANGE, render_mode=mode)
        renderer = None
    frames = [] if args.save_gif else None

    for name, policy_fn in available:
        # Every policy gets the same seed, i.e. the same start position, rolling
        # resistance and disturbance schedule.
        summary = run_episode(env, name, policy_fn, args.seed, frames, renderer)
        print_summary(summary)
        if renderer is not None and frames is None:
            print_render_stats(summary, env.DT)
        if _window_closed(env, renderer):
            break
    if renderer is not None:
        renderer.close()
    env.close()

    if args.save_gif:
        save_gif(frames, args.save_gif)
        print("saved {} frames to {}".format(len(frames), args.save_gif))


if __name__ == "__main__":
    main()
