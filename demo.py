"""Demo of the Ball-on-Tray environment.

Plays one episode per selected policy (random, PD controller, LQR controller),
using the full randomisation ranges (rolling resistance and base disturbances),
in the 3D view of ``TrayRenderer3D``: the tray really tilts and the ball sits
on its surface. The controllers live in ``controllers.py``.

Examples:
    python demo.py                                  # random episode, then PD episode
    python demo.py --policy all                     # random, then PD, then LQR
    python demo.py --policy lqr --seed 3            # LQR only, another seed
    python demo.py --realistic                      # with noise, actuator lag and delay
    python demo.py --save-gif media/demo_3d.gif     # no window, write a gif instead
"""

import argparse
import os
import time

import numpy as np

from ball_on_tray_gym import BallOnTrayEnv
from controllers import design_lqr, make_lqr_policy, pd_policy, print_design
from renderer_3d import TrayRenderer3D

# Full randomisation ranges of the environment.
FRICTION_RANGE = (0.005, 0.05)          # c_rr [-]
DISTURBANCE_RANGE = (0.5, 2.0)          # ||a_base|| [m/s^2]

DEFAULT_SEED = 4

# Environment arguments switched on by --realistic.
REALISTIC_KWARGS = {
    "pos_noise_std": 0.002,             # position measurement noise [m]
    "vel_noise_std": 0.03,              # velocity measurement noise [m/s]
    "actuator_tau": 0.05,               # actuator lag time constant [s]
    "action_delay_steps": 1,            # action delay [control steps]
}

GIF_FRAME_SKIP = 2      # keep every 2nd frame -> 25 fps gif
END_PAUSE = 1.0         # how long the last frame of an episode stays visible [s]

# Policies played by each --policy choice, in order.
POLICY_CHOICES = {
    "random": ("random",),
    "pd": ("pd",),
    "lqr": ("lqr",),
    "both": ("random", "pd"),
    "all": ("random", "pd", "lqr"),
}


def run_episode(env, renderer, policy_name, policy_fn, seed, frames=None):
    """Run one episode under a policy function and return a summary dict.

    Args:
        env: the environment (created with render_mode=None).
        renderer: the TrayRenderer3D that draws ``env``.
        policy_name: name used in the summary.
        policy_fn: ``policy_fn(obs) -> action``, or None for the random policy.
        seed: seed of the episode (and of the random policy).
        frames: if a list is given, rgb_array frames are appended to it.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    first = renderer.render(0.0, 0.0)
    if frames is not None:
        frames.append(first)
    # Timing starts after the first frame, which includes opening the window.
    drawn_before = renderer.frames_drawn
    wall_start = time.perf_counter()

    total_reward, terminated, truncated = 0.0, False, False
    while not (terminated or truncated or renderer.window_closed):
        action = env.action_space.sample() if policy_fn is None else policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if frames is None:
            # On screen the renderer keeps real-time pace and skips frames.
            renderer.render(reward, total_reward)
        elif env.step_count % GIF_FRAME_SKIP == 0 or terminated or truncated:
            frames.append(renderer.render(reward, total_reward))

    wall_time = time.perf_counter() - wall_start

    # Keep the final frame visible for a moment.
    fps = env.metadata["render_fps"]
    if frames is not None:
        frames.extend([frames[-1]] * int(END_PAUSE * fps / GIF_FRAME_SKIP))
    else:
        renderer.hold(END_PAUSE)

    if terminated:
        reason = "ball left the tray"
    elif truncated:
        reason = "step limit reached"
    else:
        reason = "window closed by user"
    return {"policy": policy_name, "steps": env.step_count, "return": total_reward,
            "reason": reason, "c_rr": env.c_rr, "disturbances": list(env.disturbances),
            "frames_drawn": renderer.frames_drawn - drawn_before,
            "wall_time": wall_time}


def print_summary(summary):
    """Print the result of one episode."""
    print("policy       : {}".format(summary["policy"]))
    print("steps        : {}".format(summary["steps"]))
    print("return       : {:.2f}".format(summary["return"]))
    print("ended because: {}".format(summary["reason"]))
    print("c_rr         : {:.4f}".format(summary["c_rr"]))
    print("disturbances : {} pulse(s)".format(len(summary["disturbances"])))
    for pulse in summary["disturbances"]:
        accel = pulse["accel"]
        print("    t = {:5.2f} s .. {:5.2f} s, |a_base| = {:.2f} m/s^2, direction = {:6.1f} deg".format(
            pulse["start"], pulse["start"] + pulse["duration"],
            np.linalg.norm(accel), np.rad2deg(np.arctan2(accel[1], accel[0]))))
    print("")


def print_render_stats(summary, dt):
    """Print how fast the episode was drawn on screen and whether frames were skipped."""
    steps, drawn, wall_time = summary["steps"], summary["frames_drawn"], summary["wall_time"]
    if steps == 0 or wall_time <= 0.0:
        return
    print("3D rendering : drew {} frames for {} steps ({:.1f} fps), "
          "{:.2f} s wall time for {:.2f} s simulated".format(
              drawn, steps, drawn / wall_time, wall_time, steps * dt))
    print("")


def print_realism(kwargs):
    """Print the realism parameters switched on by --realistic."""
    print("realistic    : pos_noise_std = {} m, vel_noise_std = {} m/s, "
          "actuator_tau = {} s, action_delay_steps = {}".format(
              kwargs["pos_noise_std"], kwargs["vel_noise_std"],
              kwargs["actuator_tau"], kwargs["action_delay_steps"]))
    print("")


def save_gif(frames, path):
    """Write rgb_array frames to an animated gif (Pillow ships with matplotlib)."""
    from PIL import Image

    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    images = [Image.fromarray(frame) for frame in frames]
    frame_ms = int(round(1000.0 * GIF_FRAME_SKIP * BallOnTrayEnv.DT))
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=frame_ms, loop=0)


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Ball-on-Tray demo.")
    parser.add_argument("--policy", choices=["random", "pd", "lqr", "both", "all"],
                        default="both",
                        help="policy to show; 'both' plays a random episode, then a PD "
                             "episode; 'all' plays random, then PD, then LQR (default: both)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="random seed of the episode(s) (default: {})".format(DEFAULT_SEED))
    parser.add_argument("--save-gif", metavar="PATH", default=None,
                        help="do not open a window; render off-screen and save the "
                             "episode(s) to this gif file")
    parser.add_argument("--realistic", action="store_true",
                        help="switch on the optional realism parameters: observation noise "
                             "(0.002 m, 0.03 m/s), actuator lag (0.05 s) and a one-step "
                             "action delay")
    return parser.parse_args()


def main():
    """Run the requested episode(s), on screen or into a gif."""
    args = parse_args()
    policy_names = POLICY_CHOICES[args.policy]

    realism = dict(REALISTIC_KWARGS) if args.realistic else {}
    if args.realistic:
        print_realism(realism)

    # None marks the random policy, which samples from the action space instead.
    policy_fns = {"random": None, "pd": pd_policy}
    if "lqr" in policy_names:
        gain, info = design_lqr()
        print_design(gain, info)
        policy_fns["lqr"] = make_lqr_policy(gain)

    # The environment itself does not render here; the 3D renderer draws it from
    # outside, which lets the demo pass the reward and control the pacing.
    env = BallOnTrayEnv(friction_range=FRICTION_RANGE, disturbance_range=DISTURBANCE_RANGE,
                        render_mode=None, **realism)
    renderer = TrayRenderer3D(env, mode="rgb_array" if args.save_gif else "human")
    frames = [] if args.save_gif else None

    for name in policy_names:
        # Every policy gets the same seed, i.e. the same start position, rolling
        # resistance and disturbance schedule.
        summary = run_episode(env, renderer, name, policy_fns[name], args.seed, frames)
        print_summary(summary)
        if frames is None:
            print_render_stats(summary, env.DT)
        if renderer.window_closed:
            break
    renderer.close()
    env.close()

    if args.save_gif:
        save_gif(frames, args.save_gif)
        print("saved {} frames to {}".format(len(frames), args.save_gif))


if __name__ == "__main__":
    main()
