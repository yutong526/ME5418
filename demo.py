"""Demo of the Ball-on-Tray environment.

Plays one episode with a random policy and/or one with a PD controller, using
the full randomisation ranges (rolling resistance and base disturbances).

Examples:
    python demo.py                                  # random episode, then PD episode
    python demo.py --policy pd --seed 3             # PD controller only, other seed
    python demo.py --save-gif media/demo.gif        # no window, write a gif instead
"""

import argparse
import os

import numpy as np

from ball_on_tray_gym import BallOnTrayEnv

# Full randomisation ranges of the environment.
FRICTION_RANGE = (0.005, 0.05)          # c_rr [-]
DISTURBANCE_RANGE = (0.5, 2.0)          # ||a_base|| [m/s^2]

DEFAULT_SEED = 4

# PD gains (tilt command per position / velocity error).
KP = 6.0                                # [rad/m]
KD = 1.5                                # [rad/(m/s)]

# Scales that undo the normalisation of the observation.
POS_SCALE = BallOnTrayEnv.TRAY_HALF_SIZE                    # [m]
VEL_SCALE = BallOnTrayEnv.VEL_SCALE                         # [m/s]
MAX_TILT = np.deg2rad(BallOnTrayEnv.MAX_TILT_DEG)           # [rad]
TILT_STEP = np.deg2rad(BallOnTrayEnv.TILT_STEP_DEG)         # [rad/step]
FRAME_DIM = BallOnTrayEnv.FRAME_DIM

GIF_FRAME_SKIP = 2      # keep every 2nd frame -> 25 fps gif
END_PAUSE = 1.0         # how long the last frame of an episode stays visible [s]


def pd_policy(obs, kp=KP, kd=KD):
    """Deterministic PD controller that only uses the observation.

    The newest frame (last 6 entries of the 18-D observation) is
    de-normalised to position [m], velocity [m/s] and tilt [rad]. Then

        tilt_target = clip(-kp * pos - kd * vel, +/- 15 deg)        [rad]
        action      = clip((tilt_target - tilt) / 3 deg, -1, 1)     [-]

    The minus signs follow the sign convention of the environment: positive
    pitch accelerates the ball towards +x, so a ball at x > 0 needs pitch < 0
    (same for roll and y).
    """
    frame = np.asarray(obs, dtype=np.float64)[-FRAME_DIM:]
    pos = frame[0:2] * POS_SCALE
    vel = frame[2:4] * VEL_SCALE
    tilt = frame[4:6] * MAX_TILT

    tilt_target = np.clip(-kp * pos - kd * vel, -MAX_TILT, MAX_TILT)
    return np.clip((tilt_target - tilt) / TILT_STEP, -1.0, 1.0).astype(np.float32)


def run_episode(env, policy_name, seed, frames=None):
    """Run one episode and return a summary dict.

    Args:
        env: the environment (rendering follows its render_mode).
        policy_name: "random" or "pd".
        seed: seed of the episode (and of the random policy).
        frames: if a list is given, rgb_array frames are appended to it.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    if frames is not None:
        frames.append(env.render())

    total_reward, terminated, truncated = 0.0, False, False
    while not (terminated or truncated or env.window_closed):
        if policy_name == "random":
            action = env.action_space.sample()
        else:
            action = pd_policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if frames is not None and (
                env.step_count % GIF_FRAME_SKIP == 0 or terminated or truncated):
            frames.append(env.render())

    # Keep the final frame visible for a moment.
    fps = env.metadata["render_fps"]
    if frames is not None:
        frames.extend([frames[-1]] * int(END_PAUSE * fps / GIF_FRAME_SKIP))
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
            "reason": reason, "c_rr": env.c_rr, "disturbances": list(env.disturbances)}


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
    parser.add_argument("--policy", choices=["random", "pd", "both"], default="both",
                        help="policy to show; 'both' plays a random episode, then a PD "
                             "episode (default: both)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="random seed of the episode(s) (default: {})".format(DEFAULT_SEED))
    parser.add_argument("--save-gif", metavar="PATH", default=None,
                        help="do not open a window; render off-screen and save the "
                             "episode(s) to this gif file")
    return parser.parse_args()


def main():
    """Run the requested episode(s), on screen or into a gif."""
    args = parse_args()
    policies = ["random", "pd"] if args.policy == "both" else [args.policy]

    env = BallOnTrayEnv(friction_range=FRICTION_RANGE, disturbance_range=DISTURBANCE_RANGE,
                        render_mode="rgb_array" if args.save_gif else "human")
    frames = [] if args.save_gif else None

    for policy_name in policies:
        # Both policies get the same seed, i.e. the same start position,
        # rolling resistance and disturbance schedule.
        print_summary(run_episode(env, policy_name, args.seed, frames))
        if env.window_closed:
            break
    env.close()

    if args.save_gif:
        save_gif(frames, args.save_gif)
        print("saved {} frames to {}".format(len(frames), args.save_gif))


if __name__ == "__main__":
    main()
