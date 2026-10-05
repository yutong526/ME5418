"""3D demo of the Ball-on-Tray environment.

Same episodes and command-line arguments as ``demo.py``, but drawn in 3D by
``TrayRenderer3D``: the tray really tilts and the ball sits on its surface.

Examples:
    python demo_3d.py                                   # random episode, then PD episode
    python demo_3d.py --policy pd --seed 3              # PD controller only, other seed
    python demo_3d.py --save-gif media/demo_3d.gif      # no window, write a gif instead
"""

import time

from ball_on_tray_gym import BallOnTrayEnv
# The policy, the argument parser and the reporting are shared with the 2D demo.
from demo import (DISTURBANCE_RANGE, END_PAUSE, FRICTION_RANGE, GIF_FRAME_SKIP,
                  parse_args, pd_policy, print_summary, save_gif)
from renderer_3d import TrayRenderer3D


def run_episode(env, renderer, policy_name, seed, frames=None):
    """Run one episode, drawing it with the 3D renderer, and return a summary dict.

    Args:
        env: the environment (created with render_mode=None).
        renderer: the TrayRenderer3D that draws ``env``.
        policy_name: "random" or "pd".
        seed: seed of the episode (and of the random policy).
        frames: if a list is given, rgb_array frames are appended to it.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    total_reward, terminated, truncated = 0.0, False, False
    frame = renderer.render(0.0, total_reward)
    if frames is not None:
        frames.append(frame)
    # Timing starts after the first frame, which includes opening the window.
    drawn_before = renderer.frames_drawn
    wall_start = time.perf_counter()

    while not (terminated or truncated or renderer.window_closed):
        if policy_name == "random":
            action = env.action_space.sample()
        else:
            action = pd_policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if frames is None:
            # On screen: the renderer keeps real-time pace and skips frames if needed.
            renderer.render(reward, total_reward)
        elif env.step_count % GIF_FRAME_SKIP == 0 or terminated or truncated:
            frames.append(renderer.render(reward, total_reward))

    wall_time = time.perf_counter() - wall_start
    drawn = renderer.frames_drawn - drawn_before

    # Keep the final frame visible for a moment.
    if frames is not None:
        fps = int(round(1.0 / env.DT))
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
            "frames_drawn": drawn, "wall_time": wall_time}


def print_render_stats(summary, dt):
    """Print how fast the episode was drawn on screen and whether frames were skipped."""
    steps, drawn, wall_time = summary["steps"], summary["frames_drawn"], summary["wall_time"]
    if steps == 0 or wall_time <= 0.0:
        return
    print("3D rendering : drew {} frames for {} steps ({:.1f} fps), "
          "{:.2f} s wall time for {:.2f} s simulated".format(
              drawn, steps, drawn / wall_time, wall_time, steps * dt))
    print("")


def main():
    """Run the requested episode(s), on screen or into a gif."""
    args = parse_args()
    policies = ["random", "pd"] if args.policy == "both" else [args.policy]

    # The environment itself does not render; the 3D renderer draws it from outside.
    env = BallOnTrayEnv(friction_range=FRICTION_RANGE, disturbance_range=DISTURBANCE_RANGE,
                        render_mode=None)
    renderer = TrayRenderer3D(env, mode="rgb_array" if args.save_gif else "human")
    frames = [] if args.save_gif else None

    for policy_name in policies:
        # Both policies get the same seed, i.e. the same start position,
        # rolling resistance and disturbance schedule.
        summary = run_episode(env, renderer, policy_name, args.seed, frames)
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
