"""Record a side-by-side 3D comparison video (PD vs LQR vs Best RL Agent) to MP4.

Runs PD, LQR and a trained RL checkpoint simultaneously on the exact same
episode seed(s) — identical initial ball position, rolling resistance ``c_rr``,
disturbance pulse schedule and sensor noise — and composites their 3D views
into a 1920x640 H.264 MP4 video with a live telemetry header and bottom
comparison bar.

Examples:
    python record_demo_mp4.py
    python record_demo_mp4.py --seeds 49 88 --out media/final_demo.mp4
    python record_demo_mp4.py --checkpoint trained/ppo_dfeat_x5_staged_seed2.pt --ideal --seeds 4
"""

import argparse
import os
import time

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agents import FeatureEnv, checkpoint_features, describe, is_default, load_policy
from ball_on_tray_gym import BallOnTrayEnv, TrayRenderer3D
from controllers import design_lqr, make_lqr_policy, pd_policy
from demo import DISTURBANCE_RANGE, FRICTION_RANGE, REALISTIC_KWARGS

# Layout geometry (1920 x 640, divisible by 16 for H.264 yuv420p).
PANEL_W = 640
PANEL_H = 428
HEADER_H = 76
FOOTER_H = 136
VIDEO_W = PANEL_W * 3          # 1920
VIDEO_H = HEADER_H + PANEL_H + FOOTER_H  # 640
MAX_RETURN = 750.0

DEFAULT_CHECKPOINT = "trained/ppo_act3_staged_realistic_seed1.pt"
DEFAULT_SEEDS = [49, 88]
DEFAULT_OUT = "media/final_demo.mp4"

# Palette
BG_HEADER = (24, 31, 44)
BG_FOOTER = (246, 248, 251)
DIVIDER_COLOR = (210, 216, 226)
TEXT_DARK = (25, 32, 44)
TEXT_MUTED = (95, 106, 124)
TEXT_WHITE = (245, 247, 250)
ACCENT_RED = (220, 53, 69)
ACCENT_GREEN = (22, 163, 74)
ACCENT_BLUE = (37, 99, 235)

PANEL_ACCENTS = [
    (100, 116, 139),   # PD: slate
    (217, 119, 6),     # LQR: amber
    (16, 185, 129),    # RL: emerald
]


def load_font(size, bold=False, mono=False):
    """Load a crisp TrueType font on Windows/Linux, falling back gracefully."""
    candidates = []
    if mono:
        candidates += [
            r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    else:
        candidates += [
            r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    # Also check matplotlib's bundled DejaVu fonts.
    try:
        import matplotlib
        mpl_ttf = os.path.join(os.path.dirname(matplotlib.__file__),
                               "mpl-data", "fonts", "ttf")
        if mono:
            candidates.append(os.path.join(
                mpl_ttf, "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"))
        else:
            candidates.append(os.path.join(
                mpl_ttf, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
    except Exception:
        pass

    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_fonts():
    """Pre-load the font set used for overlays."""
    return {
        "title": load_font(24, bold=True),
        "subtitle": load_font(17, bold=False),
        "badge": load_font(16, bold=True),
        "card_title": load_font(20, bold=True),
        "metric": load_font(17, bold=False, mono=True),
        "metric_bold": load_font(18, bold=True, mono=True),
        "fall": load_font(22, bold=True),
    }


def compose_frame(panel_imgs, states, meta, fonts):
    """Assemble the 1920x640 comparison frame from three 3D renders and telemetry."""
    canvas = Image.new("RGB", (VIDEO_W, VIDEO_H), BG_FOOTER)
    draw = ImageDraw.Draw(canvas)

    # ------------------------------------------------------------- top header
    draw.rectangle([0, 0, VIDEO_W, HEADER_H], fill=BG_HEADER)
    mode_tag = ("Realistic (Noise + 50 ms Lag + 1-Step Delay)"
                if meta["realistic"] else "Ideal (Full Friction & Pulse Randomisation)")
    draw.text((22, 11),
              "ME5418 Ball-on-Tray  |  PD vs LQR vs Best RL Agent",
              fill=TEXT_WHITE, font=fonts["title"])
    draw.text((22, 44),
              "Condition: {}   |   Clip {}/{} (Seed {})   |   c_rr = {:.4f}".format(
                  mode_tag, meta["clip_idx"], meta["n_clips"], meta["seed"], meta["c_rr"]),
              fill=(185, 196, 214), font=fonts["subtitle"])

    # Right side of header: simulation clock + active disturbance badge
    clock_text = "t = {:5.2f} / {:5.2f} s   (step {:3d}/500)".format(
        meta["sim_time"], meta["max_time"], meta["step"])
    draw.text((VIDEO_W - 390, 14), clock_text, fill=TEXT_WHITE, font=fonts["metric_bold"])

    a_mag = meta["a_base_mag"]
    if a_mag > 0.0:
        pill_box = [VIDEO_W - 390, 42, VIDEO_W - 22, 68]
        draw.rounded_rectangle(pill_box, radius=6, fill=ACCENT_RED)
        draw.text((VIDEO_W - 376, 46),
                  "DISTURBANCE PULSE ACTIVE: {:.2f} m/s^2".format(a_mag),
                  fill=TEXT_WHITE, font=fonts["badge"])
    else:
        pill_box = [VIDEO_W - 390, 42, VIDEO_W - 22, 68]
        draw.rounded_rectangle(pill_box, radius=6, fill=(44, 56, 76))
        draw.text((VIDEO_W - 376, 46),
                  "Disturbance: none  (pulses scheduled: {})".format(meta["n_pulses"]),
                  fill=(185, 196, 214), font=fonts["badge"])

    # ------------------------------------------------------ 3D panels + cards
    for idx, (img, st) in enumerate(zip(panel_imgs, states)):
        x0 = idx * PANEL_W
        accent = PANEL_ACCENTS[idx]

        # Resize 900x600 renderer output cleanly to 640x428
        pil_panel = Image.fromarray(img).resize((PANEL_W, PANEL_H), Image.LANCZOS)
        canvas.paste(pil_panel, (x0, HEADER_H))

        # Top accent stripe on each panel
        draw.rectangle([x0, HEADER_H, x0 + PANEL_W, HEADER_H + 5], fill=accent)

        # If ball fell off tray, draw a clear red banner on the frozen panel
        if st["terminated"]:
            overlay = Image.new("RGBA", (PANEL_W, 46), (185, 28, 28, 215))
            canvas.paste(overlay, (x0, HEADER_H + PANEL_H - 54), overlay)
            draw.text((x0 + 18, HEADER_H + PANEL_H - 44),
                      "BALL OFF TRAY at t = {:.2f} s  (step {})".format(
                          st["steps"] * BallOnTrayEnv.DT, st["steps"]),
                      fill=TEXT_WHITE, font=fonts["fall"])

        # Footer telemetry card
        fy = HEADER_H + PANEL_H
        draw.rectangle([x0, fy, x0 + PANEL_W, VIDEO_H], fill=BG_FOOTER)
        draw.line([(x0, fy), (x0 + PANEL_W, fy)], fill=DIVIDER_COLOR, width=2)

        # Card header: colored dot + title + status pill
        draw.ellipse([x0 + 18, fy + 15, x0 + 32, fy + 29], fill=accent)
        draw.text((x0 + 42, fy + 10), st["title"], fill=TEXT_DARK, font=fonts["card_title"])

        if st["terminated"]:
            badge_bg, badge_txt = ACCENT_RED, "FAILED"
        elif st["dist_mm"] < 20.0:
            badge_bg, badge_txt = ACCENT_GREEN, "IN GOAL (<20 mm)"
        else:
            badge_bg, badge_txt = ACCENT_BLUE, "TRACKING"
        bx1 = x0 + PANEL_W - 18
        bx0 = bx1 - (168 if badge_txt.startswith("IN") else 112)
        draw.rounded_rectangle([bx0, fy + 11, bx1, fy + 36], radius=6, fill=badge_bg)
        draw.text((bx0 + 10, fy + 14), badge_txt, fill=TEXT_WHITE, font=fonts["badge"])

        # Return progress bar
        bar_x0, bar_y0 = x0 + 18, fy + 48
        bar_w, bar_h = PANEL_W - 36, 18
        frac = float(np.clip(st["total_reward"] / MAX_RETURN, 0.0, 1.0))
        draw.rounded_rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h],
                               radius=5, fill=(224, 230, 238))
        if frac > 0.0:
            fill_w = max(12, int(round(bar_w * frac)))
            draw.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y0 + bar_h],
                                   radius=min(5, fill_w // 2), fill=accent)

        # Numeric telemetry rows
        goal_pct = (100.0 * st["goal_steps"] / st["steps"]) if st["steps"] > 0 else 0.0
        dist_color = ACCENT_GREEN if (not st["terminated"] and st["dist_mm"] < 20.0) else TEXT_DARK
        draw.text((x0 + 18, fy + 76),
                  "Return: {:6.1f} / 750  ({:4.1f}%)".format(
                      st["total_reward"], 100.0 * st["total_reward"] / MAX_RETURN),
                  fill=TEXT_DARK, font=fonts["metric_bold"])
        draw.text((x0 + 18, fy + 103),
                  "Ball Dist: {:5.1f} mm   |   Goal Time: {:5.1f}%".format(
                      st["dist_mm"], goal_pct),
                  fill=dist_color, font=fonts["metric"])

        # Vertical dividers between panels
        if idx > 0:
            draw.line([(x0, HEADER_H), (x0, VIDEO_H)], fill=DIVIDER_COLOR, width=2)

    return np.asarray(canvas, dtype=np.uint8)


def record_clip(writer, seed, specs, env_kwargs, step_skip, end_pause_frames,
                clip_idx, n_clips, fonts):
    """Run all three controllers in lockstep on one seed and stream frames to MP4."""
    envs = []
    step_envs = []
    renderers = []
    obs_list = []
    states = []
    panel_imgs = []

    for spec in specs:
        base_env = BallOnTrayEnv(
            friction_range=FRICTION_RANGE,
            disturbance_range=DISTURBANCE_RANGE,
            render_mode=None,
            **env_kwargs,
        )
        wrapped = (FeatureEnv(base_env, **spec["features"])
                   if spec["features"] else base_env)
        obs, _ = wrapped.reset(seed=seed)
        renderer = TrayRenderer3D(base_env, mode="rgb_array")
        renderer.set_label(spec["panel_label"])
        first_img = renderer.render(0.0, 0.0)

        envs.append(base_env)
        step_envs.append(wrapped)
        renderers.append(renderer)
        obs_list.append(obs)
        panel_imgs.append(first_img)
        states.append({
            "title": spec["card_title"],
            "total_reward": 0.0,
            "last_reward": 0.0,
            "steps": 0,
            "goal_steps": 0,
            "dist_mm": 1000.0 * float(np.linalg.norm(base_env.ball_pos)),
            "terminated": False,
            "truncated": False,
        })

    # Reference environment for shared episode metadata (same seed for all 3).
    ref = envs[0]
    max_steps = ref.max_steps
    max_time = max_steps * ref.DT
    c_rr = ref.c_rr
    n_pulses = len(ref.disturbances)
    realistic = bool(env_kwargs)

    def current_a_base_mag(step_idx):
        # Compute shared disturbance magnitude at current sim time from the schedule.
        t_now = step_idx * ref.DT
        for pulse in ref.disturbances:
            if pulse["start"] <= t_now < pulse["start"] + pulse["duration"]:
                return float(np.linalg.norm(pulse["accel"]))
        return 0.0

    # Write frame 0
    meta = {
        "realistic": realistic,
        "clip_idx": clip_idx,
        "n_clips": n_clips,
        "seed": seed,
        "c_rr": c_rr,
        "n_pulses": n_pulses,
        "step": 0,
        "sim_time": 0.0,
        "max_time": max_time,
        "a_base_mag": current_a_base_mag(0),
    }
    last_composed = compose_frame(panel_imgs, states, meta, fonts)
    writer.append_data(last_composed)
    frames_written = 1

    for step in range(1, max_steps + 1):
        any_running = False
        for i, spec in enumerate(specs):
            st = states[i]
            if st["terminated"] or st["truncated"]:
                continue
            any_running = True
            action = spec["policy"](obs_list[i])
            obs, reward, terminated, truncated, _ = step_envs[i].step(action)
            obs_list[i] = obs
            st["last_reward"] = float(reward)
            st["total_reward"] += float(reward)
            st["steps"] = envs[i].step_count
            st["terminated"] = bool(terminated)
            st["truncated"] = bool(truncated)
            dist = float(np.linalg.norm(envs[i].ball_pos))
            st["dist_mm"] = 1000.0 * dist
            if dist < envs[i].GOAL_RADIUS:
                st["goal_steps"] += 1

        if not any_running:
            break

        should_draw = (
            step % step_skip == 0
            or step == max_steps
            or any(st["terminated"] and st["steps"] == step for st in states)
        )
        if should_draw:
            for i in range(len(specs)):
                st = states[i]
                # Keep rendering active envs, or render once on the exact failure step.
                if (not st["terminated"] and not st["truncated"]) or st["steps"] == step:
                    panel_imgs[i] = renderers[i].render(st["last_reward"], st["total_reward"])
            meta["step"] = step
            meta["sim_time"] = step * ref.DT
            meta["a_base_mag"] = current_a_base_mag(step)
            last_composed = compose_frame(panel_imgs, states, meta, fonts)
            writer.append_data(last_composed)
            frames_written += 1

    # Hold the final frame briefly so viewers can read the final returns.
    for _ in range(end_pause_frames):
        writer.append_data(last_composed)
        frames_written += 1

    # Print per-clip summary
    print("Clip {}/{} | Seed {} | c_rr={:.4f} | {} pulse(s) | {} frames".format(
        clip_idx, n_clips, seed, c_rr, n_pulses, frames_written))
    for st in states:
        goal_pct = 100.0 * st["goal_steps"] / max(st["steps"], 1)
        status = "BALL OFF TRAY" if st["terminated"] else "SURVIVED"
        print("  {:<24} steps={:3d}  return={:6.2f}  goal={:5.1f}%  [{}]".format(
            st["title"], st["steps"], st["total_reward"], goal_pct, status))

    for renderer, env in zip(renderers, envs):
        renderer.close()
        env.close()

    return frames_written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record a side-by-side MP4 comparing PD, LQR and Best RL Agent.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="path to trained RL checkpoint (default: {})".format(
                            DEFAULT_CHECKPOINT))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="episode seed(s) to record (default: {})".format(DEFAULT_SEEDS))
    parser.add_argument("--ideal", action="store_true",
                        help="use ideal environment instead of realistic (noise, lag, delay)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output MP4 path (default: {})".format(DEFAULT_OUT))
    parser.add_argument("--fps", type=int, default=25,
                        help="video frame rate in fps; 25 = real-time with skip 2 (default: 25)")
    parser.add_argument("--end-pause", type=float, default=1.5,
                        help="seconds to hold the final frame of each clip (default: 1.5)")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()

    gain, _ = design_lqr()
    lqr_policy = make_lqr_policy(gain)

    algo, rl_policy, checkpoint = load_policy(args.checkpoint)
    features = checkpoint_features(checkpoint)
    rl_features = None if is_default(features) else features
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    print("Checkpoint : {} ({}, step {:,})".format(
        args.checkpoint, algo.upper(), checkpoint.get("step", 0)))
    print("Agent input: {}".format(describe(features)))
    print("Condition  : {}".format("IDEAL" if args.ideal else "REALISTIC (noise, lag, delay)"))
    print("Seeds      : {}".format(args.seeds))
    print("Output     : {}".format(args.out))
    print("")

    specs = [
        {
            "panel_label": "PD Controller",
            "card_title": "PD Baseline",
            "policy": pd_policy,
            "features": None,
        },
        {
            "panel_label": "LQR Controller",
            "card_title": "LQR Baseline",
            "policy": lqr_policy,
            "features": None,
        },
        {
            "panel_label": "{} Agent (Best RL)".format(algo.upper()),
            "card_title": "{} Best ({})".format(algo.upper(), ckpt_name),
            "policy": rl_policy,
            "features": rl_features,
        },
    ]

    # Shorten card title if too long for the 640 px card header.
    if len(specs[2]["card_title"]) > 28:
        specs[2]["card_title"] = "{} Best RL Agent".format(algo.upper())

    env_kwargs = {} if args.ideal else dict(REALISTIC_KWARGS)
    step_skip = max(1, int(round((1.0 / BallOnTrayEnv.DT) / args.fps)))
    actual_fps = int(round((1.0 / BallOnTrayEnv.DT) / step_skip))
    end_pause_frames = int(round(args.end_pause * actual_fps))
    fonts = build_fonts()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_frames = 0
    writer = imageio.get_writer(
        args.out,
        fps=actual_fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=16,
    )
    try:
        for idx, seed in enumerate(args.seeds, start=1):
            total_frames += record_clip(
                writer, seed, specs, env_kwargs, step_skip, end_pause_frames,
                idx, len(args.seeds), fonts,
            )
    finally:
        writer.close()

    size_mb = os.path.getsize(args.out) / (1024.0 * 1024.0)
    print("\nSaved {} ({} frames, {:.1f} s @ {} fps, {:.2f} MB) in {:.1f} s".format(
        args.out, total_frames, total_frames / float(actual_fps), actual_fps,
        size_mb, time.perf_counter() - t0))


if __name__ == "__main__":
    main()
