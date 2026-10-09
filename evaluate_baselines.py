"""Batch evaluation of the classical baselines (PD and LQR) on Ball-on-Tray.

Runs both controllers, with exactly the gains used by the demos, over a grid of
fixed rolling-resistance and disturbance conditions plus the full randomisation
range, and reports how hard each condition is for them. Nothing is rendered.

Grid:
    c_rr         0.005, 0.0275, 0.05                (friction_range = (c, c))
    disturbance  none, 0.5, 1.0, 2.0, 3.0 m/s^2     (disturbance_range = (m, m))
    plus one "full range" cell: c_rr in (0.005, 0.05), disturbance in (0.5, 2.0)

Only the magnitude of the pulses is fixed; their number, start time, duration
and direction are still drawn by the environment. 3.0 m/s^2 is deliberately
outside the training range: holding the ball against it needs a tilt of about
asin(3.0 / 9.81) = 17.8 deg, more than the 15 deg tilt limit.

Every cell runs the same seeds for PD and LQR, so the two are compared on
identical episodes (same start position and disturbance schedule). The random
policy is only run on the full-range cell as a lower reference.

Outputs:
    terminal                              summary table, failure analysis, run time
    results/eval_baselines.csv            one row per episode
    results/survival_heatmap_pd.png       survival rate, c_rr vs disturbance
    results/survival_heatmap_lqr.png

Examples:
    python evaluate_baselines.py                    # full evaluation, 100 seeds per cell
    python evaluate_baselines.py --episodes 10      # quick check
"""

import argparse
import csv
import os
import time

import numpy as np

from ball_on_tray_gym import BallOnTrayEnv
# The controllers are the ones of the demos, with unchanged gains.
from demo import pd_policy
from demo_lqr import design_lqr, make_lqr_policy

# ------------------------------------------------------------------------- grid
FRICTION_VALUES = (0.005, 0.0275, 0.05)             # c_rr [-]
DISTURBANCE_VALUES = (None, 0.5, 1.0, 2.0, 3.0)     # ||a_base|| [m/s^2], None = no pulses
FULL_FRICTION_RANGE = (0.005, 0.05)                 # c_rr [-]
FULL_DISTURBANCE_RANGE = (0.5, 2.0)                 # ||a_base|| [m/s^2]
DEFAULT_EPISODES = 100                              # seeds 0 .. N-1 per cell

# ---------------------------------------------------------------------- metrics
MAX_RETURN = BallOnTrayEnv().max_steps * (1.0 + BallOnTrayEnv.GOAL_BONUS)   # 750
GOAL_RADIUS = BallOnTrayEnv.GOAL_RADIUS             # radius of the goal zone [m]
CALM_START_TIME = 1.0       # steps before this episode time are never "calm" [s]
CALM_AFTER_PULSE = 1.0      # nor are steps this soon after a disturbance ended [s]
SATURATION_TOL_DEG = 1e-6   # |tilt| >= limit - tol counts as saturated [deg]
FAILURE_THRESHOLD = 95.0    # survival below this is reported as "clearly failing" [%]

CSV_FIELDS = [
    "condition", "c_rr_setting", "disturbance_setting", "policy", "seed",
    "survived", "steps", "return", "return_pct_of_max",
    "steady_state_error_mm", "calm_steps", "time_to_goal_s",
    "goal_zone_fraction", "tilt_saturation_fraction",
    "c_rr", "n_pulses",
    "fail_disturbance_active", "fail_disturbance_magnitude", "fail_time_since_pulse_s",
]


# ------------------------------------------------------------------- conditions
def make_conditions():
    """Return the list of evaluation cells.

    Each cell is a dict with a label, the environment arguments and the values
    shown in the tables (``c_rr`` / ``disturbance`` are None for the full range).
    """
    conditions = []
    for c_rr in FRICTION_VALUES:
        for magnitude in DISTURBANCE_VALUES:
            conditions.append({
                "label": "c_rr={:g}, dist={}".format(
                    c_rr, "none" if magnitude is None else "{:g}".format(magnitude)),
                "c_rr": c_rr,
                "disturbance": magnitude,
                "full_range": False,
                "env_kwargs": {
                    "friction_range": (c_rr, c_rr),
                    "disturbance_range": None if magnitude is None else (magnitude, magnitude),
                },
            })
    conditions.append({
        "label": "full range",
        "c_rr": None,
        "disturbance": None,
        "full_range": True,
        "env_kwargs": {"friction_range": FULL_FRICTION_RANGE,
                       "disturbance_range": FULL_DISTURBANCE_RANGE},
    })
    return conditions


# --------------------------------------------------------------------- episodes
def run_episode(env, policy_fn, seed):
    """Run one episode without rendering and return its metrics.

    Args:
        env: the environment of the cell.
        policy_fn: ``policy_fn(obs) -> action``, or None for the random policy.
        seed: seed of the episode (and of the random policy).

    Returns:
        dict with the per-episode metrics (see ``CSV_FIELDS``). Metrics that
        are undefined for the episode are NaN.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    dt = env.DT
    max_tilt_deg = env.MAX_TILT_DEG

    dist = float(np.linalg.norm(env.ball_pos))                  # [m]
    time_to_goal = 0.0 if dist < GOAL_RADIUS else np.nan        # [s]
    total_reward = 0.0
    goal_steps = 0
    saturated_steps = 0
    calm_dist_sum, calm_steps = 0.0, 0
    last_active_time = None     # episode time at which a pulse was last active [s]
    terminated = truncated = False
    info = {}

    while not (terminated or truncated):
        action = env.action_space.sample() if policy_fn is None else policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        now = env.step_count * dt                               # time after this step [s]
        dist = float(np.linalg.norm(env.ball_pos))
        active = bool(np.any(info["a_base"] != 0.0))            # pulse applied in this step
        if active:
            last_active_time = now

        if dist < GOAL_RADIUS:
            goal_steps += 1
            if np.isnan(time_to_goal):
                time_to_goal = now
        if np.max(np.abs(np.rad2deg(env.tilt))) >= max_tilt_deg - SATURATION_TOL_DEG:
            saturated_steps += 1

        # Calm step: after the first second, no pulse now, and more than one
        # second since the last pulse ended.
        calm = (now > CALM_START_TIME and not active
                and (last_active_time is None or now - last_active_time > CALM_AFTER_PULSE))
        if calm and not terminated:
            calm_dist_sum += dist
            calm_steps += 1

    steps = env.step_count
    survived = bool(truncated and not terminated)

    result = {
        "seed": seed,
        "survived": int(survived),
        "steps": steps,
        "return": total_reward,
        "return_pct_of_max": 100.0 * total_reward / MAX_RETURN,
        # Steady-state error only for surviving episodes that had calm steps.
        "steady_state_error_mm": (1000.0 * calm_dist_sum / calm_steps
                                  if survived and calm_steps > 0 else np.nan),
        "calm_steps": calm_steps,
        "time_to_goal_s": time_to_goal,
        "goal_zone_fraction": goal_steps / float(steps),
        "tilt_saturation_fraction": saturated_steps / float(steps),
        "c_rr": env.c_rr,
        "n_pulses": len(env.disturbances),
        "fail_disturbance_active": np.nan,
        "fail_disturbance_magnitude": np.nan,
        "fail_time_since_pulse_s": np.nan,
    }
    if not survived:
        # State of the disturbance in the step in which the ball left the tray.
        magnitude = float(np.linalg.norm(info["a_base"]))
        result["fail_disturbance_active"] = int(magnitude > 0.0)
        result["fail_disturbance_magnitude"] = magnitude
        if magnitude > 0.0:
            result["fail_time_since_pulse_s"] = 0.0
        elif last_active_time is not None:
            result["fail_time_since_pulse_s"] = steps * dt - last_active_time
    return result


def evaluate(conditions, policies, n_episodes):
    """Run every policy on every cell.

    Args:
        conditions: list from ``make_conditions``.
        policies: list of (name, policy_fn, full_range_only).
        n_episodes: seeds 0 .. n_episodes-1 are used in every cell.

    Returns:
        list of per-episode dicts (rows of the csv file).
    """
    rows = []
    for condition in conditions:
        env = BallOnTrayEnv(render_mode=None, **condition["env_kwargs"])
        for name, policy_fn, full_range_only in policies:
            if full_range_only and not condition["full_range"]:
                continue
            for seed in range(n_episodes):
                row = run_episode(env, policy_fn, seed)
                row["condition"] = condition["label"]
                row["c_rr_setting"] = ("full" if condition["full_range"]
                                       else condition["c_rr"])
                row["disturbance_setting"] = (
                    "full" if condition["full_range"]
                    else 0.0 if condition["disturbance"] is None
                    else condition["disturbance"])
                row["policy"] = name
                rows.append(row)
        env.close()
    return rows


# ------------------------------------------------------------------ aggregation
def select(rows, condition_label, policy):
    """Return the rows of one cell and one policy."""
    return [r for r in rows if r["condition"] == condition_label and r["policy"] == policy]


def nan_mean(values):
    """Mean over the non-NaN entries, or NaN if there are none."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else np.nan


def summarise(cell_rows):
    """Aggregate the per-episode rows of one cell and one policy."""
    returns = np.array([r["return"] for r in cell_rows])
    reached = [r["time_to_goal_s"] for r in cell_rows]
    return {
        "n": len(cell_rows),
        "survival_pct": 100.0 * np.mean([r["survived"] for r in cell_rows]),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_pct": 100.0 * float(returns.mean()) / MAX_RETURN,
        "sse_mm": nan_mean([r["steady_state_error_mm"] for r in cell_rows]),
        "time_to_goal_s": nan_mean(reached),
        "never_reached": int(np.sum(np.isnan(reached))),
        "goal_pct": 100.0 * np.mean([r["goal_zone_fraction"] for r in cell_rows]),
        "saturation_pct": 100.0 * np.mean([r["tilt_saturation_fraction"] for r in cell_rows]),
    }


def fmt(value, pattern):
    """Format a number, printing '-' for NaN."""
    return "-" if np.isnan(value) else pattern.format(value)


# --------------------------------------------------------------------- printing
def print_summary_table(rows, conditions, policy_names):
    """Print one line per cell with the metrics of the policies side by side."""
    left, right = policy_names
    print("Metrics are shown as {} | {}. {} seeds per cell.".format(
        left.upper(), right.upper(), len(select(rows, conditions[0]["label"], left))))
    print("  surv %   episodes that lasted all 500 steps")
    print("  return   mean +- std of the episode return (maximum {:.0f})".format(MAX_RETURN))
    print("  % max    mean return as a percentage of the maximum")
    print("  sse mm   mean distance to the centre during calm steps, surviving episodes only")
    print("  t_goal   mean time until the ball first enters the 2 cm goal zone [s]")
    print("  goal %   share of steps inside the goal zone")
    print("  sat %    share of steps with |pitch| or |roll| at the 15 deg limit")
    print("")
    header = "{:<8}{:<6} {:^13} {:^27} {:^13} {:^13} {:^13} {:^13} {:^13}".format(
        "c_rr", "dist", "surv %", "return", "% max", "sse mm", "t_goal s", "goal %", "sat %")
    print(header)
    print("-" * len(header))
    for condition in conditions:
        a = summarise(select(rows, condition["label"], left))
        b = summarise(select(rows, condition["label"], right))
        if condition["full_range"]:
            c_rr_text, dist_text = "full", "full"
        else:
            c_rr_text = "{:g}".format(condition["c_rr"])
            dist_text = "none" if condition["disturbance"] is None else "{:g}".format(
                condition["disturbance"])
        print("{:<8}{:<6} {:>5.0f} | {:<5.0f} {:>6.1f}+-{:<5.1f}|{:>6.1f}+-{:<5.1f} "
              "{:>5.1f} | {:<5.1f} {:>5} | {:<5} {:>5} | {:<5} {:>5.1f} | {:<5.1f} "
              "{:>5.1f} | {:<5.1f}".format(
                  c_rr_text, dist_text,
                  a["survival_pct"], b["survival_pct"],
                  a["return_mean"], a["return_std"], b["return_mean"], b["return_std"],
                  a["return_pct"], b["return_pct"],
                  fmt(a["sse_mm"], "{:.1f}"), fmt(b["sse_mm"], "{:.1f}"),
                  fmt(a["time_to_goal_s"], "{:.2f}"), fmt(b["time_to_goal_s"], "{:.2f}"),
                  a["goal_pct"], b["goal_pct"],
                  a["saturation_pct"], b["saturation_pct"]))
    print("")


def print_reference(rows, conditions, name):
    """Print the full-range result of the lower-reference policy."""
    label = [c["label"] for c in conditions if c["full_range"]][0]
    cell = select(rows, label, name)
    if not cell:
        return
    s = summarise(cell)
    print("Reference, {} policy on the full range: survival {:.0f} %, return {:.1f} +- {:.1f} "
          "({:.1f} % of max), mean episode length {:.1f} steps, goal zone {:.1f} %".format(
              name, s["survival_pct"], s["return_mean"], s["return_std"], s["return_pct"],
              np.mean([r["steps"] for r in cell]), s["goal_pct"]))
    print("")


def print_failures(rows, conditions, policy_names):
    """Print, per cell and policy, when the failed episodes failed."""
    print("Failures (cells with at least one failed episode)")
    header = "{:<22}{:<7}{:>8}{:>18}{:>20}{:>22}{:>14}".format(
        "condition", "policy", "failed", "during a pulse", "mean |a_base| then",
        "others: s after pulse", "mean c_rr")
    print(header)
    print("-" * len(header))
    any_failure = False
    for condition in conditions:
        for name in policy_names:
            failed = [r for r in select(rows, condition["label"], name) if not r["survived"]]
            if not failed:
                continue
            any_failure = True
            during = [r for r in failed if r["fail_disturbance_active"] == 1]
            after = [r["fail_time_since_pulse_s"] for r in failed
                     if r["fail_disturbance_active"] == 0]
            print("{:<22}{:<7}{:>8d}{:>12d} / {:<3d}{:>20}{:>22}{:>14.4f}".format(
                condition["label"], name, len(failed), len(during), len(failed),
                fmt(nan_mean([r["fail_disturbance_magnitude"] for r in during]), "{:.2f}"),
                fmt(nan_mean(after), "{:.2f}"),
                np.mean([r["c_rr"] for r in failed])))
    if not any_failure:
        print("(no failed episodes)")
    print("")


def print_threshold_report(rows, conditions, policy_names):
    """List the cells where a policy survives less often than the threshold."""
    print("Cells with survival below {:.0f} %".format(FAILURE_THRESHOLD))
    for name in policy_names:
        bad = []
        for condition in conditions:
            s = summarise(select(rows, condition["label"], name))
            if s["survival_pct"] < FAILURE_THRESHOLD:
                bad.append("{} ({:.0f} %)".format(condition["label"], s["survival_pct"]))
        print("  {:<4}: {}".format(name, "; ".join(bad) if bad else "none"))
    print("")


# ----------------------------------------------------------------------- output
def save_csv(rows, path):
    """Write one row per episode."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_FIELDS})


def save_heatmap(rows, conditions, policy, path):
    """Save the survival-rate heat map of one policy (c_rr vs disturbance)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.zeros((len(FRICTION_VALUES), len(DISTURBANCE_VALUES)))
    for condition in conditions:
        if condition["full_range"]:
            continue
        i = FRICTION_VALUES.index(condition["c_rr"])
        j = DISTURBANCE_VALUES.index(condition["disturbance"])
        grid[i, j] = summarise(select(rows, condition["label"], policy))["survival_pct"]
    n_episodes = len(select(rows, conditions[0]["label"], policy))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=120)
    # One hue, light to dark: darker means more episodes survived.
    image = ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_xticks(range(len(DISTURBANCE_VALUES)))
    ax.set_xticklabels(["none" if m is None else "{:g}".format(m) for m in DISTURBANCE_VALUES])
    ax.set_yticks(range(len(FRICTION_VALUES)))
    ax.set_yticklabels(["{:g}".format(c) for c in FRICTION_VALUES])
    ax.set_xlabel("disturbance magnitude [m/s$^2$]")
    ax.set_ylabel("rolling resistance c_rr [-]")
    ax.set_title("{}: survival rate [%] ({} seeds per cell)".format(policy.upper(), n_episodes))
    # White gaps between the cells and a value in every cell.
    ax.set_xticks(np.arange(-0.5, len(DISTURBANCE_VALUES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FRICTION_VALUES)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, "{:.0f}".format(grid[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if grid[i, j] > 60.0 else "#222222")
    fig.colorbar(image, ax=ax, label="survival rate [%]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------- main
def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Batch evaluation of the PD and LQR baselines.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="episodes per cell and policy, seeds 0..N-1 (default: {})".format(
                            DEFAULT_EPISODES))
    parser.add_argument("--out-dir", default="results",
                        help="directory for the csv file and the heat maps (default: results)")
    return parser.parse_args()


def main():
    """Run the evaluation, print the tables and save the files."""
    args = parse_args()
    start = time.perf_counter()

    gain, _ = design_lqr()
    # (name, policy function, only on the full-range cell)
    policies = [("pd", pd_policy, False),
                ("lqr", make_lqr_policy(gain), False),
                ("random", None, True)]
    conditions = make_conditions()
    rows = evaluate(conditions, policies, args.episodes)

    print_summary_table(rows, conditions, ("pd", "lqr"))
    print_reference(rows, conditions, "random")
    print_threshold_report(rows, conditions, ("pd", "lqr"))
    print_failures(rows, conditions, ("pd", "lqr"))

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    csv_path = os.path.join(args.out_dir, "eval_baselines.csv")
    save_csv(rows, csv_path)
    print("saved {} episodes to {}".format(len(rows), csv_path))
    for name in ("pd", "lqr"):
        path = os.path.join(args.out_dir, "survival_heatmap_{}.png".format(name))
        save_heatmap(rows, conditions, name, path)
        print("saved {}".format(path))

    print("\ntotal time: {:.1f} s".format(time.perf_counter() - start))


if __name__ == "__main__":
    main()
