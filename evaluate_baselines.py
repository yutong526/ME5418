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

Realism evaluation (optional, selected by command-line arguments):
    ``--realism-sweep`` runs both controllers, with unchanged gains, on the
    full-range cell under a fixed list of realism settings (observation noise,
    actuator lag, action delay and their combination) and writes
    ``results/eval_realism.csv``. Giving any of ``--pos-noise-std``,
    ``--vel-noise-std``, ``--actuator-tau`` or ``--action-delay-steps``
    instead evaluates that single setting next to the ideal one and writes
    ``results/eval_realism_custom.csv``. Without these arguments the script
    behaves exactly as described above and writes the same files as before.

Output directory: ``results/`` for the standard 100 episodes per cell. A run
with a different ``--episodes`` value writes to ``results_quick/`` instead, so
that a quick check never replaces the full results. ``--out-dir`` overrides
both.

Examples:
    python evaluate_baselines.py                    # full evaluation, 100 seeds per cell
    python evaluate_baselines.py --episodes 10      # quick check, written to results_quick/
    python evaluate_baselines.py --realism-sweep    # the realism comparison
    python evaluate_baselines.py --actuator-tau 0.08 --action-delay-steps 1
"""

import argparse
import csv
import os
import time

import numpy as np

from ball_on_tray_gym import BallOnTrayEnv
# The controllers are the ones of the demo, with unchanged gains.
from controllers import design_lqr, make_lqr_policy, pd_policy

# ------------------------------------------------------------------------- grid
FRICTION_VALUES = (0.005, 0.0275, 0.05)             # c_rr [-]
DISTURBANCE_VALUES = (None, 0.5, 1.0, 2.0, 3.0)     # ||a_base|| [m/s^2], None = no pulses
FULL_FRICTION_RANGE = (0.005, 0.05)                 # c_rr [-]
FULL_DISTURBANCE_RANGE = (0.5, 2.0)                 # ||a_base|| [m/s^2]
DEFAULT_EPISODES = 100                              # seeds 0 .. N-1 per cell
DEFAULT_OUT_DIR = "results"             # output of a run with DEFAULT_EPISODES
QUICK_OUT_DIR = "results_quick"         # output of a run with any other episode count

# ---------------------------------------------------------------------- metrics
MAX_RETURN = BallOnTrayEnv().max_steps * (1.0 + BallOnTrayEnv.GOAL_BONUS)   # 750
GOAL_RADIUS = BallOnTrayEnv.GOAL_RADIUS             # radius of the goal zone [m]
CALM_START_TIME = 1.0       # steps before this episode time are never "calm" [s]
CALM_AFTER_PULSE = 1.0      # nor are steps this soon after a disturbance ended [s]
SATURATION_TOL_DEG = 1e-6   # |tilt| >= limit - tol counts as saturated [deg]
FAILURE_THRESHOLD = 95.0    # survival below this is reported as "clearly failing" [%]
# A sign flip of the action is only counted if both actions are larger than
# this, so that floating-point residue around a zero action is not counted [-].
SIGN_FLIP_MIN_ACTION = 1e-3

# ---------------------------------------------------------------------- realism
# Settings of the realism sweep: (label, extra environment arguments).
REALISM_SETTINGS = (
    ("ideal", {}),
    ("noise", {"pos_noise_std": 0.002, "vel_noise_std": 0.03}),
    ("lag 0.05 s", {"actuator_tau": 0.05}),
    ("lag 0.10 s", {"actuator_tau": 0.10}),
    ("delay 1 step", {"action_delay_steps": 1}),
    ("delay 2 steps", {"action_delay_steps": 2}),
    ("noise+lag+delay", {"pos_noise_std": 0.002, "vel_noise_std": 0.03,
                         "actuator_tau": 0.05, "action_delay_steps": 1}),
)
REALISM_DEFAULTS = {"pos_noise_std": 0.0, "vel_noise_std": 0.0,
                    "actuator_tau": 0.0, "action_delay_steps": 0}
REALISM_CSV_FIELDS = [
    "setting", "pos_noise_std", "vel_noise_std", "actuator_tau", "action_delay_steps",
    "policy", "seed", "survived", "steps", "return", "return_pct_of_max",
    "steady_state_error_mm", "calm_steps", "goal_zone_fraction",
    "tilt_saturation_fraction", "action_change_calm", "sign_flips_per_s",
    "c_rr", "n_pulses",
    "fail_disturbance_active", "fail_disturbance_magnitude", "fail_time_since_pulse_s",
]

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
    # Action jitter, measured over pairs of consecutive calm steps.
    prev_action, prev_calm = None, False
    calm_pairs, change_sum, sign_flips = 0, 0.0, 0

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

        action = np.asarray(action, dtype=np.float64)
        if calm and prev_calm:
            change_sum += float(np.mean(np.abs(action - prev_action)))  # mean over both axes
            sign_flips += int(np.sum(
                (action * prev_action < 0.0)
                & (np.abs(action) > SIGN_FLIP_MIN_ACTION)
                & (np.abs(prev_action) > SIGN_FLIP_MIN_ACTION)))        # both axes
            calm_pairs += 1
        prev_action, prev_calm = action, calm

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
        # Mean |a_t - a_{t-1}| per axis over calm steps [-], and sign changes of
        # the action per second of calm time, averaged over the two axes [1/s].
        "action_change_calm": change_sum / calm_pairs if calm_pairs > 0 else np.nan,
        "sign_flips_per_s": (sign_flips / (2.0 * calm_pairs * dt)
                             if calm_pairs > 0 else np.nan),
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


# ---------------------------------------------------------- realism evaluation
def evaluate_realism(settings, policies, n_episodes):
    """Run the policies on the full-range cell under each realism setting.

    Args:
        settings: sequence of (label, extra environment arguments).
        policies: list of (name, policy_fn).
        n_episodes: seeds 0 .. n_episodes-1 are used for every setting.

    Returns:
        list of per-episode dicts (rows of the realism csv file).
    """
    rows = []
    for label, extra in settings:
        parameters = dict(REALISM_DEFAULTS, **extra)
        env = BallOnTrayEnv(friction_range=FULL_FRICTION_RANGE,
                            disturbance_range=FULL_DISTURBANCE_RANGE,
                            render_mode=None, **parameters)
        for name, policy_fn in policies:
            for seed in range(n_episodes):
                row = run_episode(env, policy_fn, seed)
                row.update(parameters)
                row["setting"] = label
                row["policy"] = name
                rows.append(row)
        env.close()
    return rows


def summarise_realism(cell_rows):
    """Aggregate the rows of one realism setting and one policy."""
    summary = summarise(cell_rows)
    summary["action_change"] = nan_mean([r["action_change_calm"] for r in cell_rows])
    summary["sign_flips"] = nan_mean([r["sign_flips_per_s"] for r in cell_rows])
    return summary


def print_realism_table(rows, settings, policy_names):
    """Print one line per realism setting with the policies side by side."""
    left, right = policy_names
    n_episodes = len([r for r in rows if r["setting"] == settings[0][0] and r["policy"] == left])
    print("Full randomisation range (c_rr in {}, |a_base| in {} m/s^2), {} seeds per setting."
          .format(FULL_FRICTION_RANGE, FULL_DISTURBANCE_RANGE, n_episodes))
    print("Metrics are shown as {} | {}; the gains are the same in every setting.".format(
        left.upper(), right.upper()))
    print("  surv %   episodes that lasted all 500 steps")
    print("  return   mean +- std of the episode return (maximum {:.0f})".format(MAX_RETURN))
    print("  % max    mean return as a percentage of the maximum")
    print("  sse mm   mean true distance to the centre during calm steps, survivors only")
    print("  goal %   share of steps inside the 2 cm goal zone")
    print("  |da|     mean |a_t - a_(t-1)| per axis during calm steps (action units)")
    print("  flips/s  sign changes of the action per second during calm steps, per axis")
    print("           (only actions larger than {:g} in magnitude are counted)".format(
        SIGN_FLIP_MIN_ACTION))
    print("")
    header = "{:<17} {:^13} {:^27} {:^13} {:^13} {:^13} {:^15} {:^13}".format(
        "setting", "surv %", "return", "% max", "sse mm", "goal %", "|da|", "flips/s")
    print(header)
    print("-" * len(header))
    for label, _ in settings:
        a = summarise_realism([r for r in rows if r["setting"] == label and r["policy"] == left])
        b = summarise_realism([r for r in rows if r["setting"] == label and r["policy"] == right])
        print("{:<17} {:>5.0f} | {:<5.0f} {:>6.1f}+-{:<5.1f}|{:>6.1f}+-{:<5.1f} "
              "{:>5.1f} | {:<5.1f} {:>5} | {:<5} {:>5.1f} | {:<5.1f} {:>6} | {:<6} "
              "{:>5} | {:<5}".format(
                  label, a["survival_pct"], b["survival_pct"],
                  a["return_mean"], a["return_std"], b["return_mean"], b["return_std"],
                  a["return_pct"], b["return_pct"],
                  fmt(a["sse_mm"], "{:.1f}"), fmt(b["sse_mm"], "{:.1f}"),
                  a["goal_pct"], b["goal_pct"],
                  fmt(a["action_change"], "{:.4f}"), fmt(b["action_change"], "{:.4f}"),
                  fmt(a["sign_flips"], "{:.1f}"), fmt(b["sign_flips"], "{:.1f}")))
    print("")


def save_realism_csv(rows, path):
    """Write one row per episode of the realism evaluation."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REALISM_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in REALISM_CSV_FIELDS})


def run_realism(args, custom, start):
    """Realism mode: the predefined sweep, or the ideal and one custom setting."""
    if args.realism_sweep:
        settings = REALISM_SETTINGS
        file_name = "eval_realism.csv"
    else:
        settings = (("ideal", {}), ("custom", custom))
        file_name = "eval_realism_custom.csv"

    gain, _ = design_lqr()
    policies = [("pd", pd_policy), ("lqr", make_lqr_policy(gain))]
    rows = evaluate_realism(settings, policies, args.episodes)
    print_realism_table(rows, settings, ("pd", "lqr"))

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    path = os.path.join(args.out_dir, file_name)
    save_realism_csv(rows, path)
    print("saved {} episodes to {}".format(len(rows), path))
    print("\ntotal time: {:.1f} s".format(time.perf_counter() - start))


# ------------------------------------------------------------------------- main
def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Batch evaluation of the PD and LQR baselines.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="episodes per cell and policy, seeds 0..N-1 (default: {})".format(
                            DEFAULT_EPISODES))
    parser.add_argument("--out-dir", default=None,
                        help="directory for the csv file and the heat maps (default: {} "
                             "with the default --episodes, otherwise {}, so that a quick "
                             "run does not replace the full results)".format(
                                 DEFAULT_OUT_DIR, QUICK_OUT_DIR))
    realism = parser.add_argument_group(
        "realism evaluation (full-range cell only; writes eval_realism*.csv instead of "
        "the default files)")
    realism.add_argument("--realism-sweep", action="store_true",
                         help="compare the predefined realism settings")
    realism.add_argument("--pos-noise-std", type=float, default=0.0, metavar="M",
                         help="position noise standard deviation [m] (default: 0)")
    realism.add_argument("--vel-noise-std", type=float, default=0.0, metavar="M_PER_S",
                         help="velocity noise standard deviation [m/s] (default: 0)")
    realism.add_argument("--actuator-tau", type=float, default=0.0, metavar="S",
                         help="actuator lag time constant [s] (default: 0)")
    realism.add_argument("--action-delay-steps", type=int, default=0, metavar="K",
                         help="action delay in control steps (default: 0)")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_DIR if args.episodes == DEFAULT_EPISODES else QUICK_OUT_DIR
    return args


def main():
    """Run the evaluation, print the tables and save the files."""
    args = parse_args()
    start = time.perf_counter()

    custom = {"pos_noise_std": args.pos_noise_std, "vel_noise_std": args.vel_noise_std,
              "actuator_tau": args.actuator_tau, "action_delay_steps": args.action_delay_steps}
    if args.realism_sweep or custom != REALISM_DEFAULTS:
        run_realism(args, custom, start)
        return

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
