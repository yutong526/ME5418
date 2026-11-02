"""Plot learning curves of one or more training runs, with the PD and LQR levels.

Two panels, both showing the evaluation return of the deterministic policy on
the evaluation seeds 500-519 (the same seeds ``train.py`` evaluates on):

    left    against environment steps    (sample efficiency)
    right   against training time        (compute cost)

The PD and LQR returns on the same seeds and conditions are drawn as reference
lines, so a curve reaching the LQR line means the agent matches the optimal
fixed-gain controller on those seeds.

Examples:
    python plot_training.py runs/ppo_staged_seed0 runs/sac_staged_seed0
    python plot_training.py runs/* --out media/learning_curves.png
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines      # noqa: E402
import matplotlib.pyplot as plt     # noqa: E402

from agents import EVAL_SEEDS, Curriculum, evaluate_policy             # noqa: E402
from controllers import design_lqr, make_lqr_policy, pd_policy          # noqa: E402

# Categorical slots in fixed order (one per run), then neutral references.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300")
LQR_COLOR, PD_COLOR = "#52514e", "#a3a29d"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_run(run_dir):
    """Return (label, steps, minutes, eval returns, config) of one run folder."""
    with open(os.path.join(run_dir, "config.json")) as handle:
        config = json.load(handle)
    steps, minutes, returns = [], [], []
    with open(os.path.join(run_dir, "progress.csv")) as handle:
        for row in csv.DictReader(handle):
            if row["eval_return"]:
                steps.append(int(row["step"]))
                minutes.append(float(row["time"]) / 60.0)
                returns.append(float(row["eval_return"]))
    label = config.get("algo_label") or _label(run_dir, config)
    return label, steps, minutes, returns, config


def _label(run_dir, config):
    """Readable legend label from the run's configuration."""
    algo = "PPO" if "gae_lambda" in config else "SAC"
    parts = [algo]
    if config.get("asymmetric"):
        parts.append("asymmetric critic")
    if config.get("curriculum", "staged") != "staged":
        parts.append("curriculum " + config["curriculum"])
    if config.get("realistic"):
        parts.append("realistic")
    return ", ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot learning curves of training runs.")
    parser.add_argument("runs", nargs="+", help="run folders written by train.py")
    parser.add_argument("--out", default="media/learning_curves.png", help="output image")
    parser.add_argument("--ymin", type=float, default=450.0,
                        help="lower end of the return axis (default: 450)")
    return parser.parse_args()


def main():
    args = parse_args()
    runs = [read_run(r) for r in args.runs]
    if len(runs) > len(SERIES_COLORS):
        raise SystemExit("at most {} runs per figure".format(len(SERIES_COLORS)))

    # Reference levels on the evaluation seeds, in the condition of the first run.
    first = runs[0][4]
    env_kwargs = Curriculum(first["curriculum"], 1, first["realistic"]).final_env_kwargs()
    gain, _ = design_lqr()
    lqr = evaluate_policy(make_lqr_policy(gain), EVAL_SEEDS, env_kwargs)["mean"]
    pd = evaluate_policy(pd_policy, EVAL_SEEDS, env_kwargs)["mean"]

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.05), sharey=True)
    for ax, xkey, xlabel in ((axes[0], 1, "environment steps"),
                             (axes[1], 2, "training time [min]")):
        for level, color, name in ((lqr, LQR_COLOR, "LQR"), (pd, PD_COLOR, "PD")):
            ax.axhline(level, color=color, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for i, run in enumerate(runs):
            ax.plot(run[xkey], run[3], color=SERIES_COLORS[i], lw=2.0, zorder=3,
                    label=run[0])
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(args.ymin, 745)
    axes[0].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: "{:g}k".format(v / 1000) if v else "0"))
    axes[0].set_ylabel("eval return (seeds 500-519)")
    # The two reference lines are 14 return units apart, too close for direct
    # labels, so they get dashed legend entries next to the runs.
    handles, labels = axes[1].get_legend_handles_labels()
    for level, color, name in ((lqr, LQR_COLOR, "LQR"), (pd, PD_COLOR, "PD")):
        handles.append(matplotlib.lines.Line2D([], [], color=color, lw=1.0, ls=(0, (4, 3))))
        labels.append("{} ({:.0f})".format(name, level))
    axes[1].legend(handles, labels, loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    directory = os.path.dirname(args.out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("saved {} (LQR {:.1f}, PD {:.1f} on the evaluation seeds)".format(args.out, lqr, pd))


if __name__ == "__main__":
    main()
