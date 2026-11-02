"""Evaluate trained agents next to the PD and LQR baselines.

Every policy runs on the same test seeds 0-99 (the seeds of
``evaluate_baselines.py``, never used in training or for picking the best
checkpoint), so each seed gives the same start position, c_rr and
disturbance schedule to every policy and the comparison is paired.

Examples:
    python evaluate_agent.py trained/*.pt                # the four reference agents
    python evaluate_agent.py runs/ppo_staged_seed0/best.pt
    python evaluate_agent.py runs/ppo_staged_seed0/best.pt runs/sac_staged_seed0/best.pt
    python evaluate_agent.py runs/ppo_staged_seed0/best.pt --realistic   # noise, lag, delay
    python evaluate_agent.py runs/*/best.pt --episodes 20                # quicker
    python evaluate_agent.py runs/ppo_staged_seed0/best.pt --csv results/agents.csv
"""

import argparse
import csv
import os

import numpy as np
import torch

from agents import TEST_SEEDS, evaluate_policy, load_policy
from controllers import design_lqr, make_lqr_policy, pd_policy
from demo import REALISTIC_KWARGS

MAX_RETURN = 750.0          # 1.5 per step x 500 steps


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Evaluate agents against PD and LQR.")
    parser.add_argument("checkpoints", nargs="+", help="checkpoints written by train.py")
    parser.add_argument("--episodes", type=int, default=len(TEST_SEEDS),
                        help="number of test seeds, from 0 (default: {})".format(
                            len(TEST_SEEDS)))
    parser.add_argument("--realistic", action="store_true",
                        help="evaluate with observation noise, actuator lag and action delay")
    parser.add_argument("--no-baselines", action="store_true", help="skip PD and LQR")
    parser.add_argument("--csv", metavar="PATH", default=None,
                        help="also write one row per policy and episode to this csv file")
    return parser.parse_args()


def main():
    """Evaluate every policy and print a table with paired differences to LQR."""
    args = parse_args()
    torch.set_num_threads(1)
    seeds = TEST_SEEDS[:args.episodes]
    env_kwargs = dict(REALISTIC_KWARGS) if args.realistic else {}

    policies = []
    if not args.no_baselines:
        gain, _ = design_lqr()
        policies += [("PD", pd_policy), ("LQR", make_lqr_policy(gain))]
    for path in args.checkpoints:
        algo, policy, checkpoint = load_policy(path)
        # runs/<run>/best.pt -> <run>; trained/<run>.pt -> <run>
        stem = os.path.splitext(os.path.basename(path))[0]
        run = (os.path.basename(os.path.dirname(os.path.abspath(path)))
               if stem in ("best", "last") else stem)
        policies.append(("{} {}".format(algo.upper(), run), policy))

    print("test seeds {}-{}, full randomisation range{}".format(
        seeds[0], seeds[-1], ", REALISTIC (noise, lag, delay)" if args.realistic else ""))
    print("  {:<38} {:>16} {:>9} {:>8} {:>8}  {}".format(
        "policy", "return", "% of max", "survival", "goal %", "vs LQR (paired)"))
    results = {}
    for name, policy in policies:
        res = evaluate_policy(policy, seeds, env_kwargs)
        results[name] = res
        paired = ""
        if "LQR" in results and name != "LQR":
            diff = res["returns"] - results["LQR"]["returns"]
            paired = "{:+7.1f}, better on {}/{}".format(diff.mean(), int(np.sum(diff > 0)),
                                                        len(diff))
        print("  {:<38} {:>8.1f} +- {:<5.1f} {:>8.1f}% {:>7.0f}% {:>7.1f}%  {}".format(
            name[:38], res["mean"], res["std"], 100.0 * res["mean"] / MAX_RETURN,
            100.0 * res["survival"], 100.0 * res["goal"], paired))

    if args.csv:
        directory = os.path.dirname(args.csv)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["policy", "seed", "return", "realistic"])
            for name, res in results.items():
                for seed, ret in zip(seeds, res["returns"]):
                    writer.writerow([name, seed, "{:.4f}".format(ret), int(args.realistic)])
        print("wrote {}".format(args.csv))


if __name__ == "__main__":
    main()
