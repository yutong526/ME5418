"""Train a PPO or SAC agent on the Ball-on-Tray environment.

Writes to ``runs/<name>/`` (``runs/`` is ignored by git):

    progress.csv   one row per PPO iteration / SAC log interval: training
                   returns, losses, and every ``--eval-interval`` steps the
                   return of the deterministic policy on seeds 500-519
    best.pt        checkpoint with the best evaluation return so far
    last.pt        checkpoint at the last evaluation

Examples:
    python train.py --algo ppo --quick              # ~1 min smoke run, checks everything runs
    python train.py --algo sac --quick
    python train.py --algo ppo                      # 1M steps, staged curriculum (~10 min)
    python train.py --algo ppo --asymmetric         # critic also sees c_rr and a_base
    python train.py --algo sac --steps 200000       # SAC is slower per step, more sample-efficient
    python train.py --algo ppo --curriculum full    # ablation: full randomisation from step 0
    python train.py --algo ppo --realistic          # noise, lag and delay in the last stage

Then ``python evaluate_agent.py runs/<name>/best.pt`` compares the agent with
PD and LQR on the test seeds 0-99, and ``python demo.py --checkpoint
runs/<name>/best.pt`` shows it in the 3D view.
"""

import argparse
import json
import os
import time

import torch

from agents import CURRICULA, PPO_DEFAULTS, SAC_DEFAULTS, train_ppo, train_sac

# --quick: a short run that exercises every code path (curriculum stages,
# updates, evaluation, checkpoints) in about a minute. It is not expected to
# learn a good policy.
QUICK = {
    "ppo": {"total_steps": 16384, "eval_interval": 8192},
    "sac": {"total_steps": 3000, "warmup_steps": 1000, "eval_interval": 1500,
            "log_interval": 500},
}


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Train a PPO or SAC agent on Ball-on-Tray.")
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True)
    parser.add_argument("--steps", type=int, default=None,
                        help="environment steps (default: PPO {:,}, SAC {:,})".format(
                            PPO_DEFAULTS["total_steps"], SAC_DEFAULTS["total_steps"]))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum", choices=sorted(CURRICULA), default="staged",
                        help="staged domain randomisation (default), full range from the "
                             "start, or the fixed easy task")
    parser.add_argument("--realistic", action="store_true",
                        help="add observation noise, actuator lag and action delay "
                             "(demo.py --realistic) to the last curriculum stage")
    parser.add_argument("--asymmetric", action="store_true",
                        help="PPO only: the critic also receives c_rr and a_base")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="environment steps between evaluations")
    parser.add_argument("--quick", action="store_true",
                        help="short smoke run (about a minute); overrides --steps")
    parser.add_argument("--name", default=None,
                        help="run name, i.e. the folder in runs/ (default: from the options)")
    parser.add_argument("--out-root", default="runs", help="parent folder of the run folders")
    args = parser.parse_args()
    if args.asymmetric and args.algo != "ppo":
        parser.error("--asymmetric applies to PPO only")
    return args


def main():
    """Build the configuration, train, and print where the results are."""
    args = parse_args()
    torch.set_num_threads(1)        # small networks: one thread is fastest and reproducible
    options = {"seed": args.seed, "curriculum": args.curriculum, "realistic": args.realistic}
    if args.algo == "ppo":
        options["asymmetric"] = args.asymmetric
    if args.steps is not None:
        options["total_steps"] = args.steps
    if args.eval_interval is not None:
        options["eval_interval"] = args.eval_interval
    if args.quick:
        options.update(QUICK[args.algo])

    name = args.name or "{}{}_{}{}_seed{}{}".format(
        args.algo, "_asym" if args.asymmetric else "", args.curriculum,
        "_realistic" if args.realistic else "", args.seed, "_quick" if args.quick else "")
    out_dir = os.path.join(args.out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    defaults = PPO_DEFAULTS if args.algo == "ppo" else SAC_DEFAULTS
    config = dict(defaults, **options)
    with open(os.path.join(out_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    print("training {} for {:,} steps -> {}".format(args.algo.upper(), config["total_steps"],
                                                     out_dir))

    start = time.perf_counter()
    if args.algo == "ppo":
        train_ppo(out_dir, **options)
    else:
        train_sac(out_dir, **options)
    print("done in {:.0f} s; results in {}".format(time.perf_counter() - start, out_dir))
    print("next: python evaluate_agent.py {}".format(os.path.join(out_dir, "best.pt")))


if __name__ == "__main__":
    main()
