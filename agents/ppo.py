"""PPO training loop for the Ball-on-Tray environment.

Builds on the stage 2 pieces: ``ActorCritic`` (networks/actor_critic.py) and
``ppo_loss`` / ``gae`` (networks/losses.py). One iteration of ``train_ppo``:

    1. curriculum: switch the environment ranges if a new stage has begun
    2. rollout: ``rollout_steps`` steps in each of ``num_envs`` environments
       with the current stochastic policy (2048 transitions by default)
    3. advantages: GAE(lambda) on the scaled rewards, bootstrapping from
       V(last observation) when an episode was truncated by the time limit
    4. update: ``epochs`` passes over the batch in shuffled minibatches,
       advantages normalised per minibatch, gradient norm clipped, learning
       rate annealed linearly to zero over the run
    5. log one CSV row; every ``eval_interval`` steps evaluate the
       deterministic policy on fixed held-out seeds and save checkpoints:
       ``last.pt`` always, ``best.pt`` only for the best evaluation within the
       last curriculum stage (see ``Curriculum.is_final_stage``)

``asymmetric=True`` trains the variant whose critic also receives c_rr and the
base acceleration (``privileged_features``), from the same ``info`` dicts the
environment returns; the actor never sees them.

Reward scale: the per-step reward is up to 1.5 and gamma = 0.99, so returns
reach ~150. Rewards are multiplied by ``reward_scale`` (0.1) for learning
only, which keeps value targets near 10 and the value loss on a scale
comparable with the policy loss. Logged returns are always unscaled.
"""

import os

import numpy as np
import torch
from torch import nn

from agents.common import (EVAL_SEEDS, Curriculum, ProgressLog, VecEnv, evaluate_policy,
                           recent_episode_stats)
from networks import PRIVILEGED_DIM, ActorCritic, gae, ppo_loss, privileged_features

PPO_DEFAULTS = {
    "total_steps": 1000000,     # environment steps, summed over all environments
    "num_envs": 8,              # parallel environments
    "rollout_steps": 256,       # steps per environment per iteration (batch = 8 x 256)
    "epochs": 10,               # passes over each batch
    "minibatch_size": 256,
    "learning_rate": 3e-4,      # Adam, annealed linearly to 0
    "gamma": 0.99,              # discount factor (horizon ~100 steps = 2 s)
    "gae_lambda": 0.95,
    "clip": 0.2,                # PPO clipping epsilon
    "value_coef": 0.5,
    "entropy_coef": 0.0,        # the std is a free parameter, which PPO lowers itself
    "max_grad_norm": 0.5,
    "reward_scale": 0.1,        # learning only; logs are unscaled
    "asymmetric": False,        # critic also gets c_rr and a_base
    "curriculum": "staged",
    "realistic": False,         # realism parameters in the last curriculum stage
    "eval_interval": 50000,     # steps between evaluations
    "seed": 0,
}

LOG_COLUMNS = ("step", "time", "stage", "lr", "train_return", "train_length", "train_falls",
               "policy_loss", "value_loss", "entropy", "std", "approx_kl", "clip_frac",
               "explained_var", "eval_return", "eval_std", "eval_survival", "eval_goal")


def _to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def save_ppo(path, network, config, step, evaluation=None, stage=None):
    """Save a PPO checkpoint: network configuration, weights and run metadata.

    ``step`` is the number of environment steps trained so far and ``stage``
    the curriculum stage of the rollout of the latest update.
    """
    torch.save({"algo": "ppo", "network_config": dict(network.config),
                "state_dict": network.state_dict(), "train_config": dict(config),
                "step": int(step), "stage": stage, "eval": evaluation}, path)


def load_ppo(checkpoint):
    """Rebuild the ``ActorCritic`` from a checkpoint dict written by ``save_ppo``."""
    network = ActorCritic(**checkpoint["network_config"])
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return network


def train_ppo(out_dir, verbose=True, **overrides):
    """Train PPO and write ``progress.csv``, ``last.pt`` and ``best.pt`` to ``out_dir``.

    Args:
        out_dir: output directory (created if needed).
        verbose: print one line per evaluation.
        **overrides: any key of ``PPO_DEFAULTS``.

    Returns:
        (network, config) after the last update.
    """
    unknown = set(overrides) - set(PPO_DEFAULTS)
    if unknown:
        raise ValueError("unknown PPO options: {}".format(sorted(unknown)))
    cfg = dict(PPO_DEFAULTS, **overrides)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg["seed"])

    n_envs, n_steps = cfg["num_envs"], cfg["rollout_steps"]
    batch_size = n_envs * n_steps
    n_iterations = max(1, cfg["total_steps"] // batch_size)
    asym = bool(cfg["asymmetric"])

    network = ActorCritic(privileged_dim=PRIVILEGED_DIM if asym else 0, seed=cfg["seed"])
    optimiser = torch.optim.Adam(network.parameters(), lr=cfg["learning_rate"], eps=1e-5)
    curriculum = Curriculum(cfg["curriculum"], n_iterations * batch_size, cfg["realistic"])
    _, stage_name, stage_kwargs = curriculum.stage_at(0)
    vec = VecEnv(n_envs, stage_kwargs, seed=cfg["seed"])
    eval_kwargs = curriculum.final_env_kwargs()
    log = ProgressLog(os.path.join(out_dir, "progress.csv"), LOG_COLUMNS)
    finished = []
    best_eval = -np.inf
    next_eval = cfg["eval_interval"]
    obs_dim = network.config["obs_dim"]

    for iteration in range(n_iterations):
        step = iteration * batch_size
        _, stage_name, stage_kwargs = curriculum.stage_at(step)
        # The rollout of this iteration starts at ``step``; it decides whether
        # the updated network has trained under the final conditions.
        in_final_stage = curriculum.is_final_stage(step)
        vec.set_env_kwargs(stage_kwargs)
        lr = cfg["learning_rate"] * (1.0 - iteration / float(n_iterations))
        for group in optimiser.param_groups:
            group["lr"] = lr

        # ------------------------------------------------------------ rollout
        buf = {"obs": np.zeros((n_steps, n_envs, obs_dim), np.float32),
               "next_obs": np.zeros((n_steps, n_envs, obs_dim), np.float32),
               "actions": np.zeros((n_steps, n_envs, 2), np.float32),
               "log_probs": np.zeros((n_steps, n_envs), np.float32),
               "rewards": np.zeros((n_steps, n_envs), np.float32),
               "terminated": np.zeros((n_steps, n_envs), np.float32),
               "truncated": np.zeros((n_steps, n_envs), np.float32)}
        priv_now, priv_next = [], []
        for t in range(n_steps):
            buf["obs"][t] = vec.obs
            if asym:
                priv_now.append(privileged_features(list(vec.infos)))
            with torch.no_grad():
                action, log_prob, env_action = network.act(_to_tensor(vec.obs))
            result = vec.step(env_action.numpy())
            buf["actions"][t] = action.numpy()
            buf["log_probs"][t] = log_prob.numpy()
            buf["rewards"][t] = result["rewards"] * cfg["reward_scale"]
            buf["terminated"][t] = result["terminated"]
            buf["truncated"][t] = result["truncated"]
            buf["next_obs"][t] = result["next_obs"]
            if asym:
                priv_next.append(privileged_features(result["next_infos"]))
            finished.extend(result["finished"])

        # --------------------------------------------------------- advantages
        flat_obs = _to_tensor(buf["obs"].reshape(batch_size, obs_dim))
        flat_priv = _to_tensor(np.concatenate(priv_now)) if asym else None
        with torch.no_grad():
            values = network.value(flat_obs, flat_priv).reshape(n_steps, n_envs)
            next_values = network.value(
                _to_tensor(buf["next_obs"].reshape(batch_size, obs_dim)),
                _to_tensor(np.concatenate(priv_next)) if asym else None).reshape(n_steps, n_envs)
        advantages, returns = gae(_to_tensor(buf["rewards"]), values, next_values,
                                  _to_tensor(buf["terminated"]), _to_tensor(buf["truncated"]),
                                  cfg["gamma"], cfg["gae_lambda"])
        flat = {"obs": flat_obs, "priv": flat_priv,
                "actions": _to_tensor(buf["actions"].reshape(batch_size, 2)),
                "log_probs": _to_tensor(buf["log_probs"].reshape(batch_size)),
                "advantages": advantages.reshape(batch_size),
                "returns": returns.reshape(batch_size)}

        # ------------------------------------------------------------- update
        losses = []
        for _ in range(cfg["epochs"]):
            order = torch.randperm(batch_size)
            for start in range(0, batch_size, cfg["minibatch_size"]):
                idx = order[start:start + cfg["minibatch_size"]]
                adv = flat["advantages"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                total, policy_loss, value_loss, entropy = ppo_loss(
                    network, flat["obs"][idx], flat["actions"][idx], flat["log_probs"][idx],
                    adv, flat["returns"][idx],
                    flat["priv"][idx] if asym else None, clip=cfg["clip"],
                    value_coef=cfg["value_coef"], entropy_coef=cfg["entropy_coef"])
                optimiser.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(network.parameters(), cfg["max_grad_norm"])
                optimiser.step()
                losses.append((policy_loss.item(), value_loss.item(), entropy.item()))

        # ---------------------------------------------------------- diagnostics
        with torch.no_grad():
            new_log_probs, _, _ = network.evaluate_actions(flat["obs"], flat["actions"],
                                                           flat["priv"])
            log_ratio = new_log_probs - flat["log_probs"]
            approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean().item()
            clip_frac = ((log_ratio.exp() - 1.0).abs() > cfg["clip"]).float().mean().item()
            residual = flat["returns"] - values.reshape(batch_size)
            explained = 1.0 - (residual.var() / (flat["returns"].var() + 1e-8)).item()
        step = (iteration + 1) * batch_size
        mean_losses = np.mean(losses, axis=0)
        row = {"step": step, "time": log.elapsed(), "stage": stage_name, "lr": lr,
               "policy_loss": mean_losses[0], "value_loss": mean_losses[1],
               "entropy": mean_losses[2], "std": network.actor.log_std.exp().mean().item(),
               "approx_kl": approx_kl, "clip_frac": clip_frac, "explained_var": explained}
        row.update(recent_episode_stats(finished))

        # --------------------------------------------------- evaluation, saving
        if step >= next_eval or iteration == n_iterations - 1:
            next_eval += cfg["eval_interval"]
            result = evaluate_policy(lambda o: network.predict(o, deterministic=True),
                                     EVAL_SEEDS, eval_kwargs)
            row.update({"eval_return": result["mean"], "eval_std": result["std"],
                        "eval_survival": result["survival"], "eval_goal": result["goal"]})
            summary = {k: result[k] for k in ("mean", "std", "survival", "goal")}
            save_ppo(os.path.join(out_dir, "last.pt"), network, cfg, step, summary, stage_name)
            # best.pt is chosen only among checkpoints of the last stage.
            if in_final_stage and result["mean"] > best_eval:
                best_eval = result["mean"]
                save_ppo(os.path.join(out_dir, "best.pt"), network, cfg, step, summary,
                         stage_name)
            if verbose:
                print("[ppo] step {:>8,}  {:<20} train {:7.1f}  eval {:7.1f} +- {:5.1f}  "
                      "survival {:4.0%}  std {:.3f}  {:5.0f} s".format(
                          step, stage_name, row.get("train_return", float("nan")),
                          result["mean"], result["std"], result["survival"], row["std"],
                          log.elapsed()), flush=True)
        log.write(row)

    vec.close()
    return network, cfg
