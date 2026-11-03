"""SAC training loop for the Ball-on-Tray environment.

Builds on the stage 2 pieces: ``SACActor`` / ``SACCritic`` (networks/sac.py)
and ``sac_critic_loss`` / ``sac_actor_loss`` / ``sac_alpha_loss``
(networks/losses.py). What stage 2 left to "the training loop" is here:

    replay buffer    off-policy: every transition is stored and reused
    target critic    a copy of the twin critic, updated by Polyak averaging
                     theta' <- (1 - tau) theta' + tau theta; the critic's
                     target y is computed with it, not with the critic itself
    entropy tuning   alpha is learned (``log_alpha``) so that the policy
                     entropy tracks ``target_entropy`` = -action_dim
    warm-up          the first ``warmup_steps`` actions are uniformly random,
                     to fill the buffer before any update

One environment step is followed by ``updates_per_step`` gradient updates of
(critic, actor, alpha) on minibatches drawn from the buffer.

Only ``terminated`` (the ball fell) is stored as ``done``. A transition that
hit the time limit keeps bootstrapping from its true next observation,
because the episode was cut off, not finished. Rewards are scaled by
``reward_scale`` for learning only, as in PPO.
"""

import copy
import os

import numpy as np
import torch

from agents.common import (EVAL_SEEDS, Curriculum, ProgressLog, VecEnv, evaluate_policy,
                           recent_episode_stats)
from networks import (ACTION_DIM, OBS_DIM, SACActor, SACCritic, sac_actor_loss,
                      sac_alpha_loss, sac_critic_loss)

SAC_DEFAULTS = {
    "total_steps": 200000,      # environment steps
    "num_envs": 1,
    "buffer_size": 300000,      # transitions kept in the replay buffer
    "batch_size": 256,
    "learning_rate": 3e-4,      # Adam, actor, critic and alpha
    "gamma": 0.99,
    "tau": 0.005,               # Polyak coefficient of the target critic
    "warmup_steps": 5000,       # uniformly random actions before learning starts
    "updates_per_step": 1,      # gradient updates per environment step
    "init_alpha": 0.2,          # initial entropy temperature (the stage 2 fixed value)
    "auto_alpha": True,         # learn alpha; False keeps init_alpha fixed
    "target_entropy": -float(ACTION_DIM),
    "reward_scale": 0.1,        # learning only; logs are unscaled
    "curriculum": "staged",
    "realistic": False,
    "eval_interval": 10000,
    "log_interval": 2000,
    "seed": 0,
}

LOG_COLUMNS = ("step", "time", "stage", "train_return", "train_length", "train_falls",
               "critic_loss", "actor_loss", "alpha", "entropy", "q_mean",
               "eval_return", "eval_std", "eval_survival", "eval_goal")


class ReplayBuffer(object):
    """Fixed-size circular buffer of transitions (s, a, r, s', terminated).

    Args:
        capacity: maximum number of transitions; the oldest are overwritten.
        seed: seed of the minibatch sampler.
    """

    def __init__(self, capacity, obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), np.float32)
        self.actions = np.zeros((self.capacity, action_dim), np.float32)
        self.rewards = np.zeros(self.capacity, np.float32)
        self.terminated = np.zeros(self.capacity, np.float32)
        self.position = 0
        self.size = 0
        self._rng = np.random.RandomState(seed)

    def __len__(self):
        return self.size

    def add(self, obs, actions, rewards, next_obs, terminated):
        """Add a batch of N transitions (arrays with a leading axis N)."""
        for i in range(len(rewards)):
            p = self.position
            self.obs[p], self.actions[p], self.rewards[p] = obs[i], actions[i], rewards[i]
            self.next_obs[p], self.terminated[p] = next_obs[i], terminated[i]
            self.position = (p + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Uniform minibatch as a dict of float32 tensors."""
        idx = self._rng.randint(0, self.size, size=batch_size)
        return {key: torch.as_tensor(getattr(self, key)[idx])
                for key in ("obs", "actions", "rewards", "next_obs", "terminated")}


@torch.no_grad()
def polyak_update(target, source, tau):
    """theta_target <- (1 - tau) theta_target + tau theta_source."""
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(1.0 - tau).add_(tau * s)


def save_sac(path, actor, critic, target, log_alpha, config, step, evaluation=None,
             stage=None):
    """Save a SAC checkpoint (the actor is all a deployed policy needs).

    ``step`` is the number of environment steps so far and ``stage`` the
    curriculum stage of the latest environment step.
    """
    torch.save({"algo": "sac", "actor": actor.state_dict(), "critic": critic.state_dict(),
                "target_critic": target.state_dict(), "log_alpha": float(log_alpha.item()),
                "train_config": dict(config), "step": int(step), "stage": stage,
                "eval": evaluation}, path)


def load_sac_actor(checkpoint):
    """Rebuild the ``SACActor`` from a checkpoint dict written by ``save_sac``."""
    actor = SACActor()
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def sac_update(batch, actor, critic, target, log_alpha, optimisers, cfg):
    """One SAC update: critic, then actor, then alpha, then the target critic.

    Returns:
        dict of floats for logging.
    """
    alpha = log_alpha.exp().detach()

    critic_loss = sac_critic_loss(actor, critic, target, batch["obs"], batch["actions"],
                                  batch["rewards"], batch["next_obs"], batch["terminated"],
                                  cfg["gamma"], alpha)
    optimisers["critic"].zero_grad()
    critic_loss.backward()
    optimisers["critic"].step()

    # The actor loss differentiates through Q(s, a~); freeze the critic so that
    # this backward pass only produces gradients for the actor.
    for p in critic.parameters():
        p.requires_grad_(False)
    actor_loss = sac_actor_loss(actor, critic, batch["obs"], alpha, use_min=True)
    optimisers["actor"].zero_grad()
    actor_loss.backward()
    optimisers["actor"].step()
    for p in critic.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        _, log_probs = actor(batch["obs"])
    if cfg["auto_alpha"]:
        alpha_loss = sac_alpha_loss(log_alpha, log_probs, cfg["target_entropy"])
        optimisers["alpha"].zero_grad()
        alpha_loss.backward()
        optimisers["alpha"].step()

    polyak_update(target, critic, cfg["tau"])
    with torch.no_grad():
        q1, _ = critic(batch["obs"], batch["actions"])
    return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
            "alpha": log_alpha.exp().item(), "entropy": -log_probs.mean().item(),
            "q_mean": q1.mean().item()}


def train_sac(out_dir, verbose=True, **overrides):
    """Train SAC and write ``progress.csv``, ``last.pt`` and ``best.pt`` to ``out_dir``.

    Args:
        out_dir: output directory (created if needed).
        verbose: print one line per evaluation.
        **overrides: any key of ``SAC_DEFAULTS``.

    Returns:
        (actor, config) after the last step.
    """
    unknown = set(overrides) - set(SAC_DEFAULTS)
    if unknown:
        raise ValueError("unknown SAC options: {}".format(sorted(unknown)))
    cfg = dict(SAC_DEFAULTS, **overrides)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    action_rng = np.random.RandomState(cfg["seed"] + 1)

    actor = SACActor(seed=cfg["seed"])
    critic = SACCritic(seed=cfg["seed"] + 1)
    target = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    log_alpha = torch.tensor(np.log(cfg["init_alpha"]), dtype=torch.float32,
                             requires_grad=True)
    optimisers = {"actor": torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"]),
                  "critic": torch.optim.Adam(critic.parameters(), lr=cfg["learning_rate"]),
                  "alpha": torch.optim.Adam([log_alpha], lr=cfg["learning_rate"])}
    buffer = ReplayBuffer(cfg["buffer_size"], seed=cfg["seed"])

    n_envs = cfg["num_envs"]
    n_iterations = max(1, cfg["total_steps"] // n_envs)
    curriculum = Curriculum(cfg["curriculum"], n_iterations * n_envs, cfg["realistic"])
    _, stage_name, stage_kwargs = curriculum.stage_at(0)
    vec = VecEnv(n_envs, stage_kwargs, seed=cfg["seed"])
    eval_kwargs = curriculum.final_env_kwargs()
    log = ProgressLog(os.path.join(out_dir, "progress.csv"), LOG_COLUMNS)
    finished, update_stats = [], []
    best_eval = -np.inf
    next_eval, next_log = cfg["eval_interval"], cfg["log_interval"]

    for iteration in range(n_iterations):
        step = iteration * n_envs
        _, stage_name, stage_kwargs = curriculum.stage_at(step)
        in_final_stage = curriculum.is_final_stage(step)
        vec.set_env_kwargs(stage_kwargs)

        obs = vec.obs.copy()
        if step < cfg["warmup_steps"]:
            actions = action_rng.uniform(-1.0, 1.0, size=(n_envs, ACTION_DIM)).astype(np.float32)
        else:
            with torch.no_grad():
                actions, _ = actor(torch.as_tensor(obs), with_log_prob=False)
            actions = actions.numpy()
        result = vec.step(actions)
        buffer.add(obs, actions, result["rewards"] * cfg["reward_scale"], result["next_obs"],
                   result["terminated"].astype(np.float32))
        finished.extend(result["finished"])

        step = (iteration + 1) * n_envs
        if step >= cfg["warmup_steps"] and len(buffer) >= cfg["batch_size"]:
            for _ in range(cfg["updates_per_step"] * n_envs):
                update_stats.append(sac_update(buffer.sample(cfg["batch_size"]), actor, critic,
                                               target, log_alpha, optimisers, cfg))

        do_eval = step >= next_eval or iteration == n_iterations - 1
        if step >= next_log or do_eval:
            next_log += cfg["log_interval"]
            row = {"step": step, "time": log.elapsed(), "stage": stage_name}
            row.update(recent_episode_stats(finished))
            if update_stats:
                for key in update_stats[0]:
                    row[key] = float(np.mean([s[key] for s in update_stats]))
                update_stats = []
            if do_eval:
                next_eval += cfg["eval_interval"]
                res = evaluate_policy(lambda o: actor.predict(o, deterministic=True),
                                      EVAL_SEEDS, eval_kwargs)
                row.update({"eval_return": res["mean"], "eval_std": res["std"],
                            "eval_survival": res["survival"], "eval_goal": res["goal"]})
                summary = {k: res[k] for k in ("mean", "std", "survival", "goal")}
                save_sac(os.path.join(out_dir, "last.pt"), actor, critic, target, log_alpha,
                         cfg, step, summary, stage_name)
                # best.pt is chosen only among checkpoints of the last stage.
                if in_final_stage and res["mean"] > best_eval:
                    best_eval = res["mean"]
                    save_sac(os.path.join(out_dir, "best.pt"), actor, critic, target,
                             log_alpha, cfg, step, summary, stage_name)
                if verbose:
                    print("[sac] step {:>8,}  {:<20} train {:7.1f}  eval {:7.1f} +- {:5.1f}  "
                          "survival {:4.0%}  alpha {:.3f}  {:5.0f} s".format(
                              step, stage_name, row.get("train_return", float("nan")),
                              res["mean"], res["std"], res["survival"],
                              log_alpha.exp().item(), log.elapsed()), flush=True)
            log.write(row)

    vec.close()
    return actor, cfg
