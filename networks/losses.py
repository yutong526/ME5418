"""Loss functions of PPO and SAC, shared by the demo, the tests and (stage 3) training.

    returns_and_advantages   discounted returns and normalised advantages of a
                             rollout segment (plain returns; GAE is stage 3)
    ppo_loss                 clipped surrogate + value loss - entropy bonus
    sac_critic_loss          twin-Q regression to the entropy-regularised target
    sac_actor_loss           E[alpha * log pi(a|s) - Q1(s, a)], reparameterised

PPO (Schulman et al., 2017), with r_t = exp(log pi_new(a|s) - log pi_old(a|s)):

    L = -E[min(r_t A_t, clip(r_t, 1 - eps, 1 + eps) A_t)]
        + c_v E[(V(s) - R_t)^2] - c_e E[H(pi(.|s))]

SAC (Haarnoja et al., 2018):

    y        = r + gamma (1 - d) [min(Q1'(s', a'), Q2'(s', a')) - alpha log pi(a'|s')]
    L_critic = E[(Q1(s, a) - y)^2] + E[(Q2(s, a) - y)^2]
    L_actor  = E[alpha log pi(a~|s) - Q1(s, a~)],    a~ ~ pi(.|s) reparameterised

``sac_critic_loss`` takes the critic used for the target as an argument; the
demo passes the critic itself, the training loop will pass its target copy.
"""

import torch
from torch import nn

PPO_CLIP = 0.2          # clipping parameter epsilon
VALUE_COEF = 0.5        # weight c_v of the value loss
ENTROPY_COEF = 0.01     # weight c_e of the entropy bonus
SAC_ALPHA = 0.2         # SAC entropy temperature (fixed; automatic tuning is stage 3)


def returns_and_advantages(rewards, values, dones, gamma):
    """Discounted returns and normalised advantages, without bootstrapping.

    The return is the discounted sum to the end of the segment and restarts
    after every step with ``dones = 1`` (the ball left the tray). A time-limit
    truncation is not a ``done``: it does not break the bootstrap.

    Args:
        rewards, values, dones: float tensors of shape (T,).
        gamma: discount factor.

    Returns:
        (returns, advantages), both (T,); advantages = returns - values,
        normalised to mean 0 and standard deviation 1 as PPO expects.
    """
    returns = torch.zeros_like(rewards)
    running = torch.zeros(())
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        returns[t] = running
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def ppo_loss(network, obs, actions, old_log_probs, advantages, returns, privileged=None,
             clip=PPO_CLIP, value_coef=VALUE_COEF, entropy_coef=ENTROPY_COEF):
    """The clipped PPO objective for an ``ActorCritic``.

    Args:
        network: ``networks.ActorCritic`` (baseline or asymmetric).
        obs: (B, 18) observations of the rollout.
        actions: (B, 2) raw (unclipped) actions returned by ``act``.
        old_log_probs: (B,) log-probabilities under the policy that acted.
        advantages, returns: (B,) from ``returns_and_advantages``.
        privileged: (B, 3) critic inputs for the asymmetric network, else None.

    Returns:
        (total, policy_loss, value_loss, entropy) as scalar tensors.
    """
    log_probs, entropy, values = network.evaluate_actions(obs, actions, privileged)
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = nn.functional.mse_loss(values, returns)
    entropy = entropy.mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, policy_loss, value_loss, entropy


def sac_critic_loss(actor, critic, target_critic, obs, actions, rewards, next_obs, dones,
                    gamma, alpha=SAC_ALPHA):
    """Twin-Q loss against the entropy-regularised target (target has no gradient).

    Args:
        actor: ``networks.SACActor``; samples a' for the target.
        critic: ``networks.SACCritic`` being trained.
        target_critic: critic used for the target (the critic itself in the
            demo, its slowly updated copy in training).
        obs, actions, rewards, next_obs, dones: transition batch, shapes
            (B, 18), (B, 2) in (-1, 1), (B,), (B, 18), (B,).

    Returns:
        scalar loss MSE(Q1, y) + MSE(Q2, y).
    """
    with torch.no_grad():
        next_actions, next_log_probs = actor(next_obs)
        target_q1, target_q2 = target_critic(next_obs, next_actions)
        target = rewards + gamma * (1.0 - dones) * (
            torch.min(target_q1, target_q2) - alpha * next_log_probs)
    q1, q2 = critic(obs, actions)
    return nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)


def sac_actor_loss(actor, critic, obs, alpha=SAC_ALPHA):
    """E[alpha * log pi(a~|s) - Q1(s, a~)] with a reparameterised sample a~."""
    actions, log_probs = actor(obs)
    return (alpha * log_probs - critic.q1(obs, actions)).mean()
