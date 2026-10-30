"""SAC networks for the Ball-on-Tray environment: squashed Gaussian actor and twin Q critics.

The proposal trains PPO as the baseline and SAC as the more sample-efficient
alternative. The PPO networks are in ``actor_critic.py``; this file holds SAC's:

    SACActor    obs (18) -> 128 -> 64 -> mean (2), log_std (2)
                a = tanh(u), u ~ N(mean, std)     state-dependent std
    SACCritic   [obs, action] (20) -> 128 -> 64 -> Q1   and the same again
                for Q2, two independently initialised networks

Hidden sizes, activations and initialisation gains are the same as for the
PPO networks (see ``actor_critic.py``).

Why the actor squashes with tanh
--------------------------------
SAC's actor loss needs log pi(a|s) of the action actually applied, and that
action must lie in the environment's range [-1, 1]. Squashing the Gaussian
sample u through tanh gives a bounded action with a tractable density; the
change of variables adds a correction term

    log pi(a|s) = log N(u; mean, std) - sum_i log(1 - tanh(u_i)^2)

which is evaluated in the numerically stable form
2 * (log 2 - u - softplus(-2u)); the direct form loses all precision once |u|
exceeds about 8. PPO does not need this: its importance ratio is computed on
the raw sample and the environment clips the action.

The target critic of the SAC update is a copy of ``SACCritic`` made by the
training loop (stage 3); it is an algorithm detail, not a network definition.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from networks.actor_critic import (ACTION_DIM, HIDDEN_INIT_GAIN, HIDDEN_SIZES, OBS_DIM,
                                   POLICY_OUTPUT_GAIN, VALUE_OUTPUT_GAIN)

# SAC clamps the state-dependent log std into this range. Without the clamp the
# actor can drive the std to 0 (log_prob -> +inf) or make it explode early in
# training.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _linear(in_dim, out_dim, gain):
    """``nn.Linear`` with orthogonal weights (given gain) and zero bias."""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _trunk(in_dim, hidden_sizes):
    """ReLU MLP without an output layer; returns (module, output width)."""
    layers, width = [], in_dim
    for size in hidden_sizes:
        layers += [_linear(width, size, HIDDEN_INIT_GAIN), nn.ReLU()]
        width = size
    return nn.Sequential(*layers), width


def tanh_log_det(u):
    """sum_i log(1 - tanh(u_i)^2) in the numerically stable form, shape (B,)."""
    return (2.0 * (math.log(2.0) - u - nn.functional.softplus(-2.0 * u))).sum(-1)


class SACActor(nn.Module):
    """Tanh-squashed Gaussian policy with a state-dependent standard deviation.

    Inputs are batches of shape (B, obs_dim) (tensor or numpy); use
    ``predict`` for a single observation.

    Args:
        obs_dim, action_dim: input and output sizes (18 and 2 for Ball-on-Tray).
        hidden_sizes: hidden layer widths.
        seed: if given, the weights are initialised from this seed without
            changing torch's global random state.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES,
                 seed=None):
        super(SACActor, self).__init__()
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        with torch.random.fork_rng(enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
            self.body, width = _trunk(self.obs_dim, hidden_sizes)
            # Small output gains: the initial policy is centred (tanh(mean) ~ 0)
            # and its std starts close to exp(0) = 1.
            self.mean_head = _linear(width, self.action_dim, POLICY_OUTPUT_GAIN)
            self.log_std_head = _linear(width, self.action_dim, POLICY_OUTPUT_GAIN)

    def distribution(self, obs):
        """Return the Gaussian over the pre-squash variable u, batch shape (B, action_dim)."""
        features = self.body(torch.as_tensor(obs, dtype=torch.float32))
        log_std = torch.clamp(self.log_std_head(features), LOG_STD_MIN, LOG_STD_MAX)
        return Normal(self.mean_head(features), log_std.exp())

    def forward(self, obs, deterministic=False, with_log_prob=True):
        """Sample a squashed action and its log-probability.

        The sample uses ``rsample`` (reparameterisation), so the gradient of
        the actor loss flows through the action into the critic and back.

        Returns:
            (action, log_prob): action of shape (B, action_dim) in (-1, 1);
            log_prob of shape (B,), or None if ``with_log_prob`` is False.
        """
        dist = self.distribution(obs)
        u = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(u)
        if not with_log_prob:
            return action, None
        return action, dist.log_prob(u).sum(-1) - tanh_log_det(u)

    @torch.no_grad()
    def predict(self, obs, deterministic=True):
        """Numpy convenience: action(s) in (-1, 1), shape (2,) or (B, 2), float32."""
        obs = torch.as_tensor(obs, dtype=torch.float32)
        single = obs.dim() == 1
        action, _ = self.forward(obs.unsqueeze(0) if single else obs, deterministic,
                                 with_log_prob=False)
        action = action.numpy().astype(np.float32)
        return action[0] if single else action


class SACCritic(nn.Module):
    """Twin Q-networks Q1(s, a), Q2(s, a) in one module.

    SAC uses the minimum of the two in its targets, which counteracts the
    overestimation bias of a single Q-network. One module means one optimiser
    and one ``state_dict`` cover both. The input width is obs_dim + action_dim
    (20 here): a Q-function is evaluated at a state-action pair.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES,
                 seed=None):
        super(SACCritic, self).__init__()
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        in_dim = self.obs_dim + self.action_dim
        with torch.random.fork_rng(enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
            self.q1_body, width = _trunk(in_dim, hidden_sizes)
            self.q1_head = _linear(width, 1, VALUE_OUTPUT_GAIN)
            self.q2_body, _ = _trunk(in_dim, hidden_sizes)
            self.q2_head = _linear(width, 1, VALUE_OUTPUT_GAIN)

    @staticmethod
    def _input(obs, action):
        return torch.cat([torch.as_tensor(obs, dtype=torch.float32),
                          torch.as_tensor(action, dtype=torch.float32)], dim=-1)

    def forward(self, obs, action):
        """Return (q1, q2), each of shape (B,)."""
        x = self._input(obs, action)
        return (self.q1_head(self.q1_body(x)).squeeze(-1),
                self.q2_head(self.q2_body(x)).squeeze(-1))

    def q1(self, obs, action):
        """Return only Q1(s, a), shape (B,); the actor loss needs a single critic."""
        return self.q1_head(self.q1_body(self._input(obs, action))).squeeze(-1)
