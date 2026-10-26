"""Actor-critic networks for PPO on the Ball-on-Tray environment.

Two separate multilayer perceptrons (MLPs), no shared trunk:

    actor   obs (18) -> 128 -> 64 -> 2    mean of a Gaussian policy, squashed
                                           to [-1, 1] by tanh; the standard
                                           deviation is a learned, state-
                                           independent parameter (log_std)
    critic  obs (18) -> 128 -> 64 -> 1    state value V(s)

Why an MLP and not a CNN or an RNN
----------------------------------
The observation is a short, low-dimensional vector (3 stacked frames of
[x, y, vx, vy, pitch, roll], already normalised), not an image, so there is no
spatial structure for convolutions to exploit. Velocity is part of every frame
and the three stacked frames already carry the recent history, so a recurrent
network is not needed to make the state observable for the policy. A small MLP
is the standard choice for low-dimensional continuous control with PPO.

Asymmetric variant
------------------
``ActorCritic(privileged_dim=3)`` gives the critic three extra inputs that the
actor never sees: the rolling resistance coefficient and the current base
acceleration (both hidden from the observation, available in the ``info`` dict
of the environment, see ``privileged_features``). The critic is only used
during training, so the deployed actor still needs nothing but the 18-D
observation. The actor is identical in both variants.

Actions
-------
A Gaussian sample can leave [-1, 1] even though its mean cannot. Following the
usual PPO practice, ``log_prob`` is computed for the raw (unclipped) sample,
and only the action passed to ``env.step`` is clipped to [-1, 1]
(``ActorCritic.act`` returns both). The environment would clip it anyway.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

# ------------------------------------------------------------------ dimensions
OBS_DIM = 18                # 3 stacked frames x [x, y, vx, vy, pitch, roll]
ACTION_DIM = 2              # (d_pitch, d_roll), normalised to [-1, 1]
PRIVILEGED_DIM = 3          # [c_rr, a_base_x, a_base_y], asymmetric critic only

# ------------------------------------------------------------- architecture
HIDDEN_SIZES = (128, 64)    # hidden layer widths of both networks
LOG_STD_INIT = -0.5         # initial log standard deviation (std ~ 0.61)

# Orthogonal initialisation gains: sqrt(2) suits ReLU hidden layers; the small
# policy-output gain starts the policy with near-zero mean actions (a level
# tray), and gain 1 is the usual choice for the value output.
HIDDEN_INIT_GAIN = math.sqrt(2.0)
POLICY_OUTPUT_GAIN = 0.01
VALUE_OUTPUT_GAIN = 1.0

# ------------------------------------------------- privileged feature scaling
# Bring the hidden quantities to roughly unit scale, like the observation:
# c_rr by the top of its full range, a_base by the top of the full disturbance
# range.
C_RR_SCALE = 0.05           # [-]
A_BASE_SCALE = 2.0          # [m/s^2]


def build_mlp(in_dim, hidden_sizes, out_dim, output_gain):
    """Build an MLP with ReLU hidden layers and a linear output layer.

    All weights are initialised orthogonally (gain ``HIDDEN_INIT_GAIN`` for the
    hidden layers, ``output_gain`` for the output layer) and all biases to 0.

    Returns:
        nn.Sequential mapping (..., in_dim) to (..., out_dim).
    """
    layers = []
    width = in_dim
    for size in hidden_sizes:
        linear = nn.Linear(width, size)
        nn.init.orthogonal_(linear.weight, gain=HIDDEN_INIT_GAIN)
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.ReLU()]
        width = size
    output = nn.Linear(width, out_dim)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def privileged_features(infos):
    """Extract the critic's privileged inputs from environment ``info`` dicts.

    Args:
        infos: one ``info`` dict, or a list of them.

    Returns:
        float32 array [c_rr / C_RR_SCALE, a_base_x / A_BASE_SCALE,
        a_base_y / A_BASE_SCALE], shape (3,) for one dict or (N, 3) for a list.
    """
    single = isinstance(infos, dict)
    rows = [[info["c_rr"] / C_RR_SCALE,
             info["a_base"][0] / A_BASE_SCALE,
             info["a_base"][1] / A_BASE_SCALE] for info in ([infos] if single else infos)]
    features = np.asarray(rows, dtype=np.float32)
    return features[0] if single else features


def _as_batch(x, width, name):
    """Convert x to a float32 tensor of shape (B, width).

    Returns:
        (tensor, single) where ``single`` is True if x was one vector of shape
        (width,), so that results can be returned without the batch axis.
    """
    tensor = torch.as_tensor(x, dtype=torch.float32)
    single = tensor.dim() == 1
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.shape[1] != width:
        raise ValueError("{} must have shape ({},) or (B, {}), got {}".format(
            name, width, width, tuple(torch.as_tensor(x).shape)))
    return tensor, single


class Actor(nn.Module):
    """Gaussian policy: tanh-squashed mean from an MLP, state-independent std."""

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 hidden_sizes=HIDDEN_SIZES, log_std_init=LOG_STD_INIT):
        super(Actor, self).__init__()
        self.body = build_mlp(obs_dim, hidden_sizes, action_dim, POLICY_OUTPUT_GAIN)
        # One learnable log standard deviation per action dimension.
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def forward(self, obs):
        """Return (mean, std), each of shape (B, action_dim); mean is in [-1, 1]."""
        mean = torch.tanh(self.body(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std


class Critic(nn.Module):
    """State-value network V(s); its input may include privileged features."""

    def __init__(self, in_dim=OBS_DIM, hidden_sizes=HIDDEN_SIZES):
        super(Critic, self).__init__()
        self.body = build_mlp(in_dim, hidden_sizes, 1, VALUE_OUTPUT_GAIN)

    def forward(self, x):
        """Return the value, shape (B,)."""
        return self.body(x).squeeze(-1)


class ActorCritic(nn.Module):
    """Separate actor and critic for PPO.

    Every method accepts one observation of shape (18,) or a batch (B, 18), as
    a numpy array or a tensor, and returns results without or with the batch
    axis accordingly.

    Args:
        obs_dim, action_dim: input and output sizes (18 and 2 for Ball-on-Tray).
        hidden_sizes: hidden layer widths of both networks.
        log_std_init: initial log standard deviation of the policy.
        privileged_dim: number of extra critic inputs; 0 is the baseline
            (symmetric) network, ``PRIVILEGED_DIM`` (3) the asymmetric variant.
        seed: if given, the weights are initialised from this seed without
            changing torch's global random state.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=HIDDEN_SIZES,
                 log_std_init=LOG_STD_INIT, privileged_dim=0, seed=None):
        super(ActorCritic, self).__init__()
        self.config = {"obs_dim": int(obs_dim), "action_dim": int(action_dim),
                       "hidden_sizes": tuple(int(h) for h in hidden_sizes),
                       "log_std_init": float(log_std_init),
                       "privileged_dim": int(privileged_dim)}
        if seed is None:
            self._build()
        else:
            with torch.random.fork_rng():
                torch.manual_seed(int(seed))
                self._build()

    def _build(self):
        """Create the two networks from ``self.config``."""
        cfg = self.config
        self.actor = Actor(cfg["obs_dim"], cfg["action_dim"], cfg["hidden_sizes"],
                           cfg["log_std_init"])
        self.critic = Critic(cfg["obs_dim"] + cfg["privileged_dim"], cfg["hidden_sizes"])

    @property
    def asymmetric(self):
        """True if the critic takes privileged inputs."""
        return self.config["privileged_dim"] > 0

    # ----------------------------------------------------------------- core
    def distribution(self, obs):
        """Return the action distribution (independent Normal per dimension) for a batch."""
        mean, std = self.actor(obs)
        return Normal(mean, std)

    def _critic_input(self, obs, privileged):
        """Concatenate the observation and, for the asymmetric critic, the privileged features."""
        if not self.asymmetric:
            if privileged is not None:
                raise ValueError("this critic is symmetric; do not pass privileged features")
            return obs
        if privileged is None:
            raise ValueError("the asymmetric critic needs the privileged features "
                             "(see privileged_features)")
        extra, _ = _as_batch(privileged, self.config["privileged_dim"], "privileged")
        if extra.shape[0] != obs.shape[0]:
            raise ValueError("privileged batch size does not match the observation batch")
        return torch.cat([obs, extra], dim=1)

    def forward(self, obs, privileged=None):
        """Return (distribution, value) for an observation or a batch.

        Args:
            obs: (18,) or (B, 18).
            privileged: (3,) or (B, 3) for the asymmetric critic, else None.

        Returns:
            (Normal distribution with batch shape (B, 2), value of shape (B,)).
            The batch axis is kept here because a distribution cannot be
            unbatched; use ``value`` or ``act`` for unbatched results.
        """
        obs, _ = _as_batch(obs, self.config["obs_dim"], "obs")
        return self.distribution(obs), self.critic(self._critic_input(obs, privileged))

    def value(self, obs, privileged=None):
        """Return V(s): a scalar tensor for one observation, shape (B,) for a batch."""
        obs, single = _as_batch(obs, self.config["obs_dim"], "obs")
        value = self.critic(self._critic_input(obs, privileged))
        return value[0] if single else value

    def act(self, obs, deterministic=False):
        """Choose an action.

        Args:
            obs: (18,) or (B, 18).
            deterministic: return the mean instead of a sample.

        Returns:
            (action, log_prob, env_action): the raw action (sample or mean),
            its log-probability summed over the action dimensions, and the
            action clipped to [-1, 1] for ``env.step``. Shapes (2,), (), (2,)
            for one observation, (B, 2), (B,), (B, 2) for a batch.
        """
        obs, single = _as_batch(obs, self.config["obs_dim"], "obs")
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        env_action = action.clamp(-1.0, 1.0)
        if single:
            return action[0], log_prob[0], env_action[0]
        return action, log_prob, env_action

    def evaluate_actions(self, obs, actions, privileged=None):
        """Return (log_prob, entropy, value) of given raw actions, each of shape (B,).

        This is the quantity a PPO update needs; ``actions`` are the raw
        (unclipped) actions returned by ``act``.
        """
        obs, _ = _as_batch(obs, self.config["obs_dim"], "obs")
        actions, _ = _as_batch(actions, self.config["action_dim"], "actions")
        dist = self.distribution(obs)
        value = self.critic(self._critic_input(obs, privileged))
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), value

    @torch.no_grad()
    def predict(self, obs, deterministic=True):
        """Numpy convenience for running the policy: env action(s) in [-1, 1].

        Returns:
            float32 numpy array, shape (2,) for one observation or (B, 2).
        """
        _, _, env_action = self.act(obs, deterministic=deterministic)
        return env_action.numpy().astype(np.float32)

    # ---------------------------------------------------------------- utils
    def num_parameters(self):
        """Return (actor parameters, critic parameters)."""
        count = lambda module: sum(p.numel() for p in module.parameters())
        return count(self.actor), count(self.critic)

    def save(self, path):
        """Save the configuration and the weights to ``path``."""
        torch.save({"config": dict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        """Rebuild a network saved with ``save``."""
        checkpoint = torch.load(path, map_location="cpu")
        network = cls(**checkpoint["config"])
        network.load_state_dict(checkpoint["state_dict"])
        return network
