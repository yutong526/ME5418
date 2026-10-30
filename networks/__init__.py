"""Neural networks for the Ball-on-Tray learning agent (PyTorch).

    actor_critic.py   PPO: actor and critic MLPs, optional asymmetric critic
    sac.py            SAC: tanh-squashed Gaussian actor, twin Q critics
    losses.py         PPO and SAC losses, returns and advantages
    experiments.py    behaviour cloning and critic comparison (imported on demand)
"""

from networks.actor_critic import (ACTION_DIM, OBS_DIM, PRIVILEGED_DIM, Actor, ActorCritic,
                                   Critic, build_mlp, privileged_features)
from networks.losses import (ppo_loss, returns_and_advantages, sac_actor_loss,
                             sac_critic_loss)
from networks.sac import LOG_STD_MAX, LOG_STD_MIN, SACActor, SACCritic, tanh_log_det

__all__ = ["ACTION_DIM", "OBS_DIM", "PRIVILEGED_DIM", "Actor", "ActorCritic", "Critic",
           "build_mlp", "privileged_features",
           "SACActor", "SACCritic", "LOG_STD_MIN", "LOG_STD_MAX", "tanh_log_det",
           "ppo_loss", "returns_and_advantages", "sac_actor_loss", "sac_critic_loss"]
