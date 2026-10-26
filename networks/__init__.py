"""Neural networks for the Ball-on-Tray learning agent (PyTorch)."""

from networks.actor_critic import (ACTION_DIM, OBS_DIM, PRIVILEGED_DIM, Actor, ActorCritic,
                                   Critic, build_mlp, privileged_features)

__all__ = ["ACTION_DIM", "OBS_DIM", "PRIVILEGED_DIM", "Actor", "ActorCritic", "Critic",
           "build_mlp", "privileged_features"]
