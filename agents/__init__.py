"""Training of the Ball-on-Tray learning agents (stage 3).

    common.py   curriculum (staged domain randomisation), parallel
                environments, evaluation on fixed seeds, CSV logging
    ppo.py      PPO training loop (GAE, minibatch epochs, LR annealing)
    sac.py      SAC training loop (replay buffer, target critic, entropy tuning)
    features.py optional extra agent inputs (past actions, disturbance
                estimate, input scale) and the ``FeatureEnv`` wrapper

``load_policy`` turns a saved checkpoint of either algorithm into a function
obs -> action, as used by ``evaluate_agent.py`` and ``demo.py --checkpoint``.
``checkpoint_features`` returns the input options the checkpoint was trained
with; pass them to ``evaluate_policy`` or wrap the environment in
``FeatureEnv`` so that the policy gets the input it expects.
"""

import torch

from agents.common import (CURRICULA, EVAL_SEEDS, TEST_SEEDS, Curriculum, VecEnv,
                           evaluate_policy)
from agents.features import (FEATURE_DEFAULTS, FeatureEnv, checkpoint_features, describe,
                             feature_config, feature_dim, is_default, make_env)
from agents.ppo import PPO_DEFAULTS, load_ppo, train_ppo
from agents.sac import SAC_DEFAULTS, ReplayBuffer, load_sac_actor, polyak_update, train_sac


def load_policy(path):
    """Load a PPO or SAC checkpoint as a deterministic policy.

    Returns:
        (algo, policy, checkpoint): algo is "ppo" or "sac"; policy maps one
        observation to an action (2,) in [-1, 1] (the PPO mean or the tanh of
        the SAC mean); checkpoint is the loaded dict. The observation is the
        18-number one unless the agent was trained with extra inputs; then it
        is the input of ``FeatureEnv`` with ``checkpoint_features(checkpoint)``.
    """
    checkpoint = torch.load(path, map_location="cpu")
    algo = checkpoint.get("algo")
    if algo == "ppo":
        network = load_ppo(checkpoint)
        return algo, (lambda obs: network.predict(obs, deterministic=True)), checkpoint
    if algo == "sac":
        actor = load_sac_actor(checkpoint)
        return algo, (lambda obs: actor.predict(obs, deterministic=True)), checkpoint
    raise ValueError("{} is not a PPO or SAC checkpoint written by train.py".format(path))


__all__ = ["CURRICULA", "EVAL_SEEDS", "TEST_SEEDS", "Curriculum", "VecEnv", "evaluate_policy",
           "PPO_DEFAULTS", "train_ppo", "load_ppo", "SAC_DEFAULTS", "train_sac",
           "ReplayBuffer", "polyak_update", "load_sac_actor", "load_policy",
           "FEATURE_DEFAULTS", "FeatureEnv", "checkpoint_features", "describe",
           "feature_config", "feature_dim", "is_default", "make_env"]
