"""Demo of the actor-critic networks used by the Ball-on-Tray learning agent.

The default run walks through one network step by step:

    1. build the network and print its structure and parameter count
    2. forward pass on a real observation from ``env.reset``: action
       distribution, sampled action, log-probability, entropy, value
    3. backward pass on an explicit demonstration loss, gradient norm of every
       layer, one optimiser step, and the loss before and after
    4. closed loop: the (untrained) network drives the environment for a few
       steps through ``env.step``

Two small experiments are available as options:

    --bc          behaviour cloning of the LQR controller: can the actor
                  network learn a good policy from data? (about half a minute)
    --critics     how much information the privileged critic inputs carry
                  (baseline vs asymmetric critic on the same returns)
    --realistic   with --bc or --critics: also run the experiment with the
                  realism parameters of demo.py --realistic, next to the
                  ideal condition

Examples:
    python demo_network.py                  # the four steps above
    python demo_network.py --variant asymmetric
    python demo_network.py --bc
    python demo_network.py --critics
    python demo_network.py --bc --realistic
"""

import argparse
import time

import numpy as np
import torch

from ball_on_tray_gym import BallOnTrayEnv
from networks import PRIVILEGED_DIM, ActorCritic, privileged_features

# Full randomisation ranges of the environment, as in demo.py.
FRICTION_RANGE = (0.005, 0.05)          # c_rr [-]
DISTURBANCE_RANGE = (0.5, 2.0)          # ||a_base|| [m/s^2]

DEMO_SEED = 0               # seed of the network and of the episode
DEMO_LEARNING_RATE = 1e-3   # Adam step size of the single demonstration update
DEMO_ADVANTAGE = 1.0        # fixed advantage of the demonstration loss
DEMO_VALUE_TARGET = 10.0    # fixed value target of the demonstration loss
CLOSED_LOOP_STEPS = 10      # steps of the closed-loop check


def demo_loss(network, obs, action, privileged):
    """Explicit demonstration loss (NOT the PPO loss).

        loss = -DEMO_ADVANTAGE * log pi(action | obs) + (V(obs) - DEMO_VALUE_TARGET)^2

    The first term increases the probability of the given action, as a
    policy-gradient step with a positive advantage would; the second pulls the
    value towards a fixed target. Together they touch every parameter of the
    actor and the critic, which is all this demo needs.

    Returns:
        (loss, policy term, value term) as tensors.
    """
    log_prob, _, value = network.evaluate_actions(obs, action, privileged)
    policy_term = -DEMO_ADVANTAGE * log_prob.mean()
    value_term = ((value - DEMO_VALUE_TARGET) ** 2).mean()
    return policy_term + value_term, policy_term, value_term


def print_structure(network):
    """Print the layers and the parameter counts."""
    n_actor, n_critic = network.num_parameters()
    print(network)
    print("parameters   : actor {:,}, critic {:,}, total {:,}".format(
        n_actor, n_critic, n_actor + n_critic))
    print("")


def show_forward(network, obs, privileged):
    """Forward pass on one observation; print what comes out."""
    print("forward pass on the observation of env.reset(seed={})".format(DEMO_SEED))
    print("  input  obs  : shape {}, dtype {}".format(obs.shape, obs.dtype))
    print("                newest frame [x, y, vx, vy, pitch, roll] = {}".format(
        np.array2string(obs[-6:], precision=3)))
    if privileged is not None:
        print("  input  priv : [c_rr/0.05, a_x/2, a_y/2] = {}".format(
            np.array2string(privileged, precision=3)))
    dist, value = network(obs, privileged)
    torch.manual_seed(DEMO_SEED)
    action, log_prob, env_action = network.act(obs)
    print("  output mean : shape {}, {} (tanh keeps it in [-1, 1])".format(
        tuple(dist.mean.shape), np.array2string(dist.mean.detach().numpy()[0], precision=4)))
    print("         std  : {} (state independent, exp(log_std))".format(
        np.array2string(dist.stddev.detach().numpy()[0], precision=4)))
    print("         sample action  : {}".format(np.array2string(action.numpy(), precision=4)))
    print("         env action     : {} (sample clipped to [-1, 1])".format(
        np.array2string(env_action.numpy(), precision=4)))
    print("         log_prob       : {:.4f}".format(log_prob.item()))
    print("         entropy        : {:.4f}".format(dist.entropy().sum(-1).item()))
    print("         value V(s)     : {:.4f}".format(value.item()))
    print("")
    return action


def show_backward(network, obs, action, privileged):
    """Backward pass on the demonstration loss, gradient norms, one optimiser step."""
    print("backward pass on the demonstration loss")
    print("  loss = -{:g} * log_prob(action) + (V - {:g})^2".format(
        DEMO_ADVANTAGE, DEMO_VALUE_TARGET))
    optimiser = torch.optim.Adam(network.parameters(), lr=DEMO_LEARNING_RATE)
    optimiser.zero_grad()
    loss, policy_term, value_term = demo_loss(network, obs, action, privileged)
    loss.backward()
    print("  loss before : {:.4f} (policy {:.4f}, value {:.4f})".format(
        loss.item(), policy_term.item(), value_term.item()))
    print("  gradient norms:")
    for name, parameter in network.named_parameters():
        print("    {:<22} shape {:<12} |grad| = {:.4e}".format(
            name, str(tuple(parameter.shape)), parameter.grad.norm().item()))
    optimiser.step()
    with torch.no_grad():
        after, policy_term, value_term = demo_loss(network, obs, action, privileged)
    print("  one Adam step (lr = {:g})".format(DEMO_LEARNING_RATE))
    print("  loss after  : {:.4f} (policy {:.4f}, value {:.4f}), change {:+.4f}".format(
        after.item(), policy_term.item(), value_term.item(), after.item() - loss.item()))
    print("")


def show_closed_loop(network, env):
    """Let the untrained network drive the environment for a few steps."""
    print("closed loop: the untrained network drives env.step for {} steps".format(
        CLOSED_LOOP_STEPS))
    obs, _ = env.reset(seed=DEMO_SEED)
    torch.manual_seed(DEMO_SEED)
    print("  {:>4} {:>18} {:>9} {:>20}".format("step", "env action", "reward", "ball position [m]"))
    for _ in range(CLOSED_LOOP_STEPS):
        _, _, env_action = network.act(obs)
        obs, reward, terminated, truncated, info = env.step(env_action.numpy())
        print("  {:>4} {:>18} {:>9.4f} {:>20}".format(
            env.step_count, np.array2string(env_action.numpy(), precision=3), reward,
            np.array2string(info["ball_pos"], precision=4)))
        if terminated or truncated:
            break
    print("")


def run_walkthrough(variant):
    """The default demo: structure, forward, backward, closed loop."""
    privileged_dim = PRIVILEGED_DIM if variant == "asymmetric" else 0
    network = ActorCritic(privileged_dim=privileged_dim, seed=DEMO_SEED)
    env = BallOnTrayEnv(friction_range=FRICTION_RANGE, disturbance_range=DISTURBANCE_RANGE)
    obs, info = env.reset(seed=DEMO_SEED)
    privileged = privileged_features(info) if network.asymmetric else None

    print("variant      : {} ({} critic)".format(
        variant, "observation + privileged" if network.asymmetric else "observation-only"))
    print("torch        : {}\n".format(torch.__version__))
    print_structure(network)
    action = show_forward(network, obs, privileged)
    show_backward(network, obs, action, privileged)
    show_closed_loop(network, env)
    env.close()


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Ball-on-Tray neural network demo.")
    parser.add_argument("--variant", choices=["baseline", "asymmetric"], default="baseline",
                        help="network shown by the walkthrough (default: baseline)")
    parser.add_argument("--bc", action="store_true",
                        help="behaviour cloning of the LQR controller instead of the walkthrough")
    parser.add_argument("--critics", action="store_true",
                        help="compare the baseline and the asymmetric critic instead of the "
                             "walkthrough")
    parser.add_argument("--realistic", action="store_true",
                        help="with --bc or --critics: also run the experiment with "
                             "observation noise, actuator lag and action delay (the settings "
                             "of demo.py --realistic), next to the ideal condition")
    args = parser.parse_args()
    if args.realistic and not (args.bc or args.critics):
        parser.error("--realistic only applies to --bc and --critics")
    return args


def main():
    """Run the walkthrough or one of the experiments."""
    args = parse_args()
    start = time.perf_counter()
    torch.set_num_threads(1)        # small networks: one thread is fastest and reproducible
    if args.bc:
        from networks.experiments import run_behaviour_cloning
        run_behaviour_cloning(realistic=args.realistic)
    elif args.critics:
        from networks.experiments import run_critic_comparison
        run_critic_comparison(realistic=args.realistic)
    else:
        run_walkthrough(args.variant)
    print("total time: {:.2f} s".format(time.perf_counter() - start))


if __name__ == "__main__":
    main()
