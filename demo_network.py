"""Demo of the neural networks of the Ball-on-Tray learning agent.

The default run builds every network of the project and shows one forward and
one backward pass of each algorithm on data collected in the environment:

    [1] construction   PPO actor-critic, SAC actor and SAC twin critics:
                       layers and parameter counts
    [2] PPO forward    one real observation from env.reset: action
                       distribution, sampled action, log-probability, entropy,
                       value
    [3] PPO backward   a 64-step rollout with the current (untrained) policy,
                       the clipped PPO loss, the gradient norm of every
                       parameter, one Adam step, the loss before and after
    [4] SAC            critic TD loss and reparameterised actor loss on the
                       same transitions, gradients, one Adam step each, and a
                       check of the tanh correction against the direct formula
    [5] closed loop    the untrained PPO policy drives one full episode

Everything is seeded, so two runs print identical numbers. Nothing is trained
to convergence here; training is stage 3.

Two small supervised experiments are available as options (networks/experiments.py):

    --bc          behaviour cloning of the LQR controller: can the actor
                  network learn a good policy from data? (about a minute)
    --critics     how much information the privileged critic inputs carry
                  (baseline vs asymmetric critic on the same returns)
    --realistic   with --bc or --critics: also run the experiment with the
                  realism parameters of demo.py --realistic

Examples:
    python demo_network.py                          # the five sections above
    python demo_network.py --variant asymmetric     # PPO critic with privileged inputs
    python demo_network.py --seed 7                 # another seed
    python demo_network.py --bc                     # behaviour cloning experiment
    python demo_network.py --critics --realistic    # critic comparison, ideal and realistic
"""

import argparse
import time

import numpy as np
import torch
from torch import nn

from ball_on_tray_gym import BallOnTrayEnv
from networks import (PRIVILEGED_DIM, ActorCritic, SACActor, SACCritic, ppo_loss,
                      privileged_features, returns_and_advantages, sac_actor_loss,
                      sac_critic_loss, tanh_log_det)
from networks.losses import ENTROPY_COEF, PPO_CLIP, SAC_ALPHA, VALUE_COEF

# Full randomisation ranges of the environment, as in demo.py.
FRICTION_RANGE = (0.005, 0.05)          # c_rr [-]
DISTURBANCE_RANGE = (0.5, 2.0)          # ||a_base|| [m/s^2]

DEFAULT_SEED = 0            # seed of the networks and of the episodes
ROLLOUT_STEPS = 64          # transitions collected for the backward passes
GAMMA = 0.99                # discount factor of the demo returns
LEARNING_RATE = 3e-4        # Adam step size of the demonstration updates
MAX_GRAD_NORM = 0.5         # global gradient-norm clip of the PPO step
SAC_NOISE_SEED = 12345      # fixed reparameterisation noise (see section [4])


# ----------------------------------------------------------------- printing
def rule(title):
    """Print a section heading."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def fmt(values, precision=4):
    """Format a small array or tensor on one line."""
    if torch.is_tensor(values):
        values = values.detach().numpy()
    return np.array2string(np.asarray(values), precision=precision, suppress_small=True,
                           max_line_width=200)


def count(module):
    """Number of trainable parameters of a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def grad_report(module, name):
    """Print the gradient norm of every parameter tensor of ``module``.

    A missing or zero gradient anywhere is the fastest sign that part of the
    computation graph is detached.
    """
    print("  gradient norms of {}:".format(name))
    total, missing = 0.0, []
    for pname, parameter in module.named_parameters():
        if parameter.grad is None:
            missing.append(pname)
            continue
        norm = parameter.grad.norm().item()
        total += norm ** 2
        print("    {:<24} shape {:<11} |grad| = {:.4e}".format(
            pname, str(tuple(parameter.shape)), norm))
    print("    global norm = {:.4e}".format(total ** 0.5))
    print("    every parameter received a gradient" if not missing
          else "    WARNING: no gradient for {}".format(missing))


# -------------------------------------------------------------------- [1]
def section_construction(variant, seed):
    """Build the three networks and print their structure."""
    rule("[1] CONSTRUCTION")
    privileged_dim = PRIVILEGED_DIM if variant == "asymmetric" else 0
    ppo = ActorCritic(privileged_dim=privileged_dim, seed=seed)
    sac_actor = SACActor(seed=seed + 1)
    sac_critic = SACCritic(seed=seed + 2)

    n_actor, n_critic = ppo.num_parameters()
    print("PPO actor-critic, {} variant ({} critic)".format(
        variant, "observation + privileged" if ppo.asymmetric else "observation-only"))
    print(ppo)
    print("parameters : actor {:,}, critic {:,}, total {:,}".format(
        n_actor, n_critic, n_actor + n_critic))
    print("\nSAC actor (tanh-squashed Gaussian, state-dependent std)")
    print(sac_actor)
    print("parameters : {:,}".format(count(sac_actor)))
    print("\nSAC critic (twin Q-networks, input width 18 + 2 = 20)")
    print("parameters : {:,} ({:,} per Q-network)".format(count(sac_critic),
                                                         count(sac_critic) // 2))
    return ppo, sac_actor, sac_critic


# -------------------------------------------------------------------- [2]
def section_ppo_forward(env, ppo, seed):
    """Forward pass of the PPO network on one real observation."""
    rule("[2] PPO FORWARD PASS")
    obs, info = env.reset(seed=seed)
    privileged = privileged_features(info) if ppo.asymmetric else None
    print("observation of env.reset(seed={}): shape {}, dtype {}".format(
        seed, obs.shape, obs.dtype))
    print("  newest frame [x, y, vx, vy, pitch, roll] = {}".format(fmt(obs[-6:], 3)))
    if privileged is not None:
        print("  privileged [c_rr/0.05, a_x/2, a_y/2]    = {}".format(fmt(privileged, 3)))

    dist, value = ppo(obs, privileged)
    torch.manual_seed(seed)
    action, log_prob, env_action = ppo.act(obs)
    print("  mean          : {} (tanh keeps it in [-1, 1])".format(fmt(dist.mean[0])))
    print("  std           : {} (state independent, exp(log_std))".format(
        fmt(dist.stddev[0])))
    print("  sample action : {}".format(fmt(action)))
    print("  env action    : {} (sample clipped to [-1, 1])".format(fmt(env_action)))
    print("  log pi(a|s)   : {:.4f}".format(log_prob.item()))
    print("  entropy       : {:.4f} nats".format(dist.entropy().sum(-1).item()))
    print("  value V(s)    : {:.4f}".format(value.item()))


# -------------------------------------------------------------------- [3]
def collect_rollout(env, ppo, seed):
    """Run the current PPO policy for ROLLOUT_STEPS steps and store the transitions.

    A miniature version of the rollout buffer of stage 3. The episode is reset
    whenever it ends, so a fixed number of transitions is returned. The raw
    (unclipped) action is stored, because that is what the log-probability
    refers to; ``done`` is 1 only when the ball left the tray.
    """
    obs, info = env.reset(seed=seed)
    torch.manual_seed(seed)
    rows = {key: [] for key in ("obs", "action", "log_prob", "value", "privileged",
                                "reward", "next_obs", "done")}
    with torch.no_grad():
        for _ in range(ROLLOUT_STEPS):
            privileged = privileged_features(info)
            action, log_prob, env_action = ppo.act(obs)
            value = ppo.value(obs, privileged if ppo.asymmetric else None)
            next_obs, reward, terminated, truncated, info = env.step(env_action.numpy())
            for key, item in (("obs", obs), ("action", action.numpy()),
                              ("log_prob", log_prob.item()), ("value", value.item()),
                              ("privileged", privileged), ("reward", reward),
                              ("next_obs", next_obs), ("done", float(terminated))):
                rows[key].append(item)
            obs = next_obs
            if terminated or truncated:
                obs, info = env.reset()
    return {key: torch.as_tensor(np.asarray(value, dtype=np.float32))
            for key, value in rows.items()}


def section_ppo_backward(env, ppo, seed):
    """Clipped PPO loss on a real rollout, backward pass, one optimiser step."""
    rule("[3] PPO BACKWARD PASS")
    batch = collect_rollout(env, ppo, seed)
    returns, advantages = returns_and_advantages(batch["reward"], batch["value"],
                                                 batch["done"], GAMMA)
    privileged = batch["privileged"] if ppo.asymmetric else None
    print("rollout of {} steps with the untrained policy: obs {}, actions {}, "
          "{} fall(s)".format(ROLLOUT_STEPS, tuple(batch["obs"].shape),
                              tuple(batch["action"].shape), int(batch["done"].sum())))
    print("  reward mean {:.4f}, return mean {:.4f} (gamma {}), advantages normalised".format(
        batch["reward"].mean().item(), returns.mean().item(), GAMMA))

    inputs = (ppo, batch["obs"], batch["action"], batch["log_prob"], advantages, returns,
              privileged)
    optimiser = torch.optim.Adam(ppo.parameters(), lr=LEARNING_RATE)
    optimiser.zero_grad()
    total, policy, value, entropy = ppo_loss(*inputs)
    print("\nclipped PPO loss = policy + {} * value - {} * entropy (clip eps {})".format(
        VALUE_COEF, ENTROPY_COEF, PPO_CLIP))
    print("  policy loss  : {:.6f}".format(policy.item()))
    print("  value loss   : {:.6f}".format(value.item()))
    print("  entropy      : {:.6f} nats".format(entropy.item()))
    print("  total        : {:.6f}".format(total.item()))
    print("  The policy loss is zero up to rounding: the rollout came from the current")
    print("  weights, so the ratio is 1 and the normalised advantages average to 0.")
    print("  Its gradient is not zero; later epochs on the same batch give a non-zero loss.")

    total.backward()
    print("")
    grad_report(ppo, "the PPO actor-critic")
    log_std_before = ppo.actor.log_std.detach().clone()
    nn.utils.clip_grad_norm_(ppo.parameters(), MAX_GRAD_NORM)
    optimiser.step()
    with torch.no_grad():
        after_total, after_policy, after_value, _ = ppo_loss(*inputs)
    print("\nafter one Adam step (lr {:g}, gradient norm clipped to {}):".format(
        LEARNING_RATE, MAX_GRAD_NORM))
    print("  total loss   : {:.6f} -> {:.6f}".format(total.item(), after_total.item()))
    print("  policy loss  : {:.6f} -> {:.6f}".format(policy.item(), after_policy.item()))
    print("  value loss   : {:.6f} -> {:.6f}".format(value.item(), after_value.item()))
    print("  log_std      : {} -> {}".format(fmt(log_std_before), fmt(ppo.actor.log_std)))
    return batch


# -------------------------------------------------------------------- [4]
def section_sac(sac_actor, sac_critic, batch):
    """SAC critic and actor losses on the rollout transitions, one step each."""
    rule("[4] SAC FORWARD / BACKWARD PASS")
    obs, next_obs = batch["obs"], batch["next_obs"]
    # SAC acts in (-1, 1); the stored PPO actions are brought into that range.
    actions = batch["action"].clamp(-0.999, 0.999)
    rewards, dones = batch["reward"], batch["done"]

    torch.manual_seed(SAC_NOISE_SEED)
    sampled, log_prob = sac_actor(obs)
    print("actor forward on the {} observations:".format(len(obs)))
    print("  action range [{:.4f}, {:.4f}] (tanh-squashed)".format(
        sampled.min().item(), sampled.max().item()))
    print("  log pi(a|s): mean {:.4f}, min {:.4f}, max {:.4f}".format(
        log_prob.mean().item(), log_prob.min().item(), log_prob.max().item()))
    q1, q2 = sac_critic(obs, actions)
    print("critic forward: Q1 mean {:.4f}, Q2 mean {:.4f}, mean |Q1 - Q2| = {:.4f}".format(
        q1.mean().item(), q2.mean().item(), (q1 - q2).abs().mean().item()))
    print("  (independently initialised twins give different estimates)")

    # Critic: regression to the entropy-regularised target. The demo uses the
    # critic itself for the target; stage 3 uses a slowly updated copy.
    critic_optimiser = torch.optim.Adam(sac_critic.parameters(), lr=LEARNING_RATE)
    critic_optimiser.zero_grad()
    torch.manual_seed(SAC_NOISE_SEED)
    critic_loss = sac_critic_loss(sac_actor, sac_critic, sac_critic, obs, actions, rewards,
                                  next_obs, dones, GAMMA)
    critic_loss.backward()
    print("\ncritic loss = MSE(Q1, y) + MSE(Q2, y) = {:.6f}".format(critic_loss.item()))
    print("  y = r + gamma (1 - d) [min(Q1', Q2') - alpha log pi(a'|s')], alpha = {}".format(
        SAC_ALPHA))
    grad_report(sac_critic, "the SAC critic")
    critic_optimiser.step()

    # Actor: the loss is a reparameterised, hence stochastic, estimate. The
    # before/after comparison only shows the effect of the step if both
    # evaluations draw the same noise, so the seed is reset before each.
    actor_optimiser = torch.optim.Adam(sac_actor.parameters(), lr=LEARNING_RATE)
    actor_optimiser.zero_grad()
    torch.manual_seed(SAC_NOISE_SEED)
    actor_loss = sac_actor_loss(sac_actor, sac_critic, obs)
    actor_loss.backward()
    print("\nactor loss = E[alpha log pi(a|s) - Q1(s, a)] = {:.6f}".format(actor_loss.item()))
    print("  (the gradient reaches the actor through the reparameterised tanh sample)")
    grad_report(sac_actor, "the SAC actor")
    actor_optimiser.step()
    with torch.no_grad():
        torch.manual_seed(SAC_NOISE_SEED)
        after = sac_actor_loss(sac_actor, sac_critic, obs)
    print("  after one Adam step, same noise: {:.6f} -> {:.6f}".format(
        actor_loss.item(), after.item()))

    # The tanh correction is the easiest part to get wrong, so it is checked
    # against the direct (less stable) formula where the latter is accurate.
    with torch.no_grad():
        dist = sac_actor.distribution(obs[:4])
        u = dist.sample()
        stable = dist.log_prob(u).sum(-1) - tanh_log_det(u)
        direct = dist.log_prob(u).sum(-1) - torch.log(1.0 - torch.tanh(u) ** 2).sum(-1)
    print("\ntanh correction, stable vs direct formula on 4 samples:")
    print("  stable : {}".format(fmt(stable, 6)))
    print("  direct : {}".format(fmt(direct, 6)))
    print("  max |difference| = {:.2e}".format((stable - direct).abs().max().item()))


# -------------------------------------------------------------------- [5]
def section_closed_loop(env, ppo, seed):
    """Let the untrained PPO policy drive one full episode."""
    rule("[5] CLOSED LOOP: UNTRAINED PPO POLICY")
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed)
    total, terminated, truncated = 0.0, False, False
    with torch.no_grad():
        while not (terminated or truncated):
            _, _, env_action = ppo.act(obs)
            obs, reward, terminated, truncated, _ = env.step(env_action.numpy())
            total += reward
    print("steps         : {}".format(env.step_count))
    print("return        : {:.2f}".format(total))
    print("ended because : {}".format("ball left the tray" if terminated
                                      else "step limit reached"))
    print("c_rr          : {:.4f}".format(env.c_rr))
    print("The untrained policy is expected to fail; this only shows that the network")
    print("takes the gym observation and returns a valid action at every step.")


# ------------------------------------------------------------------- main
def run_walkthrough(variant, seed):
    """The default demo: sections [1] to [5]."""
    print("torch {} | variant {} | seed {}".format(torch.__version__, variant, seed))
    env = BallOnTrayEnv(friction_range=FRICTION_RANGE, disturbance_range=DISTURBANCE_RANGE)
    ppo, sac_actor, sac_critic = section_construction(variant, seed)
    section_ppo_forward(env, ppo, seed)
    batch = section_ppo_backward(env, ppo, seed)
    section_sac(sac_actor, sac_critic, batch)
    section_closed_loop(env, ppo, seed)
    env.close()


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Ball-on-Tray neural network demo.")
    parser.add_argument("--variant", choices=["baseline", "asymmetric"], default="baseline",
                        help="PPO critic of the walkthrough: observation only (baseline) "
                             "or observation + privileged inputs (default: baseline)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed of the networks and episodes (default: {})".format(
                            DEFAULT_SEED))
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
        run_walkthrough(args.variant, args.seed)
    print("\ntotal time: {:.2f} s".format(time.perf_counter() - start))


if __name__ == "__main__":
    main()
