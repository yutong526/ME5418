"""Tests for the networks and losses in ``networks/``.

The PPO tests run on both variants: the baseline (symmetric) network and the
asymmetric one whose critic also gets the privileged features. The SAC tests
check the squashed actor (tanh correction, log std clamp, bounds) and the twin
critics; the loss tests check the PPO objective and the return computation
against hand-written formulas.

Run with ``python tests/test_networks.py`` (no test framework required). The
functions are also named so that pytest can collect them if it is installed.
"""

import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch.distributions import Normal

# Make the project root importable, wherever this file is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ball_on_tray_gym import BallOnTrayEnv  # noqa: E402
from networks import (ACTION_DIM, LOG_STD_MAX, LOG_STD_MIN, OBS_DIM,  # noqa: E402
                      PRIVILEGED_DIM, ActorCritic, SACActor, SACCritic, ppo_loss,
                      privileged_features, returns_and_advantages, sac_actor_loss,
                      sac_critic_loss, tanh_log_det)

FULL = {"friction_range": (0.005, 0.05), "disturbance_range": (0.5, 2.0)}
VARIANTS = (("baseline", 0), ("asymmetric", PRIVILEGED_DIM))
BATCH = 32


def _batch(seed, n=BATCH):
    """Reproducible random observations, privileged features and target actions."""
    rng = np.random.RandomState(seed)
    obs = rng.normal(0.0, 1.0, size=(n, OBS_DIM)).astype(np.float32)
    privileged = rng.uniform(-1.0, 1.0, size=(n, PRIVILEGED_DIM)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(n, ACTION_DIM)).astype(np.float32)
    returns = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    return obs, privileged, actions, returns


def _privileged(network, privileged):
    """The privileged argument a variant expects (None for the baseline)."""
    return privileged if network.asymmetric else None


def _loss(network, obs, privileged, actions, returns):
    """A simple differentiable loss over every parameter: action NLL + value MSE."""
    log_prob, entropy, value = network.evaluate_actions(
        obs, actions, _privileged(network, privileged))
    return -log_prob.mean() + ((value - torch.as_tensor(returns)) ** 2).mean()


# ------------------------------------------------------------------ shapes
def test_output_shapes_single_and_batch():
    """Single observations give unbatched results, batches give batched ones."""
    obs, privileged, actions, _ = _batch(0)
    for name, dim in VARIANTS:
        network = ActorCritic(privileged_dim=dim, seed=0)
        extra = _privileged(network, privileged)

        action, log_prob, env_action = network.act(obs[0])
        assert action.shape == (ACTION_DIM,) and env_action.shape == (ACTION_DIM,)
        assert log_prob.shape == ()
        assert network.value(obs[0], None if extra is None else extra[0]).shape == ()

        action, log_prob, env_action = network.act(obs)
        assert action.shape == (BATCH, ACTION_DIM) and log_prob.shape == (BATCH,)
        dist, value = network(obs, extra)
        assert dist.mean.shape == (BATCH, ACTION_DIM) and value.shape == (BATCH,)
        log_prob, entropy, value = network.evaluate_actions(obs, actions, extra)
        assert log_prob.shape == entropy.shape == value.shape == (BATCH,)
        assert network.predict(obs[0]).shape == (ACTION_DIM,)
        assert network.predict(obs[0]).dtype == np.float32

        n_actor, n_critic = network.num_parameters()
        # 18*128+128 + 128*64+64 + 64*2+2 + 2 (log_std) = 10,820 for the actor;
        # the critic has 18*128+128 + 128*64+64 + 64+1 = 10,753 (+128 per extra input).
        assert n_actor == 10820
        assert n_critic == 10753 + 128 * dim
        print("    {:<10}: actor {} parameters, critic {} parameters".format(
            name, n_actor, n_critic))


def test_action_range_and_no_nan():
    """Means and env actions stay in [-1, 1], also for extreme inputs; nothing is NaN."""
    rng = np.random.RandomState(1)
    for _, dim in VARIANTS:
        network = ActorCritic(privileged_dim=dim, seed=1)
        # Push the actor towards saturation so the test is not trivial.
        with torch.no_grad():
            network.actor.body[-1].weight.mul_(1000.0)
        for scale in (1.0, 100.0, 1e4):
            obs = (scale * rng.normal(size=(256, OBS_DIM))).astype(np.float32)
            extra = rng.normal(size=(256, PRIVILEGED_DIM)).astype(np.float32) * scale
            dist, value = network(obs, extra if network.asymmetric else None)
            assert torch.all(dist.mean.abs() <= 1.0)
            action, log_prob, env_action = network.act(obs)
            assert torch.all(env_action.abs() <= 1.0)
            for tensor in (dist.mean, dist.stddev, value, action, log_prob):
                assert torch.all(torch.isfinite(tensor))
        # Raw samples may leave [-1, 1]; only the env action is clipped.
        torch.manual_seed(0)
        raw, _, clipped = network.act(np.zeros((2000, OBS_DIM), dtype=np.float32))
        assert torch.equal(clipped, raw.clamp(-1.0, 1.0))


def test_log_prob_and_entropy_formulas():
    """log_prob and entropy match the closed-form Gaussian expressions."""
    obs, privileged, actions, _ = _batch(2)
    for _, dim in VARIANTS:
        network = ActorCritic(privileged_dim=dim, seed=2)
        log_prob, entropy, _ = network.evaluate_actions(
            obs, actions, _privileged(network, privileged))
        mean = torch.tanh(network.actor.body(torch.as_tensor(obs))).detach().numpy()
        std = np.exp(network.actor.log_std.detach().numpy())
        expected = np.sum(-0.5 * ((actions - mean) / std) ** 2 - np.log(std)
                          - 0.5 * math.log(2.0 * math.pi), axis=1)
        assert np.allclose(log_prob.detach().numpy(), expected, atol=1e-5)
        expected_entropy = np.sum(0.5 + 0.5 * math.log(2.0 * math.pi) + np.log(std))
        assert np.allclose(entropy.detach().numpy(), expected_entropy, atol=1e-5)
        # Initial std = exp(-0.5) and the mean starts close to zero (small output gain).
        assert np.allclose(std, math.exp(-0.5))
        assert np.max(np.abs(mean)) < 0.1


# --------------------------------------------------------------- gradients
def test_every_parameter_gets_a_gradient():
    """One backward pass gives every parameter a finite, non-zero gradient."""
    obs, privileged, actions, returns = _batch(3)
    for _, dim in VARIANTS:
        network = ActorCritic(privileged_dim=dim, seed=3)
        _loss(network, obs, privileged, actions, returns).backward()
        for name, parameter in network.named_parameters():
            assert parameter.grad is not None, name
            assert torch.all(torch.isfinite(parameter.grad)), name
            assert parameter.grad.abs().sum() > 0.0, name


def test_loss_decreases_on_a_fixed_batch():
    """A few Adam steps on one fixed minibatch lower the loss."""
    obs, privileged, actions, returns = _batch(4)
    for _, dim in VARIANTS:
        network = ActorCritic(privileged_dim=dim, seed=4)
        optimiser = torch.optim.Adam(network.parameters(), lr=1e-3)
        first = _loss(network, obs, privileged, actions, returns).item()
        for _ in range(50):
            optimiser.zero_grad()
            loss = _loss(network, obs, privileged, actions, returns)
            loss.backward()
            optimiser.step()
        last = _loss(network, obs, privileged, actions, returns).item()
        assert last < first, (first, last)
    print("    loss on a fixed batch: {:.3f} -> {:.3f} after 50 Adam steps".format(first, last))


# ------------------------------------------------------------ determinism
def test_same_seed_same_network():
    """The same seed gives identical weights and outputs; a different seed does not."""
    obs, privileged, _, _ = _batch(5)
    for _, dim in VARIANTS:
        state = torch.random.get_rng_state()
        a = ActorCritic(privileged_dim=dim, seed=7)
        b = ActorCritic(privileged_dim=dim, seed=7)
        c = ActorCritic(privileged_dim=dim, seed=8)
        # Seeded construction leaves torch's global random state untouched.
        assert torch.equal(state, torch.random.get_rng_state())
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)
        extra = _privileged(a, privileged)
        assert torch.equal(a.value(obs, extra), b.value(obs, extra))
        assert not torch.equal(a.value(obs, extra), c.value(obs, extra))
        # Sampling is reproducible through torch's global seed.
        torch.manual_seed(11)
        first = a.act(obs)[0]
        torch.manual_seed(11)
        assert torch.equal(first, a.act(obs)[0])


def test_batch_matches_single():
    """Evaluating a batch gives the same results as evaluating one by one."""
    obs, privileged, actions, _ = _batch(6, n=8)
    for _, dim in VARIANTS:
        network = ActorCritic(privileged_dim=dim, seed=6)
        extra = _privileged(network, privileged)
        values = network.value(obs, extra)
        means = network.act(obs, deterministic=True)[0]
        log_probs = network.evaluate_actions(obs, actions, extra)[0]
        for i in range(len(obs)):
            single_extra = None if extra is None else extra[i]
            assert torch.allclose(network.value(obs[i], single_extra), values[i], atol=1e-6)
            assert torch.allclose(network.act(obs[i], deterministic=True)[0], means[i],
                                  atol=1e-6)
            single = network.evaluate_actions(obs[i], actions[i], single_extra)[0]
            assert torch.allclose(single[0], log_probs[i], atol=1e-6)


def test_save_and_load():
    """A saved and reloaded network gives identical outputs and configuration."""
    obs, privileged, actions, _ = _batch(7)
    folder = tempfile.mkdtemp()
    try:
        for name, dim in VARIANTS:
            network = ActorCritic(privileged_dim=dim, seed=9)
            with torch.no_grad():
                network.actor.log_std.fill_(-1.2)       # a non-default value is kept
            path = os.path.join(folder, name + ".pt")
            network.save(path)
            loaded = ActorCritic.load(path)
            assert loaded.config == network.config
            extra = _privileged(network, privileged)
            for x, y in zip(network.evaluate_actions(obs, actions, extra),
                            loaded.evaluate_actions(obs, actions, extra)):
                assert torch.equal(x, y)
    finally:
        shutil.rmtree(folder)


# --------------------------------------------------------- environment link
def test_works_with_the_environment():
    """Observations from the environment go in, actions go into env.step."""
    env = BallOnTrayEnv(**FULL)
    for _, dim in VARIANTS:
        network = ActorCritic(privileged_dim=dim, seed=10)
        obs, info = env.reset(seed=0)
        extra = privileged_features(info)
        assert extra.shape == (PRIVILEGED_DIM,) and extra.dtype == np.float32
        assert np.isclose(extra[0], info["c_rr"] / 0.05)
        for _ in range(20):
            torch.manual_seed(env.step_count)
            _, _, env_action = network.act(obs)
            assert env.action_space.contains(env_action.numpy().astype(np.float32))
            value = network.value(obs, privileged_features(info) if network.asymmetric else None)
            assert torch.isfinite(value)
            obs, _, terminated, truncated, info = env.step(env_action.numpy())
            if terminated or truncated:
                break
    assert privileged_features([info, info]).shape == (2, PRIVILEGED_DIM)


def test_wrong_inputs_are_rejected():
    """Missing or unexpected privileged features and wrong shapes raise ValueError."""
    obs, privileged, _, _ = _batch(8)
    cases = [
        lambda: ActorCritic(seed=0).value(obs, privileged),                 # baseline + extra
        lambda: ActorCritic(privileged_dim=3, seed=0).value(obs),           # asymmetric, none
        lambda: ActorCritic(seed=0).act(obs[:, :10]),                       # wrong width
        lambda: ActorCritic(privileged_dim=3, seed=0).value(obs, privileged[:5]),
    ]
    for case in cases:
        try:
            case()
        except ValueError:
            continue
        raise AssertionError("no ValueError")


# ------------------------------------------------------------------- losses
def test_ppo_loss_matches_formula():
    """ppo_loss equals the clipped objective written out by hand, also when clipping is active."""
    obs, privileged, actions, returns = _batch(12)
    for _, dim in VARIANTS:
        network = ActorCritic(privileged_dim=dim, seed=12)
        extra = _privileged(network, privileged)
        with torch.no_grad():
            log_probs, _, _ = network.evaluate_actions(obs, actions, extra)
        # Old log-probabilities shifted by up to +-0.5: some ratios leave [0.8, 1.2].
        old = log_probs + torch.linspace(-0.5, 0.5, BATCH)
        advantages = torch.as_tensor(np.random.RandomState(13).normal(size=BATCH),
                                     dtype=torch.float32)
        total, policy, value, entropy = ppo_loss(network, obs, actions, old, advantages,
                                                 torch.as_tensor(returns), extra)
        new_log_probs, new_entropy, values = network.evaluate_actions(obs, actions, extra)
        ratio = torch.exp(new_log_probs - old)
        assert torch.any((ratio < 0.8) | (ratio > 1.2))
        expected_policy = -torch.min(ratio * advantages,
                                     ratio.clamp(0.8, 1.2) * advantages).mean()
        expected_value = ((values - torch.as_tensor(returns)) ** 2).mean()
        assert torch.allclose(policy, expected_policy, atol=1e-6)
        assert torch.allclose(value, expected_value, atol=1e-6)
        assert torch.allclose(entropy, new_entropy.mean(), atol=1e-6)
        assert torch.allclose(total, policy + 0.5 * value - 0.01 * entropy, atol=1e-6)
        # Every parameter receives a gradient from the full PPO loss.
        total.backward()
        for name, parameter in network.named_parameters():
            assert parameter.grad is not None and torch.all(torch.isfinite(parameter.grad)), name
            assert parameter.grad.abs().sum() > 0.0, name


def test_returns_and_advantages():
    """Discounted returns restart after a fall; advantages are normalised."""
    rewards = torch.tensor([1.0, 1.0, -10.0, 1.0, 1.0])
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    values = torch.tensor([0.5, 0.0, -1.0, 2.0, 0.0])
    gamma = 0.9
    returns, advantages = returns_and_advantages(rewards, values, dones, gamma)
    expected = torch.tensor([1.0 + 0.9 * (1.0 + 0.9 * -10.0), 1.0 + 0.9 * -10.0, -10.0,
                             1.0 + 0.9 * 1.0, 1.0])
    assert torch.allclose(returns, expected, atol=1e-6)
    raw = expected - values
    assert torch.allclose(advantages, (raw - raw.mean()) / raw.std(), atol=1e-5)
    assert abs(advantages.mean().item()) < 1e-6 and abs(advantages.std().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------- SAC
def test_sac_shapes_and_parameter_counts():
    """SAC actor and critic: shapes, input widths and parameter counts."""
    obs = torch.as_tensor(_batch(14)[0])
    actor, critic = SACActor(seed=14), SACCritic(seed=15)
    action, log_prob = actor(obs)
    assert action.shape == (BATCH, ACTION_DIM) and log_prob.shape == (BATCH,)
    q1, q2 = critic(obs, action)
    assert q1.shape == q2.shape == (BATCH,)
    assert critic.q1_body[0].in_features == OBS_DIM + ACTION_DIM
    count = lambda module: sum(p.numel() for p in module.parameters())
    # Trunk 18*128+128 + 128*64+64 = 10,688, plus two heads of 64*2+2: 10,948.
    assert count(actor) == 10948
    # Two Q-networks of 20*128+128 + 128*64+64 + 64+1 = 11,009 each.
    assert count(critic) == 22018
    single = actor.predict(np.zeros(OBS_DIM, dtype=np.float32))
    assert single.shape == (ACTION_DIM,) and single.dtype == np.float32
    print("    SAC actor {:,} parameters, twin critics {:,} parameters".format(
        count(actor), count(critic)))


def test_sac_action_is_bounded():
    """The tanh squash keeps every action strictly inside (-1, 1); nothing is NaN."""
    actor = SACActor(seed=16)
    for scale in (1.0, 100.0):
        obs = torch.as_tensor(_batch(16, n=512)[0]) * scale
        action, log_prob = actor(obs)
        assert float(action.abs().max()) < 1.0
        assert torch.all(torch.isfinite(log_prob))
    deterministic, none = actor(torch.as_tensor(_batch(17)[0]), deterministic=True,
                                with_log_prob=False)
    assert none is None and float(deterministic.abs().max()) < 1.0


def test_sac_log_std_is_clamped():
    """Extreme inputs cannot push the std outside [exp(-20), exp(2)]."""
    actor = SACActor(seed=18)
    with torch.no_grad():
        actor.log_std_head.weight.mul_(1e4)
        actor.log_std_head.bias.fill_(500.0)
    obs = torch.as_tensor(_batch(18, n=64)[0]) * 100.0
    std = actor.distribution(obs).stddev
    assert float(std.max()) <= np.exp(LOG_STD_MAX) + 1e-4
    assert float(std.min()) >= np.exp(LOG_STD_MIN)
    _, log_prob = actor(obs)
    assert torch.all(torch.isfinite(log_prob))


def test_sac_tanh_correction_is_correct():
    """The stable correction equals log N(u) - sum log(1 - tanh(u)^2) and raises log pi.

    Checked against the direct formula where the latter is still accurate
    (|u| <= 3). tanh compresses the real line onto (-1, 1), so the squashed
    density is higher than the Gaussian one: log(1 - tanh(u)^2) <= 0 is
    subtracted, which can only raise the log-probability.
    """
    generator = torch.Generator().manual_seed(19)
    mean = torch.randn(32, ACTION_DIM, generator=generator)
    log_std = torch.randn(32, ACTION_DIM, generator=generator).clamp(-1.0, 1.0)
    dist = Normal(mean, log_std.exp())
    u = (mean + log_std.exp() * torch.randn(32, ACTION_DIM, generator=generator)).clamp(-3, 3)
    stable = dist.log_prob(u).sum(-1) - tanh_log_det(u)
    direct = dist.log_prob(u).sum(-1) - torch.log(1.0 - torch.tanh(u) ** 2).sum(-1)
    assert torch.allclose(stable, direct, atol=1e-5)
    assert bool(torch.all(stable >= dist.log_prob(u).sum(-1) - 1e-6))


def test_sac_critic_twins():
    """The twin Q-networks are independent; q1() matches the first output."""
    obs = torch.as_tensor(_batch(20)[0])
    action = torch.tanh(torch.randn(BATCH, ACTION_DIM,
                                    generator=torch.Generator().manual_seed(20)))
    critic = SACCritic(seed=20)
    q1, q2 = critic(obs, action)
    assert not torch.allclose(q1, q2), "twin critics must not be identical"
    assert torch.allclose(critic.q1(obs, action), q1)


def test_sac_losses_reach_every_parameter():
    """The critic loss trains both Q-networks; the actor loss reaches the actor through a~."""
    obs, _, actions, returns = _batch(21)
    obs = torch.as_tensor(obs)
    actor, critic = SACActor(seed=21), SACCritic(seed=22)
    critic_loss = sac_critic_loss(actor, critic, critic, obs, torch.as_tensor(actions),
                                  torch.as_tensor(returns), obs, torch.zeros(BATCH),
                                  gamma=0.99)
    critic_loss.backward()
    for name, parameter in critic.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0.0, name
    # The target is computed without gradient, so the actor gets none from it.
    assert all(p.grad is None for p in actor.parameters())
    critic.zero_grad()
    sac_actor_loss(actor, critic, obs).backward()
    for name, parameter in actor.named_parameters():
        assert parameter.grad is not None and torch.all(torch.isfinite(parameter.grad)), name
        assert parameter.grad.abs().sum() > 0.0, name


def test_sac_same_seed_same_network():
    """Seeded SAC networks are identical and leave torch's global random state alone."""
    state = torch.random.get_rng_state()
    a, b, c = SACActor(seed=23), SACActor(seed=23), SACActor(seed=24)
    critic_a, critic_b = SACCritic(seed=23), SACCritic(seed=23)
    assert torch.equal(state, torch.random.get_rng_state())
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(critic_a.parameters(), critic_b.parameters()):
        assert torch.equal(pa, pb)
    assert not torch.equal(a.mean_head.weight, c.mean_head.weight)


if __name__ == "__main__":
    tests = [
        test_output_shapes_single_and_batch,
        test_action_range_and_no_nan,
        test_log_prob_and_entropy_formulas,
        test_every_parameter_gets_a_gradient,
        test_loss_decreases_on_a_fixed_batch,
        test_same_seed_same_network,
        test_batch_matches_single,
        test_save_and_load,
        test_works_with_the_environment,
        test_wrong_inputs_are_rejected,
        test_ppo_loss_matches_formula,
        test_returns_and_advantages,
        test_sac_shapes_and_parameter_counts,
        test_sac_action_is_bounded,
        test_sac_log_std_is_clamped,
        test_sac_tanh_correction_is_correct,
        test_sac_critic_twins,
        test_sac_losses_reach_every_parameter,
        test_sac_same_seed_same_network,
    ]
    for test in tests:
        print("[RUN ] {}".format(test.__name__))
        test()
        print("[ OK ] {}".format(test.__name__))
    print("\nAll {} tests passed.".format(len(tests)))
