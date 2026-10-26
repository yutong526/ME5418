"""Tests for the actor-critic networks in ``networks/``.

Every test runs on both variants: the baseline (symmetric) network and the
asymmetric one whose critic also gets the privileged features.

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

# Make the project root importable, wherever this file is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ball_on_tray_gym import BallOnTrayEnv  # noqa: E402
from networks import (ACTION_DIM, OBS_DIM, PRIVILEGED_DIM, ActorCritic,  # noqa: E402
                      privileged_features)

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
    ]
    for test in tests:
        print("[RUN ] {}".format(test.__name__))
        test()
        print("[ OK ] {}".format(test.__name__))
    print("\nAll {} tests passed.".format(len(tests)))
