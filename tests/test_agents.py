"""Tests for the training code in ``agents/`` and the stage 3 losses.

The unit tests check the pieces the training loops are built from against
hand-written expectations: GAE with termination and truncation, the parallel
environments (automatic reset, true last observation, seeds, curriculum
switches), the replay buffer, the Polyak update and the entropy temperature.
The smoke tests run both training loops for a few thousand steps and check
that they finish, write their outputs, save loadable checkpoints and are
reproducible. They do not test that a good policy is learned.

Run with ``python tests/test_agents.py`` (no test framework required). The
functions are also named so that pytest can collect them if it is installed.
"""

import copy
import csv
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

# Make the project root importable, wherever this file is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (CURRICULA, Curriculum, ReplayBuffer, VecEnv, load_policy,  # noqa: E402
                    polyak_update, train_ppo, train_sac)
from agents.common import TRAIN_SEED_BASE, evaluate_policy  # noqa: E402
from ball_on_tray_gym import BallOnTrayEnv  # noqa: E402
from networks import (SACActor, SACCritic, gae, sac_actor_loss,  # noqa: E402
                      sac_alpha_loss)

TOL = 1e-6
FIXED = {"friction_range": (0.02, 0.02), "disturbance_range": None}


# ------------------------------------------------------------------- GAE
def test_gae_matches_hand_computation():
    """Three steps: normal, truncated (bootstrap, chain cut), terminated (no bootstrap)."""
    gamma, lam = 0.9, 0.5
    r = torch.tensor([1.0, 2.0, 3.0])
    v = torch.tensor([0.5, 1.0, 1.5])
    nv = torch.tensor([1.0, 4.0, 7.0])         # V(true next obs); 7.0 must be ignored
    term = torch.tensor([0.0, 0.0, 1.0])
    trunc = torch.tensor([0.0, 1.0, 0.0])
    adv, ret = gae(r, v, nv, term, trunc, gamma, lam)
    d0 = 1.0 + gamma * 1.0 - 0.5
    d1 = 2.0 + gamma * 4.0 - 1.0               # truncated: bootstraps from V(last obs)
    d2 = 3.0 - 1.5                             # terminated: nothing to bootstrap
    expected = torch.tensor([d0 + gamma * lam * d1, d1, d2])
    assert torch.allclose(adv, expected, atol=TOL), (adv, expected)
    assert torch.allclose(ret, adv + v, atol=TOL)


def test_gae_lambda_one_is_discounted_return():
    """With lambda = 1 and no episode end, returns are the discounted sum + bootstrap."""
    gamma = 0.95
    r = torch.tensor([1.0, 0.0, 2.0, -1.0])
    v = torch.tensor([0.3, -0.2, 0.7, 0.1])
    nv = torch.cat([v[1:], torch.tensor([5.0])])
    zeros = torch.zeros(4)
    _, ret = gae(r, v, nv, zeros, zeros, gamma, 1.0)
    expected0 = sum(gamma ** k * r[k].item() for k in range(4)) + gamma ** 4 * 5.0
    assert abs(ret[0].item() - expected0) < 1e-5


def test_gae_parallel_envs_are_independent():
    """(T, N) input gives, per column, the same result as N separate (T,) calls."""
    rng = np.random.RandomState(0)
    shape = (6, 3)
    r, v, nv = [torch.tensor(rng.normal(size=shape), dtype=torch.float32) for _ in range(3)]
    term = torch.tensor(rng.rand(*shape) < 0.2, dtype=torch.float32)
    trunc = torch.tensor(rng.rand(*shape) < 0.2, dtype=torch.float32)
    adv, _ = gae(r, v, nv, term, trunc, 0.99, 0.95)
    for j in range(shape[1]):
        col, _ = gae(r[:, j], v[:, j], nv[:, j], term[:, j], trunc[:, j], 0.99, 0.95)
        assert torch.allclose(adv[:, j], col, atol=1e-6)


# --------------------------------------------------------------- VecEnv
def test_vecenv_returns_true_last_observation():
    """On the step an episode ends, next_obs is its last observation, obs is the reset one."""
    vec = VecEnv(2, dict(FIXED, max_steps=5), seed=0)
    for t in range(5):
        result = vec.step(np.zeros((2, 2), dtype=np.float32))
    assert result["truncated"].all() and not result["terminated"].any()
    assert len(result["finished"]) == 2
    for i in range(2):
        # The observation after the reset is a fresh episode (step_count 0) ...
        assert vec.envs[i].step_count == 0
        # ... and differs from the episode's last observation that was returned.
        assert not np.allclose(vec.obs[i], result["next_obs"][i])
        assert result["next_infos"][i]["step_count"] == 5
    vec.close()


def test_vecenv_episode_seeds_reproducible_and_disjoint():
    """Same seed -> same episodes; episode seeds never overlap the evaluation seeds."""
    starts = []
    for _ in range(2):
        vec = VecEnv(3, FIXED, seed=7)
        starts.append(vec.obs.copy())
        vec.close()
    assert np.array_equal(starts[0], starts[1])
    rng = np.random.RandomState(7)
    first_seed = TRAIN_SEED_BASE + int(rng.randint(0, 2 ** 30))
    assert first_seed >= TRAIN_SEED_BASE > 2049


def test_vecenv_switches_ranges_at_episode_boundaries():
    """New environment arguments apply to the next episode, not the running one."""
    vec = VecEnv(1, dict(FIXED, max_steps=3), seed=0)
    assert vec.envs[0].c_rr == 0.02
    vec.set_env_kwargs({"friction_range": (0.04, 0.04), "disturbance_range": None,
                        "max_steps": 3})
    vec.step(np.zeros((1, 2), dtype=np.float32))
    assert vec.envs[0].c_rr == 0.02              # same episode, unchanged
    for _ in range(2):
        vec.step(np.zeros((1, 2), dtype=np.float32))
    assert vec.envs[0].c_rr == 0.04              # the episode ended; new range in use
    vec.close()


# ----------------------------------------------------------- curriculum
def test_curriculum_stages_follow_the_proposal():
    """staged: fixed -> random friction -> half pulses -> full; boundaries at the fractions."""
    cur = Curriculum("staged", 1000)
    names = [cur.stage_at(s)[1] for s in (0, 99, 100, 249, 250, 399, 400, 999)]
    assert names == ["fixed", "fixed", "friction", "friction", "half-pulses", "half-pulses",
                     "full", "full"], names
    assert cur.stage_at(0)[2]["disturbance_range"] is None
    assert cur.final_env_kwargs()["disturbance_range"] == (0.5, 2.0)
    for name, stages in CURRICULA.items():
        assert abs(sum(f for _, f, _ in stages) - 1.0) < 1e-9, name


def test_curriculum_realistic_only_in_last_stage():
    cur = Curriculum("staged", 1000, realistic=True)
    assert "actuator_tau" not in cur.stage_at(0)[2]
    last = cur.final_env_kwargs()
    assert last["actuator_tau"] > 0 and last["action_delay_steps"] == 1
    env = BallOnTrayEnv(**last)                 # the arguments are valid for the env
    env.close()


# --------------------------------------------------------- SAC pieces
def test_replay_buffer_wraps_and_samples():
    buf = ReplayBuffer(5, seed=0)
    for k in range(7):
        buf.add(np.full((1, 18), k, np.float32), np.full((1, 2), 0.1 * k, np.float32),
                np.array([k], np.float32), np.full((1, 18), k + 1, np.float32),
                np.array([k % 2], np.float32))
    assert len(buf) == 5
    assert sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0, 5.0, 6.0]   # 0 and 1 overwritten
    batch = buf.sample(64)
    assert batch["obs"].shape == (64, 18) and batch["actions"].shape == (64, 2)
    # Every sampled transition is internally consistent.
    assert torch.allclose(batch["next_obs"][:, 0], batch["obs"][:, 0] + 1)
    assert torch.allclose(batch["rewards"], batch["obs"][:, 0])


def test_polyak_update():
    torch.manual_seed(0)
    source, target = SACCritic(seed=1), SACCritic(seed=2)
    before = [p.clone() for p in target.parameters()]
    polyak_update(target, source, 0.25)
    for b, t, s in zip(before, target.parameters(), source.parameters()):
        assert torch.allclose(t, 0.75 * b + 0.25 * s, atol=TOL)
    polyak_update(target, source, 1.0)
    for t, s in zip(target.parameters(), source.parameters()):
        assert torch.allclose(t, s, atol=TOL)


def test_alpha_moves_towards_target_entropy():
    """Entropy below target -> alpha grows; above target -> alpha shrinks."""
    for log_prob, sign in ((5.0, +1), (-5.0, -1)):     # entropy -5 (low) / +5 (high)
        log_alpha = torch.zeros((), requires_grad=True)
        sac_alpha_loss(log_alpha, torch.full((8,), log_prob), -2.0).backward()
        # A gradient step moves log_alpha by -grad.
        assert np.sign(-log_alpha.grad.item()) == sign


def test_sac_actor_loss_min_variant():
    """use_min=True uses min(Q1, Q2) and differs from the Q1-only stage 2 version."""
    actor, critic = SACActor(seed=0), SACCritic(seed=1)
    obs = torch.randn(16, 18)
    torch.manual_seed(3)
    with_min = sac_actor_loss(actor, critic, obs, 0.2, use_min=True)
    torch.manual_seed(3)
    actions, log_probs = actor(obs)
    expected = (0.2 * log_probs - torch.min(*critic(obs, actions))).mean()
    assert torch.allclose(with_min, expected, atol=TOL)
    torch.manual_seed(3)
    assert not torch.allclose(with_min, sac_actor_loss(actor, critic, obs, 0.2))


# ------------------------------------------------------------- smoke runs
def _rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def test_ppo_smoke_run_saves_loadable_checkpoint():
    out = tempfile.mkdtemp()
    try:
        train_ppo(out, verbose=False, total_steps=1024, num_envs=2, rollout_steps=128,
                  epochs=2, eval_interval=512, seed=0)
        rows = _rows(os.path.join(out, "progress.csv"))
        assert len(rows) == 4 and rows[-1]["step"] == "1024"
        # Each iteration is logged under the stage active when its rollout began.
        cur = Curriculum("staged", 1024)
        assert [r["stage"] for r in rows] == [cur.stage_at(s)[1] for s in (0, 256, 512, 768)]
        assert rows[0]["stage"] == "fixed" and rows[-1]["stage"] == "full"
        assert rows[-1]["eval_return"] != ""
        for name in ("best.pt", "last.pt"):
            algo, policy, _ = load_policy(os.path.join(out, name))
            assert algo == "ppo"
            action = policy(np.zeros(18, np.float32))
            assert action.shape == (2,) and np.all(np.abs(action) <= 1.0)
    finally:
        shutil.rmtree(out)


def test_ppo_asymmetric_smoke_run():
    out = tempfile.mkdtemp()
    try:
        network, _ = train_ppo(out, verbose=False, total_steps=512, num_envs=2,
                               rollout_steps=128, epochs=1, eval_interval=512,
                               asymmetric=True, curriculum="full", seed=0)
        assert network.asymmetric
        algo, policy, _ = load_policy(os.path.join(out, "last.pt"))
        policy(np.zeros(18, np.float32))     # the deployed actor needs no privileged input
    finally:
        shutil.rmtree(out)


def test_ppo_is_reproducible():
    """Same seed -> identical weights after training."""
    nets = []
    for _ in range(2):
        out = tempfile.mkdtemp()
        try:
            net, _ = train_ppo(out, verbose=False, total_steps=512, num_envs=2,
                               rollout_steps=128, epochs=2, eval_interval=10 ** 9, seed=3)
            nets.append(net)
        finally:
            shutil.rmtree(out)
    for a, b in zip(nets[0].parameters(), nets[1].parameters()):
        assert torch.equal(a, b)


def test_sac_smoke_run_learns_alpha_and_saves():
    out = tempfile.mkdtemp()
    try:
        actor, cfg = train_sac(out, verbose=False, total_steps=600, warmup_steps=300,
                               batch_size=64, eval_interval=600, log_interval=200, seed=0)
        rows = _rows(os.path.join(out, "progress.csv"))
        alphas = [float(r["alpha"]) for r in rows if r["alpha"]]
        assert alphas and alphas[-1] != cfg["init_alpha"]      # alpha is being tuned
        algo, policy, ckpt = load_policy(os.path.join(out, "last.pt"))
        assert algo == "sac" and ckpt["step"] == 600
        assert np.all(np.abs(policy(np.zeros(18, np.float32))) < 1.0)
        # The target critic differs from the critic (Polyak, not a copy) but is close.
        critic, target = SACCritic(), SACCritic()
        critic.load_state_dict(ckpt["critic"])
        target.load_state_dict(ckpt["target_critic"])
        diffs = [(c - t).abs().max().item()
                 for c, t in zip(critic.parameters(), target.parameters())]
        assert 0.0 < max(diffs)
    finally:
        shutil.rmtree(out)


def test_evaluate_policy_matches_manual_rollout():
    """evaluate_policy returns exactly the return of a hand-run episode."""
    policy = lambda obs: np.array([0.1, -0.1], np.float32)      # noqa: E731
    res = evaluate_policy(policy, seeds=(3,))
    env = BallOnTrayEnv(friction_range=(0.005, 0.05), disturbance_range=(0.5, 2.0))
    obs, _ = env.reset(seed=3)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(policy(obs))
        total += r
        done = term or trunc
    assert abs(res["mean"] - total) < 1e-9


if __name__ == "__main__":
    tests = [
        test_gae_matches_hand_computation,
        test_gae_lambda_one_is_discounted_return,
        test_gae_parallel_envs_are_independent,
        test_vecenv_returns_true_last_observation,
        test_vecenv_episode_seeds_reproducible_and_disjoint,
        test_vecenv_switches_ranges_at_episode_boundaries,
        test_curriculum_stages_follow_the_proposal,
        test_curriculum_realistic_only_in_last_stage,
        test_replay_buffer_wraps_and_samples,
        test_polyak_update,
        test_alpha_moves_towards_target_entropy,
        test_sac_actor_loss_min_variant,
        test_ppo_smoke_run_saves_loadable_checkpoint,
        test_ppo_asymmetric_smoke_run,
        test_ppo_is_reproducible,
        test_sac_smoke_run_learns_alpha_and_saves,
        test_evaluate_policy_matches_manual_rollout,
    ]
    torch.set_num_threads(1)
    for test in tests:
        print("[RUN ] {}".format(test.__name__))
        test()
        print("[ OK ] {}".format(test.__name__))
    print("\nAll {} tests passed.".format(len(tests)))
