"""Two small supervised experiments with the actor-critic networks.

Both use the same data: rollouts of the LQR controller from ``controllers.py``
on the full randomisation range, with Gaussian exploration noise added to the
executed action so that the data also covers states slightly off the LQR's own
trajectory. Every sample stores the clean LQR action as the label.

Seeds are split by episode, never by step, and are disjoint from the seeds
0-99 that ``evaluate_baselines.py`` uses:

    training     seeds 1000-1199
    validation   seeds 2000-2049
    closed loop  seeds 0-99 (behaviour cloning only; the same seeds as the
                 baseline evaluation, so the LQR numbers are comparable)

Behaviour cloning (``run_behaviour_cloning``)
    Fit the actor's mean action to the LQR action (mean squared error), then
    run the cloned actor's mean action in closed loop next to the LQR. This
    checks that the network can represent and learn a good policy; it is not
    the reinforcement learning method itself.

Critic comparison (``run_critic_comparison``)
    Fit the baseline critic (18 inputs) and the asymmetric critic (18 + 3
    privileged inputs) to the same discounted returns. The validation error
    measures how much information the privileged inputs carry about the return
    of this behaviour policy; it says nothing yet about PPO training.

Realistic condition (``realistic=True``)
    Repeats the experiment in the environment with the realism parameters of
    ``demo.py --realistic`` (observation noise 0.002 m and 0.03 m/s, actuator
    lag 0.05 s, one-step action delay), reported next to the ideal condition.
    Behaviour cloning keeps the actor trained on the clean data and only moves
    the closed-loop evaluation (cloned actor and LQR alike) into the realistic
    environment. The critic comparison collects new data in the realistic
    environment, because the critics must be trained on the observation
    distribution they are tested on; there the LQR that collects the data acts
    on the noisy observation. The return labels are always the true rewards,
    and the privileged inputs are always the true c_rr and a_base.

Return targets and the 500-step time limit
    The discounted return G_t = sum_k GAMMA^k r_(t+k) is computed up to the end
    of the episode. When an episode ends at the time limit (truncation), the
    rewards after step 500 are missing and G_t near the end is biased low. Only
    steps t < RETURN_STEPS = 250 are therefore used as samples: their returns
    include at least 250 later steps, and the missing tail weighs at most
    GAMMA^250 = 0.6 % (for GAMMA = 0.98, i.e. a horizon of about one second).
    Episodes that end because the ball falls are genuinely terminal and need no
    correction.
"""

import time

import numpy as np
import torch

from ball_on_tray_gym import BallOnTrayEnv
from controllers import design_lqr, make_lqr_policy
from demo import REALISTIC_KWARGS       # the same settings as demo.py --realistic
from networks.actor_critic import PRIVILEGED_DIM, ActorCritic, privileged_features

# ------------------------------------------------------------------- data
FRICTION_RANGE = (0.005, 0.05)          # c_rr [-]
DISTURBANCE_RANGE = (0.5, 2.0)          # ||a_base|| [m/s^2]
TRAIN_SEEDS = tuple(range(1000, 1200))  # 200 training episodes
VAL_SEEDS = tuple(range(2000, 2050))    # 50 validation episodes
EVAL_SEEDS = tuple(range(0, 100))       # closed-loop evaluation episodes
EXPLORATION_STD = 0.3                   # noise on the executed action [action units]
NOISE_SEED_OFFSET = 7919                # noise generator seed = episode seed + offset

# --------------------------------------------------------- behaviour cloning
BC_EPOCHS = 15
BC_BATCH_SIZE = 256
BC_LEARNING_RATE = 1e-3
BC_NETWORK_SEED = 0

# --------------------------------------------------------- critic comparison
GAMMA = 0.98                    # discount factor of the return targets
RETURN_STEPS = 250              # only steps t < RETURN_STEPS are used (see above)
CRITIC_EPOCHS = 30
CRITIC_BATCH_SIZE = 256
CRITIC_LEARNING_RATE = 1e-3
CRITIC_NETWORK_SEEDS = (0, 1, 2)   # every network seed is reported, none is picked


# ===================================================================== data
def collect_rollouts(seeds, exploration_std=EXPLORATION_STD, env_kwargs=None):
    """Roll out the noisy LQR controller and record one sample per observation.

    Args:
        seeds: episode seeds.
        exploration_std: standard deviation of the noise added to the executed
            action; the label is always the LQR action for the observation.
        env_kwargs: extra environment arguments (the realism parameters), or
            None for the ideal environment. With observation noise, the stored
            observation and the LQR's input are the noisy observation.

    Returns:
        dict of numpy arrays, one row per observation the policy acted on:
        "obs" (N, 18), "label" (N, 2) LQR action, "privileged" (N, 3),
        "reward" (N,) reward that followed, "step" (N,) index within the
        episode, "episode" (N,) index into ``seeds``, "pulse" (N,) whether a
        disturbance pulse was active, "terminated" (N,) whether the episode
        ended by a fall after this step.
    """
    gain, _ = design_lqr()
    lqr = make_lqr_policy(gain)
    env = BallOnTrayEnv(friction_range=FRICTION_RANGE, disturbance_range=DISTURBANCE_RANGE,
                        **(env_kwargs or {}))
    columns = {key: [] for key in ("obs", "label", "privileged", "reward", "step",
                                   "episode", "pulse", "terminated")}
    for index, seed in enumerate(seeds):
        noise = np.random.RandomState(seed + NOISE_SEED_OFFSET)
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            label = lqr(obs)
            executed = np.clip(label + noise.normal(0.0, exploration_std, size=2), -1.0, 1.0)
            columns["obs"].append(obs)
            columns["label"].append(label)
            columns["privileged"].append(privileged_features(info))
            columns["pulse"].append(bool(np.any(info["a_base"] != 0.0)))
            columns["step"].append(env.step_count)
            columns["episode"].append(index)
            obs, reward, terminated, truncated, info = env.step(executed)
            columns["reward"].append(reward)
            columns["terminated"].append(terminated)
    env.close()
    dtypes = {"obs": np.float32, "label": np.float32, "privileged": np.float32,
              "reward": np.float64, "step": np.int64, "episode": np.int64,
              "pulse": np.bool_, "terminated": np.bool_}
    return {key: np.asarray(value, dtype=dtypes[key]) for key, value in columns.items()}


def discounted_returns(data, gamma=GAMMA):
    """G_t = r_t + gamma * G_(t+1), computed backwards within every episode."""
    returns = np.zeros_like(data["reward"])
    running = 0.0
    for i in range(len(returns) - 1, -1, -1):
        last_of_episode = (i == len(returns) - 1 or data["episode"][i + 1] != data["episode"][i])
        if last_of_episode:
            running = 0.0
        running = data["reward"][i] + gamma * running
        returns[i] = running
    return returns


def _minibatches(n, batch_size, generator):
    """Yield index batches of a random permutation of range(n)."""
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


# ======================================================= behaviour cloning
def train_behaviour_cloning(train, val, seed=BC_NETWORK_SEED):
    """Fit the actor's mean action to the LQR labels (mean squared error).

    Only the actor's parameters are trained; the critic is not used.

    Returns:
        (network, history) where history lists (epoch, train MSE, val MSE).
    """
    network = ActorCritic(seed=seed)
    optimiser = torch.optim.Adam(network.actor.parameters(), lr=BC_LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
    x_train, y_train = torch.as_tensor(train["obs"]), torch.as_tensor(train["label"])
    x_val, y_val = torch.as_tensor(val["obs"]), torch.as_tensor(val["label"])

    def mse(x, y):
        return ((network.actor(x)[0] - y) ** 2).mean()

    history = []
    for epoch in range(1, BC_EPOCHS + 1):
        for batch in _minibatches(len(x_train), BC_BATCH_SIZE, generator):
            optimiser.zero_grad()
            loss = mse(x_train[batch], y_train[batch])
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            history.append((epoch, mse(x_train, y_train).item(), mse(x_val, y_val).item()))
    return network, history


def closed_loop(policy, seeds, env_kwargs=None):
    """Run a policy (obs -> action) on the given seeds; return per-episode results.

    ``env_kwargs`` are extra environment arguments (the realism parameters), or
    None for the ideal environment.
    """
    env = BallOnTrayEnv(friction_range=FRICTION_RANGE, disturbance_range=DISTURBANCE_RANGE,
                        **(env_kwargs or {}))
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total, in_goal = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            in_goal += int(np.linalg.norm(info["ball_pos"]) < env.GOAL_RADIUS)
        results.append({"return": total, "survived": bool(truncated and not terminated),
                        "goal": in_goal / float(env.step_count)})
    env.close()
    return results


def describe_realism():
    """One line with the realism parameters."""
    return ("observation noise {pos_noise_std:g} m / {vel_noise_std:g} m/s, actuator lag "
            "{actuator_tau:g} s, action delay {action_delay_steps} step(s)".format(**REALISTIC_KWARGS))


def print_closed_loop(rows):
    """Print a closed-loop table (LQR and cloned actor) and their paired difference."""
    print("  {:<14} {:>10} {:>18} {:>10} {:>10}".format(
        "policy", "survived", "return", "% of 750", "goal %"))
    for name, results in rows:
        returns = np.array([r["return"] for r in results])
        print("  {:<14} {:>9.0f}% {:>10.1f} +- {:<5.1f} {:>9.1f}% {:>9.1f}%".format(
            name, 100.0 * np.mean([r["survived"] for r in results]), returns.mean(),
            returns.std(), 100.0 * returns.mean() / 750.0,
            100.0 * np.mean([r["goal"] for r in results])))
    lqr_returns = np.array([r["return"] for r in rows[0][1]])
    bc_returns = np.array([r["return"] for r in rows[1][1]])
    print("  paired difference (cloned - LQR): {:+.1f} mean, cloned better on {} of {} "
          "seeds".format((bc_returns - lqr_returns).mean(),
                         int(np.sum(bc_returns > lqr_returns)), len(lqr_returns)))


def run_behaviour_cloning(realistic=False):
    """Behaviour cloning of the LQR controller, with a closed-loop comparison.

    With ``realistic=True`` the same cloned actor (trained on the clean data)
    is also evaluated, next to the LQR, in the realistic environment.
    """
    start = time.perf_counter()
    train = collect_rollouts(TRAIN_SEEDS)
    val = collect_rollouts(VAL_SEEDS)
    print("behaviour cloning of the LQR controller")
    print("  data         : noisy-LQR rollouts (exploration std {:g}), label = clean LQR "
          "action".format(EXPLORATION_STD))
    print("  training     : seeds {}-{}, {} episodes, {:,} samples".format(
        TRAIN_SEEDS[0], TRAIN_SEEDS[-1], len(TRAIN_SEEDS), len(train["obs"])))
    print("  validation   : seeds {}-{}, {} episodes, {:,} samples".format(
        VAL_SEEDS[0], VAL_SEEDS[-1], len(VAL_SEEDS), len(val["obs"])))
    print("  collected in {:.1f} s\n".format(time.perf_counter() - start))

    start = time.perf_counter()
    network, history = train_behaviour_cloning(train, val)
    zero_mse = float(np.mean(val["label"] ** 2))
    print("  actor mean action vs LQR action, mean squared error (action units^2)")
    print("  {:>5} {:>12} {:>12}".format("epoch", "train", "validation"))
    for epoch, train_mse, val_mse in history:
        if epoch in (1, 2, 3, 5) or epoch % 5 == 0:
            print("  {:>5} {:>12.5f} {:>12.5f}".format(epoch, train_mse, val_mse))
    print("  reference: always predicting a zero action gives {:.5f} on validation".format(
        zero_mse))
    print("  trained in {:.1f} s ({} epochs, batch {}, Adam lr {:g})\n".format(
        time.perf_counter() - start, BC_EPOCHS, BC_BATCH_SIZE, BC_LEARNING_RATE))

    start = time.perf_counter()
    gain, _ = design_lqr()
    lqr = make_lqr_policy(gain)
    cloned = lambda obs: network.predict(obs)
    print("  closed loop on seeds {}-{} (full randomisation range, mean action, "
          "no noise)".format(EVAL_SEEDS[0], EVAL_SEEDS[-1]))
    print_closed_loop([("LQR", closed_loop(lqr, EVAL_SEEDS)),
                       ("cloned actor", closed_loop(cloned, EVAL_SEEDS))])
    print("  evaluated in {:.1f} s\n".format(time.perf_counter() - start))

    if realistic:
        start = time.perf_counter()
        print("  closed loop on seeds {}-{}, REALISTIC environment: {}".format(
            EVAL_SEEDS[0], EVAL_SEEDS[-1], describe_realism()))
        print("  (the same cloned actor, trained on the clean data above; both policies "
              "see the noisy observation)")
        print_closed_loop([("LQR", closed_loop(lqr, EVAL_SEEDS, REALISTIC_KWARGS)),
                           ("cloned actor", closed_loop(cloned, EVAL_SEEDS, REALISTIC_KWARGS))])
        print("  evaluated in {:.1f} s\n".format(time.perf_counter() - start))
    return network


# ======================================================= critic comparison
def _return_samples(data):
    """Select the samples with an unbiased return target (see the module docstring)."""
    returns = discounted_returns(data)
    keep = data["step"] < RETURN_STEPS
    return {key: value[keep] for key, value in data.items()}, returns[keep]


def train_critic(privileged_dim, train, train_returns, val, val_returns, seed):
    """Fit one critic to the return targets (standardised with training statistics).

    Returns:
        dict with the validation MSE overall, on steps with and without an
        active pulse (all in return units squared), and the explained variance.
    """
    network = ActorCritic(privileged_dim=privileged_dim, seed=seed)
    optimiser = torch.optim.Adam(network.critic.parameters(), lr=CRITIC_LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
    mean, std = float(train_returns.mean()), float(train_returns.std())

    def inputs(data):
        x = torch.as_tensor(data["obs"])
        if privileged_dim:
            x = torch.cat([x, torch.as_tensor(data["privileged"])], dim=1)
        return x

    x_train, x_val = inputs(train), inputs(val)
    y_train = torch.as_tensor((train_returns - mean) / std, dtype=torch.float32)
    for _ in range(CRITIC_EPOCHS):
        for batch in _minibatches(len(x_train), CRITIC_BATCH_SIZE, generator):
            optimiser.zero_grad()
            loss = ((network.critic(x_train[batch]) - y_train[batch]) ** 2).mean()
            loss.backward()
            optimiser.step()
    with torch.no_grad():
        prediction = network.critic(x_val).numpy() * std + mean
    error = (prediction - val_returns) ** 2
    pulse = val["pulse"]
    return {"mse": float(error.mean()), "mse_pulse": float(error[pulse].mean()),
            "mse_quiet": float(error[~pulse].mean()),
            "explained": 1.0 - float(error.mean()) / float(val_returns.var())}


def run_critic_comparison(realistic=False):
    """Baseline critic vs asymmetric critic on the same return targets.

    With ``realistic=True`` the whole comparison is repeated on new data
    collected in the realistic environment, and the two conditions are
    summarised side by side.
    """
    summaries = {"ideal": _compare_critics(None)}
    if realistic:
        summaries["realistic"] = _compare_critics(REALISTIC_KWARGS)
        print("summary: mean validation MSE over {} network seeds (return units^2)".format(
            len(CRITIC_NETWORK_SEEDS)))
        print("  {:<10} {:<15} {:>10} {:>12} {:>10}".format(
            "condition", "steps", "baseline", "asymmetric", "change"))
        for condition, summary in summaries.items():
            for key, label in (("mse", "all"), ("mse_pulse", "during a pulse"),
                               ("mse_quiet", "no pulse")):
                base = np.mean([r[key] for r in summary["baseline"]])
                asym = np.mean([r[key] for r in summary["asymmetric"]])
                print("  {:<10} {:<15} {:>10.3f} {:>12.3f} {:>9.0%}".format(
                    condition, label, base, asym, asym / base - 1.0))
        print("")
    return summaries


def _compare_critics(env_kwargs):
    """Collect data in one environment condition and compare the two critics on it."""
    start = time.perf_counter()
    train, train_returns = _return_samples(collect_rollouts(TRAIN_SEEDS, env_kwargs=env_kwargs))
    val, val_returns = _return_samples(collect_rollouts(VAL_SEEDS, env_kwargs=env_kwargs))
    if env_kwargs is None:
        print("critic comparison: how much do the privileged inputs tell about the return?")
    else:
        print("critic comparison, REALISTIC environment: {}".format(describe_realism()))
        print("  new data collected in this environment; the LQR collecting it acts on the "
              "noisy observation,")
        print("  the critics see the noisy observation, the return labels are the true "
              "rewards and the")
        print("  privileged inputs are the true c_rr and a_base")
    print("  data         : noisy-LQR rollouts (exploration std {:g}), the same for both "
          "critics".format(EXPLORATION_STD))
    print("  target       : discounted return, gamma {:g}, computed to the episode end; "
          "only steps t < {} are used,".format(GAMMA, RETURN_STEPS))
    print("                 so the part cut off by the 500-step limit weighs at most "
          "gamma^{} = {:.2%}".format(500 - RETURN_STEPS, GAMMA ** (500 - RETURN_STEPS)))
    print("  training     : seeds {}-{}, {:,} samples ({:.0%} during a pulse)".format(
        TRAIN_SEEDS[0], TRAIN_SEEDS[-1], len(train_returns), train["pulse"].mean()))
    print("  validation   : seeds {}-{}, {:,} samples ({:.0%} during a pulse)".format(
        VAL_SEEDS[0], VAL_SEEDS[-1], len(val_returns), val["pulse"].mean()))
    print("  return       : validation mean {:.2f}, std {:.2f}".format(
        val_returns.mean(), val_returns.std()))
    print("  collected in {:.1f} s\n".format(time.perf_counter() - start))

    start = time.perf_counter()
    print("  validation error of the return prediction (return units^2), "
          "{} epochs, Adam lr {:g}".format(CRITIC_EPOCHS, CRITIC_LEARNING_RATE))
    print("  {:<12} {:>5} {:>10} {:>12} {:>12} {:>11}".format(
        "critic", "seed", "MSE", "MSE pulse", "MSE quiet", "expl. var"))
    summary = {}
    for name, dim in (("baseline", 0), ("asymmetric", PRIVILEGED_DIM)):
        rows = []
        for seed in CRITIC_NETWORK_SEEDS:
            result = train_critic(dim, train, train_returns, val, val_returns, seed)
            rows.append(result)
            print("  {:<12} {:>5} {:>10.3f} {:>12.3f} {:>12.3f} {:>10.1%}".format(
                name, seed, result["mse"], result["mse_pulse"], result["mse_quiet"],
                result["explained"]))
        summary[name] = rows
    for key, label in (("mse", "overall"), ("mse_pulse", "during a pulse"),
                       ("mse_quiet", "no pulse")):
        base = np.mean([r[key] for r in summary["baseline"]])
        asym = np.mean([r[key] for r in summary["asymmetric"]])
        print("  mean MSE {:<15}: baseline {:.3f}, asymmetric {:.3f} ({:+.0%})".format(
            label, base, asym, asym / base - 1.0))
    print("  trained in {:.1f} s\n".format(time.perf_counter() - start))
    return summary
