# ME5418 Ball on Tray

## Introduction

A square tray (0.3 m × 0.3 m) can tilt about two axes, pitch and roll, and a solid ball rolls on it. The agent tilts the tray to bring the ball to the tray centre and keep it there without letting it fall off, while the mobile base under the tray occasionally accelerates and pushes the ball away. The environment is `BallOnTrayEnv` in `ball_on_tray_gym.py` (gym 0.26 API, control rate 50 Hz).

**Observation** (18 numbers): the last 3 frames, oldest first. One frame is `[x, y, vx, vy, pitch, roll]`, normalised: position divided by the tray half size (0.15 m), velocity by 1 m/s, tilt by the tilt limit (15°). Values are not clipped. The rolling resistance and the base acceleration are not observed.

**Action** (2 numbers in [−1, 1]): the tilt increment for pitch and roll. An action of ±1 changes the tilt by ±3° in that step; the accumulated tilt is limited to ±15°. Positive pitch accelerates the ball towards +x, positive roll towards +y.

**Reward** per step, with `dist` the distance of the ball from the centre and `a_t` the clipped action:

```
r = (1 − dist / d_max) + 0.5 · [dist < 0.02 m] − 0.05 · ‖a_t − a_(t−1)‖² − 0.01 · ‖a_t‖²
```

- `1 − dist / d_max`: 1 at the centre, 0 in a corner (`d_max` = half diagonal of the tray, 0.212 m).
- `0.5 · [dist < 0.02 m]`: bonus while the ball is within 2 cm of the centre.
- `0.05 · ‖a_t − a_(t−1)‖²`: penalty for changing the action (smoothness); `a_(t−1)` is zero after a reset.
- `0.01 · ‖a_t‖²`: penalty for the action size (effort).
- If the ball leaves the tray, the reward of that step is −10 instead. The maximum is 1.5 per step, 750 per episode.

**Episode end**: `terminated` when the ball leaves the tray (|x| or |y| above 0.15 m); `truncated` after 500 steps (`max_steps`). Each episode starts with a level tray and the ball at rest at a random position in the central 80 % of the tray.

**Randomisation**, sampled at every reset:

- Rolling resistance coefficient `c_rr`, uniform in `friction_range` and constant during the episode (default: fixed 0.02; full range 0.005–0.05).
- Base acceleration disturbance: 1 to 3 pulses with random start time, a duration of 0.2–1.0 s, a magnitude uniform in `disturbance_range` and a random direction (default: no disturbance; full range 0.5–2.0 m/s²). The pulses never overlap.

**Optional realism parameters** add observation noise, actuator lag and action delay. They are off by default; see [Optional realism parameters](#optional-realism-parameters).

**Differences from the proposal**:

- The robot arm is not simulated. The tray is an ideal two-axis platform whose tilt follows the command directly.
- A rolling resistance coefficient is used instead of a "friction coefficient", because the ball rolls without slipping.
- The distance reward is a normalised distance combined with a survival term: `1 − dist / d_max` is 1 per step survived minus the distance as a fraction of `d_max`.
- The disturbance pulses do not overlap.

## Quick start

```bash
conda env create -f environment.yml     # 1. create the environment (only once)
conda activate me5418-ballontray        # 2. activate it (in every new terminal)
python train.py --algo ppo --quick      # 3. training smoke run: every code path in ~15 s
python tests/test_agents.py             # 4. training tests (~20 s)
python demo.py                          # 5. environment demo: random policy, then LQR controller
```

A full training run and its evaluation (see [Training the agents](#training-the-agents)):

```bash
python train.py --algo ppo                              # ~10 min on one CPU core
python evaluate_agent.py runs/ppo_staged_seed0/best.pt  # agent vs PD and LQR, seeds 0-99
python demo.py --checkpoint runs/ppo_staged_seed0/best.pt --policy lqr+agent
```

Activate the environment in every new terminal before running anything, otherwise `gym` will not be found or a wrong version will be used; in VS Code, also select `me5418-ballontray` as the Python interpreter.

The demo opens a 3D view of the tilting tray. The default run shows the random policy and the LQR controller. To also see the PD controller:

```bash
python demo.py --policy all             # random, then PD, then LQR
```

All commands are run from the project root.

## Project structure

```
ME5418_BallOnTray/
├── README.md
├── environment.yml
├── demo.py
├── demo_network.py
├── train.py
├── evaluate_agent.py
├── plot_training.py
├── ball_on_tray_gym.py
├── controllers.py
├── evaluate_baselines.py
├── networks/
│   ├── __init__.py
│   ├── actor_critic.py
│   ├── sac.py
│   ├── losses.py
│   └── experiments.py
├── agents/
│   ├── __init__.py
│   ├── common.py
│   ├── ppo.py
│   └── sac.py
├── tests/
│   ├── test_env.py
│   ├── test_renderer_3d.py
│   ├── test_realism.py
│   ├── test_networks.py
│   ├── test_agents.py
│   ├── make_golden.py
│   └── golden_v1.npz
├── trained/
├── media/
├── results/
└── report/
```

| File | Content |
|---|---|
| `demo.py` | The demo: random, PD and LQR policies, or a trained agent (`--checkpoint`), in the 3D view. |
| `ball_on_tray_gym.py` | The environment class `BallOnTrayEnv` (dynamics, reward, observation) and the 3D renderer `TrayRenderer3D`, drawn with matplotlib's mplot3d. The renderer only reads the environment and does not change it; it is used by `BallOnTrayEnv.render()` and by `demo.py`. |
| `controllers.py` | The PD and LQR baseline controllers, used by `demo.py` and `evaluate_baselines.py`. |
| `evaluate_baselines.py` | Batch evaluation of the PD and LQR controllers without rendering. |
| `networks/actor_critic.py` | The PPO actor-critic networks (PyTorch): baseline and asymmetric variant. |
| `networks/sac.py` | The SAC networks: tanh-squashed Gaussian actor and twin Q critics. |
| `networks/losses.py` | The PPO clipped loss, GAE(λ), the SAC critic, actor and temperature losses, and the plain returns and advantages used by the demo. |
| `networks/experiments.py` | Two small supervised experiments with the networks: behaviour cloning of the LQR, and the critic comparison. |
| `demo_network.py` | Demo of the networks: construction, PPO forward and backward pass, SAC forward and backward pass, closed loop; `--bc` and `--critics` run the experiments. |
| `agents/common.py` | Training infrastructure: the curriculum (staged domain randomisation), parallel environments with automatic reset, evaluation on fixed seeds, CSV logging. |
| `agents/ppo.py` | The PPO training loop. |
| `agents/sac.py` | The SAC training loop and its replay buffer. |
| `train.py` | Command line for training PPO or SAC; writes `runs/<name>/`. |
| `evaluate_agent.py` | Trained agents next to PD and LQR on the test seeds 0–99, with paired differences. |
| `plot_training.py` | Learning curves of one or more runs, with the PD and LQR levels. |
| `tests/test_env.py` | Tests of the environment. |
| `tests/test_renderer_3d.py` | Tests of the 3D renderer (image format, tilt direction, ball on the tray surface). |
| `tests/test_realism.py` | Tests of the optional realism parameters, including the regression check against `tests/golden_v1.npz`. |
| `tests/test_networks.py` | Tests of the networks and losses (shapes, action range, probability formulas, tanh correction, gradients, determinism, save and load). |
| `tests/test_agents.py` | Tests of the training code (GAE, parallel environments, curriculum, replay buffer, target update, entropy tuning) and short smoke runs of both training loops. |
| `tests/make_golden.py`, `tests/golden_v1.npz` | The script that recorded the trajectory snapshot, and the snapshot itself. They are kept as a record: `make_golden.py` does not need to be run again and the snapshot must not be regenerated. |
| `environment.yml` | Conda environment. |
| `media/` | Recorded demos (`demo_3d.gif`: random and LQR; `demo_ppo_3d.gif`: the reference PPO agent on seed 4) and the learning curves of the reference runs (`learning_curves.png`). |
| `results/` | Output of `evaluate_baselines.py`, and of `evaluate_agent.py --csv` for the reference runs (`agents_ideal.csv`, `agents_realistic.csv`). |
| `runs/` | Output of `train.py` (progress logs and checkpoints); not tracked by git. |
| `trained/` | Checkpoints of the four reference runs (see [Reference runs](#reference-runs)). |
| `report/` | Project reports: `ME5418_NeuralNetwork_Report.pdf` (stage 2) and `ME5418_Agent_Report.pdf` (stage 3, written from the `.html` file next to it). |

## How to Run the Demo

```bash
python demo.py
```

With no arguments this opens a window with a 3D view of the tray and plays two episodes in real time: first a random policy, then the LQR controller, whose gain and closed-loop poles are printed first. All policies of one run use the same seed, so they face the same start position, rolling resistance and disturbance schedule. The environment uses the full randomisation ranges (`c_rr` in 0.005–0.05, disturbance pulses of 0.5–2.0 m/s²). After each episode a summary is printed in the terminal: policy, number of steps, return, why the episode ended, `c_rr` and the disturbance schedule.

### Command-line arguments

| Argument | Values | Default | Meaning |
|---|---|---|---|
| `--policy` | `random`, `pd`, `lqr`, `both`, `all` | `both` | Which policy to show. `both` plays a random episode, then an LQR episode. `all` plays random, then PD, then LQR. |
| `--seed` | integer | `4` | Seed of the episode(s). The same seed always gives the same demo. |
| `--save-gif PATH` | file path | not set | Do not open a window; render off-screen and save the episode(s) to a gif. |
| `--realistic` | switch | off | Turn on the optional realism parameters (see below): `pos_noise_std=0.002`, `vel_noise_std=0.03`, `actuator_tau=0.05`, `action_delay_steps=1`. |

Examples:

```bash
python demo.py --policy pd                          # PD controller only
python demo.py --policy lqr --seed 7                # LQR controller, another seed
python demo.py --policy all                         # random, then PD, then LQR
python demo.py --realistic --policy all             # with noise, actuator lag and delay
python demo.py --save-gif media/demo_3d.gif         # write a gif instead of opening a window
```

When the LQR controller is used (`lqr` or `all`), its gain and closed-loop poles are printed first. With `--realistic`, one extra line lists the realism parameters. Closing the window stops the demo early. On screen the demo runs in real time. If the computer cannot draw 50 frames per second, some frames are skipped so that the playback speed stays correct; the terminal reports how many frames were drawn.

### What the picture shows

| Element | Meaning |
|---|---|
| Beige slab | The tray, tilted by the real pitch and roll angles (not exaggerated). The lower side is where the ball is accelerated to: positive pitch lowers the +x side, positive roll lowers the +y side. |
| Dark pillar and base | The support. The tray rotates about the top of the pillar (its centre). |
| Grey dotted square | Where the tray would be at zero tilt, as a reference. |
| `x`, `y` on the ground | Directions of the +x and +y axes. |
| Green `+` and dashed circle | Goal and goal radius (0.02 m), drawn on the tray surface. |
| Orange ball and orange line | The ball, resting on the tray surface, and its path over the last 1.5 s. |
| Red arrow at the ball ("disturbance push") | Inertial force caused by the base acceleration (opposite to `a_base`, length proportional to its magnitude). Only visible during a disturbance pulse. |
| Caption (top left) | The policy that is running: random, PD controller or LQR controller. With `--realistic` it also says so. |
| Text panel | Step and time, pitch and roll in degrees, `c_rr` of the episode, reward of the last step, return so far, and whether a disturbance is active (with its magnitude). `BALL OFF TRAY` appears when the ball has left the tray. |

## Tests

```bash
python tests/test_env.py            # the environment
python tests/test_renderer_3d.py    # the 3D renderer
python tests/test_realism.py        # the optional realism parameters and the regression check
python tests/test_networks.py       # the neural networks and their losses
python tests/test_agents.py         # the training code, including short training runs (~20 s)
```

Each script prints one line per test and ends with `All N tests passed.` (14, 7, 8, 19 and 17 tests). No test framework is needed. `tests/test_realism.py` compares the environment with the recorded snapshot `tests/golden_v1.npz`; `tests/make_golden.py` only documents how that snapshot was recorded and is not run again.

## Evaluating the baselines

```bash
python evaluate_baselines.py --episodes 10      # quick comparison of PD and LQR (about 30 s), written to results_quick/
python evaluate_baselines.py                    # full evaluation, 100 seeds per condition
python evaluate_baselines.py --realism-sweep    # PD and LQR under the realism parameters
```

Nothing is rendered. The script prints a table that compares the PD and LQR controllers over a grid of rolling resistance and disturbance strength. The full evaluation (100 seeds per condition) writes its csv file and heat maps to `results/`. A run with any other `--episodes` value writes to `results_quick/` instead, so a quick check never replaces the full results; `results_quick/` is ignored by git. `--out-dir some_folder` chooses another folder in either case.

## Neural network

The proposal trains PPO as the baseline and SAC as the more sample-efficient alternative. The networks of both are in `networks/`, written directly with PyTorch 1.13.1 (CPU), without a reinforcement-learning framework. Every network is a multilayer perceptron with the hidden sizes of the proposal (128 and 64, ReLU):

| Network | File | Layers | Output | Parameters |
|---|---|---|---|---|
| PPO actor | `actor_critic.py` | 18 → 128 → 64 → 2 | Mean action, squashed to [−1, 1] by tanh. The standard deviation is a learned parameter that does not depend on the state (initial log std −0.5). | 10,820 |
| PPO critic | `actor_critic.py` | 18 → 128 → 64 → 1 | State value V(s). | 10,753 |
| SAC actor | `sac.py` | 18 → 128 → 64 → 2 + 2 | Mean and log std of a Gaussian (log std clamped to [−20, 2]); the sample is squashed by tanh into (−1, 1). | 10,948 |
| SAC critic | `sac.py` | (18 + 2) → 128 → 64 → 1, twice | Two independent Q-values Q1(s, a), Q2(s, a). | 22,018 |

The observation is a short vector of 18 numbers, not an image, so a convolutional network has nothing to exploit, and the three stacked frames already contain velocity and recent history, so a recurrent network is not needed. Weights are initialised orthogonally (gain √2 on hidden layers, 0.01 on the policy outputs, 1 on value and Q outputs), so the first actions are close to zero (a level tray).

**Action range.** The two algorithms bound the action differently. For PPO a sampled action can leave [−1, 1]; its log-probability is computed for the raw sample, and only the action passed to `env.step` is clipped (`ActorCritic.act` returns both). SAC needs the density of the action it applies, so its actor squashes the sample with tanh and corrects the log-probability by the log-determinant of the squashing, computed in the numerically stable form `2 (log 2 − u − softplus(−2u))`.

**Asymmetric variant.** `ActorCritic(privileged_dim=3)` gives the PPO critic three extra inputs that the actor never sees: the rolling resistance coefficient and the current base acceleration, read from the `info` dict of the environment (`privileged_features(info)`). The critic is only needed during training, so the actor still uses only the 18-number observation (critic: 11,137 parameters).

**Losses.** `networks/losses.py` holds the clipped PPO objective (policy + 0.5 · value − 0.01 · entropy, ε = 0.2), the SAC critic loss against the entropy-regularised twin-Q target, the SAC actor loss E[α log π − Q1] (α = 0.2) and the discounted returns and normalised advantages of a rollout. The demo, the tests and the training loop of the next stage use the same functions.

Run the demo and the tests:

```bash
python demo_network.py                          # the five sections below
python demo_network.py --variant asymmetric     # the same with the asymmetric PPO critic
python demo_network.py --seed 7                 # another seed
python demo_network.py --bc                     # behaviour cloning of the LQR (about 1 minute)
python demo_network.py --critics                # baseline vs asymmetric critic (about 1 minute)
python demo_network.py --bc --realistic         # the same, plus the realistic condition (also --critics)
python tests/test_networks.py
```

`python demo_network.py` prints five sections, all seeded so that two runs print identical numbers:

1. **Construction**: the layers and parameter counts of the PPO actor-critic, the SAC actor and the SAC critics.
2. **PPO forward pass** on a real observation from `env.reset`: mean, standard deviation, sampled and clipped action, log-probability, entropy, value.
3. **PPO backward pass**: a 64-step rollout with the untrained policy, the clipped PPO loss, `.backward()`, the gradient norm of every parameter tensor, one Adam step, and the loss before and after.
4. **SAC forward and backward pass** on the same transitions: twin-Q critic loss and reparameterised actor loss, their gradients, one Adam step each, and a check of the tanh correction against the direct formula.
5. **Closed loop**: the untrained PPO policy drives one full episode.

The two experiments train on LQR rollouts from seeds 1000–1199, validate on seeds 2000–2049, and (behaviour cloning only) evaluate in closed loop on seeds 0–99, the same seeds as `evaluate_baselines.py`:

- **Behaviour cloning**: the PPO actor learns to imitate the LQR controller from data. In closed loop the cloned actor survives all 100 episodes with a return of 724.3 ± 8.5, against 724.4 ± 8.3 for the LQR itself. This shows that the network can represent and learn a good policy; it is not the learning method of the project.
- **Critic comparison**: both PPO critics are fitted to the same discounted returns (γ = 0.98, only the first 250 steps of each episode, so that the 500-step limit does not bias the targets). The asymmetric critic has a 24 % lower validation error on average over three network seeds. This measures how much information the hidden quantities carry about the return of this data; it does not show anything about PPO training yet.

With `--realistic`, both experiments are also run with the settings of `demo.py --realistic` (observation noise, actuator lag, one-step delay) and reported next to the ideal condition. Behaviour cloning keeps the actor trained on the clean data and only evaluates it, next to the LQR, in the realistic environment: there the cloned actor reaches 616.0 ± 132.0 (94 % survival) against 627.4 ± 111.1 (97 %) for the LQR. The critic comparison collects new data in the realistic environment, where the LQR that collects it sees the noisy observation; the asymmetric critic's validation error is 20 % lower overall (38 % lower during disturbance pulses, 13 % lower without). As in the ideal case, this measures information content only.

The numbers above were measured on Windows 11 with the pinned environment. Every run on one machine prints identical numbers, but floating-point details differ between platforms. The ideal-condition results were the same on another machine; the realistic-condition results differed slightly there (cloned actor 615.6 ± 130.7 with 95 % survival; critic errors 21 %, 41 % and 13 % lower).

The training loops of PPO and SAC are described in [Training the agents](#training-the-agents).

## Training the agents

Stage 3 trains the networks of stage 2 with the two algorithms of the proposal. Everything is written directly with PyTorch, without a reinforcement-learning framework, and runs on one CPU core.

```bash
python train.py --algo ppo                       # PPO, 1M steps, staged curriculum
python train.py --algo sac                       # SAC, 200k steps, staged curriculum
python train.py --algo ppo --asymmetric          # critic also receives c_rr and a_base
python train.py --algo ppo --realistic           # noise, lag and delay in the last stage
python train.py --algo ppo --curriculum full     # no curriculum: full range from step 0
python train.py --algo ppo --quick               # ~15 s smoke run of every code path
```

| Argument | Default | Meaning |
|---|---|---|
| `--algo` | required | `ppo` or `sac`. |
| `--steps` | PPO 1,000,000; SAC 200,000 | Environment steps. |
| `--seed` | `0` | Seed of the networks, the sampling and the episode seeds. A run is reproducible on one machine. |
| `--curriculum` | `staged` | `staged`, `full` or `fixed` (below). |
| `--realistic` | off | Adds observation noise, actuator lag and a one-step delay (`demo.py --realistic`) to the last stage; evaluation then also uses them. |
| `--asymmetric` | off | PPO only: the critic also gets `c_rr` and `a_base`. The actor, which is what is deployed, still sees only the observation. |
| `--eval-interval` | PPO 50,000; SAC 10,000 | Steps between evaluations. |
| `--quick` | off | Short run that goes through every stage, update, evaluation and checkpoint; it is not expected to learn a good policy. |
| `--name`, `--out-root` | from the options, `runs` | Output folder `runs/<name>/`. |

Each run writes `config.json` (all hyper-parameters), `progress.csv` (one row per PPO iteration or per 2,000 SAC steps: training returns, losses, and at every evaluation the return, survival and goal fraction of the deterministic policy on seeds 500–519), `best.pt` (the checkpoint with the best evaluation return) and `last.pt`.

**Curriculum.** The proposal trains "first with a fixed friction coefficient and no disturbances, then progressively randomizing" both. `--curriculum staged` does this in four stages, as fractions of the run: fixed `c_rr` = 0.02 without pulses (10 %), random `c_rr` in 0.005–0.05 (15 %), plus pulses of 0.5–1.25 m/s² (15 %), then the full range with pulses of 0.5–2.0 m/s² (60 %). A new stage applies from the next episode of each environment. Evaluation always uses the conditions of the last stage, i.e. the task itself.

**PPO** (`agents/ppo.py`): 8 parallel environments × 256 steps per iteration; GAE with γ = 0.99, λ = 0.95; 10 epochs over each batch of 2,048 in minibatches of 256, advantages normalised per minibatch; clipped loss with ε = 0.2, value weight 0.5, no entropy bonus (the log std is a free parameter and decays on its own); gradient norm clipped to 0.5; Adam with learning rate 3·10⁻⁴ annealed linearly to 0.

**SAC** (`agents/sac.py`): one environment; the first 5,000 actions are uniformly random; then one update per step on a minibatch of 256 from a replay buffer of 300,000 transitions; target critic updated by Polyak averaging with τ = 0.005; the actor maximises min(Q1, Q2) − α log π; α is learned so that the policy entropy tracks −2 (= −action dimension); Adam with learning rate 3·10⁻⁴ for actor, critics and α.

Both algorithms scale the reward by 0.1 for learning (logged returns are unscaled), and both treat the two ways an episode ends differently: a fall (`terminated`) is terminal, but the 500-step limit (`truncated`) is not, so the value of the last observation is still bootstrapped. `VecEnv` in `agents/common.py` returns that last observation separately from the observation after the automatic reset. Training episodes use seeds from 100,000 upwards, disjoint from every evaluation seed.

### Evaluating and watching a trained agent

```bash
python evaluate_agent.py runs/ppo_staged_seed0/best.pt runs/sac_staged_seed0/best.pt
python evaluate_agent.py runs/ppo_staged_seed0/best.pt --realistic
python plot_training.py runs/ppo_staged_seed0 runs/sac_staged_seed0 --out media/learning_curves.png
python demo.py --checkpoint runs/ppo_staged_seed0/best.pt --policy lqr+agent
```

`evaluate_agent.py` runs PD, LQR and every given checkpoint on the test seeds 0–99 (the seeds of `evaluate_baselines.py`, never seen in training or used to pick `best.pt`) and prints the return, survival, the fraction of steps inside the goal radius, and the paired difference to LQR. `--csv PATH` also writes every episode. `demo.py --checkpoint` plays the agent in the 3D view; `--policy lqr+agent` plays LQR first on the same seed.

### Reference runs

`trained/` holds the `best.pt` of four reference runs (one training seed each, about 0.1 MB per file), so the evaluation and the demo work without training first:

```bash
python evaluate_agent.py trained/ppo_staged_seed0.pt trained/ppo_asym_staged_seed0.pt trained/sac_staged_seed0.pt
python evaluate_agent.py trained/ppo_staged_seed0.pt trained/ppo_staged_realistic_seed0.pt --realistic
python demo.py --checkpoint trained/ppo_staged_seed0.pt --policy lqr+agent
```

They were trained with `python train.py --algo ppo`, `--algo ppo --asymmetric`, `--algo ppo --realistic` and `--algo sac` (seed 0), on one CPU core each: PPO 1M steps in about 6–7 minutes, SAC 200k steps in 22 minutes. Results on the test seeds 0–99 (`evaluate_agent.py`; maximum return 750):

| Condition | Policy | Return | Survival | Goal % | vs LQR, paired |
|---|---|---|---|---|---|
| Ideal | PD | 712.6 ± 11.7 | 100 % | 93.4 | |
| | LQR | 724.4 ± 8.3 | 100 % | 96.1 | |
| | PPO | 720.0 ± 71.8 | 99 % | 95.1 | −4.4, better on 77/100 seeds |
| | PPO, asymmetric critic | 720.6 ± 71.8 | 99 % | 95.3 | −3.8, better on 83/100 |
| | SAC | 692.0 ± 21.0 | 100 % | 89.8 | −32.4, better on 1/100 |
| Realistic (noise, lag 0.05 s, 1-step delay) | PD | 658.5 ± 68.3 | 99 % | 86.7 | |
| | LQR | 627.4 ± 111.1 | 97 % | 78.8 | |
| | PPO trained on the ideal task | 386.7 ± 118.3 | 78 % | 13.8 | −240.7, better on 2/100 |
| | PPO trained with `--realistic` | 657.3 ± 69.8 | 99 % | 81.1 | +29.9, better on 77/100 |

- PPO beats LQR on most seeds but drops the ball on seed 0 (both PPO variants), which starts near a corner with a 1.41 m/s² push from t = 0.03 s; LQR saves that episode. Without seed 0, PPO averages 727.2 against 724.8 for LQR.
- Under noise, lag and delay, the PPO policy trained on the ideal task fails often, while the one trained with the realism parameters in its last stage beats LQR by 29.9 and matches PD.
- SAC reaches 681 after 20k steps (PPO: 674 after 51k) but then stays at 680–690, below PD; the cause is still open.
- `media/learning_curves.png` shows the learning curves (`plot_training.py`). These are single runs; several seeds per configuration are needed before drawing conclusions, in particular about the asymmetric critic.


## Optional realism parameters

`BallOnTrayEnv` has four optional constructor arguments that make the task more realistic. All of them are off by default, and with the defaults the environment behaves exactly as before (same trajectories, observations and rewards for the same seed and actions). The observation stays 18-dimensional.

```python
env = BallOnTrayEnv(pos_noise_std=0.002, vel_noise_std=0.03,
                    actuator_tau=0.05, action_delay_steps=1)
```

| Parameter | Meaning | Default | Assumption of the proposal it relaxes |
|---|---|---|---|
| `pos_noise_std` | Standard deviation [m] of Gaussian noise on the measured ball position: measurement error of the camera or pressure sensor. | `0.0` (no noise) | Ideal sensor |
| `vel_noise_std` | Standard deviation [m/s] of Gaussian noise on the measured ball velocity. | `0.0` (no noise) | Ideal sensor |
| `actuator_tau` | Time constant [s] of a first-order lag between the commanded and the actual tray tilt: motors and arm cannot reach a commanded angle instantly. | `0.0` (instant) | Ideal actuator |
| `action_delay_steps` | Number of control steps (20 ms each) before an action takes effect: latency of sensing, computation and communication. | `0` (no delay) | An action takes effect in the same control step |

Details:

- The noise only changes the observation. The physical state, the reward, the out-of-bounds check and the state in `info` use the true values. Each new frame gets noise once; frames already in the stack are not disturbed again.
- The noise has its own random generator, derived from the `reset` seed. Changing the noise settings does not change the start position, `c_rr` or the disturbance plan of a seed.
- With `actuator_tau > 0`, actions accumulate into the commanded tilt `env.tilt_cmd`, and the actual tilt `env.tilt` follows it. The physics, the observation and the renderer use the actual tilt; `info` contains both (`tilt`, `tilt_cmd`).
- With a delay, the action penalties of the reward still use the action the agent gave, not the delayed one.
- Order inside one step: delay queue, commanded tilt, actuator lag, actual tilt, ball dynamics.

`python tests/test_realism.py` tests these parameters. `python evaluate_baselines.py --realism-sweep` compares the PD and LQR baselines under them and writes `results/eval_realism.csv`.
