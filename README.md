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
python demo_network.py                  # 3. network demo: forward and backward pass of PPO and SAC
python tests/test_networks.py           # 4. network tests
python demo.py                          # 5. environment demo: random policy, then LQR controller
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
├── ball_on_tray_gym.py
├── controllers.py
├── evaluate_baselines.py
├── networks/
│   ├── __init__.py
│   ├── actor_critic.py
│   ├── sac.py
│   ├── losses.py
│   └── experiments.py
├── tests/
│   ├── test_env.py
│   ├── test_renderer_3d.py
│   ├── test_realism.py
│   ├── test_networks.py
│   ├── make_golden.py
│   └── golden_v1.npz
├── media/
├── results/
└── report/
```

| File | Content |
|---|---|
| `demo.py` | The demo: random, PD and LQR policies in the 3D view. |
| `ball_on_tray_gym.py` | The environment class `BallOnTrayEnv` (dynamics, reward, observation) and the 3D renderer `TrayRenderer3D`, drawn with matplotlib's mplot3d. The renderer only reads the environment and does not change it; it is used by `BallOnTrayEnv.render()` and by `demo.py`. |
| `controllers.py` | The PD and LQR baseline controllers, used by `demo.py` and `evaluate_baselines.py`. |
| `evaluate_baselines.py` | Batch evaluation of the PD and LQR controllers without rendering. |
| `networks/actor_critic.py` | The PPO actor-critic networks (PyTorch): baseline and asymmetric variant. |
| `networks/sac.py` | The SAC networks: tanh-squashed Gaussian actor and twin Q critics. |
| `networks/losses.py` | The PPO clipped loss, the SAC critic and actor losses, and the returns and advantages of a rollout. |
| `networks/experiments.py` | Two small supervised experiments with the networks: behaviour cloning of the LQR, and the critic comparison. |
| `demo_network.py` | Demo of the networks: construction, PPO forward and backward pass, SAC forward and backward pass, closed loop; `--bc` and `--critics` run the experiments. |
| `tests/test_env.py` | Tests of the environment. |
| `tests/test_renderer_3d.py` | Tests of the 3D renderer (image format, tilt direction, ball on the tray surface). |
| `tests/test_realism.py` | Tests of the optional realism parameters, including the regression check against `tests/golden_v1.npz`. |
| `tests/test_networks.py` | Tests of the networks and losses (shapes, action range, probability formulas, tanh correction, gradients, determinism, save and load). |
| `tests/make_golden.py`, `tests/golden_v1.npz` | The script that recorded the trajectory snapshot, and the snapshot itself. They are kept as a record: `make_golden.py` does not need to be run again and the snapshot must not be regenerated. |
| `environment.yml` | Conda environment. |
| `media/` | Recorded demo (`demo_3d.gif`). |
| `results/` | Output of `evaluate_baselines.py`. |
| `report/` | Project report (`ME5418_NeuralNetwork_Report.pdf`, written from the `.html` file next to it). |

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
```

Each script prints one line per test and ends with `All N tests passed.` (14, 7, 8 and 19 tests). No test framework is needed. `tests/test_realism.py` compares the environment with the recorded snapshot `tests/golden_v1.npz`; `tests/make_golden.py` only documents how that snapshot was recorded and is not run again.

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

The training loops of PPO and SAC are the next part of the project.

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
