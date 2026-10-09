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
python demo.py                          # 3. run the demo: random policy, then PD controller
python tests/test_env.py                # 4. run the environment tests
```

Activate the environment in every new terminal before running anything, otherwise `gym` will not be found or a wrong version will be used; in VS Code, also select `me5418-ballontray` as the Python interpreter.

The default demo does not include the LQR controller and uses the 2D view. To see all three policies, or the 3D view:

```bash
python demo.py --policy all             # random, then PD, then LQR
python demo.py --view 3d                # the same episodes in the 3D view
python demo.py --policy all --view 3d   # all three policies in 3D
```

All commands are run from the project root.

## Project structure

```
ME5418_BallOnTray/
├── README.md
├── environment.yml
├── demo.py
├── ball_on_tray_gym.py
├── renderer_3d.py
├── controllers.py
├── evaluate_baselines.py
├── tests/
│   ├── test_env.py
│   ├── test_renderer_3d.py
│   ├── test_realism.py
│   ├── make_golden.py
│   └── golden_v1.npz
├── media/
├── results/
└── report/
```

| File | Content |
|---|---|
| `demo.py` | The demo: random, PD and LQR policies in the 2D or 3D view. |
| `ball_on_tray_gym.py` | The environment class `BallOnTrayEnv` (dynamics, reward, observation, built-in 2D rendering). |
| `renderer_3d.py` | `TrayRenderer3D`, a 3D view of the environment drawn from outside with matplotlib's mplot3d. It only reads the environment and does not change it. |
| `controllers.py` | The PD and LQR baseline controllers, used by `demo.py` and `evaluate_baselines.py`. |
| `evaluate_baselines.py` | Batch evaluation of the PD and LQR controllers without rendering. |
| `tests/test_env.py` | Tests of the environment. |
| `tests/test_renderer_3d.py` | Tests of the 3D renderer (image format, tilt direction, ball on the tray surface). |
| `tests/test_realism.py` | Tests of the optional realism parameters, including the regression check against `tests/golden_v1.npz`. |
| `tests/make_golden.py`, `tests/golden_v1.npz` | The script that recorded the trajectory snapshot, and the snapshot itself. They are kept as a record: `make_golden.py` does not need to be run again and the snapshot must not be regenerated. |
| `environment.yml` | Conda environment. |
| `media/` | Recorded demos (`demo.gif`, `demo_3d.gif`). |
| `results/` | Output of `evaluate_baselines.py`. |
| `report/` | Project report. |

## How to Run the Demo

```bash
python demo.py
```

With no arguments this opens a window and plays two episodes in real time: first a random policy, then a PD controller. All policies of one run use the same seed, so they face the same start position, rolling resistance and disturbance schedule. The environment uses the full randomisation ranges (`c_rr` in 0.005–0.05, disturbance pulses of 0.5–2.0 m/s²). After each episode a summary is printed in the terminal: policy, number of steps, return, why the episode ended, `c_rr` and the disturbance schedule.

### Command-line arguments

| Argument | Values | Default | Meaning |
|---|---|---|---|
| `--view` | `2d`, `3d` | `2d` | `2d` is the top view built into the environment, `3d` shows the tilting tray. |
| `--policy` | `random`, `pd`, `lqr`, `both`, `all` | `both` | Which policy to show. `both` plays a random episode, then a PD episode. `all` plays random, then PD, then LQR. |
| `--seed` | integer | `4` | Seed of the episode(s). The same seed always gives the same demo. |
| `--save-gif PATH` | file path | not set | Do not open a window; render off-screen and save the episode(s) to a gif. |
| `--realistic` | switch | off | Turn on the optional realism parameters (see below): `pos_noise_std=0.002`, `vel_noise_std=0.03`, `actuator_tau=0.05`, `action_delay_steps=1`. |

Examples:

```bash
python demo.py --policy pd                          # PD controller only
python demo.py --policy lqr --seed 7                # LQR controller, another seed
python demo.py --policy all --view 3d               # all three policies in the 3D view
python demo.py --realistic --policy all             # with noise, actuator lag and delay
python demo.py --save-gif media/demo.gif            # write a gif instead of opening a window
python demo.py --view 3d --save-gif media/demo_3d.gif
```

When the LQR controller is used (`lqr` or `all`), its gain and closed-loop poles are printed first. With `--realistic`, one extra line lists the realism parameters. Closing the window stops the demo early.

### What the 2D picture shows

The view is from above, in the tray frame (origin at the tray centre).

| Element | Meaning |
|---|---|
| Beige square with dark outline | The tray (0.3 m × 0.3 m). The episode ends when the ball centre crosses the outline. |
| Green `+` | The goal (tray centre). |
| Green dashed circle | Goal radius (0.02 m). Inside it the reward gets a bonus. |
| Orange disc | The ball. |
| Orange line | Path of the ball over the last 1.5 s. |
| Blue arrow at the centre | Downhill direction of the tray, i.e. where the tilt accelerates the ball. Its x and y components are proportional to pitch and roll; longer means more tilt. |
| Red arrow at the ball ("disturbance push") | Inertial force on the ball caused by the base acceleration (direction opposite to `a_base`, length proportional to its magnitude). Only visible while a disturbance pulse is active. |
| Text panel | Step and time, pitch and roll in degrees, `c_rr` of the episode, reward of the last step, return so far, and whether a disturbance is active (with its magnitude). `BALL OFF TRAY` appears when the ball has left the tray. |

### What the 3D picture shows

`--view 3d` plays the same episodes in a 3D view. On screen it runs in real time. If the computer cannot draw 50 frames per second, some frames are skipped so that the playback speed stays correct; the terminal reports how many frames were drawn.

| Element | Meaning |
|---|---|
| Beige slab | The tray, tilted by the real pitch and roll angles (not exaggerated). The lower side is where the ball is accelerated to: positive pitch lowers the +x side, positive roll lowers the +y side. |
| Dark pillar and base | The support. The tray rotates about the top of the pillar (its centre). |
| Grey dotted square | Where the tray would be at zero tilt, as a reference. |
| `x`, `y` on the ground | Directions of the +x and +y axes. |
| Green `+` and dashed circle | Goal and goal radius (0.02 m), drawn on the tray surface. |
| Orange ball and orange line | The ball, resting on the tray surface, and its path over the last 1.5 s. |
| Red arrow at the ball ("disturbance push") | Inertial force caused by the base acceleration (opposite to `a_base`, length proportional to its magnitude). Only visible during a disturbance pulse. |
| Text panel | Same information as in the 2D view. |

## Tests

```bash
python tests/test_env.py            # the environment
python tests/test_renderer_3d.py    # the 3D renderer
python tests/test_realism.py        # the optional realism parameters and the regression check
```

Each script prints one line per test and ends with `All N tests passed.` No test framework is needed. `tests/test_realism.py` compares the environment with the recorded snapshot `tests/golden_v1.npz`; `tests/make_golden.py` only documents how that snapshot was recorded and is not run again.

## Evaluating the baselines

```bash
python evaluate_baselines.py --episodes 10      # quick comparison of PD and LQR (about 30 s), written to results_quick/
python evaluate_baselines.py                    # full evaluation, 100 seeds per condition
python evaluate_baselines.py --realism-sweep    # PD and LQR under the realism parameters
```

Nothing is rendered. The script prints a table that compares the PD and LQR controllers over a grid of rolling resistance and disturbance strength. The full evaluation (100 seeds per condition) writes its csv file and heat maps to `results/`. A run with any other `--episodes` value writes to `results_quick/` instead, so a quick check never replaces the full results; `results_quick/` is ignored by git. `--out-dir some_folder` chooses another folder in either case.

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
- With `actuator_tau > 0`, actions accumulate into the commanded tilt `env.tilt_cmd`, and the actual tilt `env.tilt` follows it. The physics, the observation and both renderers use the actual tilt; `info` contains both (`tilt`, `tilt_cmd`).
- With a delay, the action penalties of the reward still use the action the agent gave, not the delayed one.
- Order inside one step: delay queue, commanded tilt, actuator lag, actual tilt, ball dynamics.

`python tests/test_realism.py` tests these parameters. `python evaluate_baselines.py --realism-sweep` compares the PD and LQR baselines under them and writes `results/eval_realism.csv`.
