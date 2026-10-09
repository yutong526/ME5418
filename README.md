# ME5418 Ball on Tray

## Introduction

TBD

## Environment Setup

### Setup (only once)

```bash
conda env create -f environment.yml
```

### Before each use

```bash
conda activate me5418-ballontray
```

Activate the environment in every new terminal before running anything, otherwise `gym` will not be found or a wrong version will be used; in VS Code, also select `me5418-ballontray` as the Python interpreter.

## File Overview

| File | Content |
|---|---|
| `ball_on_tray_gym.py` | The environment class `BallOnTrayEnv` (dynamics, reward, observation, built-in 2D rendering). |
| `demo.py` | 2D demo: random policy and PD controller. |
| `test_env.py` | Tests of the environment. |
| `renderer_3d.py` | `TrayRenderer3D`, a 3D view of the environment drawn from outside with matplotlib's mplot3d. It only reads the environment and does not change it. |
| `demo_3d.py` | 3D demo: same episodes and arguments as `demo.py`, drawn with `TrayRenderer3D`. |
| `test_renderer_3d.py` | Tests of the 3D renderer (image format, tilt direction, ball on the tray surface). |
| `test_realism.py` | Tests of the optional realism parameters, including the regression check against `tests/golden_v1.npz`. |
| `tests/` | `make_golden.py` and the trajectory snapshot `golden_v1.npz` recorded before the realism parameters were added. Do not regenerate the snapshot. |
| `environment.yml` | Conda environment. |
| `media/` | Recorded demos (`demo.gif`, `demo_3d.gif`). |
| `report/` | Project report. |

## How to Run the Demo

```bash
python demo.py
```

With no arguments this opens a window and plays two episodes in real time: first a random policy, then a PD controller. Both use the same seed, so they face the same start position, rolling resistance and disturbance schedule. The environment uses the full randomisation ranges (`c_rr` in 0.005–0.05, disturbance pulses of 0.5–2.0 m/s²). After each episode a summary is printed in the terminal: policy, number of steps, return, why the episode ended, `c_rr` and the disturbance schedule.

### Command-line arguments

| Argument | Values | Default | Meaning |
|---|---|---|---|
| `--policy` | `random`, `pd`, `both` | `both` | Which policy to show. `both` plays a random episode, then a PD episode. |
| `--seed` | integer | `4` | Seed of the episode(s). The same seed always gives the same demo. |
| `--save-gif PATH` | file path | not set | Do not open a window; render off-screen and save the episode(s) to a gif. |

Examples:

```bash
python demo.py --policy pd                  # PD controller only
python demo.py --policy random --seed 7     # random policy, another seed
python demo.py --save-gif media/demo.gif    # write a gif instead of opening a window
```

Closing the window stops the demo early.

### What the picture shows

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

### 3D demo

```bash
python demo_3d.py
```

This plays the same two episodes as `demo.py` in a 3D view. The arguments are identical (`--policy`, `--seed`, `--save-gif PATH`, same defaults):

```bash
python demo_3d.py --policy pd                     # PD controller only
python demo_3d.py --policy random --seed 7        # random policy, another seed
python demo_3d.py --save-gif media/demo_3d.gif    # write a gif instead of opening a window
```

On screen the demo runs in real time. If the computer cannot draw 50 frames per second, some frames are skipped so that the playback speed stays correct; the terminal reports how many frames were drawn.

What the 3D picture shows:

| Element | Meaning |
|---|---|
| Beige slab | The tray, tilted by the real pitch and roll angles (not exaggerated). The lower side is where the ball is accelerated to: positive pitch lowers the +x side, positive roll lowers the +y side. |
| Dark pillar and base | The support. The tray rotates about the top of the pillar (its centre). |
| Grey dotted square | Where the tray would be at zero tilt, as a reference. |
| `x`, `y` on the ground | Directions of the +x and +y axes. |
| Green `+` and dashed circle | Goal and goal radius (0.02 m), drawn on the tray surface. |
| Orange ball and orange line | The ball, resting on the tray surface, and its path over the last 1.5 s. |
| Red arrow at the ball ("disturbance push") | Inertial force caused by the base acceleration (opposite to `a_base`, length proportional to its magnitude). Only visible during a disturbance pulse. |
| Text panel | Same information as in the 2D demo. |

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

`python test_realism.py` tests these parameters. `python evaluate_baselines.py --realism-sweep` compares the PD and LQR baselines under them and writes `results/eval_realism.csv`.
