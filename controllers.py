"""Classical baseline controllers for the Ball-on-Tray environment.

Two deterministic policies that only use the observation (they never read the
internal state of the environment), shared by ``demo.py`` and
``evaluate_baselines.py``:

    pd_policy                               PD controller on the tilt command
    design_lqr / make_lqr_policy            infinite-horizon discrete-time LQR

LQR model
---------
The two axes of the environment are exactly decoupled: the ball's x motion is
driven by the pitch only and its y motion by the roll only, so one single-axis
design is computed and the same gain is applied to both axes.

Because the action is a tilt *increment*, the tilt is itself a state. With
k = (5/7)*g [m/s^2 per rad], c = TILT_STEP [rad per unit action] and the
small-angle approximation sin(theta) ~ theta, the per-axis state is

    z = [x, v, theta]^T     [m, m/s, rad],      u = action in [-1, 1]

The environment updates the tilt *before* integrating the ball, and integrates
one control period as N semi-implicit Euler sub-steps of h = DT / N with the
acceleration held constant:

    v_i = v_0 + i*a*h
    x_N = x_0 + h*sum_i v_i = x_0 + v_0*DT + a*DT^2*(N+1)/(2N)

so the position coefficient is p = (N+1)/(2N) = 0.625 for N = 4, not the 0.5 of
an exact double integrator. This gives

        | 1   DT   p*k*DT^2 |         | p*k*c*DT^2 |
    A = | 0   1    k*DT     |     B = | k*c*DT     |
        | 0   0    1        |         | c          |

which reproduces one environment step to ~1e-7 m; the only residual error is
the small-angle approximation itself.

The base acceleration ``a_base`` is an *unmodelled* disturbance (it is not part
of the observation), so the controller has no feed-forward term for it and
shows a transient excursion during each disturbance pulse. The rolling
resistance ``c_rr`` is likewise unmodelled, which is deliberate: like
``pd_policy``, this controller reads nothing but the newest observation frame,
so the comparison with a future RL agent stays fair.

Note that [x, v, theta] is fully contained in a single observation frame, so no
state estimator is needed. The three stacked frames of the observation only
help infer the hidden ``c_rr`` and ``a_base``, which neither LQR nor PD uses.

LQR design
----------
Q and R are specified as dimensionless weights divided by the square of a
characteristic scale of each state, so the weights are directly comparable:
position by the tray half size, velocity by ``VEL_SCALE``, tilt by the tilt
limit, and the action is already normalised to [-1, 1].

The discrete algebraic Riccati equation is solved by iterating

    P <- A'PA - A'PB (R + B'PB)^-1 B'PA + Q

to convergence. This avoids a scipy dependency, which would otherwise have to
be added to ``environment.yml``; the system is 3x1, so the iteration costs
microseconds.
"""

import numpy as np

from ball_on_tray_gym import BallOnTrayEnv

# Scales that undo the normalisation of the observation.
POS_SCALE = BallOnTrayEnv.TRAY_HALF_SIZE                    # [m]
VEL_SCALE = BallOnTrayEnv.VEL_SCALE                         # [m/s]
MAX_TILT = np.deg2rad(BallOnTrayEnv.MAX_TILT_DEG)           # [rad]
TILT_STEP = np.deg2rad(BallOnTrayEnv.TILT_STEP_DEG)         # [rad/step]
FRAME_DIM = BallOnTrayEnv.FRAME_DIM

# ------------------------------------------------------------------- PD gains
# Tilt command per position / velocity error.
KP = 6.0                                # [rad/m]
KD = 1.5                                # [rad/(m/s)]

# ------------------------------------------------------------------ LQR weights
# Dimensionless; each is divided by the square of the matching scale below.
# Chosen by a grid search over the environment's own return (the quadratic cost
# is only a proxy for it: the reward's distance term is linear and its goal
# bonus is a step), then validated on held-out seeds not used for the search.
# The result is insensitive to W_VEL; W_TILT and W_ACTION do the work.
W_POS = 1.0         # weight on the position error [-]
W_VEL = 0.05        # weight on the velocity [-]
W_TILT = 0.02       # weight on the tilt (keeps the tray near level) [-]
W_ACTION = 0.002    # weight on the action (control effort) [-]

# Characteristic scales used to normalise the states.
SCALE_POS = BallOnTrayEnv.TRAY_HALF_SIZE                    # [m]
SCALE_VEL = BallOnTrayEnv.VEL_SCALE                         # [m/s]
SCALE_TILT = np.deg2rad(BallOnTrayEnv.MAX_TILT_DEG)         # [rad]

# Riccati iteration.
DARE_TOL = 1e-12        # convergence threshold on ||P_new - P||_inf
DARE_MAX_ITER = 10000   # iteration cap


# ------------------------------------------------------------------------- PD
def pd_policy(obs, kp=KP, kd=KD):
    """Deterministic PD controller that only uses the observation.

    The newest frame (last 6 entries of the 18-D observation) is
    de-normalised to position [m], velocity [m/s] and tilt [rad]. Then

        tilt_target = clip(-kp * pos - kd * vel, +/- 15 deg)        [rad]
        action      = clip((tilt_target - tilt) / 3 deg, -1, 1)     [-]

    The minus signs follow the sign convention of the environment: positive
    pitch accelerates the ball towards +x, so a ball at x > 0 needs pitch < 0
    (same for roll and y).
    """
    frame = np.asarray(obs, dtype=np.float64)[-FRAME_DIM:]
    pos = frame[0:2] * POS_SCALE
    vel = frame[2:4] * VEL_SCALE
    tilt = frame[4:6] * MAX_TILT

    tilt_target = np.clip(-kp * pos - kd * vel, -MAX_TILT, MAX_TILT)
    return np.clip((tilt_target - tilt) / TILT_STEP, -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------ LQR
def build_model():
    """Build the discrete single-axis model (A, B) described in the docstring.

    Returns:
        (A, B) with shapes (3, 3) and (3, 1); state [x, v, theta], input u.
    """
    dt = BallOnTrayEnv.DT                                       # [s]
    n_sub = BallOnTrayEnv.N_SUBSTEPS                            # [-]
    k = BallOnTrayEnv.ROLLING_FACTOR * BallOnTrayEnv.GRAVITY    # [m/s^2 per rad]
    c = np.deg2rad(BallOnTrayEnv.TILT_STEP_DEG)                 # [rad per unit action]
    p = (n_sub + 1.0) / (2.0 * n_sub)                           # position coefficient [-]

    A = np.array([[1.0, dt, p * k * dt ** 2],
                  [0.0, 1.0, k * dt],
                  [0.0, 0.0, 1.0]])
    B = np.array([[p * k * c * dt ** 2],
                  [k * c * dt],
                  [c]])
    return A, B


def build_weights():
    """Build the LQR cost matrices.

    Returns:
        (Q, R) with shapes (3, 3) and (1, 1).
    """
    Q = np.diag([W_POS / SCALE_POS ** 2,
                 W_VEL / SCALE_VEL ** 2,
                 W_TILT / SCALE_TILT ** 2])
    R = np.array([[W_ACTION]])
    return Q, R


def solve_dare(A, B, Q, R, tol=DARE_TOL, max_iter=DARE_MAX_ITER):
    """Solve the discrete algebraic Riccati equation by value iteration.

    Iterates ``P <- A'PA - A'PB (R + B'PB)^-1 B'PA + Q`` from ``P = Q`` until
    ``P`` stops changing, then returns the optimal gain.

    Args:
        A, B: system matrices, shapes (n, n) and (n, m).
        Q, R: cost matrices, shapes (n, n) and (m, m).
        tol: convergence threshold on the largest entry change of P.
        max_iter: iteration cap.

    Returns:
        (K, P, n_iter) with K of shape (m, n). The optimal input is u = -K z.

    Raises:
        RuntimeError: if the iteration did not converge within ``max_iter``.
    """
    P = Q.copy()
    for i in range(1, max_iter + 1):
        BtP = B.T.dot(P)
        # Gain of this iteration: K = (R + B'PB)^-1 B'PA
        K = np.linalg.solve(R + BtP.dot(B), BtP.dot(A))
        P_next = A.T.dot(P).dot(A) - A.T.dot(P).dot(B).dot(K) + Q
        if np.max(np.abs(P_next - P)) < tol:
            return K, P_next, i
        P = P_next
    raise RuntimeError("Riccati iteration did not converge in {} steps".format(max_iter))


def design_lqr():
    """Run the full design and report it.

    Returns:
        (K, info) where K has shape (1, 3) and info holds the model, the cost
        matrices, the iteration count and the closed-loop poles.
    """
    A, B = build_model()
    Q, R = build_weights()
    K, P, n_iter = solve_dare(A, B, Q, R)
    poles = np.linalg.eigvals(A - B.dot(K))
    return K, {"A": A, "B": B, "Q": Q, "R": R, "P": P,
               "n_iter": n_iter, "poles": poles}


def print_design(K, info):
    """Print the gain, the closed-loop poles and the stability check."""
    radii = np.abs(info["poles"])
    print("LQR design (one axis, applied to both)")
    print("  weights      : pos {:.3g}, vel {:.3g}, tilt {:.3g}, action {:.3g}".format(
        W_POS, W_VEL, W_TILT, W_ACTION))
    print("  Riccati      : converged in {} iterations".format(info["n_iter"]))
    print("  gain K       : x {:8.3f} [1/m]   v {:8.3f} [s/m]   theta {:8.3f} [1/rad]".format(
        K[0, 0], K[0, 1], K[0, 2]))
    print("  closed loop  : |z| = {}".format(
        ", ".join("{:.4f}".format(r) for r in sorted(radii, reverse=True))))
    print("  stable       : {} (all poles inside the unit circle)".format(
        "yes" if np.max(radii) < 1.0 else "NO"))
    # Tilt reachable in one step from rest at the tray edge, as a sanity figure.
    print("  settling     : ~{:.2f} s to decay by 1/e (slowest pole)".format(
        -BallOnTrayEnv.DT / np.log(np.max(radii))))
    print("")


def make_lqr_policy(K):
    """Build a policy function ``policy(obs) -> action`` from the gain K.

    The newest observation frame (last 6 entries) is de-normalised to position
    [m], velocity [m/s] and tilt [rad], the per-axis state z = [x, v, theta] is
    assembled for each axis, and the same gain is applied to both:

        u = clip(-K z, -1, 1)

    Args:
        K: gain of shape (1, 3) from ``design_lqr``.

    Returns:
        A function mapping an observation to a (2,) float32 action.
    """
    gain = K.ravel()        # [k_x, k_v, k_theta]

    def policy(obs):
        frame = np.asarray(obs, dtype=np.float64)[-FRAME_DIM:]
        pos = frame[0:2] * POS_SCALE            # [m]
        vel = frame[2:4] * VEL_SCALE            # [m/s]
        tilt = frame[4:6] * MAX_TILT            # [rad]
        # One axis per column: z[:, 0] is the x/pitch axis, z[:, 1] the y/roll axis.
        z = np.vstack([pos, vel, tilt])         # (3, 2)
        return np.clip(-gain.dot(z), -1.0, 1.0).astype(np.float32)

    return policy
