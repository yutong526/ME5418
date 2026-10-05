"""3D visualisation of the Ball-on-Tray environment (matplotlib mplot3d only).

``TrayRenderer3D`` draws a ``BallOnTrayEnv`` from outside: it only reads the
public attributes of the environment and never changes its state. The existing
2D ``BallOnTrayEnv.render`` is not used and not affected.

Geometry (derived from the sign convention of the environment)
--------------------------------------------------------------
World frame W: origin at the pivot (centre of the tray surface), z up.
Tray frame  T: x_t, y_t in the tray surface, n = surface normal.
R is the rotation matrix whose columns are x_t, y_t, n expressed in W, so a
tray-frame point p_T is drawn at p_W = R @ p_T.

Gravity is g_W = (0, 0, -g). Its component along a tray axis is what
accelerates the ball along that axis:

    a_x  ~  g_W . x_t = -g * (x_t)_z ,      a_y  ~  g_W . y_t = -g * (y_t)_z

The environment uses a_x ~ g * sin(pitch) and a_y ~ g * sin(roll) (positive
pitch -> ball accelerates towards +x, positive roll -> towards +y). Both hold
exactly if and only if the third row of R is

    R[2, :] = (-sin(pitch), -sin(roll), c),   c = sqrt(1 - sin^2(pitch) - sin^2(roll))

i.e. the +x side of the tray is lower by sin(pitch) per metre and the +y side
by sin(roll) per metre. This fixes R up to a rotation about the vertical axis;
the tray has no yaw, so the tilt without twist (rotation about a horizontal
axis that takes e_z to n) is used. With s_x = sin(pitch), s_y = sin(roll):

        | 1 - s_x^2/(1+c)    -s_x*s_y/(1+c)     s_x |
    R = | -s_x*s_y/(1+c)     1 - s_y^2/(1+c)    s_y |
        | -s_x               -s_y               c   |

The normal n = (s_x, s_y, c) leans towards the downhill direction. For a single
axis this reduces to a rotation by +pitch about the world y axis, or by -roll
about the world x axis. Angles are drawn to scale (no exaggeration).
"""

import time
from collections import deque

import numpy as np


class TrayRenderer3D(object):
    """Draws a BallOnTrayEnv as a tilting tray with a ball in 3D.

    Read-only inputs taken from the environment: ``ball_pos``, ``ball_vel``,
    ``tilt``, ``a_base``, ``c_rr``, ``tray_half_size``, ``step_count`` (plus
    the constants ``DT``, ``GOAL_RADIUS``, ``max_steps``, ``BALL_RADIUS``).

    Args:
        env: the BallOnTrayEnv to draw (create it with ``render_mode=None``).
        mode: "human" opens a window that plays in real time;
            "rgb_array" draws off-screen and ``render`` returns an image.
        ball_radius: drawn ball radius [m]; defaults to ``env.BALL_RADIUS``.

    Public attributes:
        window_closed: True once the user has closed the "human" window.
        frames_drawn / frames_skipped: counters of ``render`` calls that were
            drawn / skipped to keep real-time pace ("human" mode).
    """

    MODES = ("human", "rgb_array")

    # ------------------------------------------------------------------ figure
    FIGSIZE = (9.0, 6.0)            # figure size [inch]
    DPI = 100                       # -> 900 x 600 pixel frames
    FONT_SIZE = 14                  # font size of the text panel [pt]
    VIEW_ELEV = 25.0                # fixed camera elevation [deg]
    VIEW_AZIM = -60.0               # fixed camera azimuth [deg]

    # ------------------------------------------------------------------- scene
    TRAY_THICKNESS = 0.006          # thickness of the drawn tray slab [m]
    PIVOT_HEIGHT = 0.12             # height of the pivot above the ground [m]
    VIEW_HALF_WIDTH = 0.19          # x and y axis limits are +/- this [m]
    VIEW_TOP = 0.10                 # upper z axis limit [m]
    VIEW_BOTTOM_MARGIN = 0.02       # visible space below the ground [m]
    LINE_LIFT = 0.0005              # lines are drawn this far above the surface [m]
    TRAIL_LENGTH = 75               # number of past positions in the trail (1.5 s)
    DISTURBANCE_ARROW_SCALE = 0.05  # arrow length per unit |a_base| [m / (m/s^2)]
    ARROW_HEAD_LENGTH = 0.015       # length of the arrow head [m]
    BALL_N_LAT = 8                  # ball mesh resolution (latitude bands)
    BALL_N_LON = 14                 # ball mesh resolution (longitude bands)

    # ------------------------------------------------------------------ timing
    MAX_SKIPPED_FRAMES = 9          # never skip more consecutive frames than this

    def __init__(self, env, mode="human", ball_radius=None):
        if mode not in self.MODES:
            raise ValueError("mode must be one of {}".format(self.MODES))
        self.env = env
        self.mode = mode
        self.ball_radius = float(
            getattr(env, "BALL_RADIUS", 0.01) if ball_radius is None else ball_radius)

        self.window_closed = False
        self.frames_drawn = 0
        self.frames_skipped = 0

        self._fig = None                # created lazily on the first render()
        self._artists = {}              # name -> artist updated every frame
        self._unit_sphere = None        # (F, 4, 3) facets of a unit sphere
        self._trail = deque(maxlen=self.TRAIL_LENGTH)   # tray-frame (x, y) points
        self._last_step = 0             # env.step_count at the last render() call
        self._wall_start = None         # wall-clock time of step 0 of the episode [s]
        self._skipped_in_a_row = 0
        self._last_values = (0.0, 0.0)  # (reward, episode_return) of the last call

    # --------------------------------------------------------------- geometry
    @staticmethod
    def tray_rotation(pitch, roll):
        """Rotation matrix R (tray frame -> world frame) for the given tilt [rad].

        See the module docstring for the derivation. The third row is
        (-sin(pitch), -sin(roll), c): the +x side of the tray is lower for
        pitch > 0 and the +y side is lower for roll > 0, which is exactly where
        the environment accelerates the ball.
        """
        s_x = np.sin(pitch)
        s_y = np.sin(roll)
        c = np.sqrt(1.0 - s_x * s_x - s_y * s_y)    # cosine of the total tilt angle
        k = 1.0 / (1.0 + c)
        return np.array([
            [1.0 - k * s_x * s_x, -k * s_x * s_y, s_x],
            [-k * s_x * s_y, 1.0 - k * s_y * s_y, s_y],
            [-s_x, -s_y, c],
        ])

    def rotation(self):
        """Rotation matrix of the tray for the current ``env.tilt``."""
        return self.tray_rotation(self.env.tilt[0], self.env.tilt[1])

    def tray_to_world(self, points):
        """Transform tray-frame points (..., 3) [m] to world coordinates [m]."""
        return np.asarray(points, dtype=np.float64).dot(self.rotation().T)

    def ball_center_world(self):
        """World position of the ball centre [m].

        The ball touches the tray surface at the tray-frame point (x, y, 0), so
        its centre is one radius away along the surface normal:
        p_T = (x, y, ball_radius).
        """
        x, y = self.env.ball_pos
        return self.tray_to_world([x, y, self.ball_radius])

    # ------------------------------------------------------------- public API
    def render(self, reward=0.0, episode_return=0.0, force=False):
        """Draw the current state of the environment.

        Call this once after ``env.reset`` and once after every ``env.step``.

        Args:
            reward: reward of the last step (only displayed, supplied by the caller).
            episode_return: return of the episode so far (only displayed).
            force: draw even if the frame would be skipped ("human" mode).

        Returns:
            "rgb_array" mode: the frame as an (H, W, 3) uint8 array.
            "human" mode: None. The window is paced to real time: the call
            sleeps when it is ahead of the simulation clock and skips drawing
            when it is more than one control period behind, so playback speed
            stays close to real time on slow machines.
        """
        if self.window_closed:
            return None
        if self._fig is None:
            self._init_figure()
        self._last_values = (float(reward), float(episode_return))
        self._track_episode()

        if self.mode == "rgb_array":
            self._update_artists()
            canvas = self._fig.canvas
            canvas.draw()
            self.frames_drawn += 1
            # buffer_rgba() is (H, W, 4) uint8; drop the alpha channel.
            return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

        period = self.env.DT                                        # [s]
        target = self._wall_start + self.env.step_count * period    # [s]
        behind = time.perf_counter() - target
        if behind > period and not force:
            if self._skipped_in_a_row < self.MAX_SKIPPED_FRAMES:
                self._skipped_in_a_row += 1
                self.frames_skipped += 1
                return None
            # Too slow even with skipping: accept the delay instead of freezing.
            self._wall_start += behind

        self._update_artists()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self.frames_drawn += 1
        self._skipped_in_a_row = 0

        if self.env.step_count == 0:
            # The first frame of an episode (window creation, first draw) is
            # slow; start the real-time clock only after it is on screen.
            self._wall_start = time.perf_counter()
            return None
        remaining = target - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        return None

    def hold(self, seconds):
        """Show the current state and keep the window responsive for a while."""
        if self.mode != "human" or self._fig is None:
            return
        self.render(self._last_values[0], self._last_values[1], force=True)
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and not self.window_closed:
            self._fig.canvas.flush_events()
            time.sleep(0.02)
        # The pause must not count as lag of the simulation clock.
        if self._wall_start is not None:
            self._wall_start += seconds

    def close(self):
        """Close the window (if any) and free the figure."""
        if self._fig is not None:
            if self.mode == "human":
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            self._fig = None
        self._artists = {}
        self._trail.clear()
        self._wall_start = None
        self.window_closed = False

    # ---------------------------------------------------------------- set-up
    def _init_figure(self):
        """Create the figure and every artist exactly once."""
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if self.mode == "human":
            import matplotlib.pyplot as plt
            self._fig = plt.figure(figsize=self.FIGSIZE, dpi=self.DPI)
            self._fig.canvas.mpl_connect("close_event", self._on_window_close)
        else:
            # Off-screen figure: no window and no GUI backend required.
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            self._fig = Figure(figsize=self.FIGSIZE, dpi=self.DPI)
            FigureCanvasAgg(self._fig)

        half = self.env.tray_half_size
        width = self.VIEW_HALF_WIDTH
        ground = -self.PIVOT_HEIGHT
        z_low = ground - self.VIEW_BOTTOM_MARGIN

        ax = self._fig.add_axes([-0.035, -0.08, 0.72, 1.16], projection="3d")
        # Fixed camera, fixed limits and equal scale on all three axes, so the
        # picture never rescales or jitters between frames.
        ax.set_proj_type("ortho")
        ax.view_init(elev=self.VIEW_ELEV, azim=self.VIEW_AZIM)
        ax.set_xlim(-width, width)
        ax.set_ylim(-width, width)
        ax.set_zlim(z_low, self.VIEW_TOP)
        ax.set_box_aspect((2.0 * width, 2.0 * width, self.VIEW_TOP - z_low))
        ax.set_axis_off()
        # Draw order is given by the explicit zorder below, not by matplotlib's
        # depth heuristic. This is valid because the camera elevation (25 deg)
        # is larger than the largest total tilt (about 21 deg), so the camera
        # always looks at the top face of the tray and everything lying on the
        # tray is in front of it.
        ax.computed_zorder = False

        # ---- static artists: ground, base, pillar, level reference, axes ----
        ground_plate = np.array([[[-width, -width, ground], [width, -width, ground],
                                  [width, width, ground], [-width, width, ground]]])
        ax.add_collection3d(Poly3DCollection(
            ground_plate, facecolors="#eeeeee", edgecolors="#cccccc", zorder=1))
        base = 0.04
        base_plate = np.array([[[-base, -base, ground], [base, -base, ground],
                                [base, base, ground], [-base, base, ground]]])
        ax.add_collection3d(Poly3DCollection(
            base_plate, facecolors="#888888", edgecolors="#555555", zorder=2))
        # Pillar from the ground up to the pivot, about which the tray rotates.
        ax.plot([0.0, 0.0], [0.0, 0.0], [ground, -self.TRAY_THICKNESS],
                color="#555555", linewidth=5.0, solid_capstyle="butt", zorder=3)
        # Outline of the level (zero tilt) tray as a reference for the tilt.
        ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half],
                [0.0] * 5, color="#999999", linewidth=0.8, linestyle=":", zorder=3)
        # World axes on the ground, to show where +x and +y are.
        corner = np.array([-width + 0.02, -width + 0.02, ground])
        for direction, label in (((0.07, 0.0, 0.0), "x"), ((0.0, 0.07, 0.0), "y")):
            tip = corner + np.array(direction)
            ax.plot([corner[0], tip[0]], [corner[1], tip[1]], [corner[2], tip[2]],
                    color="#333333", linewidth=1.5, zorder=3)
            ax.text(tip[0], tip[1], tip[2], " " + label, fontsize=self.FONT_SIZE - 1, zorder=3)

        # ---- dynamic artists (their vertices are updated every frame) ----
        tray = Poly3DCollection(self._tray_faces(np.eye(3)), zorder=4,
                                facecolors=["#f3ecdc"] + ["#b9a98a"] * 5,
                                edgecolors="#333333", linewidths=1.0)
        ax.add_collection3d(tray)
        goal_circle, = ax.plot([], [], [], color="#2ca02c", linestyle="--",
                               linewidth=1.5, zorder=5)
        goal_point, = ax.plot([], [], [], color="#2ca02c", linestyle="none",
                              marker="+", markersize=9, zorder=5)
        trail, = ax.plot([], [], [], color="#ff7f0e", linewidth=1.5, alpha=0.7, zorder=6)
        self._unit_sphere, ball_colors = self._make_unit_sphere()
        ball = Poly3DCollection(self._unit_sphere * self.ball_radius, zorder=7,
                                facecolors=ball_colors, edgecolors="none")
        ax.add_collection3d(ball)
        arrow, = ax.plot([], [], [], color="#d62728", linewidth=2.0, zorder=8)
        text = self._fig.text(0.68, 0.93, "", va="top", ha="left",
                              family="monospace", fontsize=self.FONT_SIZE)

        self._fig.legend(
            handles=[
                Line2D([], [], linestyle="none", marker="o", markersize=8,
                       markerfacecolor="#ff7f0e", markeredgecolor="none", label="ball"),
                Line2D([], [], color="#ff7f0e", linewidth=1.5, alpha=0.7, label="recent path"),
                Line2D([], [], color="#2ca02c", linestyle="--", label="goal radius"),
                Line2D([], [], color="#999999", linestyle=":", label="level reference"),
                Line2D([], [], color="#d62728", linewidth=2.0, marker=">",
                       label="disturbance push"),
            ],
            loc="lower left", bbox_to_anchor=(0.665, 0.06),
            fontsize=self.FONT_SIZE - 3, frameon=False)

        self._artists = {"tray": tray, "goal_circle": goal_circle,
                         "goal_point": goal_point, "trail": trail, "ball": ball,
                         "arrow": arrow, "text": text}
        if self.mode == "human":
            plt.show(block=False)

    def _tray_faces(self, rot):
        """World vertices (6, 4, 3) of the tray slab; face 0 is the top surface.

        The top surface is the tray-frame plane z = 0, the slab extends to
        z = -TRAY_THICKNESS.
        """
        h = self.env.tray_half_size
        t = self.TRAY_THICKNESS
        corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        top = np.column_stack([corners, np.zeros(4)])
        bottom = np.column_stack([corners, -t * np.ones(4)])
        faces = [top, bottom[::-1]]
        for i in range(4):
            j = (i + 1) % 4
            faces.append(np.array([top[i], top[j], bottom[j], bottom[i]]))
        return np.array(faces).dot(rot.T)

    def _make_unit_sphere(self):
        """Facets (F, 4, 3) of a unit sphere and one shaded colour per facet.

        The ball only translates, so the facet colours (simple diffuse shading
        from a fixed light) are computed once.
        """
        lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, self.BALL_N_LAT + 1)
        lon = np.linspace(0.0, 2.0 * np.pi, self.BALL_N_LON + 1)

        def point(la, lo):
            return [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]

        facets = []
        for i in range(self.BALL_N_LAT):
            for j in range(self.BALL_N_LON):
                facets.append([point(lat[i], lon[j]), point(lat[i], lon[j + 1]),
                               point(lat[i + 1], lon[j + 1]), point(lat[i + 1], lon[j])])
        facets = np.array(facets)

        normals = facets.mean(axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        light = np.array([-0.3, -0.5, 0.8])
        light /= np.linalg.norm(light)
        shade = 0.55 + 0.45 * np.clip(normals.dot(light), 0.0, 1.0)
        colors = np.outer(shade, np.array([1.0, 0.5, 0.05]))        # orange
        return facets, np.clip(colors, 0.0, 1.0)

    # ---------------------------------------------------------------- updates
    def _track_episode(self):
        """Restart trail and clock on a new episode; record the ball position."""
        step = self.env.step_count
        if self._wall_start is None or step == 0 or step < self._last_step:
            self._trail.clear()
            self._wall_start = time.perf_counter() - step * self.env.DT
            self._skipped_in_a_row = 0
        if not self._trail or step != self._last_step:
            self._trail.append(np.array(self.env.ball_pos, dtype=np.float64))
        self._last_step = step

    def _update_artists(self):
        """Move the existing artists to the current state (nothing is re-created)."""
        env = self.env
        rot = self.rotation()
        lift = self.LINE_LIFT

        self._artists["tray"].set_verts(self._tray_faces(rot))

        # Goal marker and goal-radius circle, drawn on the tilted surface.
        angle = np.linspace(0.0, 2.0 * np.pi, 41)
        circle = np.column_stack([env.GOAL_RADIUS * np.cos(angle),
                                  env.GOAL_RADIUS * np.sin(angle),
                                  lift * np.ones_like(angle)]).dot(rot.T)
        self._artists["goal_circle"].set_data_3d(circle[:, 0], circle[:, 1], circle[:, 2])
        centre = rot.dot([0.0, 0.0, lift])
        self._artists["goal_point"].set_data_3d([centre[0]], [centre[1]], [centre[2]])

        # Trail: stored in the tray frame, so it tilts together with the tray.
        trail_t = np.array(self._trail)
        trail = np.column_stack([trail_t, lift * np.ones(len(trail_t))]).dot(rot.T)
        self._artists["trail"].set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])

        # Ball: centre one radius above the surface along the tray normal.
        ball_t = np.array([env.ball_pos[0], env.ball_pos[1], self.ball_radius])
        ball_w = rot.dot(ball_t)
        self._artists["ball"].set_verts(self._unit_sphere * self.ball_radius + ball_w)

        # Inertial force on the ball: opposite to a_base, in the tray plane.
        push = -np.asarray(env.a_base, dtype=np.float64) * self.DISTURBANCE_ARROW_SCALE
        length = np.linalg.norm(push)
        arrow = self._artists["arrow"]
        if length < 1e-4:
            arrow.set_visible(False)
        else:
            along = push / length
            across = np.array([-along[1], along[0]])
            tip = ball_t[:2] + push
            back = tip - self.ARROW_HEAD_LENGTH * along
            head = 0.4 * self.ARROW_HEAD_LENGTH * across
            points_t = np.array([ball_t[:2], tip, back + head, tip, back - head])
            points = np.column_stack(
                [points_t, self.ball_radius * np.ones(len(points_t))]).dot(rot.T)
            arrow.set_data_3d(points[:, 0], points[:, 1], points[:, 2])
            arrow.set_visible(True)

        reward, episode_return = self._last_values
        a_base_norm = float(np.linalg.norm(env.a_base))
        pitch_deg, roll_deg = np.rad2deg(env.tilt)
        lines = [
            "step   {:d} / {:d}".format(env.step_count, env.max_steps),
            "time   {:.2f} s".format(env.step_count * env.DT),
            "",
            "pitch  {:+6.2f} deg".format(pitch_deg),
            "roll   {:+6.2f} deg".format(roll_deg),
            "c_rr   {:.4f}".format(env.c_rr),
            "",
            "reward {:+.3f}".format(reward),
            "return {:+.2f}".format(episode_return),
            "",
            "disturbance",
            "ACTIVE {:.2f} m/s^2".format(a_base_norm) if a_base_norm > 0.0 else "none",
        ]
        if np.any(np.abs(env.ball_pos) > env.tray_half_size):
            lines += ["", "BALL OFF TRAY"]
        self._artists["text"].set_text("\n".join(lines))

    def _on_window_close(self, event):
        """Matplotlib callback: the user closed the window."""
        self.window_closed = True
