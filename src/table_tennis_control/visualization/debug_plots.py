"""Per-serve debug plots for offline analysis (opt-in via ``--debug-plots``).

Renders one PNG per serve, spanning exactly the ball's flight from its launch
up to (but not including) the next launch -- the same span
:meth:`~table_tennis_control.agent.RallyAgent.serve` itself resets state
over, and the one a rally review actually cares about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..agent import AgentDiagnostics, Phase, RallyAgent
from ..config import GRAVITY, SimulationConfig, TableSpec
from .colors import Color, JOINT_COLORS, PHASE_COLOR, phase_runs

__all__ = ["ServeDebugPlotter"]


@dataclass
class _ServeRecord:
    index: int
    start_time: float
    landing_error_index: int
    strikes_before: int
    time: list = field(default_factory=list)
    phase: list = field(default_factory=list)

    # Ball path from launch to its first floor bounce only
    path_position: list = field(default_factory=list)
    position_error: list = field(default_factory=list)
    orientation_error: list = field(default_factory=list)

    # Each joints torque as a fraction of its limit, for the torque plot
    joint_torque_fraction: list = field(default_factory=list)
    joint_names: tuple = ()

    # Measured paddle speed (magnitude of the velocity vector) at each physics tick, for the speed plot
    paddle_speed: list = field(default_factory=list)

    # Times of genuine ball-paddle contact
    strike_times: list = field(default_factory=list)

    # Planned impact speed at each entry in :attr:`strike_times`
    strike_target_speeds: list = field(default_factory=list)

    # Measured paddle position and normal at each entry in :attr:`strike_times`
    strike_paddle_position: list = field(default_factory=list)
    strike_paddle_normal: list = field(default_factory=list)

    # Points where the ball bounces off the table and the last plan that predicted them, for the trajectory plots
    bounce_points: list = field(default_factory=list)
    _last_bounce_plan_id: int | None = None

    # Whether the ball and paddle are in contact at the last recorded tick, for detecting new strikes
    _in_paddle_contact: bool = False

    # The landing error(s) recorded during this serve, and how many strikes the robot made before the next launch
    landing_errors: list = field(default_factory=list)
    strikes_this_serve: int = 0
    target: np.ndarray | None = None


class ServeDebugPlotter:
    """Buffers one serve's diagnostics at a time and renders it to a PNG."""

    _PADDLE_RADIUS = 0.08

    _CELL_ASPECT = 1.5

    def __init__(self, output_dir: Path, config: SimulationConfig | None = None):
        """Create a new debug plotter, clearing any existing plots in ``output_dir``.
        
        The ``config`` is only used to derive the nominal view bounds for the trajectory plots; if ``None``, 
        a fixed default is used instead. The bounds are still allowed to grow past that if a serve's actual 
        path doesn't fit, but the default is more likely to be too small than too large, so it is better to 
        derive them from the actual configuration.
        
        Args:
            output_dir: Folder to write the serve plots to. Any existing serve plots in that folder are deleted.
            config: Simulation configuration to derive view bounds from.
        """
        # Create and/or clear the output folder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clear_folder(self.output_dir)

        self.table = config.table if config is not None else None
        self._x_limits, self._y_limits, self._z_limits = self._nominal_limits(config)

        self._torque_smoothing_period = 1.0 / config.planner.replan_rate if config is not None else 0.02

        self._count = 0
        self._current: _ServeRecord | None = None
        self._paddle_geom_id: int | None = None

    def begin_serve(self, agent: RallyAgent, time: float) -> None:
        """Finalise the previous serve (if any) and start recording a new one.
        
        Args:
            agent: The agent whose diagnostics are being recorded.
            time: The simulation time at the start of the serve.
        """
        self.close()
        self._count += 1
        self._current = _ServeRecord(index=self._count, start_time=time, landing_error_index=len(agent.statistics.landing_errors), strikes_before=agent.statistics.strikes)

    def record(self, agent: RallyAgent, diagnostics: AgentDiagnostics) -> None:
        """Add the current tick's diagnostics to the buffer for the current serve.
        
        Args:
            agent: The agent whose diagnostics are being recorded.
            diagnostics: The diagnostics for the current tick.
        """
        record = self._current
        if record is None:
            return

        # Record the current tick's diagnostics into the serve record
        record.time.append(diagnostics.time)
        record.phase.append(diagnostics.phase)
        record.path_position = list(agent.actual_path)
        record.position_error.append(diagnostics.arm.position_error)
        record.orientation_error.append(diagnostics.arm.orientation_error)
        torque_limits = agent.arm.controller.torque_limits
        record.joint_torque_fraction.append(np.abs(diagnostics.arm.controller.torque) / torque_limits)

        # Fill in the joint names if they haven't been set yet (they don't change between serves)
        if not record.joint_names:
            record.joint_names = tuple(agent.arm.spec.joint_names)

        record.paddle_speed.append(float(np.linalg.norm(diagnostics.measured.velocity)))

        # Record the planned bounce point if it exists and is new (not the same plan as last tick)
        plan = diagnostics.plan
        if plan is not None and plan.bounce_point is not None and id(plan) != record._last_bounce_plan_id:
            record.bounce_points.append(plan.bounce_point.copy())
            record._last_bounce_plan_id = id(plan)

        # Record the target position if it hasn't been set yet (it doesn't change between serves)
        if record.target is None:
            record.target = agent.scene.target.position.copy()

        # Detect new ball-paddle contact and record the strike time, target speed, paddle position, and normal
        in_contact = self._ball_paddle_contact(agent)
        if in_contact and not record._in_paddle_contact:
            record.strike_times.append(diagnostics.time)
            record.strike_target_speeds.append(plan.paddle_speed if plan is not None else float("nan"))
            record.strike_paddle_position.append(diagnostics.measured.position.copy())
            record.strike_paddle_normal.append(diagnostics.measured.normal.copy())
        record._in_paddle_contact = in_contact

        # Record the number of strikes and landing errors that occurred during this serve
        record.strikes_this_serve = agent.statistics.strikes - record.strikes_before
        record.landing_errors = list(agent.statistics.landing_errors[record.landing_error_index :])

    def close(self) -> None:
        """Render and save whatever serve is currently buffered, if any."""
        record = self._current
        self._current = None
        if record is None or len(record.time) < 2:
            return
        self._render(record)

    def _clear_folder(self, path: Path) -> None:
        """Delete all existing serve debug plots in the given output folder, if any."""
        for file in path.glob("serve_*.png"):
            file.unlink()

    @staticmethod
    def _padded_bounds(values, margin_fraction: float = 0.08, min_margin: float = 0.1) -> tuple[float, float]:
        """``(min, max)`` of ``values`` padded by whichever margin is larger."""
        lo, hi = min(values), max(values)
        margin = max(min_margin, margin_fraction * (hi - lo))
        return lo - margin, hi + margin

    @staticmethod
    def _match_aspect(
        vertical: tuple[float, float], horizontal: tuple[float, float], cell_aspect: float
    ) -> tuple[float, float]:
        """Widen ``vertical`` (never narrow it) to the span ``horizontal`` needs at ``cell_aspect``."""
        required_span = (horizontal[1] - horizontal[0]) / cell_aspect
        lo, hi = vertical

        span = hi - lo
        if span >= required_span:
            return lo, hi

        pad = (required_span - span) / 2
        return lo - pad, hi + pad

    @classmethod
    def _nominal_limits(cls, config: SimulationConfig | None) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Fixed default ``(x, y, z)`` view bounds for the trajectory plots."""

        # If no config is provided, use a fixed default for the y-bounds and derive the x and z bounds from that
        if config is None:
            y_bounds = (-3.0, 4.1)
            return (
                cls._match_aspect((-1.2, 1.2), y_bounds, cls._CELL_ASPECT),
                y_bounds,
                cls._match_aspect((-0.15, 2.3), y_bounds, cls._CELL_ASPECT),
            )

        # Derive the nominal bounds from the config, so that the plots are more likely to fit the actual trajectories without needing to grow
        launcher, target, arm, table = config.launcher, config.target, config.arm, config.table
        opponent_side = table.opponent_side
        target_y = [opponent_side * (table.half_length + margin) for margin in target.floor_margin_range]

        x_candidates = [
            -table.half_width,
            table.half_width,
            *launcher.x_range,
            *launcher.bounce_x_range,
            *target.floor_x_range,
            *arm.workspace[0],
        ]

        y_candidates = [
            -table.half_length,
            table.half_length,
            *launcher.y_range,
            *launcher.bounce_y_range,
            *arm.workspace[1],
            *target_y,
        ]

        # Calculate the apex of the ball's trajectory to include it in the z-bounds
        gravity = float(abs(GRAVITY[2]))
        z0 = launcher.z_range[1]
        duration = launcher.flight_time_range[1]
        v0 = (table.height - z0 + 0.5 * gravity * duration**2) / duration
        apex = z0 + max(v0, 0.0) ** 2 / (2 * gravity)
        z_candidates = [0.0, table.height, table.net_height, *launcher.z_range, *arm.workspace[2], apex]

        # Pad the bounds and match the aspect ratio for the trajectory plots, so that they are more likely to fit the actual trajectories without needing to grow
        y_bounds = cls._padded_bounds(y_candidates)
        x_bounds = cls._match_aspect(cls._padded_bounds(x_candidates), y_bounds, cls._CELL_ASPECT)
        z_bounds = cls._match_aspect(cls._padded_bounds(z_candidates), y_bounds, cls._CELL_ASPECT)
        return x_bounds, y_bounds, z_bounds

    def _ball_paddle_contact(self, agent: RallyAgent) -> bool:
        """Whether the ball and the paddle geom are touching right now."""
        data = agent.scene.data
        if self._paddle_geom_id is None:
            self._paddle_geom_id = int(agent.scene.model.geom("paddle").id)
        ball_geom = agent.scene.ball.geom_id

        # Loop through all contacts in the simulation and check if the ball geom is in contact with the paddle geom
        for index in range(data.ncon):
            contact = data.contact[index]

            # Check if the ball geom is one of the two geoms in contact
            geoms = (int(contact.geom1), int(contact.geom2))
            if ball_geom not in geoms:
                continue

            # Check if the other geom in contact is the paddle geom
            other = geoms[0] if geoms[1] == ball_geom else geoms[1]
            if other == self._paddle_geom_id:
                return True
        return False

    # ------------------------------------------------------------------ plotting
    def _render(self, record: _ServeRecord) -> None:
        """Render the current serve's diagnostics to a PNG file."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        time = np.asarray(record.time)
        phases = record.phase
        positions = np.asarray(record.path_position).reshape(-1, 3)

        # Create a 3x2 grid of subplots for the serve diagnostics, with shared x-axes for the position/orientation and torques/speed plots
        fig, axes = plt.subplot_mosaic(
            [["top", "side"], ["position", "torques"], ["orientation", "speed"]],
            figsize=(13, 13),
            constrained_layout=True,
        )
        axes["position"].sharex(axes["orientation"])
        axes["torques"].sharex(axes["speed"])

        # Add a title to the figure with the serve index, launch time, number of strikes, and any landing error
        outcome = f"strikes {record.strikes_this_serve}"
        if record.landing_errors:
            outcome += f"   landing error {record.landing_errors[-1]:.3f} m"
        fig.suptitle(f"Serve {record.index}  (launch t = {record.start_time:.2f} s)   {outcome}")

        # Plot the top-down view of the ball's path, with the table and net, and the paddle at impact if it exists
        self._plot_path(axes["top"], positions[:, 1], positions[:, 0], record, 1, 0, "y [m]", "x [m]", "Top-down", xlim=self._y_limits, ylim=self._x_limits)

        # Plot the side view of the ball's path, with the table and net, and the paddle at impact if it exists
        side_a, side_b = positions[:, 1], positions[:, 2]
        self._plot_path(axes["side"], side_a, side_b, record, 1, 2, "y [m]", "z [m]", "Side view", show_floor=True, xlim=self._y_limits, ylim=self._z_limits)

        # Plot the position error over time, with the strike times marked
        self._plot_metric(axes["position"], time, record.position_error, phases, record, "position error [m]", "Position error", hide_xlabel=True)

        # Plot the orientation error over time, with the strike times marked
        self._plot_metric(axes["orientation"], time, record.orientation_error, phases, record, "orientation error [rad]", "Orientation error")

        # Plot the joint torques over time, with the strike times marked
        self._plot_joint_torques(axes["torques"], time, record, phases, hide_xlabel=True)

        # Plot the paddle speed over time, with the strike times and target speeds marked
        self._plot_metric(axes["speed"], time, record.paddle_speed, phases, record, "paddle speed [m/s]", "Paddle speed", target_markers=(record.strike_times, record.strike_target_speeds))

        # Save the figure to a PNG file in the output directory, with a filename based on the serve index
        path = self.output_dir / f"serve_{record.index:03d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _table_corners(table: TableSpec) -> np.ndarray:
        """Closed outline of the full table (both half-courts) at play height."""
        hw, hl, h = table.half_width, table.half_length, table.height
        return np.array([[-hw, -hl, h], [hw, -hl, h], [hw, hl, h], [-hw, hl, h], [-hw, -hl, h]])

    @staticmethod
    def _net_corners(table: TableSpec) -> np.ndarray:
        """Closed outline of the net sheet, spanning x at y = 0 up to its top."""
        nw, h, nh = table.net_half_width, table.height, table.net_height
        return np.array([[-nw, 0.0, h], [nw, 0.0, h], [nw, 0.0, nh], [-nw, 0.0, nh], [-nw, 0.0, h]])

    @staticmethod
    def _paddle_indicator(position: np.ndarray, normal: np.ndarray, idx_a: int, idx_b: int, half_length: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Endpoints of a line depicting the paddle face edge-on, in this 2-D projection."""
        normal_2d = np.array([normal[idx_a], normal[idx_b]])
        norm = float(np.linalg.norm(normal_2d))
        if norm < 0.05:
            return None

        # Rotate the normal 90 degrees to get the tangent direction, then scale it to half the paddle's length and offset from the paddle's centre to get the endpoints
        tangent = np.array([-normal_2d[1], normal_2d[0]]) / norm
        centre = np.array([position[idx_a], position[idx_b]])
        return centre - tangent * half_length, centre + tangent * half_length

    @staticmethod
    def _phase_legend_handles(linewidth: float):
        """Legend handles for the four phases, with the given line width."""
        from matplotlib.lines import Line2D

        return [Line2D([0], [0], color=PHASE_COLOR[phase][:3], linewidth=linewidth, label=phase.value.capitalize()) for phase in Phase]

    def _plot_path(self, ax, a, b, record: _ServeRecord, idx_a, idx_b, xlabel, ylabel, title, show_floor: bool = False, xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None) -> None:
        """One 2-D projection of the ball's path, with the table and net, and the paddle at impact if it exists."""
        if self.table is not None:
            # Plot table outline
            corners = self._table_corners(self.table)
            ax.plot(corners[:, idx_a], corners[:, idx_b], color="0.4", linewidth=1.3, zorder=1, label="table")

            # Plot net outline
            net = self._net_corners(self.table)
            ax.plot(net[:, idx_a], net[:, idx_b], color="0.4", linewidth=1.0, linestyle="--", zorder=1, label="net")

        if show_floor:
            # Plot the floor at z=0 (or y=0 in the top-down view) as a horizontal line
            ax.axhline(0.0, color="0.2", linewidth=1.3, zorder=0, label="floor")

        # The ball's actual path, in one constant color
        ax.plot(a, b, color=Color.ACTUAL[:3], alpha=Color.ACTUAL[3], linewidth=2.0, zorder=2, label="ball path")

        # Mark the target position (if it exists) and the planned bounce points (if they exist), with the last one highlighted as the final plan
        if record.target is not None:
            ax.scatter([record.target[idx_a]], [record.target[idx_b]], marker="+", color=Color.TARGET, s=120, linewidths=2.5, zorder=3, label="target")

        # Plot the planned bounce points, with the last one highlighted as the final plan
        if record.bounce_points:
            bounce = np.asarray(record.bounce_points)

            if bounce.shape[0] > 1:
                ax.scatter(bounce[:-1, idx_a], bounce[:-1, idx_b], marker="o", color=Color.BOUNCE, s=16, alpha=0.35, linewidths=0, zorder=3, label="planned bounce (refining)")
            ax.scatter([bounce[-1, idx_a]], [bounce[-1, idx_b]], marker="o", facecolor=Color.BOUNCE, edgecolor="black", linewidths=0.8, s=25, zorder=4, label="planned bounce (final)")

        # Plot the paddle at impact, if it exists, as a line segment edge-on to the camera
        if record.strike_paddle_position:
            label = "paddle at impact"

            for position, normal in zip(record.strike_paddle_position, record.strike_paddle_normal):
                endpoints = self._paddle_indicator(position, normal, idx_a, idx_b, self._PADDLE_RADIUS)
                if endpoints is None:
                    continue

                p0, p1 = endpoints
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=Color.STRIKE, linewidth=1.8, solid_capstyle="round", zorder=5, label=label)

                label = None

        # Add axis labels, title, grid, and legend
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, linewidth=0.5)

        if xlim is not None:
            ax.set_xlim(self._grow_to_fit(xlim, ax.dataLim.intervalx))
        if ylim is not None:
            ax.set_ylim(self._grow_to_fit(ylim, ax.dataLim.intervaly))

        ax.set_aspect("equal", adjustable="box")

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + self._phase_legend_handles(2.0), loc="best", fontsize="small", ncol=2)

    @staticmethod
    def _grow_to_fit(nominal: tuple[float, float], data_interval, margin_fraction: float = 0.05) -> tuple[float, float]:
        """``nominal``, widened just enough to include ``data_interval`` if it doesn't already fit."""
        data_min, data_max = float(data_interval[0]), float(data_interval[1])
        span = nominal[1] - nominal[0]
        margin = margin_fraction * span

        lo = nominal[0] if data_min >= nominal[0] else data_min - margin
        hi = nominal[1] if data_max <= nominal[1] else data_max + margin
        return lo, hi

    @staticmethod
    def _shade_phases(ax, time, phases) -> None:
        """Shade the background of a metric plot by phase, so that the line colors match the trajectory plots above."""
        for phase, start, end in phase_runs(phases):
            ax.axvspan(time[start], time[end], color=PHASE_COLOR[phase][:3], alpha=0.22, lw=0)

    def _plot_metric(self, ax, time, values, phases, record: _ServeRecord, ylabel: str, title: str, hide_xlabel: bool = False, target_markers: tuple[list, list] | None = None) -> None:
        """Plot a metric over time, with the strike times marked and the background shaded by phase."""
        # Shade the background by phase, so that the line colors match the trajectory plots above
        self._shade_phases(ax, time, phases)

        # Plot the metric line, colored by phase to match the background shading and the trajectory plots above
        for phase, start, end in phase_runs(phases):
            ax.plot(time[start : end + 1], values[start : end + 1], color=PHASE_COLOR[phase][:3], linewidth=1.8, zorder=2)

        # Mark the target values (if any) as diamond markers
        target_handle = None
        if target_markers is not None and target_markers[0]:
            marker_times, marker_values = target_markers
            (target_handle,) = ax.plot(marker_times, marker_values, linestyle="none", marker="D", markersize=7, markerfacecolor="none", markeredgecolor="0.2", markeredgewidth=1.4, zorder=4, label="target @ impact")

        # Mark the strike times as vertical dashed lines, so that the line colors match the trajectory plots above
        strike_line = None
        for strike in record.strike_times:
            strike_line = ax.axvline(strike, color="black", linestyle="--", linewidth=1.0, label="strike")

        # Add axis labels, title, grid, and legend
        if hide_xlabel:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, linewidth=0.5)

        # Add a legend with the phase colors, the target markers (if any), and the strike lines (if any)
        handles = self._phase_legend_handles(1.8)

        if target_handle is not None:
            handles.append(target_handle)

        if strike_line is not None:
            handles.append(strike_line)

        ax.legend(handles=handles, loc="best", fontsize="small", ncol=2)

    @staticmethod
    def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
        """Centred moving average, edge-corrected so it stays unbiased at the ends."""
        if window <= 1:
            return values
        
        kernel = np.ones(window)
        summed = np.convolve(values, kernel, mode="same")
        counts = np.convolve(np.ones_like(values), kernel, mode="same")
        return summed / counts

    def _plot_joint_torques(self, ax, time, record: _ServeRecord, phases, hide_xlabel: bool = False) -> None:
        """Every joint's own ``|torque| / limit``, fixed to a 0-1 axis."""
        # Shade the background by phase, so that the line colors match the trajectory plots above
        self._shade_phases(ax, time, phases)

        fractions = np.asarray(record.joint_torque_fraction)
        dt = float(np.median(np.diff(time))) if len(time) > 1 else self._torque_smoothing_period
        window = max(1, round(self._torque_smoothing_period / dt)) if dt > 0 else 1

        # Plot each joint's torque fraction, smoothed with a moving average over the torque smoothing period, and color-coded by joint
        for index, name in enumerate(record.joint_names):
            color = JOINT_COLORS[index % len(JOINT_COLORS)]
            smoothed = self._moving_average(fractions[:, index], window)
            ax.plot(time, smoothed, color=color, linewidth=1.5, label=name)

        # Mark the strike times as vertical dashed lines, so that the line colors match the trajectory plots above
        for strike in record.strike_times:
            ax.axvline(strike, color="black", linestyle="--", linewidth=1.0, label="strike")

        # Add axis labels, title, grid, and legend
        if hide_xlabel:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("time [s]")
        ax.set_ylabel("torque used [fraction of limit], smoothed")
        ax.set_title("Joint torques")
        ax.grid(alpha=0.25, linewidth=0.5)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc="upper right", fontsize="small", ncol=2)
