"""Implements the ballistic forward model that predicts the ball's future flight, including table bounces and net contacts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import GRAVITY, BallSpec, TableSpec
from ..physics import bounce_velocity, flight_position, flight_velocity, time_to_plane


@dataclass
class BounceEvent:
    """A single impact along a predicted trajectory."""

    time: float
    position: np.ndarray
    incoming_velocity: np.ndarray
    outgoing_velocity: np.ndarray
    surface: str  # "table" | "floor" | "net"


@dataclass
class Trajectory:
    """A densely sampled prediction of the ball's flight."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    events: list[BounceEvent] = field(default_factory=list)
    terminal: str = "horizon"

    def __len__(self) -> int:
        return int(self.times.shape[0])

    @property
    def landing_point(self) -> np.ndarray | None:
        """Position of the first table/floor contact, if there is one."""
        for event in self.events:
            if event.surface in ("table", "floor"):
                return event.position
        return None

    def segment_after(self, time: float) -> "Trajectory":
        """The part of the trajectory that lies after ``time``."""
        mask = self.times >= time
        return Trajectory(self.times[mask], self.positions[mask], self.velocities[mask], self.events, self.terminal)


class BallPredictor:
    """Rolls the ball forward through free flight and bounces."""

    def __init__(self, table: TableSpec, ball: BallSpec, gravity: np.ndarray | None = None, sample_time: float = 0.01, horizon: float = 2.5, max_bounces: int = 3):
        """Initialize the ball predictor.
        
        Args:
            table: Table geometry.
            ball: Ball geometry and material properties.
            gravity: Gravity vector, defaults to ``GRAVITY``.
            sample_time: Time step for the dense trajectory samples.
            horizon: Maximum prediction horizon.
            max_bounces: Maximum number of bounces to simulate.
        """
        self.table = table
        self.ball = ball
        self.gravity = np.asarray(GRAVITY if gravity is None else gravity, dtype=float)
        self.sample_time = float(sample_time)
        self.horizon = float(horizon)
        self.max_bounces = int(max_bounces)

    # ------------------------------------------------------------------ events
    def _next_event(self, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray):
        """Time and kind of the next impact, or ``(inf, None)``."""
        table_plane = self.ball.bounce_plane(self.table)
        floor_plane = self.ball.radius

        candidates: list[tuple[float, str]] = []

        # Check if the ball is already touching the table and falling: if so, bounce immediately
        on_table_now = bool(self.table.is_on_half(position, +1)) or bool(self.table.is_on_half(position, -1))
        if on_table_now and position[2] <= table_plane + 1e-9 and velocity[2] < 0.0:
            return 0.0, "table"

        # Table top: only counts if the crossing happens above the table
        t_table = float(time_to_plane(position, velocity, table_plane, acceleration))
        if np.isfinite(t_table):
            contact = flight_position(np.array([t_table]), position, velocity, acceleration)[0]
            if bool(self.table.is_on_half(contact, +1)) or bool(self.table.is_on_half(contact, -1)):
                candidates.append((t_table, "table"))

        # Net: crossing the y=0 plane below the top of the net
        if abs(velocity[1]) > 1e-9:
            t_net = -position[1] / velocity[1]
            if t_net > 1e-6:
                crossing = flight_position(np.array([t_net]), position, velocity, acceleration)[0]
                if crossing[2] < self.table.net_height and abs(crossing[0]) < self.table.net_half_width:
                    candidates.append((float(t_net), "net"))

        # Floor: only counts if the crossing happens below the table
        t_floor = float(time_to_plane(position, velocity, floor_plane, acceleration))
        if np.isfinite(t_floor):
            candidates.append((t_floor, "floor"))

        # If there are no candidates, return infinity; otherwise, return the soonest one
        if not candidates:
            return float("inf"), None
        return min(candidates, key=lambda item: item[0])

    def _rebound_velocity(self, surface: str, impact_velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
        """Outgoing velocity of a bounce off ``surface``, shared by :meth:`predict` and :meth:`state_after`."""
        if surface == "table":
            # Calculate the rebound velocity using the ball's restitution and tangential properties
            rebound = bounce_velocity(impact_velocity, self.ball.table_restitution, self.ball.table_tangential)

            # Apply the ball's table impact duration to account for the effect of acceleration during the contact time
            return rebound + acceleration * self.ball.table_impact_duration
        
        if surface == "floor":
            return bounce_velocity(impact_velocity, 0.5, 0.8)

        return np.zeros(3)  # net

    # -------------------------------------------------------------- prediction
    def predict(self, position, velocity, acceleration=None, horizon: float | None = None) -> Trajectory:
        """Sample the future flight of a ball starting at ``position``.

        Args:
            position: Current (estimated) position.
            velocity: Current (estimated) velocity.
            acceleration: Constant acceleration, defaults to gravity. Pass ``observer.acceleration`` to fold the estimated disturbance in.
            horizon: Prediction horizon, defaults to the configured one.
        
        Returns:
            A :class:`Trajectory` object containing the sampled times, positions, velocities, and bounce events, as well as the terminal condition ("horizon", "floor", or "net").
        """
        position = np.asarray(position, dtype=float).copy()
        velocity = np.asarray(velocity, dtype=float).copy()
        acceleration = self.gravity if acceleration is None else np.asarray(acceleration, dtype=float)
        horizon = self.horizon if horizon is None else float(horizon)

        times: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        events: list[BounceEvent] = []

        elapsed = 0.0
        terminal = "horizon"

        # Simulate the ball's flight and bounces until the horizon is reached or a terminal event occurs
        for _ in range(self.max_bounces + 1):
            # Check if the remaining time is less than or equal to zero; if so, break the loop
            remaining = horizon - elapsed
            if remaining <= 0.0:
                break

            # Determine the time and surface of the next event (bounce or terminal condition)
            event_time, surface = self._next_event(position, velocity, acceleration)
            segment_end = min(event_time, remaining)

            # Sample the trajectory at regular intervals up to the next event or the remaining time
            sample_times = np.arange(0.0, segment_end, self.sample_time)
            if sample_times.size == 0:
                sample_times = np.zeros(1)

            # Append the sampled times, positions, and velocities to the respective lists
            times.append(sample_times + elapsed)
            positions.append(flight_position(sample_times, position, velocity, acceleration))
            velocities.append(flight_velocity(sample_times, velocity, acceleration))

            # If the next event time is not finite or exceeds the remaining time, break the loop
            if not np.isfinite(event_time) or event_time > remaining:
                break

            # Calculate the impact position and velocity at the event time, and determine the rebound velocity
            impact_position = flight_position(np.array([event_time]), position, velocity, acceleration)[0]
            impact_velocity = flight_velocity(np.array([event_time]), velocity, acceleration)[0]
            rebound = self._rebound_velocity(str(surface), impact_velocity, acceleration)

            # Append a BounceEvent to the events list, including the time, position, incoming and outgoing velocities, and surface type
            events.append(BounceEvent(
                time=elapsed + event_time,
                position=impact_position,
                incoming_velocity=impact_velocity,
                outgoing_velocity=rebound,
                surface=str(surface),
            ))

            # Update the elapsed time, position, and velocity for the next iteration of the loop
            elapsed += event_time
            position, velocity = impact_position, rebound

            # If the surface is "floor" or "net", set the terminal condition and break the loop
            if surface in ("floor", "net"):
                terminal = str(surface)
                break

        return Trajectory(
            times=np.concatenate(times) if times else np.zeros(0),
            positions=np.concatenate(positions) if positions else np.zeros((0, 3)),
            velocities=np.concatenate(velocities) if velocities else np.zeros((0, 3)),
            events=events,
            terminal=terminal,
        )

    def state_after(self, position, velocity, acceleration, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Ball state exactly ``dt`` seconds ahead, applying any bounce in between.

        Used for delay compensation (:meth:`RallyAgent._estimated_state`):
        plain constant-acceleration ballistics (`flight_position`/
        `flight_velocity`) silently extrapolate *through* a bounce that
        happened inside the roll-forward window instead of applying it --
        exactly wrong right when a delayed estimate is being rolled forward
        across a contact. Unlike :meth:`predict`, this jumps straight to the
        state at ``dt`` instead of densely sampling a whole trajectory, so it
        stays accurate even for a roll-forward window much shorter than
        :attr:`sample_time`.
        """
        position = np.asarray(position, dtype=float).copy()
        velocity = np.asarray(velocity, dtype=float).copy()
        acceleration = np.asarray(acceleration, dtype=float)
        remaining = float(dt)

        # Simulate the ball's flight and bounces until the remaining time is exhausted or a terminal event occurs
        for _ in range(self.max_bounces + 1):
            if remaining <= 0.0:
                break

            # Determine the time and surface of the next event (bounce or terminal condition)
            event_time, surface = self._next_event(position, velocity, acceleration)
            if not np.isfinite(event_time) or event_time >= remaining:
                break

            # Calculate the impact position and velocity at the event time
            position = flight_position(np.array([event_time]), position, velocity, acceleration)[0]
            impact_velocity = flight_velocity(np.array([event_time]), velocity, acceleration)[0]

            # Update the velocity after the bounce using the rebound velocity based on the surface type
            velocity = self._rebound_velocity(str(surface), impact_velocity, acceleration)
            remaining -= event_time

            # If the surface is "floor" or "net", return the position and velocity immediately, as no further simulation is needed
            if surface in ("floor", "net"):
                return position, velocity

        # After processing all bounces, calculate the final position and velocity after the remaining time
        position = flight_position(np.array([remaining]), position, velocity, acceleration)[0]
        velocity = flight_velocity(np.array([remaining]), velocity, acceleration)[0]
        return position, velocity
