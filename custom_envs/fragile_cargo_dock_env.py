"""FragileCargoDock-v0 — top-down fragile-cargo precision docking environment.

Task
----
A differential-throttle warehouse cart must push a freely-moving, fragile cargo
crate through a narrow gap in an interior partition wall and deliver it into a
designated dock, arriving slowly, gently and with a bounded heading error.

Design intent (for the CREATE reward-engineering study)
-------------------------------------------------------
This environment is deliberately sized between ``LunarLander-v3`` (single rigid
body, discrete actions) and ``BipedalWalker-v3`` (multi-joint coordination). It
keeps exactly one coupled sub-system (cart <-> crate contact) plus one staged
objective (approach -> traverse gap -> align -> hold), which is enough to make
reward misspecification produce *interpretable* failure behaviour:

  * over-rewarding approach        -> cart parks next to the crate
  * over-rewarding contact         -> repeated ramming
  * over-rewarding dock distance   -> high-speed shoving
  * over-rewarding heading         -> spin-in-place
  * over-weighting action cost     -> cart never moves
  * per-step dock bonus            -> jitter farming on the dock lip
  * missing low-speed constraint   -> crate bounces back out of the dock

All physics quantities needed to diagnose those behaviours are exposed either in
the observation vector or in ``info`` (official reward terms are masked and
forbidden for generated reward functions, see ``envs/env_fragilecargo/``).

Units are SI: metres, seconds, kilograms, newtons, radians.
The world is top-down, gravity is zero; floor drag is emulated with damping.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Box2D import (
    b2ContactListener,
    b2FixtureDef,
    b2PolygonShape,
    b2RayCastCallback,
    b2Vec2,
    b2World,
)


# --------------------------------------------------------------------------- #
# Scene constants (Standard difficulty)
# --------------------------------------------------------------------------- #

FIELD_HALF_W = 5.0          # warehouse half-width  (x in [-5, 5])
FIELD_HALF_H = 4.0          # warehouse half-height (y in [-4, 4])

PARTITION_X = 0.0           # x of the interior wall
PARTITION_THICK = 0.20
GAP_HALF_H = 0.80           # half-height of the door  -> 1.6 m opening
FUNNEL_THICK = 0.10
FUNNEL_MOUTH_X = 1.15       # how far the guide rails reach back from the gap
FUNNEL_MOUTH_Y = 1.75

ROBOT_HALF_L = 0.25         # cart half-length (x, along heading)
ROBOT_HALF_W = 0.20         # cart half-width  (y)
ROBOT_MASS = 8.0
ROBOT_LIN_DAMP = 1.6
ROBOT_ANG_DAMP = 3.0
ROBOT_MAX_FORCE = 34.0      # N, along heading  -> v_max ~ 2.7 m/s
ROBOT_MAX_TORQUE = 4.5      # N*m               -> w_max ~ 4.9 rad/s

CARGO_HALF = 0.30           # crate is 0.60 x 0.60 m
CARGO_MASS = 3.0
CARGO_LIN_DAMP = 2.0
CARGO_ANG_DAMP = 2.5
CARGO_FRICTION = 0.40

DOCK_X = 2.6
DOCK_Y = 0.0
DOCK_HALF = 0.42            # dock is 0.84 x 0.84 m == 1.40 x crate
DOCK_ANGLE = 0.0            # rad, dock is axis aligned

# Start ranges are sized so that a competent slow-push policy can complete the
# whole approach -> traverse -> align -> hold sequence inside MAX_EPISODE_STEPS.
# Calibrated empirically with scripts/smoke_fragile_cargo_dock.py.
ROBOT_START_X = (-3.5, -2.8)
ROBOT_START_Y = (-1.2, 1.2)
CARGO_START_X = (-1.9, -1.1)
CARGO_START_Y = (-1.1, 1.1)

SPEED_SCALE = 3.0           # obs normaliser for linear velocities
OMEGA_SCALE = 8.0           # obs normaliser for angular velocities
REL_SCALE = 3.0             # obs normaliser for cart->crate relative position
OBS_CLIP = 2.0

SENSOR_RANGE = 1.6          # m, short-range obstacle sensors
N_SENSORS = 3

# Success criteria
ANGLE_TOL = math.radians(30.0)
CARGO_SPEED_TOL = 0.05      # m/s
STABLE_STEPS_REQUIRED = 10

# Fragility criteria
HARD_IMPULSE = 5.0          # N*s, a single contact above this is a "hard hit"
MAX_HARD_COLLISIONS = 3

PHYSICS_DT = 1.0 / 120.0
FRAME_SKIP = 4              # env dt = 1/30 s
MAX_EPISODE_STEPS = 400     # 13.33 s

# Episode time fractions are capped just below 1.0 so that the observation never
# saturates before the truncation wrapper fires.
_TIME_FRACTION_CAP = 0.999

OBS_DIM = 19  # see _observation() for the per-index semantics


# --------------------------------------------------------------------------- #
# Official (native) reward weights — masked from the reward-design LLM
# --------------------------------------------------------------------------- #

OFFICIAL_W = {
    "approach_cargo": 1.0,    # per metre of cart->crate distance closed
    "progress": 1.0,          # per metre of crate->dock distance closed
    "dock_enter": 5.0,        # one-off, first step the crate is fully inside
    "roughness": 0.02,        # per N*s of cart<->crate contact impulse
    "action_cost": 0.0005,
    "time_cost": 0.002,
    "hard_hit": 0.5,
    "success": 300.0,
    "failure": 100.0,
}


# --------------------------------------------------------------------------- #
# Box2D helpers
# --------------------------------------------------------------------------- #

class _ClosestRayCast(b2RayCastCallback):
    """Record the nearest *static* fixture hit along a ray."""

    def __init__(self, static_bodies):
        super().__init__()
        self._static_bodies = static_bodies
        self.hit = False
        self.fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):  # noqa: N802
        try:
            if fixture.body not in self._static_bodies:
                return -1.0
        except Exception:
            return -1.0
        self.hit = True
        self.fraction = float(fraction)
        return float(fraction)


class _CartCargoContactMonitor(b2ContactListener):
    """Accumulate the peak normal impulse of cart<->crate contacts per step.

    Note: pybox2d hands out a *fresh* Python wrapper for the same underlying
    C++ body on every ``fixture.body`` access, so ``a is body`` is always False
    (``a == body`` compares the native pointer and does work). Tagging the two
    bodies with ``userData`` is the unambiguous way to identify the pair.
    """

    CART_TAG = "fragile_cargo_dock.cart"
    CARGO_TAG = "fragile_cargo_dock.crate"

    def __init__(self):
        super().__init__()
        self.contact = False
        self.peak_impulse = 0.0

    def begin_step(self):
        self.contact = False
        self.peak_impulse = 0.0

    def PostSolve(self, contact, impulse):  # noqa: N802
        try:
            tags = {
                getattr(contact.fixtureA.body, "userData", None),
                getattr(contact.fixtureB.body, "userData", None),
            }
        except Exception:
            return
        if tags != {self.CART_TAG, self.CARGO_TAG}:
            return
        self.contact = True
        try:
            peaks = list(impulse.normalImpulses)
        except Exception:
            return
        if peaks:
            peak = max(float(p) for p in peaks)
            if peak > self.peak_impulse:
                self.peak_impulse = peak


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

class FragileCargoDockEnv(gym.Env):
    """Top-down cart / fragile-crate precision docking task."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # ---------------------------------------------------------------- setup
    def __init__(self, render_mode: Optional[str] = None, difficulty: str = "standard"):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if difficulty not in ("easy", "standard", "hard"):
            raise ValueError(f"unsupported difficulty: {difficulty}")

        self.render_mode = render_mode
        self.difficulty = difficulty
        self.max_episode_steps = MAX_EPISODE_STEPS

        low = np.full((OBS_DIM,), -OBS_CLIP, dtype=np.float32)
        high = np.full((OBS_DIM,), OBS_CLIP, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._world: Optional[b2World] = None
        self._robot_body = None
        self._cargo_body = None
        self._static_bodies = []
        self._monitor: Optional[_CartCargoContactMonitor] = None

        self._surface = None
        self._clock = None

        self._elapsed_steps = 0
        self._prev_cargo_dock_dist = 0.0
        self._stable_steps = 0
        self._hard_collision_count = 0
        self._stagnation_steps = 0
        self._dock_entered = False
        self._action_energy = 0.0
        self._component_returns: Dict[str, float] = {}

    # ------------------------------------------------------------- building
    def _build_world(self) -> None:
        # doSleep=False keeps every body awake so contact reporting and the
        # official reward stay deterministic regardless of when motion stops.
        world = b2World(gravity=(0.0, 0.0), doSleep=False)
        self._world = world
        self._static_bodies = []

        def make_static(x, y, hw, hh, angle=0.0):
            body = world.CreateStaticBody(position=(x, y), angle=angle)
            body.CreateFixture(
                b2FixtureDef(
                    shape=b2PolygonShape(box=(hw, hh)),
                    friction=CARGO_FRICTION,
                    restitution=0.0,
                )
            )
            self._static_bodies.append(body)
            return body

        def make_static_segment(x1, y1, x2, y2, thickness=FUNNEL_THICK):
            """Thin static wall running from (x1,y1) to (x2,y2)."""
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.atan2(y2 - y1, x2 - x1)
            body = world.CreateStaticBody(position=(cx, cy), angle=angle)
            body.CreateFixture(
                b2FixtureDef(
                    shape=b2PolygonShape(box=(length * 0.5, thickness * 0.5)),
                    friction=0.2,
                    restitution=0.0,
                )
            )
            self._static_bodies.append(body)
            return body

        # Interior partition with a door.
        half_h = FIELD_HALF_H
        segment = half_h - GAP_HALF_H
        make_static(PARTITION_X, GAP_HALF_H + segment * 0.5, PARTITION_THICK * 0.5, segment * 0.5)
        make_static(PARTITION_X, -GAP_HALF_H - segment * 0.5, PARTITION_THICK * 0.5, segment * 0.5)

        # Angled guide rails funnel the crate into and out of the door. Without
        # them a rotated crate wedges on the door edge, which turns the task
        # into a de-jamming puzzle rather than a docking problem.
        for sign in (1.0, -1.0):
            make_static_segment(
                -FUNNEL_MOUTH_X, sign * FUNNEL_MOUTH_Y,
                -PARTITION_THICK * 0.5, sign * GAP_HALF_H,
            )
            make_static_segment(
                PARTITION_THICK * 0.5, sign * GAP_HALF_H,
                FUNNEL_MOUTH_X, sign * FUNNEL_MOUTH_Y,
            )

        # --- dynamic bodies -------------------------------------------------
        robot = world.CreateDynamicBody(
            position=(0.0, 0.0),
            linearDamping=ROBOT_LIN_DAMP,
            angularDamping=ROBOT_ANG_DAMP,
            bullet=True,
        )
        robot.CreateFixture(
            b2FixtureDef(
                shape=b2PolygonShape(box=(ROBOT_HALF_L, ROBOT_HALF_W)),
                density=ROBOT_MASS / (4.0 * ROBOT_HALF_L * ROBOT_HALF_W),
                friction=0.5,
                restitution=0.0,
            )
        )
        self._robot_body = robot
        robot.userData = _CartCargoContactMonitor.CART_TAG

        cargo = world.CreateDynamicBody(
            position=(0.0, 0.0),
            linearDamping=CARGO_LIN_DAMP,
            angularDamping=CARGO_ANG_DAMP,
            bullet=True,
        )
        cargo.CreateFixture(
            b2FixtureDef(
                shape=b2PolygonShape(box=(CARGO_HALF, CARGO_HALF)),
                density=CARGO_MASS / (4.0 * CARGO_HALF * CARGO_HALF),
                friction=CARGO_FRICTION,
                restitution=0.05,
            )
        )
        self._cargo_body = cargo
        cargo.userData = _CartCargoContactMonitor.CARGO_TAG

        monitor = _CartCargoContactMonitor()
        world.contactListener = monitor
        self._monitor = monitor

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        self._build_world()

        rx = float(self.np_random.uniform(*ROBOT_START_X))
        ry = float(self.np_random.uniform(*ROBOT_START_Y))
        cx = float(self.np_random.uniform(*CARGO_START_X))
        cy = float(self.np_random.uniform(*CARGO_START_Y))

        spread = math.radians(20.0)
        self._robot_body.position = (rx, ry)
        self._robot_body.angle = float(self.np_random.uniform(-spread, spread))
        self._robot_body.linearVelocity = (0.0, 0.0)
        self._robot_body.angularVelocity = 0.0

        self._cargo_body.position = (cx, cy)
        self._cargo_body.angle = float(self.np_random.uniform(-spread, spread))
        self._cargo_body.linearVelocity = (0.0, 0.0)
        self._cargo_body.angularVelocity = 0.0

        self._elapsed_steps = 0
        self._stable_steps = 0
        self._hard_collision_count = 0
        self._stagnation_steps = 0
        self._dock_entered = False
        self._action_energy = 0.0
        self._component_returns = {
            "approach_cargo": 0.0,
            "progress": 0.0,
            "dock_enter": 0.0,
            "roughness": 0.0,
            "action_cost": 0.0,
            "time_cost": 0.0,
            "hard_hit": 0.0,
            "terminal_success": 0.0,
            "terminal_failure": 0.0,
        }
        self._prev_cargo_dock_dist = self._cargo_dock_distance()
        self._prev_robot_cargo_dist = self._robot_cargo_distance()

        obs = self._observation()
        info = self._info(is_success=False, terminated=False, truncated=False, contact_impulse=0.0)
        info["official_reward_terms"] = {}
        info["component_returns"] = dict(self._component_returns)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != 2:
            raise ValueError(f"expected 2-D action, got shape {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        throttle, steer = float(action[0]), float(action[1])

        heading = float(self._robot_body.angle)
        force = (math.cos(heading) * throttle * ROBOT_MAX_FORCE,
                 math.sin(heading) * throttle * ROBOT_MAX_FORCE)
        torque = steer * ROBOT_MAX_TORQUE

        self._monitor.begin_step()
        peak_impulse = 0.0
        for _ in range(FRAME_SKIP):
            self._robot_body.ApplyForceToCenter(b2Vec2(*force), wake=True)
            self._robot_body.ApplyTorque(torque, wake=True)
            self._world.Step(PHYSICS_DT, 8, 3)
            if self._monitor.peak_impulse > peak_impulse:
                peak_impulse = self._monitor.peak_impulse

        self._elapsed_steps += 1
        self._action_energy += float(np.sum(np.square(action)))

        in_contact = bool(self._monitor.contact)
        hard_hit = peak_impulse > HARD_IMPULSE
        if hard_hit:
            self._hard_collision_count += 1

        # ----- task state ---------------------------------------------------
        cargo_pos = self._cargo_body.position
        cargo_dock_dist = self._cargo_dock_distance()
        progress = self._prev_cargo_dock_dist - cargo_dock_dist
        self._prev_cargo_dock_dist = cargo_dock_dist

        robot_cargo_dist = self._robot_cargo_distance()
        approach = self._prev_robot_cargo_dist - robot_cargo_dist
        self._prev_robot_cargo_dist = robot_cargo_dist

        if progress < 0.005:
            self._stagnation_steps += 1
        else:
            self._stagnation_steps = 0

        inside_dock = self._cargo_fully_inside_dock()
        newly_entered = inside_dock and not self._dock_entered
        if inside_dock:
            self._dock_entered = True

        angle_error = abs(_wrap_pi(float(self._cargo_body.angle) - DOCK_ANGLE))
        cargo_speed = float(math.hypot(self._cargo_body.linearVelocity.x,
                                       self._cargo_body.linearVelocity.y))

        stable_now = inside_dock and angle_error < ANGLE_TOL and cargo_speed < CARGO_SPEED_TOL
        self._stable_steps = self._stable_steps + 1 if stable_now else 0
        is_success = self._stable_steps >= STABLE_STEPS_REQUIRED

        robot_pos = self._robot_body.position
        out_of_bounds = (
            abs(float(cargo_pos.x)) > FIELD_HALF_W + CARGO_HALF
            or abs(float(cargo_pos.y)) > FIELD_HALF_H + CARGO_HALF
            or abs(float(robot_pos.x)) > FIELD_HALF_W + ROBOT_HALF_L
            or abs(float(robot_pos.y)) > FIELD_HALF_H + ROBOT_HALF_W
        )
        too_many_hits = self._hard_collision_count >= MAX_HARD_COLLISIONS
        is_failure = out_of_bounds or too_many_hits

        terminated = bool(is_success or is_failure)
        truncated = bool(self._elapsed_steps >= self.max_episode_steps and not terminated)

        # ----- official (native) reward -------------------------------------
        w = OFFICIAL_W
        terms = {
            "approach_cargo": w["approach_cargo"] * approach,
            "progress": w["progress"] * progress,
            "dock_enter": w["dock_enter"] * (1.0 if newly_entered else 0.0),
            "roughness": (
                -w["roughness"] * peak_impulse if in_contact else 0.0
            ),
            "action_cost": -w["action_cost"] * float(np.sum(np.square(action))),
            "time_cost": -w["time_cost"],
            "hard_hit": -w["hard_hit"] * (1.0 if hard_hit else 0.0),
            "terminal_success": w["success"] if is_success else 0.0,
            "terminal_failure": -w["failure"] if (is_failure and not is_success) else 0.0,
        }
        for key, value in terms.items():
            self._component_returns[key] = self._component_returns.get(key, 0.0) + float(value)
        reward = float(sum(terms.values()))

        obs = self._observation()
        info = self._info(
            is_success=is_success,
            terminated=terminated,
            truncated=truncated,
            contact_impulse=peak_impulse,
        )
        info["official_reward_terms"] = terms
        info["component_returns"] = dict(self._component_returns)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def close(self):
        self._surface = None
        self._clock = None
        self._world = None
        self._monitor = None

    # ------------------------------------------------------------ internals
    def _cargo_dock_distance(self) -> float:
        p = self._cargo_body.position
        return float(math.hypot(float(p.x) - DOCK_X, float(p.y) - DOCK_Y))

    def _robot_cargo_distance(self) -> float:
        r = self._robot_body.position
        c = self._cargo_body.position
        return float(math.hypot(float(c.x) - float(r.x), float(c.y) - float(r.y)))

    def _cargo_fully_inside_dock(self) -> bool:
        p = self._cargo_body.position
        limit = DOCK_HALF - CARGO_HALF
        return abs(float(p.x) - DOCK_X) <= limit and abs(float(p.y) - DOCK_Y) <= limit

    def _sensor_distances(self):
        """Normalised proximity of static obstacles ahead / left / right."""
        heading = float(self._robot_body.angle)
        origin = self._robot_body.position
        readings = []
        for offset in (0.0, math.pi / 2.0, -math.pi / 2.0):
            ang = heading + offset
            p1 = (float(origin.x), float(origin.y))
            p2 = (p1[0] + math.cos(ang) * SENSOR_RANGE, p1[1] + math.sin(ang) * SENSOR_RANGE)
            cb = _ClosestRayCast(self._static_bodies)
            self._world.RayCast(cb, b2Vec2(*p1), b2Vec2(*p2))
            dist = cb.fraction * SENSOR_RANGE
            readings.append(float(np.clip(1.0 - dist / SENSOR_RANGE, 0.0, 1.0)))
        return readings

    def _observation(self) -> np.ndarray:
        robot = self._robot_body
        cargo = self._cargo_body

        heading = float(robot.angle)
        cos_h, sin_h = math.cos(heading), math.sin(heading)

        rvx, rvy = float(robot.linearVelocity.x), float(robot.linearVelocity.y)
        forward_speed = rvx * cos_h + rvy * sin_h

        rel_x = float(cargo.position.x) - float(robot.position.x)
        rel_y = float(cargo.position.y) - float(robot.position.y)
        body_x = rel_x * cos_h + rel_y * sin_h
        body_y = -rel_x * sin_h + rel_y * cos_h

        cargo_angle = float(cargo.angle)
        time_fraction = min(_TIME_FRACTION_CAP, self._elapsed_steps / self.max_episode_steps)
        sensors = self._sensor_distances()

        def clip(v):
            return float(np.clip(v, -OBS_CLIP, OBS_CLIP))

        obs = np.array([
            clip(float(robot.position.x) / FIELD_HALF_W),          # 0  cart x
            clip(float(robot.position.y) / FIELD_HALF_H),          # 1  cart y
            cos_h,                                                 # 2  heading cos
            sin_h,                                                 # 3  heading sin
            clip(forward_speed / SPEED_SCALE),                     # 4  cart forward speed
            clip(float(robot.angularVelocity) / OMEGA_SCALE),      # 5  cart yaw rate
            clip(body_x / REL_SCALE),                              # 6  crate rel x (body frame)
            clip(body_y / REL_SCALE),                              # 7  crate rel y (body frame)
            clip(float(cargo.linearVelocity.x) / SPEED_SCALE),     # 8  crate vx
            clip(float(cargo.linearVelocity.y) / SPEED_SCALE),     # 9  crate vy
            math.cos(cargo_angle),                                 # 10 crate heading cos
            math.sin(cargo_angle),                                 # 11 crate heading sin
            clip((float(cargo.position.x) - DOCK_X) / FIELD_HALF_W),   # 12 crate->dock x
            clip((float(cargo.position.y) - DOCK_Y) / FIELD_HALF_H),   # 13 crate->dock y
            1.0 if self._monitor.contact else 0.0,                 # 14 contact flag
            sensors[0],                                            # 15 front sensor
            sensors[1],                                            # 16 left sensor
            sensors[2],                                            # 17 right sensor
            time_fraction,                                         # 18 time fraction
        ], dtype=np.float32)

        return obs

    def _info(self, *, is_success, terminated, truncated, contact_impulse) -> Dict[str, Any]:
        cargo = self._cargo_body
        robot = self._robot_body
        angle_error = abs(_wrap_pi(float(cargo.angle) - DOCK_ANGLE))
        cargo_speed = float(math.hypot(float(cargo.linearVelocity.x),
                                       float(cargo.linearVelocity.y)))
        robot_cargo_distance = float(math.hypot(
            float(cargo.position.x) - float(robot.position.x),
            float(cargo.position.y) - float(robot.position.y),
        ))

        if is_success:
            reason = "success"
        elif abs(float(cargo.position.x)) > FIELD_HALF_W + CARGO_HALF or \
                abs(float(cargo.position.y)) > FIELD_HALF_H + CARGO_HALF:
            reason = "cargo_out_of_bounds"
        elif abs(float(robot.position.x)) > FIELD_HALF_W + ROBOT_HALF_L or \
                abs(float(robot.position.y)) > FIELD_HALF_H + ROBOT_HALF_W:
            reason = "robot_out_of_bounds"
        elif self._hard_collision_count >= MAX_HARD_COLLISIONS:
            reason = "cargo_damaged"
        elif truncated:
            reason = "time_limit"
        else:
            reason = "running"

        return {
            "is_success": bool(is_success),
            "cargo_goal_distance": float(self._cargo_dock_distance()),
            "cargo_angle_error": float(angle_error),
            "cargo_speed": cargo_speed,
            "robot_cargo_distance": robot_cargo_distance,
            "contact_impulse": float(contact_impulse),
            "hard_collision_count": int(self._hard_collision_count),
            "stagnation_steps": int(self._stagnation_steps),
            "action_energy": float(self._action_energy),
            "dock_entered": bool(self._dock_entered),
            "stable_steps": int(self._stable_steps),
            "cargo_inside_dock": bool(self._cargo_fully_inside_dock()),
            "elapsed_steps": int(self._elapsed_steps),
            "time_fraction": float(min(1.0, self._elapsed_steps / self.max_episode_steps)),
            "termination_reason": reason,
        }

    # -------------------------------------------------------------- render
    def render(self):
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover
            raise gym.error.DependencyNotInstalled(
                "pygame is required for rendering FragileCargoDock-v0"
            ) from exc

        scale = 70
        width = int(2 * FIELD_HALF_W * scale) + 40
        height = int(2 * FIELD_HALF_H * scale) + 40

        if self._surface is None:
            pygame.init()
            pygame.display.init()
            self._surface = pygame.display.set_mode((width, height))
            pygame.display.set_caption("FragileCargoDock-v0")
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface((width, height))
        canvas.fill((250, 250, 248))

        def to_px(x, y):
            return int(width / 2 + x * scale), int(height / 2 - y * scale)

        # Field border
        pygame.draw.rect(canvas, (60, 60, 60),
                         pygame.Rect(*to_px(-FIELD_HALF_W, FIELD_HALF_H),
                                     int(2 * FIELD_HALF_W * scale),
                                     int(2 * FIELD_HALF_H * scale)), 2)

        # Dock
        dock_rect = pygame.Rect(0, 0, int(2 * DOCK_HALF * scale), int(2 * DOCK_HALF * scale))
        dock_rect.center = to_px(DOCK_X, DOCK_Y)
        pygame.draw.rect(canvas, (150, 220, 150), dock_rect, 2)

        # Static bodies
        for body in self._static_bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                try:
                    hw, hh = shape.vertices[2]
                except Exception:
                    continue
                cx, cy = body.position
                rect = pygame.Rect(0, 0, int(2 * hw * scale), int(2 * hh * scale))
                rect.center = to_px(cx, cy)
                pygame.draw.rect(canvas, (90, 90, 110), rect)

        # Cargo
        cargo = self._cargo_body
        cx, cy = cargo.position
        size = int(2 * CARGO_HALF * scale)
        pts = []
        for dx, dy in ((-CARGO_HALF, -CARGO_HALF), (CARGO_HALF, -CARGO_HALF),
                       (CARGO_HALF, CARGO_HALF), (-CARGO_HALF, CARGO_HALF)):
            rx = cx + dx * math.cos(cargo.angle) - dy * math.sin(cargo.angle)
            ry = cy + dx * math.sin(cargo.angle) + dy * math.cos(cargo.angle)
            pts.append(to_px(rx, ry))
        pygame.draw.polygon(canvas, (205, 150, 70), pts)

        # Cart
        robot = self._robot_body
        rx, ry = robot.position
        pts = []
        for dx, dy in ((-ROBOT_HALF_L, -ROBOT_HALF_W), (ROBOT_HALF_L, -ROBOT_HALF_W),
                       (ROBOT_HALF_L, ROBOT_HALF_W), (-ROBOT_HALF_L, ROBOT_HALF_W)):
            px = rx + dx * math.cos(robot.angle) - dy * math.sin(robot.angle)
            py = ry + dx * math.sin(robot.angle) + dy * math.cos(robot.angle)
            pts.append(to_px(px, py))
        pygame.draw.polygon(canvas, (70, 110, 200), pts)

        if self.render_mode == "human":
            self._surface.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
