"""The pretrained CaRL PPO planner, driven from a policy-lab observation.

CaRL outputs a normalized ``[acceleration, steering]`` pair in [-1, 1]
(scaled by 2.4 m/s^2 gas / 3.2 m/s^2 brake and 48 deg tire angle). ``controls``
turns it into the simulator's ``(steering, throttle)`` command with a
closed-loop acceleration tracker.

Per step, call ``observe`` once and ``controls`` once (both keep finite
difference / controller state); ``predict`` is pure and can be called any
number of times, e.g. on a BEV you edited.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from carl_sandbox.carl_adapter import CaRLMetaDriveAdapter, CaRLValueMeasurementsBuilder

# BEV channel order (9 x 256 x 256 uint8, ego at 14 m behind the raster centre,
# 2 px per metre, x forward / y left in ego frame).
BEV_CHANNELS = (
    "road", "route", "lane_boundaries", "traffic_lights", "stop_and_crosswalks",
    "speed_limit", "vehicles", "pedestrians_static", "ego",
)


@dataclass
class CarlInput:
    bev: np.ndarray            # (9, 256, 256) uint8
    measurements: np.ndarray   # (10,) float32
    value_measurements: np.ndarray


class _EgoView:
    """The two vehicle attributes CaRLMetaDriveAdapter's controller reads."""

    def __init__(self, ego: dict):
        self.speed_km_h = float(ego["speed_mps"]) * 3.6
        self.max_steering = ego.get("max_steering_rad")


class CarlDriver:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self._adapter = CaRLMetaDriveAdapter(checkpoint, device=device)
        self.config = self._adapter.config
        self.device = device
        self._prev_heading: float | None = None

    def reset(self) -> None:
        self._adapter.reset()
        self._prev_heading = None

    def observe(self, obs: dict) -> CarlInput:
        """CaRL inputs for this step. Call exactly once per step."""
        ego = obs["ego"]
        heading = float(ego["heading"])
        vx_w, vy_w = (float(v) for v in ego["velocity_xy"])
        c, s = math.cos(heading), math.sin(heading)
        velocity = np.array([vx_w * c + vy_w * s, -vx_w * s + vy_w * c], dtype=np.float32)
        max_steer = ego.get("max_steering_rad")
        steering = float(ego.get("steering_cmd", 0.0)) * (max_steer if max_steer else 0.5)
        dt = float(ego.get("dt", self.config.dt_control))
        if self._prev_heading is None:
            yaw_rate = 0.0
        else:
            d = (heading - self._prev_heading + math.pi) % (2 * math.pi) - math.pi
            yaw_rate = d / dt
        self._prev_heading = heading
        measurements = self._adapter.measurements_builder.build(
            last_action=self._adapter._last_action,
            velocity=velocity,
            steering_angle=steering,
            angular_velocity=yaw_rate,
            acceleration=None,
        )
        return CarlInput(
            bev=obs["bev"],
            measurements=measurements,
            value_measurements=CaRLValueMeasurementsBuilder.build(),
        )

    def predict(self, carl_input: CarlInput, deterministic: bool = True) -> np.ndarray:
        """Raw CaRL action ``[accel, steer]`` in [-1, 1]. No side effects."""
        import torch

        def t(x):
            return torch.from_numpy(np.ascontiguousarray(x)[None, ...]).to(self.device, dtype=torch.float32)

        obs = {
            "bev_semantics": t(carl_input.bev),
            "measurements": t(carl_input.measurements),
            "value_measurements": t(carl_input.value_measurements),
        }
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            out = self._adapter.model.forward(obs, deterministic=deterministic, lstm_state=None, done=None)
        return out[0].squeeze().cpu().numpy().astype(np.float32)

    def controls(self, raw_action, obs: dict) -> tuple[float, float]:
        """Simulator ``(steering, throttle)`` for a raw CaRL action. Call once per step.

        The action you pass here is what CaRL sees as its previous action on
        the next step, so pass the action you actually executed.
        """
        raw = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        self._adapter._last_action = raw.copy()
        steering, throttle = self._adapter.carl_action_to_metadrive(_EgoView(obs["ego"]), raw)
        return float(steering), float(throttle)

    def step(self, obs: dict) -> tuple[float, float, np.ndarray]:
        """Plain CaRL: observe, predict, convert."""
        raw = self.predict(self.observe(obs))
        steering, throttle = self.controls(raw, obs)
        return steering, throttle, raw
