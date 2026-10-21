"""Baseline: the pretrained CaRL planner with nothing on top."""
from carl_sandbox.carl import CarlDriver


class Policy:
    def __init__(self, carl: CarlDriver):
        self.carl = carl

    def reset(self, info: dict) -> None:
        pass

    def act(self, obs: dict) -> tuple[float, float]:
        steering, throttle, _ = self.carl.step(obs)
        return steering, throttle
