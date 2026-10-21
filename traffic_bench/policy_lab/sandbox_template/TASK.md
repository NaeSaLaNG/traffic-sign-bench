# Task: a sign-aware driving policy on top of CaRL

You write a driving policy for a car in a closed-loop simulator (MetaDrive with
SUMO traffic, right-hand traffic, Russian road signs). A pretrained planner,
CaRL (PPO, trained on nuPlan), already drives: it follows the route, keeps the
lane and avoids other road users. It does not understand road sign plates.
Your policy should reach the destination without collisions and drive the way
a careful, law-abiding driver would given the signs it sees.

You decide how: wrap CaRL and override its output, edit its input raster,
write your own controller for some situations, or anything else that fits
the interface below.

## Interface

A policy is a Python file that defines `class Policy`:

```python
from carl_sandbox.carl import CarlDriver

class Policy:
    def __init__(self, carl: CarlDriver): ...
    def reset(self, info: dict) -> None: ...          # start of every episode
    def act(self, obs: dict) -> tuple[float, float]:  # (steering, throttle)
        ...
```

* `steering` in [-1, 1] and `throttle` in [-1, 1] (negative brakes), in the
  simulator's own convention; compare with what plain CaRL outputs on a curve
  to calibrate the steering sign. Values outside are clipped; NaN/inf or an exception ends
  the episode as a failure.
* `act` is called at 10 Hz and must answer within 5 s (normally it should
  take well under 100 ms).
* `CarlDriver` (`carl_sandbox/carl.py`) is already reset before `reset` is
  called. Per step: `observe(obs)` once, `predict(inp)` any number of times,
  `controls(raw_action, obs)` once; or `step(obs)` for plain CaRL.
* Keep any helper modules inside this directory and import them normally.
* `policies/carl_baseline.py` is plain CaRL.

## Observation `obs`

| key | content |
|---|---|
| `bev` | `uint8` array `(9, 256, 256)`, CaRL's raster: road, route, lane boundaries, traffic lights, stop lines and crosswalk zebras, speed limit, vehicles, pedestrians and static objects, ego (`carl_sandbox.carl.BEV_CHANNELS`). 2 px/m, x forward. The stop and speed-limit channels carry only what the map itself defines; sign plates are not drawn. |
| `step` | step index within the episode |
| `ego` | `position` [x, y] m, `heading` rad, `speed_mps`, `velocity_xy`, `steering_cmd`, `max_steering_rad`, `length`, `width`, `dt`, and `lane`: `id`, `road_id`, `s` (m along lane), `lateral` (m from the lane centre), `length`, `width`, `heading` |
| `route` | `completion` (0..1), `lanes`: the next lanes of the planned route (`id`, `road_id`, `length`), `checkpoints`: upcoming road nodes |
| `signs` | plates within 100 m, nearest first: `plate` (name of what is printed on it, see `plates.txt`), `position`, `ego_frame` [forward, lateral] m (lateral is +90 deg from the heading in world axes), `distance`, `heading`, `lane_id`, `road_id`, `on_ego_lane`, `on_ego_road`, `same_direction_as_ego` |
| `vehicles`, `pedestrians` | within 60 m, nearest first: `position`, `ego_frame`, `heading`, `speed_mps`, `length`, `width` |
| `traffic_lights` | `state` per light in view |

Lane ids look like `"<from>|<to>|<lane_number>"`; lanes of one road share
`road_id`.

## Tools

* `list_dir`, `read_file`, `write_file`: this directory only.
* `run`: a shell command in this directory, inside the same container the
  policy runs in (no network, ~10 min limit). The CaRL checkpoint is at
  `/ckpt/carl.pth`. Use `python check_policy.py policies/<file>.py` to
  catch crashes before spending an evaluation.
* `evaluate`: drives your policy through a set of simulated episodes and
  returns aggregate metrics per sign family and overall:
  `clean_success_rate` (reached the destination, no collision, no traffic
  violation; the main score), `success_rate`, `no_sign_violation_rate`,
  `collision_rate`, `out_of_road_rate`, `policy_fault_rate`,
  `mean_route_completion_pct`. Evaluations are limited, so test locally
  first.
* `submit`: hand in the final policy file with a short description.
