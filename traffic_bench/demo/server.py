"""Serve a gallery of real MetaDrive traffic-sign scenarios to a browser.

MetaDrive owns process-global Panda3D state. Each browser session therefore
runs in its own spawned worker process while this module's aiohttp server acts
as the lightweight WebSocket gateway.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from aiohttp import WSMsgType, web
from PIL import Image

from traffic_bench.eval.engine.sim.metadrive_sumo_patch import (
    apply_metadrive_sumo_via_patch,
)
from traffic_bench.eval.engine.sim.top_down_local_film_patch import (
    apply_top_down_local_film_patch,
    set_top_down_local_film,
)
from traffic_bench.eval.engine.sim.top_down_path_conflict_patch import (
    apply_top_down_destination_cap_patch,
)
from traffic_bench.eval.engine.expand.manifest_config import (
    DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
    DEFAULT_AUX_LANES_OCCUPIED_MAX,
)
from traffic_bench.eval.engine.map.lane_keys import make_lane_key
from traffic_bench.eval.engine.spawn.auxiliary_agent import (
    DEFAULT_CONVOY_GAP_M,
    DEFAULT_CONVOY_SIZE,
    DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END,
    DEFAULT_SPAWN_VELOCITY_MS,
    add_auxiliary_agents,
    resolve_aux_spawn_plan,
)
from traffic_bench.eval.run.env import (
    _apply_destination_along_cap,
    _apply_manifest_ego_destination,
    _apply_manifest_ego_spawn_lane,
    _analyze_junction_lanes,
    _ego_reached_capped_destination,
    _reposition_ego_before_lane_end,
)
from traffic_bench.eval.run.place import place_signs_for_row
from traffic_bench.eval.signs.crosswalk.place import (
    install_segment_crosswalk_geometry,
    row_is_crosswalk,
)
from traffic_bench.eval.signs.blocked.place import (
    ego_compliant_stop_before_blocked_road,
)

apply_metadrive_sumo_via_patch()
apply_top_down_local_film_patch()
apply_top_down_destination_cap_patch()

DEFAULT_DOCS_ROOT = Path("docs")
DEFAULT_ORIGINS = {
    "http://127.0.0.1:8765",
    "http://localhost:8765",
    "https://emb-ai.github.io",
}
CONTROL_KEYS = ("left", "right", "up", "down")


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    data_subdir: str
    title: str
    violation: str
    penalty: str
    source_subdir: str | None = None
    scene_index: int | None = None
    row_overrides: tuple[tuple[str, Any], ...] = ()

    @property
    def manifest(self) -> Path:
        source = self.source_subdir or self.data_subdir
        return Path("data/runs") / source / "test" / "real_manifest.jsonl"

    @property
    def scenes_root(self) -> Path:
        return Path("data/scenes") / (self.source_subdir or self.data_subdir)

    def prepare_row(self, row: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(row)
        prepared.update(dict(self.row_overrides))
        return prepared


# Demo gallery: four signs only. Crosswalk uses a compact Moscow segment and
# places ego ~40 m before the zebra so the crossing/peds are in frame.
_DEMO_CROSSWALK_PEDS: dict[str, Any] = {
    "enabled": True,
    "spawn_mode": "ego_proximity",
    "ego_spawn_distance_m": 90.0,
    "target_pedestrian_count": 3,
    "pedestrian_spawn_gap_s": 1.2,
    "pedestrian_spawn_chain": "time_gap",
    "initial_pedestrians": 0,
    "max_pedestrians": 3,
    "spawn_by_interval": False,
    "spawn_probability": 0.0,
    "crossing_interval_range": [5.0, 10.0],
    "max_active_per_crosswalk": 3,
    "speed_mean": 1.9,
    "speed_std": 0.25,
    "yield_distance": 8.0,
    "no_stop_before_crosswalk_m": 3.0,
    "yield_to_vehicles": True,
    "yield_on_crosswalk": False,
}

SCENARIO_SPECS = (
    ScenarioSpec(
        "yield", "yield", "Yield",
        "Failed to yield to traffic on the priority road",
        "",
        row_overrides=(
            # Red finish marker ~60 m along the exit lane.
            ("destination_max_along_m", 60.0),
        ),
    ),
    ScenarioSpec(
        "crosswalk", "crosswalk", "Pedestrian crossing",
        "Failed to yield to a pedestrian on the crossing",
        "",
        # Different road from the default index-0 late_slow scene; ego is
        # repositioned before the zebra in DemoSession._prepare_crosswalk_spawn.
        scene_index=40,
        row_overrides=(
            ("pedestrian_preset_name", "three_gap_55"),
            ("pedestrian_count", 3),
            ("pedestrian_manager", _DEMO_CROSSWALK_PEDS),
            ("auxiliary_agent", False),
        ),
    ),
    ScenarioSpec(
        "detour", "detour_right", "Detour",
        "Passed the obstacle on the prohibited side",
        "",
        source_subdir="main_road",
        row_overrides=(
            ("pdd_code", "4.2.1"), ("sign_code", "4.2.1"),
            ("sign_type", "detour"), ("sign_family", "detour"),
            ("place_detour_sign", True), ("sign_s", 145.0),
            ("sign_lane_index", 1), ("spawn_lane_num", 1),
            ("spawn_distance_before_end", 90.0),
            ("auxiliary_agent", False),
            ("traffic_density", 0.0),
            ("show_npc_vehicles", False),
        ),
    ),
    ScenarioSpec(
        "speed_limit", "speed_limit", "Maximum speed",
        "Exceeded the posted maximum speed",
        "",
        source_subdir="blocked_road",
        row_overrides=(
            ("pdd_code", "3.24"), ("sign_code", "3.24"),
            ("sign_type", "speed"), ("place_speed_sign", True),
            ("v_target_kmh", 20.0), ("sign_s", 170.0),
            ("spawn_distance_before_end", 70.0),
            ("spawn_velocity_ms", 8.0), ("auxiliary_agent", False),
        ),
    ),
)
SCENARIOS_BY_ID = {scenario.id: scenario for scenario in SCENARIO_SPECS}
FILM_HALF_M = 18.0
CROSSWALK_APPROACH_M = 14.0
# Finish markers measured ahead of the demo ego spawn (segment scenarios).
CROSSWALK_DEST_AHEAD_M = 40.0
DETOUR_DEST_AHEAD_M = 65.0  # +10 m past the previous demo finish
SPEED_DEST_AHEAD_M = 60.0
# Detour checker normally starts ~15 m before cones; demo only fails on contact.
DETOUR_CONTACT_ZONE_M = 2.5
# Same-edge demos need the along-cap arrive check (MetaDrive route is degenerate).
SAME_EDGE_ARRIVE_SCENARIOS = frozenset({"crosswalk", "detour", "speed_limit"})


def load_manifest_row(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("scene index must be non-negative")
    with path.open("r", encoding="utf-8") as source:
        for current, line in enumerate(source):
            if current == index:
                row = json.loads(line)
                if not row.get("valid", True):
                    raise ValueError(f"manifest row {index} is marked invalid")
                return row
    raise IndexError(f"manifest {path} has no row {index}")


def controls_to_action(controls: dict[str, bool]) -> list[float]:
    # MetaDrive uses positive steering for left and negative for right.
    steering = float(controls["left"]) - float(controls["right"])
    throttle_brake = float(controls["up"]) - float(controls["down"])
    return [steering, throttle_brake]


class DemoSession:
    """One deterministic browser-controlled episode."""

    def __init__(
        self,
        scenario: ScenarioSpec,
        row: dict[str, Any],
        scenes_root: Path,
        *,
        screen_size: tuple[int, int] = (960, 960),
        jpeg_quality: int = 90,
    ) -> None:
        self.scenario = scenario
        self.row = dict(row)
        # Demo gallery keeps the road quiet unless a scenario opts into traffic.
        self.row["traffic_density"] = float(self.row.get("traffic_density") or 0.0)
        if row_is_crosswalk(self.row):
            self._prepare_crosswalk_spawn()
        self.scenes_root = scenes_root.resolve()
        self.screen_size = screen_size
        self.jpeg_quality = jpeg_quality
        self.env = None
        self.aux_agent_mgr = None
        self.controls = {key: False for key in CONTROL_KEYS}
        self.last_control_at = time.monotonic()
        self.steps = 0
        self.violation = False
        self.violation_checker: str | None = None
        self.compliant_stop_steps = 0
        self.reached = False
        self.crashed = False
        self.timed_out = False
        self.finished = False
        self.last_info: dict[str, Any] = {}
        self._build()

    @property
    def base_env(self):
        return self.env

    def _build(self) -> None:
        from traffic_bench.eval.run.env import _build_sumo_env

        net_path = self.scenes_root / str(self.row["net_path"])
        if not net_path.is_file():
            raise FileNotFoundError(
                f"scene map not found: {net_path}. Download the matching scene subset first."
            )

        max_steps = int(self.row.get("horizon_steps") or self.row.get("horizon") or 600)
        self.env = _build_sumo_env(
            self.row,
            scenes_root=self.scenes_root,
            max_steps=max_steps,
        )
        try:
            env_seed = (
                int(self.row.get("sign_id") or 0)
                + int(self.row.get("var_idx") or 0)
            ) % 100000
            self.env.reset(seed=env_seed)
            _apply_manifest_ego_spawn_lane(self.base_env, self.row)
            spawn_distance = float(
                self.row.get("spawn_distance_before_end") or 0.0
            )
            if spawn_distance > 0:
                _reposition_ego_before_lane_end(self.base_env, spawn_distance)
            _apply_manifest_ego_destination(self.base_env, self.row)
            _apply_destination_along_cap(self.base_env, self.row)
            place_signs_for_row(
                self.base_env,
                self.row,
                scenes_root=self.scenes_root,
                distance_before_end=float(
                    self.row.get("sign_distance_before_end", 20.0)
                ),
                show_model=True,
            )
            # Episode runner installs the zebra after reset; the demo must too
            # or the top-down film has no crossing and no pedestrians.
            if row_is_crosswalk(self.row):
                install_segment_crosswalk_geometry(self.base_env, self.row)
                self._prime_extra_crosswalk_pedestrians()
            if self.scenario.id == "detour":
                self._soften_detour_approach_rule()
                # After signs exist so we can park the finish past the cones.
                self._set_destination_ahead_of_ego(DETOUR_DEST_AHEAD_M)
            elif self.scenario.id == "speed_limit":
                self._set_destination_ahead_of_ego(SPEED_DEST_AHEAD_M)
            self._spawn_auxiliary_agents()
            self._update_follow_cam()
        except Exception:
            self.close()
            raise

    def _set_destination_ahead_of_ego(self, ahead_m: float) -> None:
        """Put the red finish marker ``ahead_m`` metres ahead on the ego lane.

        Detour/speed manifests route through a later edge; the demo instead
        finishes on the current lane so the marker is visible on the GIF.
        Along-metres are MetaDrive coordinates — do not SUMO-remap them.
        """
        vehicle = self.base_env.vehicle
        lane = getattr(vehicle, "lane", None)
        nav = getattr(vehicle, "navigation", None)
        if lane is None:
            return
        lane_index = getattr(lane, "index", None)
        if lane_index is None:
            return
        try:
            along = float(lane.local_coordinates(vehicle.position)[0])
            lane_len = float(lane.length)
        except Exception:
            return
        if lane_len <= 1.0:
            return
        dest = min(max(along + float(ahead_m), along + 5.0), max(0.5, lane_len - 2.0))
        # Detour: the 40 m mark lands on the cone cluster; push the finish past
        # the obstacle so the red marker is not covered by cones/sign icons.
        if self.scenario.id == "detour":
            manager = getattr(self.base_env.engine, "traffic_sign_manager", None)
            for sign in getattr(manager, "signs", []) or []:
                if "Detour" not in type(sign).__name__:
                    continue
                try:
                    obstacle = float(getattr(sign, "obstacle_long", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if obstacle > 0.0:
                    dest = max(dest, min(lane_len - 2.0, obstacle + 12.0))
        self.row["destination_lane_id"] = str(lane_index)
        self.row["destination_max_along_m"] = float(dest)

        # Same-lane route: red marker + arrive check stay on the driven road.
        try:
            if nav is not None:
                nav.set_route(lane_index, lane_index)
        except Exception as exc:
            print(f"[EgoDest] same-lane route failed: {exc}")

        final = getattr(nav, "final_lane", None) if nav is not None else None
        if final is None:
            final = lane
        try:
            vehicle._priority_bench_dest_along_m = float(dest)
        except Exception:
            pass
        try:
            if nav is not None:
                nav._priority_bench_dest_along_m = float(dest)
        except Exception:
            pass
        try:
            from metadrive.utils.coordinates_shift import panda_vector

            dest_path = getattr(nav, "_dest_node_path", None) if nav is not None else None
            if dest_path is not None:
                check_point = final.position(float(dest), 0.0)
                height = float(getattr(nav, "MARK_HEIGHT", 1.0) or 1.0)
                dest_path.setPos(
                    panda_vector(float(check_point[0]), float(check_point[1]), height)
                )
        except Exception:
            pass
        print(
            f"[EgoDest] Demo destination {ahead_m:.0f}m ahead at {dest:.1f}m "
            f"on {lane_index} (len={lane_len:.1f}m)"
        )

    def _soften_detour_approach_rule(self) -> None:
        """Only treat driving into the cone cluster as a detour violation."""
        manager = getattr(self.base_env.engine, "traffic_sign_manager", None)
        if manager is None:
            return
        for sign in getattr(manager, "signs", []) or []:
            if "Detour" not in type(sign).__name__:
                continue
            obstacle = float(getattr(sign, "obstacle_long", 0.0) or 0.0)
            lane_len = float(getattr(getattr(sign, "lane", None), "length", 0.0) or 0.0)
            sign.zone_start = max(0.0, obstacle - DETOUR_CONTACT_ZONE_M)
            sign.zone_end = (
                min(lane_len, obstacle + DETOUR_CONTACT_ZONE_M)
                if lane_len > 0.0
                else obstacle + DETOUR_CONTACT_ZONE_M
            )
            sign.zone_length = max(0.0, sign.zone_end - sign.zone_start)
            sign.violation_long = min(
                lane_len - 0.1 if lane_len > 0.1 else obstacle,
                obstacle,
            )

    def _prime_extra_crosswalk_pedestrians(self) -> None:
        """Schedule a few more walkers so the crossing reads clearly at start."""
        engine = getattr(self.base_env, "engine", None)
        pm = getattr(engine, "pedestrian_manager", None) if engine else None
        if pm is None or not getattr(pm, "_crosswalks", None):
            return
        cw_id = getattr(pm, "_ego_trigger_crosswalk_id", None) or next(
            iter(pm._crosswalks)
        )
        target = min(
            int(getattr(pm, "target_pedestrian_count", 1) or 1),
            int(getattr(pm, "max_pedestrians", 1) or 1),
            3,
        )
        while int(getattr(pm, "_ego_spawns_scheduled", 0) or 0) < target:
            if not pm._schedule_track(cw_id, on_crosswalk=True, immediate=True):
                break
            pm._ego_spawns_scheduled = int(pm._ego_spawns_scheduled) + 1
            pm._ego_trigger_crosswalk_id = cw_id
        if hasattr(pm, "_spawn_due_tracks"):
            pm._spawn_due_tracks()

    def _prepare_crosswalk_spawn(self) -> None:
        """Put ego ~CROSSWALK_APPROACH_M before the zebra (manifest spawns past it)."""
        try:
            edge_len = float(self.row.get("edge_length_m") or 0.0)
            zebra_s = float(self.row.get("crosswalk_position_m") or 0.0)
        except (TypeError, ValueError):
            return
        if edge_len <= 0.0 or zebra_s <= CROSSWALK_APPROACH_M:
            return
        ego_along = max(5.0, zebra_s - CROSSWALK_APPROACH_M)
        self.row["spawn_distance_before_end"] = max(5.0, edge_len - ego_along)
        # Red finish marker ~40 m ahead of the demo spawn.
        dest_cap = min(edge_len - 2.0, ego_along + CROSSWALK_DEST_AHEAD_M)
        self.row["destination_max_along_m"] = dest_cap

    def _update_follow_cam(self) -> None:
        vehicle = self.base_env.vehicle
        # Same tight follow-cam as yield for every gallery scenario.
        heading = float(getattr(vehicle, "heading_theta", 0.0) or 0.0)
        look_ahead = FILM_HALF_M * 0.35
        cx = float(vehicle.position[0]) + look_ahead * float(np.cos(heading))
        cy = float(vehicle.position[1]) + look_ahead * float(np.sin(heading))
        set_top_down_local_film((cx, cy), half_m=FILM_HALF_M)

    def _spawn_auxiliary_agents(self) -> None:
        if not bool(self.row.get("auxiliary_agent", True)):
            return

        incoming_lanes, outgoing_lanes = _analyze_junction_lanes(self.base_env)
        distance = float(
            self.row.get(
                "aux_distance_from_intersection",
                DEFAULT_AUX_DISTANCE_FROM_INTERSECTION,
            )
        )
        convoy_size = int(
            self.row.get("aux_convoy_size", DEFAULT_CONVOY_SIZE)
        )
        convoy_gap = float(
            self.row.get("aux_convoy_gap_m", DEFAULT_CONVOY_GAP_M)
        )
        lanes_occupied = int(
            self.row.get(
                "aux_lanes_occupied",
                DEFAULT_AUX_LANES_OCCUPIED_MAX,
            )
        )
        ego_lane_index = make_lane_key(
            str(self.row["road_id"]),
            int(self.row.get("spawn_lane_num", 0) or 0),
        )
        (
            spawn_lanes,
            destination_lanes,
            alternate_spawn_dest_map,
            spawn_longs,
            ring_circulate,
        ) = resolve_aux_spawn_plan(
            self.row,
            ego_lane_index=str(ego_lane_index),
            incoming_lanes=incoming_lanes,
            aux_lanes_occupied=lanes_occupied,
            aux_distance_from_intersection=distance,
            scenes_root=self.scenes_root,
        )
        if not spawn_lanes:
            raise RuntimeError("demo manifest produced no auxiliary spawn lanes")

        release_distance = max(
            float(DEFAULT_EGO_RELEASE_DISTANCE_BEFORE_END),
            float(self.row.get("spawn_distance_before_end", 0.0) or 0.0),
        )
        self.aux_agent_mgr = add_auxiliary_agents(
            self.base_env,
            spawn_lane_indices=spawn_lanes,
            outgoing_lanes=outgoing_lanes,
            distance_from_intersection=distance,
            policy="idm",
            spawn_velocity_ms=float(
                self.row.get(
                    "aux_spawn_velocity_ms",
                    DEFAULT_SPAWN_VELOCITY_MS,
                )
            ),
            destination_lanes=[destination or None for destination in destination_lanes],
            ego_vehicle=self.base_env.vehicle,
            ego_spawn_lane_index=ego_lane_index,
            ego_release_distance_before_end=release_distance,
            convoy_size=convoy_size,
            convoy_gap_m=convoy_gap,
            alternate_spawn_dest_map=alternate_spawn_dest_map,
            spawn_longitudinal_by_lane=spawn_longs,
            ring_circulate_by_lane=ring_circulate,
            junction_layout=self.row.get("junction_layout"),
        )
        if self.aux_agent_mgr is None:
            raise RuntimeError("failed to create demo auxiliary agents")

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
                self.aux_agent_mgr = None
        set_top_down_local_film(None)

    def reset(self) -> None:
        self.close()
        self.controls = {key: False for key in CONTROL_KEYS}
        self.last_control_at = time.monotonic()
        self.steps = 0
        self.violation = False
        self.violation_checker = None
        self.compliant_stop_steps = 0
        self.reached = False
        self.crashed = False
        self.timed_out = False
        self.finished = False
        self.last_info = {}
        self._build()

    def set_controls(self, message: dict[str, Any]) -> None:
        for key in CONTROL_KEYS:
            self.controls[key] = bool(message.get(key, False))
        self.last_control_at = time.monotonic()

    def release_stale_controls(self) -> None:
        if time.monotonic() - self.last_control_at > 0.6:
            for key in CONTROL_KEYS:
                self.controls[key] = False

    def _ego_stop_sign(self):
        manager = getattr(self.base_env.engine, "traffic_sign_manager", None)
        vehicle = self.base_env.vehicle
        if manager is None:
            return None
        for sign in manager.signs:
            if type(sign).__name__ != "StopSign":
                continue
            try:
                if sign._is_on_sign_road(vehicle):
                    return sign
            except Exception:
                continue
        return None

    def _check_violation(self) -> None:
        manager = getattr(self.base_env.engine, "traffic_sign_manager", None)
        if manager is None:
            return
        for checker, violated in manager.check_all_violations(
            self.base_env.vehicle,
            for_reward=False,
        ):
            if violated:
                self.violation = True
                self.violation_checker = type(checker).__name__
                return

    def step(self) -> None:
        if self.finished:
            return
        self.release_stale_controls()
        _, _, terminated, truncated, info = self.env.step(
            controls_to_action(self.controls)
        )
        self.steps += 1
        self.last_info = dict(info)
        self._check_violation()

        vehicle = self.base_env.vehicle
        blocked_road_success = False
        if self.scenario.id == "blocked_road" and not self.violation:
            stop_ok = ego_compliant_stop_before_blocked_road(
                self.base_env,
                vehicle,
                max_dist_before_sign_m=float(
                    self.row.get("compliant_stop_max_dist_m", 12.0) or 12.0
                ),
                speed_max_mps=float(
                    self.row.get("compliant_stop_speed_mps", 0.5) or 0.5
                ),
            )
            self.compliant_stop_steps = (
                self.compliant_stop_steps + 1 if stop_ok else 0
            )
            engine_config = self.base_env.engine.global_config
            decision_dt = float(engine_config["physics_world_step_size"]) * float(
                engine_config["decision_repeat"]
            )
            required_steps = max(
                1,
                int(
                    round(
                        float(
                            self.row.get(
                                "compliant_stop_success_seconds", 3.0
                            )
                            or 3.0
                        )
                        / decision_dt
                    )
                ),
            )
            blocked_road_success = self.compliant_stop_steps >= required_steps

        destination_cap = float(
            getattr(vehicle, "_priority_bench_dest_along_m", 0.0) or 0.0
        )
        reached_cap = _ego_reached_capped_destination(
            vehicle,
            max_along_m=destination_cap,
            allow_same_lane=self.scenario.id in SAME_EDGE_ARRIVE_SCENARIOS,
        )
        self.reached = bool(
            info.get("arrive_dest", False)
            or reached_cap
            or blocked_road_success
        )
        self.crashed = bool(
            info.get("crash", False) or info.get("out_of_road", False)
        )
        self.timed_out = bool(truncated and not self.reached)
        self.finished = bool(
            self.violation
            or self.reached
            or self.crashed
            or terminated
            or truncated
        )

    def _rule_status(self) -> tuple[float | None, str, bool]:
        vehicle = self.base_env.vehicle
        manager = getattr(self.base_env.engine, "traffic_sign_manager", None)
        if manager is None:
            return None, "Checker active", False

        if self.scenario.id == "crosswalk":
            for rule in manager.rules:
                if not hasattr(rule, "get_status"):
                    continue
                try:
                    status = rule.get_status(vehicle)
                except Exception:
                    continue
                distance = status.get("target_distance_m")
                if status.get("must_stop"):
                    return distance, "Pedestrian on crossing — yield", False
                if distance is not None:
                    return distance, "Crossing ahead", False
            return None, "Watch for pedestrians", False

        distance_to_line = None
        stopped = False
        for sign in manager.signs:
            line = getattr(
                sign,
                "stop_line_position",
                getattr(
                    sign,
                    "sign_line_position",
                    getattr(sign, "placement_long", None),
                ),
            )
            lane = getattr(sign, "lane", None)
            if line is None or lane is None:
                continue
            try:
                along = float(lane.local_coordinates(vehicle.position)[0])
                half_length = 0.5 * float(
                    getattr(vehicle, "LENGTH", 0.0) or 0.0
                )
                candidate = float(line) - (along + half_length)
                if distance_to_line is None or abs(candidate) < abs(distance_to_line):
                    distance_to_line = candidate
                    state = getattr(sign, "_vehicle_states_stop", {}).get(
                        vehicle.id, {}
                    )
                    stopped = bool(state.get("stopped_before_line", False))
            except Exception:
                continue

        if self.scenario.id == "stop":
            progress = (
                "Complete stop registered"
                if stopped
                else "Stop before the line"
            )
        elif self.scenario.id == "yield":
            progress = "Check priority traffic, then proceed"
        elif self.scenario.id in {"no_entry", "blocked_road"}:
            progress = "Do not cross the restriction line"
        elif self.scenario.id == "speed_limit":
            progress = "Brake to 20 km/h before entering the zone"
        elif self.scenario.id == "detour":
            progress = "Move right to pass the obstacle"
        else:
            progress = "Follow the signed rule"
        return distance_to_line, progress, stopped

    def telemetry(self) -> dict[str, Any]:
        vehicle = self.base_env.vehicle
        distance_to_line, progress, stopped = self._rule_status()

        if self.violation:
            state = "violation"
            message = self.scenario.violation
        elif self.crashed:
            state = "violation"
            message = "Collision or departure from the drivable road"
        elif self.reached:
            state = "safe"
            message = "Compliant destination reached"
        elif self.timed_out:
            state = "violation"
            message = "Time expired before the destination was reached"
        else:
            state = "ready"
            message = progress

        return {
            "type": "telemetry",
            "state": state,
            "message": message,
            "speed_mps": round(float(vehicle.speed), 2),
            "distance_to_line_m": (
                round(distance_to_line, 1) if distance_to_line is not None else None
            ),
            "stopped": stopped,
            "step": self.steps,
            "finished": self.finished,
            "result": (
                "passed"
                if self.reached and not self.violation and not self.crashed
                else "failed"
                if self.finished
                else None
            ),
            "scenario_id": self.scenario.id,
            "scenario_title": self.scenario.title,
            "violation_type": self.violation_checker,
            "penalty": None,
            "show_penalty": False,
            "scene": self.row.get("scene_id"),
            "latitude": self.row.get("latitude"),
            "longitude": self.row.get("longitude"),
            "auxiliary_agents": (
                int(self.aux_agent_mgr.get_status().get("count", 0))
                if self.aux_agent_mgr is not None
                else 0
            ),
        }

    def render_jpeg(self) -> bytes:
        self._update_follow_cam()
        telemetry = self.telemetry()
        frame = self.base_env.render(
            mode="top_down",
            film_size=(2400, 2400),
            # Match yield's close ego-centered crop for all gallery scenarios.
            scaling=16.0,
            screen_size=self.screen_size,
            semantic_map=True,
            semantic_broken_line=True,
            draw_target_vehicle_trajectory=False,
            target_agent_heading_up=True,
            screen_record=False,
            window=False,
            text={
                "MetaDrive": str(self.row.get("scene_id", "Moscow")),
                "Speed": f"{telemetry['speed_mps'] * 3.6:.1f} km/h",
                "Checker": telemetry["state"],
            },
        )
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
        output = io.BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=False,
            subsampling=0,
        )
        return output.getvalue()


def run_session_worker(
    connection: Any,
    scenario: ScenarioSpec,
    row: dict[str, Any],
    scenes_root: Path,
    frame_interval: float,
    idle_timeout: float,
) -> None:
    """Run one MetaDrive engine in an isolated process."""
    session: DemoSession | None = None
    try:
        session = DemoSession(scenario, row, scenes_root)
        connection.send(
            (
                "json",
                {
                    "type": "ready",
                    "scenario_id": scenario.id,
                    "scenario_title": scenario.title,
                    "scene": row.get("scene_id"),
                    "sign": row.get("pdd_code"),
                },
            )
        )
        last_activity_at = time.monotonic()
        next_frame = last_activity_at

        while True:
            while connection.poll():
                command = connection.recv()
                command_type = command.get("type")
                if command_type == "close":
                    return
                if command_type == "control":
                    session.set_controls(command)
                    last_activity_at = time.monotonic()
                elif command_type == "restart":
                    session.reset()
                    last_activity_at = time.monotonic()
                    next_frame = last_activity_at

            if idle_timeout > 0 and time.monotonic() - last_activity_at >= idle_timeout:
                connection.send(
                    (
                        "json",
                        {
                            "type": "error",
                            "code": "idle_timeout",
                            "message": (
                                "The demo session was inactive for too long. "
                                "Reconnect to start a new session."
                            ),
                        },
                    )
                )
                return

            session.step()
            connection.send(("json", session.telemetry()))
            connection.send(("frame", session.render_jpeg()))

            next_frame += frame_interval
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Do not accumulate lag when rendering takes longer than a frame.
                next_frame = time.monotonic()
    except (BrokenPipeError, ConnectionResetError, EOFError):
        pass
    except Exception as exc:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, EOFError):
            connection.send(
                (
                    "json",
                    {
                        "type": "error",
                        "code": "worker_error",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
    finally:
        if session is not None:
            session.close()
        connection.close()


class DemoServer:
    def __init__(
        self,
        *,
        scenarios: dict[str, tuple[ScenarioSpec, dict[str, Any], Path]],
        default_scenario: str,
        docs_root: Path,
        fps: float,
        allowed_origins: set[str],
        allow_any_origin: bool,
        max_sessions: int,
        idle_timeout: float,
    ) -> None:
        self.scenarios = scenarios
        self.default_scenario = default_scenario
        self.docs_root = docs_root
        self.frame_interval = 1.0 / fps
        self.allowed_origins = allowed_origins
        self.allow_any_origin = allow_any_origin
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout
        self.active_sessions = 0
        self.process_context = mp.get_context("spawn")

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/api/health", self.health)
        app.router.add_get("/api/drive", self.drive)
        app.router.add_get("/api/drive/{scenario}", self.drive)
        app.router.add_get("/", self.index)
        app.router.add_static("/static", self.docs_root / "static")
        return app

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "simulator": "MetaDrive",
                "scenarios": sorted(self.scenarios),
                "default_scenario": self.default_scenario,
                "active_sessions": self.active_sessions,
                "max_sessions": self.max_sessions,
                "available_sessions": self.max_sessions - self.active_sessions,
                "busy": self.active_sessions >= self.max_sessions,
            }
        )

    async def index(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(self.docs_root / "index.html")

    def _origin_allowed(self, request: web.Request) -> bool:
        origin = request.headers.get("Origin")
        if self.allow_any_origin or origin is None:
            return True
        if origin in self.allowed_origins:
            return True
        # Same-origin via node IP / hostname (e.g. http://10.x.x.x:8765).
        host = request.headers.get("Host")
        if host:
            for scheme in ("http://", "https://"):
                if origin == f"{scheme}{host}":
                    return True
        # Local / Cursor / Codespaces port-forward origins vary; allow private
        # LAN and localhost browsers talking to this demo host.
        if origin.startswith(("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost")):
            return True
        if origin.startswith(("http://10.", "https://10.", "http://192.168.", "https://192.168.")):
            return True
        return False

    async def drive(self, request: web.Request) -> web.StreamResponse:
        if not self._origin_allowed(request):
            raise web.HTTPForbidden(text="WebSocket origin is not allowed")

        scenario_id = (
            request.match_info.get("scenario")
            or request.query.get("scenario")
            or self.default_scenario
        )
        selected = self.scenarios.get(scenario_id)
        if selected is None:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json(
                {
                    "type": "error",
                    "code": "unknown_scenario",
                    "message": f"Unknown scenario: {scenario_id}",
                }
            )
            await ws.close(code=1008, message=b"unknown scenario")
            return ws

        # No await between the check and increment: aiohttp runs both atomically
        # on this event loop, so the hard capacity cannot be oversubscribed.
        if self.active_sessions >= self.max_sessions:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json(
                {
                    "type": "error",
                    "code": "capacity",
                    "message": (
                        f"All {self.max_sessions} demo sessions are currently in use. "
                        "Please try again shortly."
                    ),
                }
            )
            await ws.close(code=1013, message=b"demo capacity reached")
            return ws

        self.active_sessions += 1
        ws = web.WebSocketResponse(heartbeat=15, max_msg_size=16_384)
        parent_connection = None
        child_connection = None
        process = None
        receiver: asyncio.Task | None = None
        worker_read: asyncio.Task | None = None
        try:
            await ws.prepare(request)
            scenario, row, scenes_root = selected
            parent_connection, child_connection = self.process_context.Pipe()
            process = self.process_context.Process(
                target=run_session_worker,
                args=(
                    child_connection,
                    scenario,
                    row,
                    scenes_root,
                    self.frame_interval,
                    self.idle_timeout,
                ),
                name=f"traffic-demo-{scenario.id}",
                daemon=True,
            )
            process.start()
            child_connection.close()
            child_connection = None

            async def receive_commands() -> None:
                async for message in ws:
                    if message.type == WSMsgType.TEXT:
                        try:
                            payload = json.loads(message.data)
                        except json.JSONDecodeError:
                            continue
                        try:
                            parent_connection.send(payload)
                        except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
                            break
                    elif message.type in (
                        WSMsgType.CLOSE,
                        WSMsgType.CLOSED,
                        WSMsgType.ERROR,
                    ):
                        break

            receiver = asyncio.create_task(receive_commands())
            while not ws.closed and not receiver.done():
                worker_read = asyncio.create_task(
                    asyncio.to_thread(parent_connection.recv)
                )
                done, _pending = await asyncio.wait(
                    (worker_read, receiver),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receiver in done:
                    worker_read.cancel()
                    break

                try:
                    packet_type, payload = worker_read.result()
                except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
                    if not ws.closed:
                        await ws.send_json(
                            {
                                "type": "error",
                                "code": "worker_stopped",
                                "message": "The simulator worker stopped unexpectedly.",
                            }
                        )
                    break
                worker_read = None

                if packet_type == "json":
                    if payload.get("type") == "ready":
                        payload["active_sessions"] = self.active_sessions
                        payload["max_sessions"] = self.max_sessions
                    await ws.send_json(payload)
                elif packet_type == "frame":
                    await ws.send_bytes(payload)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        except Exception as exc:
            if not ws.closed:
                await ws.send_json(
                    {
                        "type": "error",
                        "code": "gateway_error",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
        finally:
            if receiver is not None:
                receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver
            if worker_read is not None:
                worker_read.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker_read
            if parent_connection is not None:
                with contextlib.suppress(
                    BrokenPipeError, ConnectionResetError, EOFError, OSError
                ):
                    parent_connection.send({"type": "close"})
            if process is not None:
                await asyncio.to_thread(process.join, 2.0)
                if process.is_alive():
                    process.terminate()
                    await asyncio.to_thread(process.join, 2.0)
                process.close()
            if parent_connection is not None:
                parent_connection.close()
            if child_connection is not None:
                child_connection.close()
            self.active_sessions -= 1
            if not ws.closed:
                await ws.close()
        return ws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TrafficSignBench browser challenge gallery."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=8,
        help="Maximum number of concurrent isolated simulator processes.",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=120.0,
        help="Close sessions with no control input after this many seconds; 0 disables.",
    )
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument(
        "--default-scenario",
        choices=sorted(SCENARIOS_BY_ID),
        default="yield",
    )
    parser.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS_ROOT)
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="Additional exact browser Origin allowed to open the WebSocket.",
    )
    parser.add_argument("--allow-any-origin", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build, step and render one frame without starting a server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (1.0 <= args.fps <= 20.0):
        raise SystemExit("--fps must be between 1 and 20")
    if not (1 <= args.max_sessions <= 64):
        raise SystemExit("--max-sessions must be between 1 and 64")
    if args.idle_timeout < 0:
        raise SystemExit("--idle-timeout must be non-negative")
    scenarios = {
        scenario.id: (
            scenario,
            scenario.prepare_row(
                load_manifest_row(
                    scenario.manifest,
                    (
                        scenario.scene_index
                        if scenario.scene_index is not None
                        else args.scene_index
                    ),
                )
            ),
            scenario.scenes_root,
        )
        for scenario in SCENARIO_SPECS
    }

    if args.smoke_test:
        scenario, row, scenes_root = scenarios[args.default_scenario]
        session = DemoSession(scenario, row, scenes_root)
        try:
            session.step()
            frame = session.render_jpeg()
            print(
                json.dumps(
                    {
                        "ok": True,
                        "jpeg_bytes": len(frame),
                        "telemetry": session.telemetry(),
                    }
                )
            )
        finally:
            session.close()
        return

    server = DemoServer(
        scenarios=scenarios,
        default_scenario=args.default_scenario,
        docs_root=args.docs_root,
        fps=args.fps,
        allowed_origins=DEFAULT_ORIGINS | set(args.allowed_origin),
        allow_any_origin=args.allow_any_origin,
        max_sessions=args.max_sessions,
        idle_timeout=args.idle_timeout,
    )
    web.run_app(server.make_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
