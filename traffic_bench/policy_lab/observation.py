"""Judge-side observation for sandboxed policies.

Everything here is plain data: the CaRL BEV raster, ego kinematics, the route
ahead, nearby agents and the sign plates the ego can see. No sign object, zone,
threshold or violation check crosses into the sandbox.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork

from traffic_bench.agents.adapters.carl.carl_adapter import CaRLConfig, CaRLRenderer
from traffic_bench.agents.adapters.carl.metadrive_extractor import MetaDriveExtractor

SIGN_FOV_M = 100.0
AGENT_FOV_M = 60.0
ROUTE_LANES_AHEAD = 6

_warned_node_network = False


class SignFreeExtractor(MetaDriveExtractor):
    """MetaDriveExtractor whose BEV carries nothing read from benchmark signs.

    Stop plates are dropped from the stop channel. Speed limits come from the
    map data of an EdgeRoadNetwork, which signs never write; on a
    NodeRoadNetwork the extractor would read ``lane.speed_limit``, which speed
    signs overwrite, so that channel is left empty there.
    """

    def get_stop_signs(self, ego_pos: np.ndarray) -> list[dict]:
        return []

    def get_speed_limits(self, ego_pos: np.ndarray) -> list[dict]:
        global _warned_node_network
        if isinstance(self.engine.current_map.road_network, EdgeRoadNetwork):
            return super().get_speed_limits(ego_pos)
        if not _warned_node_network:
            print("[policy_lab] NodeRoadNetwork: speed channel left empty (lane limits carry sign overrides)")
            _warned_node_network = True
        return []


def _lane_id(lane) -> str | None:
    index = getattr(lane, "index", None)
    if index is None:
        return None
    if isinstance(index, (tuple, list)):
        return "|".join(str(p) for p in index)
    return str(index)


def _road_id(lane) -> str | None:
    index = getattr(lane, "index", None)
    if isinstance(index, (tuple, list)) and len(index) >= 2:
        return f"{index[0]}|{index[1]}"
    return None


def _wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _to_ego_frame(xy, ego_xy, ego_heading: float) -> list[float]:
    dx = float(xy[0]) - float(ego_xy[0])
    dy = float(xy[1]) - float(ego_xy[1])
    c, s = math.cos(ego_heading), math.sin(ego_heading)
    return [dx * c + dy * s, -dx * s + dy * c]


def _plate(sign) -> str:
    """What is printed on the plate, the way a detector would name it."""
    icon = getattr(sign, "icon_path", None)
    if icon:
        return Path(str(icon)).stem
    return type(sign).__name__


def _lane_geometry(lane, position) -> dict[str, Any] | None:
    if lane is None:
        return None
    try:
        long, lat = lane.local_coordinates(position)
    except Exception:
        return None
    width = None
    try:
        width = float(lane.width_at(long))
    except Exception:
        width = getattr(lane, "width", None)
    return {
        "id": _lane_id(lane),
        "road_id": _road_id(lane),
        "s": float(long),
        "lateral": float(lat),
        "length": float(getattr(lane, "length", 0.0) or 0.0),
        "width": float(width) if width is not None else None,
        "heading": float(lane.heading_theta_at(long)) if hasattr(lane, "heading_theta_at") else None,
    }


def ego_state(vehicle, dt: float) -> dict[str, Any]:
    max_steer = getattr(vehicle, "max_steering", None)
    try:
        max_steer = float(max_steer)
        max_steer = math.radians(max_steer) if max_steer > math.pi else max_steer
    except (TypeError, ValueError):
        max_steer = None
    return {
        "position": [float(v) for v in vehicle.position[:2]],
        "heading": float(vehicle.heading_theta),
        "speed_mps": float(vehicle.speed_km_h) / 3.6,
        "velocity_xy": [float(v) for v in np.asarray(vehicle.velocity)[:2]],
        "steering_cmd": float(getattr(vehicle, "steering", 0.0) or 0.0),
        "max_steering_rad": max_steer,
        "length": float(getattr(vehicle, "LENGTH", 4.5)),
        "width": float(getattr(vehicle, "WIDTH", 2.0)),
        "lane": _lane_geometry(getattr(vehicle, "lane", None), vehicle.position),
        "dt": float(dt),
    }


def route_ahead(vehicle) -> dict[str, Any]:
    nav = getattr(vehicle, "navigation", None)
    out: dict[str, Any] = {"lanes": [], "completion": None}
    if nav is None:
        return out
    try:
        out["completion"] = float(nav.route_completion)
    except Exception:
        pass
    try:
        ref = list(getattr(nav, "current_ref_lanes", []) or [])
        nxt = list(getattr(nav, "next_ref_lanes", []) or [])
        seen = set()
        for lane in ref + nxt:
            lid = _lane_id(lane)
            if lid is None or lid in seen:
                continue
            seen.add(lid)
            out["lanes"].append({
                "id": lid,
                "road_id": _road_id(lane),
                "length": float(getattr(lane, "length", 0.0) or 0.0),
            })
            if len(out["lanes"]) >= ROUTE_LANES_AHEAD:
                break
        checkpoints = list(getattr(nav, "checkpoints", []) or [])
        out["checkpoints"] = [str(c) for c in checkpoints[:ROUTE_LANES_AHEAD * 2]]
    except Exception:
        pass
    return out


def visible_signs(engine, vehicle) -> list[dict[str, Any]]:
    mgr = getattr(engine, "traffic_sign_manager", None)
    signs = list(getattr(mgr, "signs", None) or [])
    ego_xy = np.asarray(vehicle.position[:2], dtype=np.float64)
    ego_heading = float(vehicle.heading_theta)
    ego_lane = getattr(vehicle, "lane", None)
    ego_lane_id, ego_road_id = _lane_id(ego_lane), _road_id(ego_lane)
    out = []
    for sign in signs:
        if sign is None or getattr(sign, "_is_destroyed", False):
            continue
        try:
            pos = np.asarray(sign.position[:2], dtype=np.float64)
        except Exception:
            continue
        dist = float(np.linalg.norm(pos - ego_xy))
        if dist > SIGN_FOV_M:
            continue
        lane = getattr(sign, "lane", None)
        heading = getattr(sign, "heading_theta", None)
        facing = None
        if lane is not None:
            try:
                long, _ = lane.local_coordinates(pos)
                lane_heading = float(lane.heading_theta_at(long))
                facing = abs(_wrap(lane_heading - ego_heading)) < math.pi / 2
            except Exception:
                pass
        out.append({
            "plate": _plate(sign),
            "position": [float(pos[0]), float(pos[1])],
            "ego_frame": _to_ego_frame(pos, ego_xy, ego_heading),
            "distance": dist,
            "heading": float(heading) if heading is not None else None,
            "lane_id": _lane_id(lane),
            "road_id": _road_id(lane),
            "on_ego_lane": _lane_id(lane) is not None and _lane_id(lane) == ego_lane_id,
            "on_ego_road": _road_id(lane) is not None and _road_id(lane) == ego_road_id,
            "same_direction_as_ego": facing,
        })
    out.sort(key=lambda s: s["distance"])
    return out


def _agents(items: list[dict], ego_xy, ego_heading: float) -> list[dict]:
    out = []
    for it in items:
        pos = np.asarray(it["position"], dtype=np.float64)[:2]
        if np.linalg.norm(pos - ego_xy) > AGENT_FOV_M:
            continue
        out.append({
            "position": [float(pos[0]), float(pos[1])],
            "ego_frame": _to_ego_frame(pos, ego_xy, ego_heading),
            "heading": float(it.get("heading", 0.0) or 0.0),
            "speed_mps": float(it.get("speed", 0.0) or 0.0),
            "length": float(it.get("length", 0.0) or 0.0),
            "width": float(it.get("width", 0.0) or 0.0),
        })
    out.sort(key=lambda a: math.hypot(*a["ego_frame"]))
    return out


class ObservationBuilder:
    """Per-episode builder: CaRL BEV without sign hints + structured state."""

    def __init__(self, engine, fov_range: float = 100.0):
        self.config = CaRLConfig()
        self.renderer = CaRLRenderer(self.config)
        self.extractor = SignFreeExtractor(engine, fov_range=fov_range)
        self.engine = engine

    def build(self, vehicle, step: int) -> tuple[dict[str, Any], bytes]:
        data = self.extractor.extract_all(vehicle)
        ego = data["ego_state"]
        bev = self.renderer.render(
            ego_pos=ego["position"],
            ego_heading=ego["heading"],
            ego_speed=ego["speed"],
            ego_length=ego.get("length", 4.5),
            ego_width=ego.get("width", 2.0),
            road_polygons=data["road_polygons"],
            route_polygons=data["route_polygons"],
            lane_boundaries=data["lane_boundaries"],
            traffic_lights=data["traffic_lights"],
            stop_signs=data["stop_signs"],
            crosswalks=data.get("crosswalks"),
            speed_limits=data["speed_limits"],
            vehicles=data["vehicles"],
            pedestrians=data["pedestrians"],
            static_objects=data.get("static_objects"),
        )
        bev = np.ascontiguousarray(bev, dtype=np.uint8)
        ego_xy = np.asarray(vehicle.position[:2], dtype=np.float64)
        heading = float(vehicle.heading_theta)
        lights = []
        for tl in data.get("traffic_lights") or []:
            state = tl.get("state") if isinstance(tl, dict) else None
            lights.append({"state": str(state) if state is not None else "unknown"})
        header = {
            "type": "step",
            "step": int(step),
            "ego": ego_state(vehicle, self.config.dt_control),
            "route": route_ahead(vehicle),
            "signs": visible_signs(self.engine, vehicle),
            "vehicles": _agents(data["vehicles"], ego_xy, heading),
            "pedestrians": _agents(data["pedestrians"], ego_xy, heading),
            "traffic_lights": lights,
            "bev_shape": list(bev.shape),
            "bev_dtype": "uint8",
        }
        return header, bev.tobytes()
