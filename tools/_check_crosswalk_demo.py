import numpy as np
from PIL import Image

from traffic_bench.demo.server import (
    FILM_HALF_M,
    SCENARIOS_BY_ID,
    DemoSession,
    load_manifest_row,
)
from traffic_bench.eval.signs.crosswalk.place import _invalidate_topdown_background

spec = SCENARIOS_BY_ID["crosswalk"]
row = spec.prepare_row(load_manifest_row(spec.manifest, spec.scene_index))
session = DemoSession(spec, row, spec.scenes_root)
try:
    vehicle = session.base_env.vehicle
    cx, cy = float(vehicle.position[0]), float(vehicle.position[1])
    print("FILM_HALF_M", FILM_HALF_M)
    print("ego", cx, cy, "heading", float(vehicle.heading_theta))
    cw = session.base_env.engine.current_map.crosswalks["segment_cw_5_19"]
    poly = np.asarray(cw["polygon"], dtype=np.float64)
    center = poly.mean(axis=0)
    print("ego_to_zebra", float(np.linalg.norm(center[:2] - np.array([cx, cy]))))
    print("peds", len(session.base_env.engine.pedestrian_manager.spawned_objects))
    _invalidate_topdown_background(session.base_env)
    print("renderer_after_invalidate", session.base_env.engine.top_down_renderer)
    frame = session.render_jpeg()
    open("/tmp/cw2.jpg", "wb").write(frame)
    im = np.asarray(Image.open("/tmp/cw2.jpg"))
    white = int(((im[:, :, 0] > 230) & (im[:, :, 1] > 230) & (im[:, :, 2] > 230)).sum())
    yellow = int(((im[:, :, 0] > 200) & (im[:, :, 1] > 180) & (im[:, :, 2] < 120)).sum())
    print("white_px", white, "yellow_px", yellow, "jpeg", len(frame))
finally:
    session.close()
