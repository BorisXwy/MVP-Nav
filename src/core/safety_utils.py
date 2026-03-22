"""
Real-time safety check: reproject short-term goal g_st from BEV to current image plane for ground-mask check.
Projection uses Habitat camera intrinsics (from sim RGB sensor config); BEV->Sim gives g_st world 3D, then project with Habitat intrinsics+extrinsics.
"""

from typing import Optional, Tuple

import numpy as np

from src.utils.my_tools import apply_transform


def _get_habitat_rgb_camera_intrinsics(envs):
    """
    Read RGB camera intrinsics (width, height, hfov) from Habitat env config.
    These are the actual sensor params used by the simulator, not BEV/map coords.
    """
    try:
        from habitat.config.default import get_agent_config
        sim_config = envs.habitat_env.habitat_config.habitat.simulator
        agent_config = get_agent_config(sim_config, agent_id=0)
        sensors = agent_config.sim_sensors
        # Support different keys: rgb_sensor / depth_sensor etc.
        rgb_cfg = getattr(sensors, "rgb_sensor", None) or sensors.get("rgb_sensor", None)
        if rgb_cfg is None and hasattr(sensors, "keys"):
            for k, v in sensors.items():
                if "rgb" in k.lower() or (hasattr(v, "hfov") and hasattr(v, "width")):
                    rgb_cfg = v
                    break
        if rgb_cfg is None:
            return None
        width = int(getattr(rgb_cfg, "width", 640))
        height = int(getattr(rgb_cfg, "height", 480))
        hfov_deg = float(getattr(rgb_cfg, "hfov", 79))
        return width, height, hfov_deg
    except Exception:
        return None


def bev_point_to_image(
    envs,
    transform_bev2sim: dict,
    stg_bev_xy: Tuple[float, float],
    camera_height: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
    hfov: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """
    Project short-term goal g_st (given in BEV pixels) onto current camera image plane.

    - Use only transform_bev2sim to go from BEV pixels -> Sim 2D -> Habitat world 3D for true scene position.
    - Camera intrinsics (width, height, hfov) from Habitat RGB sensor config; width/height/hfov args are fallback.
    - Extrinsics: camera pose from envs.habitat_env.sim.get_agent_state(0).
    - Consistent with get_sim_location: sim_x = -hab_z, sim_y = -hab_x, so hab_x = -sim_y, hab_z = -sim_x.

    Args:
        envs: Habitat env (InstanceImageGoal_Env)
        transform_bev2sim: BEV -> Sim similarity transform (only for g_st world 3D)
        stg_bev_xy: short-term goal (x, z) in BEV pixels
        camera_height: ground/camera height (m) for g_st 3D height
        width, height, hfov: optional; if not passed, read from Habitat RGB sensor config

    Returns:
        Image coords (u, v), or None if point is behind camera or not visible
    """
    # 1) Camera intrinsics: prefer Habitat RGB sensor config
    intrinsics = _get_habitat_rgb_camera_intrinsics(envs)
    if intrinsics is not None:
        w_hab, h_hab, hfov_deg = intrinsics
        width = width if width is not None else w_hab
        height = height if height is not None else h_hab
        hfov = hfov if hfov is not None else hfov_deg
    else:
        width = width or 640
        height = height or 480
        hfov = hfov if hfov is not None else 79.0

    # 2) g_st: BEV pixels -> Sim 2D (m) -> Habitat world 3D (coordinate transform only, not intrinsics)
    x_sim, z_sim, _ = apply_transform(
        (float(stg_bev_xy[0]), float(stg_bev_xy[1]), 0.0), transform_bev2sim
    )
    hab_x = -z_sim
    hab_z = -x_sim
    hab_y = camera_height
    point_hab = np.array([hab_x, hab_y, hab_z], dtype=np.float64)

    # 3) Extrinsics: current agent (camera) pose from Habitat
    try:
        agent_state = envs.habitat_env.sim.get_agent_state(0)
    except Exception:
        return None
    cam_hab = np.array(agent_state.position, dtype=np.float64)

    try:
        import quaternion
        R = quaternion.as_rotation_matrix(agent_state.rotation)  # body to world
    except Exception:
        return None
    R = np.array(R, dtype=np.float64)
    p_body = R.T @ (point_hab - cam_hab)

    # Habitat camera forward is -Z; visible point has p_body[2] < 0
    depth = -float(p_body[2])
    if depth <= 1e-6:
        return None

    # 4) Perspective projection with Habitat intrinsics (same as Habitat rendering: fx = (width/2)/tan(hfov/2))
    hfov_rad = np.deg2rad(float(hfov))
    fx = (float(width) / 2.0) / np.tan(hfov_rad / 2.0)
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    u = cx + fx * p_body[0] / depth
    v = cy + fx * p_body[1] / depth
    return (float(u), float(v))


def is_point_in_floor_mask(
    u: float, v: float, floor_mask: np.ndarray, margin: int = 2
) -> bool:
    """
    Check whether image coords (u, v) fall inside floor mask (traversable).
    Optional margin dilates neighborhood to avoid boundary false negatives.

    Args:
        u, v: image coordinates (float)
        floor_mask: (H, W), non-zero = traversable
        margin: pixel neighborhood radius; if any pixel in range is floor, treat as traversable

    Returns:
        True if g_st is in traversable region
    """
    H, W = floor_mask.shape
    i, j = int(round(v)), int(round(u))
    if i < 0 or i >= H or j < 0 or j >= W:
        return False
    if margin <= 0:
        return floor_mask[i, j] > 0
    i0, i1 = max(0, i - margin), min(H, i + margin + 1)
    j0, j1 = max(0, j - margin), min(W, j + margin + 1)
    return np.any(floor_mask[i0:i1, j0:j1] > 0)
