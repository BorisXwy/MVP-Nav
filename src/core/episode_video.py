"""
Save summary video after each episode: each frame has observation, semantic segmentation, current BEV, per-step pose and short/mid-term goals;
optional safety visualization (short-term goal projection and floor, Safe/Unsafe).
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.safety_utils import bev_point_to_image


# Action ID -> label
ACTION_STR = {0: "Stop", 1: "Forward", 2: "Left", 3: "Right"}


def _resize_to_max(img: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((max_h, max_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(max_h / max(1, h), max_w / max(1, w))
    nw, nh = int(w * scale), int(h * scale)
    if nw <= 0 or nh <= 0:
        return np.zeros((max_h, max_w, 3), dtype=np.uint8)
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_RGBA2BGR)
    else:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR) if out.shape[2] == 3 else out
    # Center on (max_h, max_w)
    pad_top = (max_h - nh) // 2
    pad_left = (max_w - nw) // 2
    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    canvas[:] = 30
    canvas[pad_top : pad_top + nh, pad_left : pad_left + nw] = out
    return canvas


def _draw_bev_with_poses(
    bev: np.ndarray,
    pose: Tuple[float, float, float],
    midterm_pose: Optional[Tuple[float, float]],
    shortterm_pose: Optional[Tuple[float, float]],
) -> np.ndarray:
    """Draw robot pose, mid-term goal, short-term goal on BEV."""
    if bev is None or bev.size == 0:
        return np.zeros((400, 400, 3), dtype=np.uint8)
    # Consistent with other BEV vis: internal array is (X, Z, C), display as (Z, X, C), X horizontal, Z vertical.
    if bev.ndim == 2:
        vis_raw = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    else:
        vis_raw = bev.copy() if bev.shape[2] >= 3 else cv2.cvtColor(bev[:, :, 0], cv2.COLOR_GRAY2BGR)
    vis = np.ascontiguousarray(np.transpose(vis_raw, (1, 0, 2)))
    H, W = vis.shape[0], vis.shape[1]
    # OpenCV uses (x, y) = (col, row); col = BEV X, row = BEV Z.
    def to_pt(x: float, z: float) -> Tuple[int, int]:
        c = int(np.clip(x, 0, W - 1))
        r = int(np.clip(z, 0, H - 1))
        return (c, r)

    # Robot position and heading
    px, pz, theta = pose[0], pose[1], pose[2]
    pt_agent = to_pt(px, pz)
    cv2.circle(vis, pt_agent, 8, (0, 255, 0), 2)
    arrow_len = 25
    dx = arrow_len * np.cos(theta)
    dz = arrow_len * np.sin(theta)
    pt_end = to_pt(px + dx, pz + dz)
    cv2.arrowedLine(vis, pt_agent, pt_end, (0, 255, 0), 2)
    cv2.putText(vis, "agent", (pt_agent[0] + 10, pt_agent[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mid-term goal
    if midterm_pose is not None:
        pt_mid = to_pt(float(midterm_pose[0]), float(midterm_pose[1]))
        cv2.circle(vis, pt_mid, 10, (255, 0, 0), 2)
        cv2.putText(vis, "g_mid", (pt_mid[0] + 5, pt_mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Short-term goal
    if shortterm_pose is not None:
        pt_short = to_pt(float(shortterm_pose[0]), float(shortterm_pose[1]))
        cv2.circle(vis, pt_short, 6, (0, 165, 255), 2)
        cv2.putText(vis, "g_st", (pt_short[0] + 5, pt_short[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    return vis


def _build_text_panel(
    decision_str: str,
    mode_str: str,
    height: int,
    width: int,
) -> np.ndarray:
    """Build text info panel (decision, mode)."""
    panel = np.ones((height, width, 3), dtype=np.uint8) * 240
    y = 30
    cv2.putText(panel, f"Decision: {decision_str}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    y += 40
    cv2.putText(panel, f"Mode: {mode_str}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return panel


def record_episode_frame(
    agent_input: Dict,
    envs: Any,
    map_inst: Any,
    midterm_goal: Any,
    agent: Any,
    args: Any,
    decision_action_id: int,
    nav_mode: str,
) -> None:
    """
    Append current step frame data to agent_input["_episode_video_frames"].
    Content: observation, segmentation, BEV, per-step pose and short/mid-term goals; includes safety_ok if safety compute/vis is on.
    Only when args.save_episode_video is True.
    """
    if not getattr(args, "save_episode_video", False):
        return
    frames = agent_input.get("_episode_video_frames")
    if frames is None:
        return

    obs_list = agent_input.get("obs") or []
    raw_obs = obs_list[-1] if obs_list else None
    if raw_obs is not None and hasattr(raw_obs, "cpu"):
        raw_obs = raw_obs.cpu().numpy().transpose(1, 2, 0)
        if raw_obs.max() <= 1.0:
            raw_obs = (raw_obs * 255).astype(np.uint8)
        else:
            raw_obs = raw_obs.astype(np.uint8)
    elif raw_obs is not None and isinstance(raw_obs, np.ndarray) and raw_obs.ndim == 3:
        raw_obs = raw_obs.astype(np.uint8)
    else:
        raw_obs = None

    segment_four = None
    if len(obs_list) >= 4:
        segment_four = []
        for i in range(4):
            img = obs_list[i]
            if isinstance(img, np.ndarray):
                segment_four.append(img)
            else:
                segment_four.append(np.zeros((100, 100, 3), dtype=np.uint8))
        if len(segment_four) == 4:
            pass
        else:
            segment_four = None
    ground_mask = getattr(agent, "_last_ground_mask", None)

    pose = agent_input.get("pose")
    if pose is not None:
        pose = (float(pose[0]), float(pose[1]), float(pose[2]) if len(pose) > 2 else 0.0)
    midterm_pose = None
    if midterm_goal is not None and isinstance(midterm_goal, dict) and "pose" in midterm_goal:
        p = midterm_goal["pose"]
        if hasattr(p, "__len__") and len(p) >= 2:
            midterm_pose = (float(p[0]), float(p[1]))
    shortterm_pose = getattr(agent, "shorterm_goal", None)

    st_proj_uv = getattr(agent, "_last_st_proj_uv", None)
    if st_proj_uv is None and shortterm_pose is not None:
        transform_bev2sim = agent_input.get("transform_bev2sim") or getattr(map_inst, "transform_bev2sim", None)
        if transform_bev2sim is not None and envs is not None:
            try:
                st_proj_uv = bev_point_to_image(
                    envs,
                    transform_bev2sim,
                    shortterm_pose,
                    getattr(args, "camera_height", 0.88),
                )
            except Exception:
                pass

    decision_str = ACTION_STR.get(decision_action_id, f"action_{decision_action_id}")
    safety_ok = getattr(agent, "_last_safety_ok", None)

    frame_data = {
        "raw_obs": raw_obs,
        "segment_four": segment_four,
        "ground_mask": ground_mask,
        "bev": map_inst.bev.copy() if map_inst is not None and getattr(map_inst, "bev", None) is not None else None,
        "pose": pose,
        "midterm_pose": midterm_pose,
        "shortterm_pose": shortterm_pose,
        "st_proj_uv": st_proj_uv,
        "safety_ok": safety_ok,
        "decision_str": decision_str,
        "mode_str": nav_mode,
    }
    frames.append(frame_data)


def write_episode_video(
    frames: List[Dict],
    output_path: str,
    fps: float = 5.0,
    panel_h: int = 320,
    panel_w: int = 480,
) -> None:
    """
    Compose frame list into video and write to output_path.
    """
    if not frames:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Frame layout: top [obs+floor proj | segment 2x2], bottom [BEV+pose goals | text]
    h1, w1 = panel_h, panel_w
    ncols, nrows = 2, 2
    out_h, out_w = nrows * h1, ncols * w1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for fd in frames:
        raw = fd.get("raw_obs")
        seg_four = fd.get("segment_four")
        ground_mask = fd.get("ground_mask")
        bev = fd.get("bev")
        pose = fd.get("pose")
        midterm_pose = fd.get("midterm_pose")
        shortterm_pose = fd.get("shortterm_pose")
        st_uv = fd.get("st_proj_uv")
        safety_ok = fd.get("safety_ok")
        decision_str = fd.get("decision_str", "")
        mode_str = fd.get("mode_str", "")

        # 1) Observation + short-term goal projection + floor mask; label Safe/Unsafe if safety result present
        if raw is not None and raw.size > 0:
            obs_vis = np.ascontiguousarray(np.asarray(raw))
            if obs_vis.shape[2] == 4:
                obs_vis = np.ascontiguousarray(obs_vis[:, :, :3])
            if obs_vis.shape[2] != 3:
                obs_vis = cv2.cvtColor(obs_vis, cv2.COLOR_GRAY2BGR)
            if st_uv is not None:
                u, v = int(round(st_uv[0])), int(round(st_uv[1]))
                if 0 <= u < obs_vis.shape[1] and 0 <= v < obs_vis.shape[0]:
                    color = (0, 200, 0) if safety_ok is True else (0, 0, 255) if safety_ok is False else (0, 255, 255)
                    cv2.circle(obs_vis, (u, v), 12, color, 2)
            if ground_mask is not None and ground_mask.size > 0:
                mask_3 = np.stack([ground_mask.astype(np.uint8) * 80] * 3, axis=-1)
                if mask_3.shape[:2] != obs_vis.shape[:2]:
                    mask_3 = cv2.resize(mask_3, (obs_vis.shape[1], obs_vis.shape[0]))
                obs_vis = cv2.addWeighted(obs_vis, 1.0, mask_3, 0.3, 0)
            if safety_ok is not None:
                label = "Safe" if safety_ok else "Unsafe"
                cv2.putText(obs_vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0) if safety_ok else (0, 0, 255), 2)
        else:
            obs_vis = np.zeros((h1, w1, 3), dtype=np.uint8)
        panel1 = _resize_to_max(obs_vis, h1, w1)

        # 2) Semantic segmentation 2x2
        if seg_four and len(seg_four) >= 4:
            seg_small = []
            for s in seg_four:
                if s is not None and s.size > 0:
                    seg_small.append(_resize_to_max(s, h1 // 2, w1 // 2))
                else:
                    seg_small.append(np.zeros((h1 // 2, w1 // 2, 3), dtype=np.uint8))
            top = np.hstack([seg_small[0], seg_small[1]])
            bot = np.hstack([seg_small[2], seg_small[3]])
            panel2 = np.vstack([top, bot])
        else:
            panel2 = np.zeros((h1, w1, 3), dtype=np.uint8)
            cv2.putText(panel2, "No segment", (w1 // 4, h1 // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

        # 3) BEV + per-step pose and short/mid-term goals
        bev_vis = _draw_bev_with_poses(bev, pose or (0, 0, 0), midterm_pose, shortterm_pose)
        panel3 = _resize_to_max(bev_vis, h1, w1)

        # 4) Text (decision, mode)
        panel4 = _build_text_panel(decision_str, mode_str, h1, w1)

        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        frame = np.ascontiguousarray(np.vstack([top_row, bottom_row]))
        writer.write(frame)

    writer.release()


def record_episode_frame_finding_turn(
    agent_input: Dict,
    map_inst: Any,
    agent: Any,
    args: Any,
    nav_mode: str,
) -> None:
    """Record one frame per step in finding/judgement branch (no midterm/shortterm, decision=right turn)."""
    if not getattr(args, "save_episode_video", False):
        return
    frames = agent_input.get("_episode_video_frames")
    if frames is None:
        return
    record_episode_frame(
        agent_input=agent_input,
        envs=getattr(agent, "envs", None),
        map_inst=map_inst,
        midterm_goal=None,
        agent=agent,
        args=args,
        decision_action_id=3,
        nav_mode=nav_mode,
    )
