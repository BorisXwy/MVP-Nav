"""
Paper §III-F Low-level Execution Loop: A* + sliding-window FMM -> short-term goal gst, semantic ground reprojection safety check.
Wraps Agent.step, control_step, and A* path generation.
Judgement success: after rotation capture, decide arrival by match count (>= succeed_match_points) between goal image and observations.
"""

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.utils.my_tools import apply_transform
from src.utils.fmm.my_fmm import compute_astar_path
from src.core.episode_video import record_episode_frame, record_episode_frame_finding_turn


def check_goal_match_in_obs(
    model_info: Dict[str, Any],
    obs_list: Any,
    goal_image: Any,
    threshold: int,
) -> bool:
    """
    Paper-aligned: after rotation in Judgement, run LightGlue match between goal image and observation sequence;
    success if any view has match count >= threshold.

    Args:
        model_info: must contain 'extractor', 'matcher', 'device'
        obs_list: list of observations, each numpy (H,W,3) or tensor (3,H,W)
        goal_image: goal image, PIL or numpy (H,W,3)
        threshold: succeed_match_points threshold

    Returns:
        True if at least one frame has match count >= threshold
    """
    extractor = model_info.get("extractor")
    matcher = model_info.get("matcher")
    device = model_info.get("device", torch.device("cpu"))
    if extractor is None or matcher is None:
        return False
    try:
        from lightglue.utils import numpy_image_to_torch
    except ImportError:
        return False

    def _to_tensor(img) -> torch.Tensor:
        if hasattr(img, "numpy"):
            img = img.cpu().numpy()
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:
                img = img.transpose(2, 0, 1)
            return torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0).unsqueeze(0).to(device)
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
            if img.ndim == 3 and img.shape[2] == 3:
                img = img.transpose(2, 0, 1)
            return torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0).unsqueeze(0).to(device)
        return img

    # Flatten obs: may be list of list or list of array
    frames = []
    if isinstance(obs_list, list):
        for x in obs_list:
            if isinstance(x, list):
                frames.extend(x)
            else:
                frames.append(x)
    else:
        frames = [obs_list]

    goal_tensor = _to_tensor(goal_image)
    if goal_tensor.dim() == 3:
        goal_tensor = goal_tensor.unsqueeze(0).to(device)

    for obs_frame in frames:
        if obs_frame is None:
            continue
        cur_tensor = _to_tensor(obs_frame)
        if cur_tensor.dim() == 3:
            cur_tensor = cur_tensor.unsqueeze(0).to(device)
        try:
            feats0 = extractor.extract(cur_tensor)
            feats1 = extractor.extract(goal_tensor)
            matches01 = matcher({"image0": feats0, "image1": feats1})
            if isinstance(matches01, dict) and "matches" in matches01:
                m = matches01["matches"]
                if hasattr(m, "shape"):
                    # May have batch dim (1, K, 2) or (K, 2)
                    num_matches = m.shape[1] if m.dim() == 3 else m.shape[0]
                else:
                    num_matches = len(m)
            else:
                num_matches = 0
            if num_matches >= threshold:
                return True
        except Exception:
            continue
    return False


def run_until_midterm_reached(
    agent: Any,
    envs: Any,
    map_inst: Any,
    agent_input: Dict,
    midterm_goal: Any,
    args: Any,
    current_step: int,
    max_steps_to_goal: int = 30,
    *,
    pretrigger_epsilon_bev: Optional[float] = None,
    on_pretrigger: Optional[Callable] = None,
    pretrigger_context: Optional[Dict] = None,
) -> tuple:
    """
    Low-level execution loop: follow A* path via FMM short-term goals until gmid reached or step limit.

    Optional pre-trigger: when distance to g_mid < pretrigger_epsilon_bev, call on_pretrigger once
    and write next-stage result to agent_input["_next_stage_result"] for seamless transition.

    Returns:
        (step_delta, done): step increment for this segment, whether episode ended.
    """
    # For safety check: pass BEV->Sim transform to agent for g_st reprojection to image
    if getattr(map_inst, "transform_bev2sim", None) is not None:
        agent_input["transform_bev2sim"] = map_inst.transform_bev2sim

    agent_input["midterm_goal"] = midterm_goal["pose"]
    pose_xy = np.array(agent_input["pose"][:2])
    goal_xy = np.array(agent_input["midterm_goal"][:2])
    dcor2goal = np.linalg.norm(pose_xy - goal_xy)
    pose_yaw = agent_input["pose"][2] if len(agent_input["pose"]) > 2 else 0.0
    goal_yaw = midterm_goal["pose"][2] if len(midterm_goal["pose"]) > 2 else 0.0
    dyaw2goal = math.degrees(goal_yaw - pose_yaw)

    use_pretrigger = (
        getattr(args, "use_pretrigger", False)
        and pretrigger_epsilon_bev is not None
        and on_pretrigger is not None
        and pretrigger_context is not None
    )
    epsilon = float(pretrigger_epsilon_bev) if pretrigger_epsilon_bev is not None else 3.0

    dstep = 0
    done = False
    step = current_step
    while (dcor2goal >= map_inst.bev_step or abs(dyaw2goal) > args.turn_angle) and dstep < max_steps_to_goal:
        step += 1
        dstep += 1
        agent_input["_video_pretrigger_this_step"] = False
        # Pre-trigger: run next-stage high-level reasoning once when close to g_mid
        if (
            use_pretrigger
            and dcor2goal < epsilon
            and agent_input.get("_next_stage_result") is None
        ):
            agent_input["_video_pretrigger_this_step"] = True
            try:
                on_pretrigger(
                    agent_input=agent_input,
                    envs=envs,
                    map_inst=map_inst,
                    step=step,
                    **pretrigger_context,
                )
            except Exception:
                pass

        agent.step_count = step
        # Shorterm/FMM debug images: my_fmm.get_local_goal_fmm only saves when visualize=True
        shorterm_dir = getattr(agent, "shorterm_save_folder", None)
        visualize_shorterm = bool(shorterm_dir)
        obs, done, info = agent.step(agent_input, visualize=visualize_shorterm)
        if getattr(args, "save_episode_video", False) and agent_input.get("_episode_video_frames") is not None:
            record_episode_frame(
                agent_input=agent_input,
                envs=envs,
                map_inst=map_inst,
                midterm_goal=midterm_goal,
                agent=agent,
                args=args,
                decision_action_id=getattr(agent, "last_action", 1),
                nav_mode=agent_input.get("_video_nav_mode", "explore"),
            )
        if done:
            break
        agent_input["pose"] = apply_transform(envs.get_sim_location(), map_inst.transform_sim2bev)
        agent_input["obs"] = obs
        dcor2goal = np.linalg.norm(
            np.array(agent_input["pose"][:2]) - np.array(agent_input["midterm_goal"][:2])
        )
        pose_yaw = agent_input["pose"][2] if len(agent_input["pose"]) > 2 else 0.0
        dyaw2goal = math.degrees(goal_yaw - pose_yaw)

    return dstep, done


def execute_finding_or_judgement_turn(
    agent: Any,
    envs: Any,
    map_inst: Any,
    agent_input: Dict,
    turn_round: int,
    current_step: int,
) -> int:
    """
    finding/judgement branch: rotate in place one full turn to collect observations; does not decide success.

    Returns:
        step_delta: step increment for this segment (turn_round + 1).
    """
    step_delta = 0
    nav_mode = agent_input.get("_video_nav_mode", "finding")
    for i in range(turn_round + 1):
        agent.step_count = current_step + 1 + i
        obs, done, infos = agent.control_step(3)
        agent_input["pose"] = apply_transform(envs.get_sim_location(), map_inst.transform_sim2bev)
        agent_input["obs"] = obs
        if getattr(agent, "args", None) and getattr(agent.args, "save_episode_video", False):
            if agent_input.get("_episode_video_frames") is not None:
                record_episode_frame_finding_turn(agent_input, map_inst, agent, agent.args, nav_mode)
        step_delta += 1
    return step_delta


def plan_astar_to_midterm(
    bev: np.ndarray,
    pose_xy: tuple,
    goal_pose,
    save_path: Optional[str] = None,
    safety_field: Optional[np.ndarray] = None,
) -> Optional[list]:
    """
    Generate A* path for explore mode.
    - If safety_field is given, use as costmap: lower safety (near walls) -> higher cost to avoid walls.
    - Normally returns list of path points;
    - Returns None when A* fails (start/goal in obstacle or no path); caller may switch to recognition to rebuild map.
    """
    bev_map = bev.transpose(1, 0, 2).copy()
    path = compute_astar_path(
        bev_map=bev_map,
        start=pose_xy,
        goal=goal_pose,
        visualize=True,
        save_path=save_path,
        safety_field=safety_field,
    )
    # compute_astar_path returns [] on failure; normalize to None for caller
    if path is None or len(path) == 0:
        return None
    return path
