"""
Paper Algorithm 1: MVP-Nav recursive navigation loop.
Order: Physical Perception -> (optional) VLM Reasoning -> MVM Planning -> Low-level Execution.
"""

import json
import os
from typing import Any, Dict, Optional, TextIO, Type

from src.utils.my_tools import apply_transform

from .physical_perception import run_physical_perception
from .vlm_reasoning import run_vlm_reasoning
from .mvm_planning import run_mvm_planning
from .low_level_execution import (
    run_until_midterm_reached,
    execute_finding_or_judgement_turn,
    plan_astar_to_midterm,
    check_goal_match_in_obs,
)
from .episode_video import write_episode_video
from .nav_metrics import coerce_nav_metrics, zero_nav_metrics


def _log_gssl_snapshot(graph: Any, loc_agent: Any, log_fn) -> None:
    """Append GSSL (object list without original_obj refs) to episode result log."""
    if not getattr(graph, "GSSL", None):
        graph.GSSL_gen(loc_agent)
    gssl = getattr(graph, "GSSL", None) or []
    rows = [{k: v for k, v in item.items() if k != "original_obj"} for item in gssl]
    log_fn("GSSL:\n" + json.dumps(rows, ensure_ascii=False, indent=2))


def _issue_stop_and_get_metrics(envs: Any, log_fn) -> Dict[str, float]:
    """Call Habitat STOP and return standard navigation metrics."""
    try:
        _, _, info = envs.step({"action": 0})
        metrics = coerce_nav_metrics(info)
    except Exception as exc:
        log_fn(f"STOP metric collection failed: {exc}")
        metrics = zero_nav_metrics()
    return metrics


def _configure_episode_outputs(
    agent: Any,
    graph: Any,
    args: Any,
    *,
    obs_save_folder: Optional[str],
    depth_save_folder: Optional[str],
    segment_save_folder: Optional[str],
    midterm_save_folder: Optional[str],
    shorterm_save_folder: Optional[str],
) -> None:
    if obs_save_folder:
        agent.obs_save_folder = obs_save_folder
    if depth_save_folder:
        agent.depth_save_folder = depth_save_folder
    if segment_save_folder:
        agent.segment_save_folder = segment_save_folder
    if shorterm_save_folder:
        agent.shorterm_save_folder = shorterm_save_folder
    if midterm_save_folder:
        args.midterm_save_folder = midterm_save_folder

    agent.track_main = graph.track_main
    agent.track_sub = graph.track_sub


def _make_agent_input(args: Any) -> Dict[str, Any]:
    agent_input = {
        "bev": None,
        "pose": None,
        "midterm_goal": None,
        "obs": None,
        "wait": None,
        "planned_path": None,
        "bev_step": None,
    }
    if getattr(args, "save_episode_video", False):
        agent_input["_episode_video_frames"] = []
    return agent_input


def _run_initial_scan(agent: Any, args: Any, agent_input: Dict[str, Any]) -> tuple:
    turn_round = int(360 / args.turn_angle)
    step = 0
    for _ in range(turn_round + 1):
        agent.step_count = step
        obs, done, infos = agent.control_step(3)
        agent_input["obs"] = obs
        step += 1
    step -= 1
    return step, turn_round


def _run_optional_vlm_reasoning(
    graph: Any,
    envs: Any,
    agent_input: Dict[str, Any],
    loc_agent: Any,
    use_vlm_per_stage: bool,
    log_fn,
) -> None:
    if not use_vlm_per_stage or agent_input.get("obs") is None:
        return
    try:
        current_img = agent_input["obs"][-1] if isinstance(agent_input["obs"], list) else agent_input["obs"]
        if hasattr(current_img, "numpy"):
            current_img = current_img.cpu().numpy().transpose(1, 2, 0)
        goal_img = getattr(envs, "instance_imagegoal", None)
        if goal_img is not None and hasattr(goal_img, "__array__"):
            goal_img = getattr(goal_img, "__array__", lambda: goal_img)()
        run_vlm_reasoning(graph, loc_agent, current_img, goal_img)
    except Exception as exc:
        log_fn(f"VLM per stage skip: {exc}")


def _update_agent_input_for_stage(agent_input: Dict[str, Any], envs: Any, map_inst: Any, nav_mode: str) -> None:
    agent_input["bev"] = map_inst.bev
    agent_input["pose"] = apply_transform(envs.get_sim_location(), map_inst.transform_sim2bev)
    agent_input["bev_step"] = map_inst.bev_step
    agent_input["_video_nav_mode"] = nav_mode


def _plan_explore_path(agent_input: Dict[str, Any], map_inst: Any, midterm_goal: Any, midterm_save_folder: Optional[str], stage: int):
    goal_pose = midterm_goal["pose"]
    pose_xy = (int(agent_input["pose"][0]), int(agent_input["pose"][1]))
    save_path = f"{midterm_save_folder}/stage_{stage}.png" if midterm_save_folder else None
    safety_field = getattr(map_inst, "stage_space_safety_field", None)
    planned_path = plan_astar_to_midterm(
        map_inst.bev, pose_xy, goal_pose, save_path, safety_field=safety_field
    )
    agent_input["planned_path"] = planned_path
    return planned_path


def _save_episode_artifacts(envs: Any, agent: Any, agent_input: Dict[str, Any], args: Any, log_fn) -> None:
    obs_dir = getattr(agent, "obs_save_folder", None)
    episode_dir = os.path.dirname(obs_dir) if obs_dir is not None else None

    if hasattr(envs, "save_topdown_traj") and episode_dir:
        try:
            envs.save_topdown_traj(episode_dir)
        except Exception as exc:
            log_fn(f"save_topdown_traj failed: {exc}")

    if getattr(args, "save_episode_video", False) and episode_dir:
        frames = agent_input.get("_episode_video_frames")
        if frames:
            try:
                video_path = os.path.join(episode_dir, "episode_video.mp4")
                write_episode_video(frames, video_path, fps=5.0)
                log_fn(f"Episode video saved: {video_path}")
            except Exception as exc:
                log_fn(f"save_episode_video failed: {exc}")


def _finalize_metrics(success: bool, final_metrics: Dict[str, float], envs: Any) -> tuple:
    if not success and final_metrics == zero_nav_metrics() and getattr(envs, "info", None):
        final_metrics = coerce_nav_metrics(envs.info)
        success = final_metrics["success"] > 0.0
    return success, final_metrics


def run_episode(
    episode_idx: int,
    envs: Any,
    agent: Any,
    graph: Any,
    MapClass: Type,
    model_info: Dict[str, Any],
    args: Any,
    *,
    use_vlm_per_stage: bool = False,
    obs_save_folder: Optional[str] = None,
    depth_save_folder: Optional[str] = None,
    segment_save_folder: Optional[str] = None,
    midterm_save_folder: Optional[str] = None,
    shorterm_save_folder: Optional[str] = None,
    result_file_handle: Optional[TextIO] = None,
) -> tuple:
    """
    Run a single episode (paper Algorithm 1).

    Per stage:
      1. Physical Perception: update GSSL / map and fields
      2. (optional) VLM Reasoning: GSSL scoring and NavMode
      3. MVM Planning: update_stage + get_nav_mode_and_goal_via_fuse(loc_agent), select gmid via fuse_fields_and_extract_goal
      4. Low-level Execution: A* + FMM to gmid or finding/judgement branch

    Returns:
        (success, step, metrics): Habitat-standard success, total step count and metrics.
    """
    def log(s: str) -> None:
        print(s)
        if result_file_handle is not None:
            result_file_handle.write(s + "\n")
            result_file_handle.flush()

    _configure_episode_outputs(
        agent,
        graph,
        args,
        obs_save_folder=obs_save_folder,
        depth_save_folder=depth_save_folder,
        segment_save_folder=segment_save_folder,
        midterm_save_folder=midterm_save_folder,
        shorterm_save_folder=shorterm_save_folder,
    )

    agent_input = _make_agent_input(args)
    step, turn_round = _run_initial_scan(agent, args, agent_input)

    stage = 0
    set2 = 0
    set1 = 0
    current_step = step
    success = False
    final_metrics = zero_nav_metrics()
    # Max steps per episode (default 200, overridable by args.max_episode_steps)
    max_steps = getattr(args, "max_episode_steps", 200) or 200
    map_inst = None

    # Paper Algorithm 1: strict per-stage order, no pre-trigger
    succeed_match_points = int(getattr(args, "succeed_match_points", 200))

    while step < max_steps:
        log(f"\n--- stage: {stage} ---")

        # ----- 1. Physical Perception -----
        loc_agent = envs.get_sim_location()
        map_inst, map_info = run_physical_perception(
            MapClass, set2, set1, current_step, stage, model_info, args, agent, loc_agent
        )

        # ----- 2. Graph state update (global objects, safety/explore fields) -----
        graph.update_stage(map_info)

        # ----- 3. (optional) VLM Reasoning -----
        _run_optional_vlm_reasoning(graph, envs, agent_input, loc_agent, use_vlm_per_stage, log)

        _log_gssl_snapshot(graph, loc_agent, log)

        # ----- 4. MVM Planning (fuse_fields_and_extract_goal) -----
        nav_mode, midterm_goal = run_mvm_planning(graph, loc_agent, vis=True)
        stage += 1

        _update_agent_input_for_stage(agent_input, envs, map_inst, nav_mode)
        log(nav_mode)

        # ----- 5. A* path for explore (safety field as costmap to avoid walls) -----
        if nav_mode == "explore" and midterm_goal is not None:
            planned_path = _plan_explore_path(agent_input, map_inst, midterm_goal, midterm_save_folder, stage)

            # If A* fails (start/goal in obstacle or no path), switch to recognition to rebuild map
            if planned_path is None or len(planned_path) == 0:
                log("A* planning failed, switching to recognition mode to rebuild map.")
                nav_mode = "recognition"
                midterm_goal = None

        # ----- 6. Low-level Execution (paper-aligned) -----
        if nav_mode in ["finding", "judgement", "recognition"]:
            step_delta = execute_finding_or_judgement_turn(
                agent, envs, map_inst, agent_input, turn_round, step
            )
            step += step_delta
            # Paper: after rotation in Judgement, decide arrival by goal-vs-observation match
            if nav_mode == "judgement":
                if getattr(args, "goal_type", None) == "object":
                    final_metrics = _issue_stop_and_get_metrics(envs, log)
                    success = final_metrics["success"] > 0.0
                    log(
                        "episode STOP metrics: "
                        f"success={final_metrics['success']:.3f}, "
                        f"spl={final_metrics['spl']:.3f}, "
                        f"soft_spl={final_metrics['soft_spl']:.3f}, "
                        f"distance_to_goal={final_metrics['distance_to_goal']:.3f}"
                    )
                    break
                goal_img = getattr(envs, "instance_imagegoal", None)
                if goal_img is not None:
                    success = check_goal_match_in_obs(
                        model_info,
                        agent_input.get("obs"),
                        goal_img,
                        succeed_match_points,
                    )
                    if success:
                        final_metrics = _issue_stop_and_get_metrics(envs, log)
                        log("episode success (goal match >= threshold)")
                        break
        elif nav_mode == "explore" and midterm_goal is not None:
            step_delta, done = run_until_midterm_reached(
                agent,
                envs,
                map_inst,
                agent_input,
                midterm_goal,
                args,
                step,
            )
            step += step_delta
            if done:
                final_metrics = coerce_nav_metrics(getattr(envs, "info", {}))
                success = final_metrics["success"] > 0.0
                break

        set2, set1, current_step = set1, current_step, step

    _save_episode_artifacts(envs, agent, agent_input, args, log)
    success, final_metrics = _finalize_metrics(success, final_metrics, envs)

    return success, step, final_metrics
