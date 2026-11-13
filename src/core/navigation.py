"""
Paper Algorithm 1: MVP-Nav recursive navigation loop.
Order: Physical Perception -> (optional) VLM Reasoning -> MVM Planning -> Low-level Execution.
"""

import json
import os
from typing import Any, Dict, Optional, TextIO, Type

import numpy as np

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
from .failure_diagnostics import FailureDiagnostics
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


def _run_initial_scan(
    agent: Any,
    args: Any,
    agent_input: Dict[str, Any],
    on_step=None,
) -> tuple:
    turn_round = int(360 / args.turn_angle)
    step = 0
    for _ in range(turn_round + 1):
        agent.step_count = step
        obs, done, infos = agent.control_step(3)
        agent_input["obs"] = obs
        if on_step is not None:
            on_step(step, infos)
        step += 1
    step -= 1
    return step, turn_round


def _read_analysis_distance(envs: Any) -> Optional[float]:
    """Read Habitat distance for offline diagnostics; never feed it to policy."""
    try:
        if hasattr(envs, "get_metrics"):
            _, _, distance, _ = envs.get_metrics()
        else:
            distance = envs.habitat_env.get_metrics().get("distance_to_goal")
        return None if distance is None else float(distance)
    except Exception:
        return None


def _make_diagnostic_step_logger(
    envs: Any,
    agent: Any,
    diagnostics: FailureDiagnostics,
    *,
    distance_interval: int = 10,
    success_distance: float = 1.0,
):
    """Record action motion on every step and sample offline distance sparsely."""
    interval = max(1, int(distance_interval))
    state = {
        "last_distance_step": -interval,
        "min_distance_to_goal": None,
        "oracle_success": 0.0,
        "sim_pose_by_step": {},
    }

    def _sample_distance(step: int, force: bool = False) -> Optional[float]:
        if not force and step - state["last_distance_step"] < interval:
            return None
        distance = _read_analysis_distance(envs)
        state["last_distance_step"] = step
        if distance is not None:
            current_min = state["min_distance_to_goal"]
            state["min_distance_to_goal"] = (
                distance if current_min is None else min(current_min, distance)
            )
            if distance <= float(success_distance):
                state["oracle_success"] = 1.0
        return distance

    def _on_step(step: int, info: Optional[Dict] = None) -> None:
        del info  # Distance is sampled directly and remains offline-only evidence.
        distance = _sample_distance(int(step))
        sim_pose = envs.get_sim_location()
        state["sim_pose_by_step"][int(step)] = sim_pose
        diagnostics.record_action(
            step=int(step),
            action_id=getattr(agent, "last_action", None),
            sim_pose=sim_pose,
            distance_to_goal=distance,
        )

    def _summary(final_step: int) -> Dict[str, Any]:
        _sample_distance(int(final_step), force=True)
        return {
            "min_distance_to_goal": state["min_distance_to_goal"],
            "oracle_success": state["oracle_success"],
        }

    _on_step.summary = _summary
    _on_step.pose_history = state["sim_pose_by_step"]
    return _on_step


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


def _goal_sim_pose(midterm_goal: Any, map_inst: Any) -> Optional[tuple]:
    """Express a selected goal in stable simulator-meter coordinates."""
    if not isinstance(midterm_goal, dict):
        return None
    center_sim = midterm_goal.get("center_sim_xy")
    if center_sim is not None:
        try:
            center = np.asarray(center_sim, dtype=np.float64).reshape(-1)
            if center.size >= 2:
                return float(center[0]), float(center[1]), 0.0
        except Exception:
            pass
    goal_pose = midterm_goal.get("pose")
    transform = getattr(map_inst, "transform_bev2sim", None)
    if goal_pose is None or transform is None:
        return None
    try:
        pose = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
        yaw = float(pose[2]) if pose.size >= 3 else 0.0
        return apply_transform((float(pose[0]), float(pose[1]), yaw), transform)
    except Exception:
        return None


def _diagnostic_global_counts(graph: Any) -> Dict[str, int]:
    objects = getattr(graph, "global_objects", {}) or {}
    return {
        str(key): len(value) if isinstance(value, list) else 0
        for key, value in objects.items()
    }


def _goal_distance_px(pose: Any, goal: Any) -> Optional[float]:
    try:
        pose_xy = np.asarray(pose, dtype=np.float64).reshape(-1)[:2]
        goal_xy = np.asarray(goal, dtype=np.float64).reshape(-1)[:2]
        if pose_xy.size < 2 or goal_xy.size < 2:
            return None
        return float(np.linalg.norm(pose_xy - goal_xy))
    except Exception:
        return None


def _record_pose_fit_diagnostics(
    map_inst: Any, map_info: Dict[str, Any], pose_history: Dict[int, Any], args: Any
) -> None:
    """Measure simulator/VGGT trajectory consistency without changing mapping."""
    frame_indices = list(getattr(map_inst, "frame_indices", []) or [])
    trajectory = getattr(map_inst, "trajectory_with_pixel", None)
    transform = getattr(map_inst, "transform_sim2bev", None)
    if trajectory is None or transform is None or not frame_indices:
        return
    try:
        bev_points = np.asarray(trajectory, dtype=np.float64)[:, :2]
        matched = [
            (idx, pose_history[idx])
            for idx in frame_indices
            if idx in pose_history
        ]
        if len(matched) < 2 or len(bev_points) != len(frame_indices):
            return
        positions = {idx: offset for offset, idx in enumerate(frame_indices)}
        sim_points = np.asarray(
            [[float(pose[0]), float(pose[1])] for _, pose in matched],
            dtype=np.float64,
        )
        observed_bev = np.asarray(
            [bev_points[positions[idx]] for idx, _ in matched], dtype=np.float64
        )
        projected_bev = np.asarray(
            [
                apply_transform((point[0], point[1], 0.0), transform)[:2]
                for point in sim_points
            ],
            dtype=np.float64,
        )
        errors = np.linalg.norm(projected_bev - observed_bev, axis=1)
        src_span = float(np.linalg.norm(np.ptp(sim_points, axis=0)))
        dst_span = float(np.linalg.norm(np.ptp(observed_bev, axis=0)))
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        reasons = []
        min_span_m = float(
            getattr(args, "pose_fit_quality_min_sim_span_m", 0.25)
        )
        if src_span < min_span_m:
            max_drift = float(
                getattr(args, "pose_fit_quality_max_stationary_drift_px", 35.0)
            )
            if dst_span > max_drift:
                reasons.append(
                    "pose_fit_stationary_drift_high:"
                    f"sim_span={src_span:.3f}m<{min_span_m:.3f}m,"
                    f"vggt_span={dst_span:.2f}px>{max_drift:.2f}px"
                )
        else:
            max_mean = float(
                getattr(args, "pose_fit_quality_max_mean_error_px", 25.0)
            )
            max_peak = float(
                getattr(args, "pose_fit_quality_max_max_error_px", 90.0)
            )
            scale = abs(float(transform.get("scale", 0.0) or 0.0))
            expected_span = src_span * scale
            span_ratio = dst_span / max(expected_span, 1e-6)
            min_ratio = float(
                getattr(args, "pose_fit_quality_min_span_ratio", 0.45)
            )
            max_ratio = float(
                getattr(args, "pose_fit_quality_max_span_ratio", 2.20)
            )
            if mean_error > max_mean:
                reasons.append(
                    f"pose_fit_mean_error_high:{mean_error:.2f}px>{max_mean:.2f}px"
                )
            if max_error > max_peak:
                reasons.append(
                    f"pose_fit_max_error_high:{max_error:.2f}px>{max_peak:.2f}px"
                )
            if span_ratio < min_ratio or span_ratio > max_ratio:
                reasons.append(
                    "pose_fit_span_ratio_bad:"
                    f"{span_ratio:.2f} not_in [{min_ratio:.2f},{max_ratio:.2f}]"
                )
        map_info["sim_bev_transform_fit"] = {
            "mode": "diagnostic_existing_transform",
            "num_points": len(matched),
            "src_span": src_span,
            "dst_span": dst_span,
            "mean_error_px": mean_error,
            "max_error_px": max_error,
            "quality_reasons": reasons,
        }
    except Exception:
        # Pose-fit diagnostics are observational and must never affect navigation.
        return


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

    episode_dir = os.path.dirname(obs_save_folder) if obs_save_folder else None
    diagnostics = FailureDiagnostics(
        os.path.join(episode_dir, "failure_diagnostics.jsonl")
        if episode_dir
        else os.devnull,
        episode_idx,
        enabled=bool(episode_dir)
        and bool(getattr(args, "failure_diagnostics_enabled", True)),
        flush_every=int(getattr(args, "failure_diagnostics_flush_every", 32)),
        collision_motion_threshold_m=float(
            getattr(args, "failure_diagnostics_collision_motion_m", 0.02)
        ),
        repeated_goal_radius_m=float(
            getattr(args, "failure_diagnostics_repeat_goal_radius_m", 0.75)
        ),
    )
    try:
        initial_sim_pose = envs.get_sim_location()
    except Exception:
        initial_sim_pose = None
    diagnostics.set_initial_pose(initial_sim_pose)
    diagnostics.emit(
        "episode_start",
        step=0,
        sim_pose=initial_sim_pose,
        goal=getattr(envs, "goal_name", getattr(envs, "object_goal", None)),
        scene=getattr(envs, "scene_path", None),
        max_steps=getattr(args, "max_episode_steps", 200),
        flush=True,
    )

    agent_input = _make_agent_input(args)
    on_step_diagnostic = _make_diagnostic_step_logger(
        envs,
        agent,
        diagnostics,
        distance_interval=int(
            getattr(args, "failure_diagnostics_distance_interval", 10)
        ),
        success_distance=float(
            getattr(args, "success_dist", getattr(args, "success_distance", 1.0))
        ),
    )
    step, turn_round = _run_initial_scan(
        agent, args, agent_input, on_step=on_step_diagnostic
    )

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
        stage_id = stage
        log(f"\n--- stage: {stage_id} ---")

        # ----- 1. Physical Perception -----
        loc_agent = envs.get_sim_location()
        map_inst, map_info = run_physical_perception(
            MapClass, set2, set1, current_step, stage_id, model_info, args, agent, loc_agent
        )

        # ----- 2. Graph state update (global objects, safety/explore fields) -----
        graph.update_stage(map_info)
        try:
            stage_bev_pose = apply_transform(loc_agent, map_inst.transform_sim2bev)
        except Exception:
            stage_bev_pose = None
        _record_pose_fit_diagnostics(
            map_inst,
            map_info,
            getattr(on_step_diagnostic, "pose_history", {}),
            args,
        )
        diagnostics.record_stage_map(
            stage=stage_id,
            step=step,
            map_info=map_info,
            sim_pose=loc_agent,
            bev_pose=stage_bev_pose,
        )

        # ----- 3. (optional) VLM Reasoning -----
        _run_optional_vlm_reasoning(graph, envs, agent_input, loc_agent, use_vlm_per_stage, log)

        _log_gssl_snapshot(graph, loc_agent, log)

        # ----- 4. MVM Planning (fuse_fields_and_extract_goal) -----
        nav_mode, midterm_goal = run_mvm_planning(graph, loc_agent, vis=True)
        stage += 1

        _update_agent_input_for_stage(agent_input, envs, map_inst, nav_mode)
        diagnostics.record_decision(
            stage=stage_id,
            step=step,
            nav_mode=nav_mode,
            goal=midterm_goal,
            goal_sim_pose=_goal_sim_pose(midterm_goal, map_inst),
            explore_direction=(getattr(graph, "plan_result", {}) or {}).get(
                "explore_direction"
            ),
            gssl_size=len(getattr(graph, "GSSL", None) or []),
            global_object_counts=_diagnostic_global_counts(graph),
        )
        log(nav_mode)

        # ----- 5. A* path for explore (safety field as costmap to avoid walls) -----
        if nav_mode == "explore" and midterm_goal is not None:
            planned_path = _plan_explore_path(agent_input, map_inst, midterm_goal, midterm_save_folder, stage)
            diagnostics.record_plan(
                stage=stage_id,
                step=step,
                success=bool(planned_path),
                path_length=len(planned_path or []),
                goal_type=midterm_goal.get("type"),
            )

            # If A* fails (start/goal in obstacle or no path), switch to recognition to rebuild map
            if planned_path is None or len(planned_path) == 0:
                log("A* planning failed, switching to recognition mode to rebuild map.")
                nav_mode = "recognition"
                midterm_goal = None

        # ----- 6. Low-level Execution (paper-aligned) -----
        if nav_mode in ["finding", "judgement", "recognition"]:
            segment_start_step = step
            segment_start_sim_pose = envs.get_sim_location()
            segment_start_bev_pose = agent_input.get("pose")
            step_delta = execute_finding_or_judgement_turn(
                agent,
                envs,
                map_inst,
                agent_input,
                turn_round,
                step,
                on_step=on_step_diagnostic,
            )
            step += step_delta
            diagnostics.record_execution_segment(
                stage=stage_id,
                start_step=segment_start_step,
                end_step=step,
                nav_mode=nav_mode,
                goal_type=midterm_goal.get("type")
                if isinstance(midterm_goal, dict)
                else None,
                start_sim_pose=segment_start_sim_pose,
                end_sim_pose=envs.get_sim_location(),
                start_bev_pose=segment_start_bev_pose,
                end_bev_pose=agent_input.get("pose"),
                start_goal_distance_px=None,
                end_goal_distance_px=None,
            )
            # Paper: after rotation in Judgement, decide arrival by goal-vs-observation match
            if nav_mode == "judgement":
                if getattr(args, "goal_type", None) == "object":
                    final_metrics = _issue_stop_and_get_metrics(envs, log)
                    success = final_metrics["success"] > 0.0
                    diagnostics.emit(
                        "verification",
                        stage=stage_id,
                        step=step,
                        goal_type="object",
                        result="accepted" if success else "rejected",
                    )
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
                    diagnostics.emit(
                        "verification",
                        stage=stage_id,
                        step=step,
                        goal_type="instance_image",
                        result="accepted" if success else "rejected",
                        threshold=succeed_match_points,
                    )
                    if success:
                        final_metrics = _issue_stop_and_get_metrics(envs, log)
                        log("episode success (goal match >= threshold)")
                        break
        elif nav_mode == "explore" and midterm_goal is not None:
            segment_start_step = step
            segment_start_sim_pose = envs.get_sim_location()
            segment_start_bev_pose = np.asarray(
                agent_input.get("pose", [])[:2], dtype=np.float64
            )
            goal_pose = midterm_goal.get("pose")
            start_goal_distance = _goal_distance_px(segment_start_bev_pose, goal_pose)
            step_delta, done = run_until_midterm_reached(
                agent,
                envs,
                map_inst,
                agent_input,
                midterm_goal,
                args,
                step,
                on_step=on_step_diagnostic,
            )
            step += step_delta
            segment_end_bev_pose = agent_input.get("pose")
            diagnostics.record_execution_segment(
                stage=stage_id,
                start_step=segment_start_step,
                end_step=step,
                nav_mode=nav_mode,
                goal_type=midterm_goal.get("type"),
                start_sim_pose=segment_start_sim_pose,
                end_sim_pose=envs.get_sim_location(),
                start_bev_pose=segment_start_bev_pose,
                end_bev_pose=segment_end_bev_pose,
                start_goal_distance_px=start_goal_distance,
                end_goal_distance_px=_goal_distance_px(
                    segment_end_bev_pose, goal_pose
                ),
            )
            if done:
                final_metrics = coerce_nav_metrics(getattr(envs, "info", {}))
                success = final_metrics["success"] > 0.0
                break

        set2, set1, current_step = set1, current_step, step

    _save_episode_artifacts(envs, agent, agent_input, args, log)
    success, final_metrics = _finalize_metrics(success, final_metrics, envs)
    analysis_metrics = on_step_diagnostic.summary(step)
    if analysis_metrics.get("min_distance_to_goal") is not None:
        final_metrics.update(analysis_metrics)
        log(
            "[analysis_metric] episode_min_distance_to_goal="
            f"{analysis_metrics['min_distance_to_goal']:.4f}, "
            f"oracle_success={analysis_metrics['oracle_success']:.3f}"
        )
    diagnostics.emit(
        "episode_end",
        step=step,
        success=bool(success),
        metrics=final_metrics,
        summary=diagnostics.summary(),
        flush=True,
    )
    diagnostics.close()

    return success, step, final_metrics
