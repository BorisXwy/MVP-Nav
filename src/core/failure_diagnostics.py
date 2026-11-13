"""Low-overhead structured diagnostics for MVP-Nav failure analysis.

The recorder only consumes values already available in the navigation loop. It
does not perform image processing, model inference, or map construction, and
logging failures are deliberately isolated from navigation execution.
"""

import json
import math
import os
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np


SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _pose_list(pose: Any) -> Optional[list]:
    if pose is None:
        return None
    try:
        values = np.asarray(pose, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if values.size < 2:
        return None
    return [float(value) for value in values[:3]]


def _translation(
    previous: Optional[Sequence[float]], current: Optional[Sequence[float]]
) -> Optional[float]:
    if previous is None or current is None or len(previous) < 2 or len(current) < 2:
        return None
    return float(
        np.linalg.norm(
            np.asarray(current[:2], dtype=np.float64)
            - np.asarray(previous[:2], dtype=np.float64)
        )
    )


class FailureDiagnostics:
    """Buffered JSONL event recorder with small diagnostic counters."""

    def __init__(
        self,
        path: str,
        episode_idx: int,
        *,
        enabled: bool = True,
        flush_every: int = 32,
        collision_motion_threshold_m: float = 0.02,
        repeated_goal_radius_m: float = 0.75,
    ) -> None:
        self.path = path
        self.episode_idx = int(episode_idx)
        self.enabled = bool(enabled)
        self.flush_every = max(1, int(flush_every))
        self.collision_motion_threshold_m = max(
            0.0, float(collision_motion_threshold_m)
        )
        self.repeated_goal_radius_m = max(0.0, float(repeated_goal_radius_m))
        self._start_time = time.perf_counter()
        self._handle = None
        self._pending = 0
        self._previous_sim_pose = None
        self._explore_goals_sim = []
        self._forward_actions = 0
        self._collision_proxy_count = 0
        self._collision_streak = 0
        self._max_collision_streak = 0
        self._stage_count = 0
        self._bad_map_stages = 0
        self._bad_pose_fit_stages = 0
        self._repeated_explore_goals = 0
        self._astar_failures = 0
        self._no_progress_segments = 0
        if self.enabled:
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                self._handle = open(path, "w", encoding="utf-8", buffering=8192)
            except OSError:
                self.enabled = False
                self._handle = None

    def emit(
        self,
        event: str,
        *,
        step: Optional[int] = None,
        stage: Optional[int] = None,
        flush: bool = False,
        **fields: Any,
    ) -> None:
        if self._handle is None:
            return
        row = {
            "schema_version": SCHEMA_VERSION,
            "event": str(event),
            "episode": self.episode_idx,
            "elapsed_s": round(time.perf_counter() - self._start_time, 6),
        }
        if step is not None:
            row["step"] = int(step)
        if stage is not None:
            row["stage"] = int(stage)
        row.update({key: _json_safe(value) for key, value in fields.items()})
        try:
            self._handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self._pending += 1
            if flush or self._pending >= self.flush_every:
                self._handle.flush()
                self._pending = 0
        except OSError:
            self.close()

    def set_initial_pose(self, sim_pose: Any) -> None:
        self._previous_sim_pose = _pose_list(sim_pose)

    def record_action(
        self,
        *,
        step: int,
        action_id: Optional[int],
        sim_pose: Any,
        distance_to_goal: Optional[float],
    ) -> None:
        current_pose = _pose_list(sim_pose)
        motion_m = _translation(self._previous_sim_pose, current_pose)
        collision_proxy = False
        if action_id == 1:
            self._forward_actions += 1
            collision_proxy = (
                motion_m is not None
                and motion_m < self.collision_motion_threshold_m
            )
            if collision_proxy:
                self._collision_proxy_count += 1
                self._collision_streak += 1
                self._max_collision_streak = max(
                    self._max_collision_streak, self._collision_streak
                )
            else:
                self._collision_streak = 0
        else:
            self._collision_streak = 0
        self.emit(
            "action",
            step=step,
            action_id=action_id,
            sim_pose=current_pose,
            translation_m=motion_m,
            distance_to_goal=distance_to_goal,
            collision_proxy=collision_proxy,
            collision_streak=self._collision_streak,
        )
        self._previous_sim_pose = current_pose

    def record_stage_map(
        self,
        *,
        stage: int,
        step: int,
        map_info: Dict[str, Any],
        sim_pose: Any,
        bev_pose: Any,
    ) -> None:
        quality = str(map_info.get("stage_map_quality", "unknown"))
        quality_metrics = map_info.get("stage_map_quality_metrics") or {}
        pose_fit = map_info.get("sim_bev_transform_fit") or {}
        reasons = list(quality_metrics.get("reasons") or []) + list(
            pose_fit.get("quality_reasons") or []
        )
        pose_fit_bad = any(str(reason).startswith("pose_fit_") for reason in reasons)
        self._stage_count += 1
        if quality != "good":
            self._bad_map_stages += 1
        if pose_fit_bad:
            self._bad_pose_fit_stages += 1
        self.emit(
            "stage_map",
            stage=stage,
            step=step,
            map_quality=quality,
            map_quality_metrics=quality_metrics,
            pose_fit=pose_fit,
            pose_fit_bad=pose_fit_bad,
            sim_pose=_pose_list(sim_pose),
            bev_pose=_pose_list(bev_pose),
            flush=True,
        )

    def record_decision(
        self,
        *,
        stage: int,
        step: int,
        nav_mode: str,
        goal: Any,
        goal_sim_pose: Any,
        explore_direction: Optional[str],
        gssl_size: int,
        global_object_counts: Dict[str, int],
    ) -> None:
        goal_type = goal.get("type") if isinstance(goal, dict) else None
        goal_pose = goal.get("pose") if isinstance(goal, dict) else None
        goal_sim = _pose_list(goal_sim_pose)
        nearest_previous_m = None
        repeat_count = 0
        if (
            nav_mode == "explore"
            and goal_type == "explore_point"
            and goal_sim is not None
        ):
            distances = [
                _translation(previous, goal_sim)
                for previous in self._explore_goals_sim
            ]
            distances = [distance for distance in distances if distance is not None]
            nearest_previous_m = min(distances) if distances else None
            repeat_count = sum(
                distance <= self.repeated_goal_radius_m for distance in distances
            )
            if repeat_count > 0:
                self._repeated_explore_goals += 1
            self._explore_goals_sim.append(goal_sim)
        self.emit(
            "decision",
            stage=stage,
            step=step,
            nav_mode=nav_mode,
            goal_type=goal_type,
            goal_pose=_pose_list(goal_pose),
            goal_sim_pose=goal_sim,
            goal_score=goal.get("score") if isinstance(goal, dict) else None,
            goal_source=goal.get("source") if isinstance(goal, dict) else None,
            explore_direction=explore_direction,
            gssl_size=int(gssl_size),
            global_object_counts=global_object_counts,
            nearest_previous_explore_goal_m=nearest_previous_m,
            repeated_explore_goal_count=repeat_count,
        )

    def record_plan(
        self,
        *,
        stage: int,
        step: int,
        success: bool,
        path_length: int,
        goal_type: Optional[str],
    ) -> None:
        if not success:
            self._astar_failures += 1
        self.emit(
            "plan",
            stage=stage,
            step=step,
            planner="astar",
            success=bool(success),
            path_length=int(path_length),
            goal_type=goal_type,
        )

    def record_execution_segment(
        self,
        *,
        stage: int,
        start_step: int,
        end_step: int,
        nav_mode: str,
        goal_type: Optional[str],
        start_sim_pose: Any,
        end_sim_pose: Any,
        start_bev_pose: Any,
        end_bev_pose: Any,
        start_goal_distance_px: Optional[float],
        end_goal_distance_px: Optional[float],
    ) -> None:
        sim_motion_m = _translation(_pose_list(start_sim_pose), _pose_list(end_sim_pose))
        bev_motion_px = _translation(_pose_list(start_bev_pose), _pose_list(end_bev_pose))
        progress_px = None
        if start_goal_distance_px is not None and end_goal_distance_px is not None:
            progress_px = float(start_goal_distance_px) - float(end_goal_distance_px)
        no_progress = (
            end_step > start_step
            and progress_px is not None
            and progress_px <= 0.0
            and (sim_motion_m is None or sim_motion_m < 0.10)
        )
        if no_progress:
            self._no_progress_segments += 1
        self.emit(
            "execution_segment",
            stage=stage,
            step=end_step,
            start_step=start_step,
            end_step=end_step,
            nav_mode=nav_mode,
            goal_type=goal_type,
            start_sim_pose=_pose_list(start_sim_pose),
            end_sim_pose=_pose_list(end_sim_pose),
            sim_motion_m=sim_motion_m,
            start_bev_pose=_pose_list(start_bev_pose),
            end_bev_pose=_pose_list(end_bev_pose),
            bev_motion_px=bev_motion_px,
            start_goal_distance_px=start_goal_distance_px,
            end_goal_distance_px=end_goal_distance_px,
            goal_progress_px=progress_px,
            no_progress=no_progress,
        )

    def summary(self) -> Dict[str, Any]:
        collision_rate = (
            float(self._collision_proxy_count) / float(self._forward_actions)
            if self._forward_actions
            else 0.0
        )
        return {
            "stage_count": self._stage_count,
            "bad_map_stages": self._bad_map_stages,
            "bad_pose_fit_stages": self._bad_pose_fit_stages,
            "forward_actions": self._forward_actions,
            "collision_proxy_count": self._collision_proxy_count,
            "collision_proxy_rate": collision_rate,
            "max_collision_streak": self._max_collision_streak,
            "repeated_explore_goals": self._repeated_explore_goals,
            "astar_failures": self._astar_failures,
            "no_progress_segments": self._no_progress_segments,
        }

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.flush()
            self._handle.close()
        except OSError:
            pass
        finally:
            self._handle = None
