"""Classify MVP-Nav failures from structured diagnostic events."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_events(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    events = []
    invalid = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if isinstance(row, dict):
                events.append(row)
            else:
                invalid += 1
    return events, invalid


def discover_logs(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    direct = target / "failure_diagnostics.jsonl"
    if direct.is_file():
        return [direct]
    return sorted(target.glob("episode_*/failure_diagnostics.jsonl"))


def _last(
    events: Iterable[Dict[str, Any]], event_name: str
) -> Optional[Dict[str, Any]]:
    rows = [row for row in events if row.get("event") == event_name]
    return rows[-1] if rows else None


def _finite_numbers(values: Iterable[Any]) -> List[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and abs(number) != float("inf"):
            result.append(number)
    return result


def _confidence(score: float, margin: float) -> str:
    if score >= 4.0 and margin >= 1.5:
        return "high"
    if score >= 2.0 and margin >= 0.5:
        return "medium"
    return "low"


def classify(
    events: List[Dict[str, Any]], source: Path, invalid_lines: int = 0
) -> Dict[str, Any]:
    stage_maps = [row for row in events if row.get("event") == "stage_map"]
    actions = [row for row in events if row.get("event") == "action"]
    decisions = [row for row in events if row.get("event") == "decision"]
    plans = [row for row in events if row.get("event") == "plan"]
    segments = [row for row in events if row.get("event") == "execution_segment"]
    guards = [row for row in events if row.get("event") == "localization_guard"]
    verifications = [row for row in events if row.get("event") == "verification"]
    end = _last(events, "episode_end") or {}
    start = _last(events, "episode_start") or {}
    episode_exception = _last(events, "episode_exception") or {}
    summary = end.get("summary") or {}
    metrics = end.get("metrics") or {}
    success = bool(end.get("success", False))

    bad_maps = sum(
        str(row.get("map_quality", "unknown")) != "good" for row in stage_maps
    )
    good_maps = sum(
        str(row.get("map_quality", "unknown")) == "good" for row in stage_maps
    )
    pose_bad = sum(bool(row.get("pose_fit_bad")) for row in stage_maps)
    astar_failures = sum(not bool(row.get("success")) for row in plans)
    forward = [row for row in actions if row.get("action_id") == 1]
    collisions = [row for row in forward if row.get("collision_proxy") is True]
    collision_rate = float(len(collisions)) / float(len(forward)) if forward else 0.0
    max_collision_streak = max(
        [int(row.get("collision_streak") or 0) for row in actions] or [0]
    )
    repeats = [
        row
        for row in decisions
        if int(row.get("repeated_explore_goal_count") or 0) > 0
    ]
    explore_decisions = [
        row
        for row in decisions
        if row.get("nav_mode") == "explore"
        and row.get("goal_type") == "explore_point"
    ]
    direction_counts: Dict[str, int] = {}
    for row in explore_decisions:
        direction = row.get("explore_direction")
        if direction:
            direction = str(direction)
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
    dominant_direction = (
        max(direction_counts, key=direction_counts.get) if direction_counts else None
    )
    dominant_direction_count = (
        direction_counts.get(dominant_direction, 0) if dominant_direction else 0
    )
    rejected_verify = sum(row.get("result") == "rejected" for row in verifications)
    inconclusive_verify = sum(
        row.get("result") == "inconclusive" for row in verifications
    )
    no_progress = sum(bool(row.get("no_progress")) for row in segments)
    nonpositive_goal_progress = sum(
        row.get("nav_mode") == "explore"
        and row.get("goal_progress_px") is not None
        and float(row.get("goal_progress_px")) <= 0.0
        for row in segments
    )
    distances = _finite_numbers(row.get("distance_to_goal") for row in actions)
    distance_improvement = distances[0] - min(distances) if distances else None

    subtype_scores = {
        "mapping_failure": 0.0,
        "localization_failure": 0.0,
        "collision_or_stall": 0.0,
        "planning_execution_failure": 0.0,
        "repeated_exploration": 0.0,
        "unproductive_direction": 0.0,
        "semantic_selection_or_verification_loop": 0.0,
    }
    evidence = {key: [] for key in subtype_scores}

    stage_count = len(stage_maps)
    if bad_maps:
        bad_ratio = float(bad_maps) / float(stage_count or 1)
        bad_stage_ids = [
            row.get("stage") for row in stage_maps if row.get("map_quality") != "good"
        ]
        subtype_scores["mapping_failure"] = 1.0 + 4.0 * bad_ratio
        evidence["mapping_failure"].append(
            "bad map stages: {}/{} ({:.0%}), stages={}".format(
                bad_maps, stage_count, bad_ratio, bad_stage_ids
            )
        )
    if pose_bad or guards:
        pose_ratio = float(pose_bad) / float(stage_count or 1)
        pose_stage_ids = [
            row.get("stage") for row in stage_maps if row.get("pose_fit_bad")
        ]
        subtype_scores["localization_failure"] = min(
            5.0, 1.5 + 3.5 * pose_ratio + len(guards)
        )
        evidence["localization_failure"].append(
            "pose-fit failures: {}/{}, stages={}; localization guards: {}".format(
                pose_bad, stage_count, pose_stage_ids, len(guards)
            )
        )
    if len(collisions) >= 2 or max_collision_streak >= 2:
        subtype_scores["collision_or_stall"] = min(
            5.0, 1.0 + 5.0 * collision_rate + 0.5 * max_collision_streak
        )
        evidence["collision_or_stall"].append(
            "forward collision proxies: {}/{} ({:.0%}); max streak: {}; first steps={}".format(
                len(collisions),
                len(forward),
                collision_rate,
                max_collision_streak,
                [row.get("step") for row in collisions[:8]],
            )
        )
    if astar_failures or guards or no_progress:
        subtype_scores["planning_execution_failure"] = min(
            5.0, 1.0 + 1.5 * astar_failures + len(guards) + no_progress
        )
        evidence["planning_execution_failure"].append(
            "A* failures: {} at stages={}; localization guards: {}; stalled segments: {}".format(
                astar_failures,
                [row.get("stage") for row in plans if not row.get("success")],
                len(guards),
                no_progress,
            )
        )

    maps_usable = stage_count > 0 and good_maps >= max(1, (stage_count + 1) // 2)
    physical_chain_dominant = (
        pose_bad > good_maps
        or (len(collisions) >= 2 and collision_rate >= 0.20)
        or bad_maps > good_maps
    )
    if repeats:
        subtype_scores["repeated_exploration"] = min(
            5.0, 1.5 + 1.25 * len(repeats)
        )
        evidence["repeated_exploration"].append(
            "repeated explore targets: {}/{} explore decisions, stages={}".format(
                len(repeats),
                len(explore_decisions),
                [row.get("stage") for row in repeats],
            )
        )
    if dominant_direction_count >= 3:
        subtype_scores["repeated_exploration"] += min(
            1.5, 0.25 * dominant_direction_count
        )
        evidence["repeated_exploration"].append(
            "dominant explore direction: {} in {}/{} decisions".format(
                dominant_direction, dominant_direction_count, len(explore_decisions)
            )
        )
    if maps_usable and len(explore_decisions) >= 2:
        low_improvement = (
            distance_improvement is not None and distance_improvement < 0.25
        )
        if low_improvement or nonpositive_goal_progress >= 2:
            subtype_scores["unproductive_direction"] = min(
                5.0,
                2.0
                + 0.75 * nonpositive_goal_progress
                + (1.0 if low_improvement else 0.0),
            )
            evidence["unproductive_direction"].append(
                "usable maps: {}/{}; episode distance improvement: {}; nonpositive goal-progress segments: {}".format(
                    good_maps,
                    stage_count,
                    "{:.3f} m".format(distance_improvement)
                    if distance_improvement is not None
                    else "unavailable",
                    nonpositive_goal_progress,
                )
            )
    if rejected_verify + inconclusive_verify >= 2:
        subtype_scores["semantic_selection_or_verification_loop"] = min(
            5.0, 1.0 + rejected_verify + 0.75 * inconclusive_verify
        )
        evidence["semantic_selection_or_verification_loop"].append(
            "verification rejected: {}; inconclusive: {}; steps={}".format(
                rejected_verify,
                inconclusive_verify,
                [
                    row.get("step")
                    for row in verifications
                    if row.get("result") != "accepted"
                ],
            )
        )

    nav_subtypes = [
        "localization_failure",
        "mapping_failure",
        "collision_or_stall",
        "planning_execution_failure",
    ]
    explore_subtypes = [
        "repeated_exploration",
        "unproductive_direction",
        "semantic_selection_or_verification_loop",
    ]
    nav_primary = max(nav_subtypes, key=lambda key: subtype_scores[key])
    explore_primary = max(explore_subtypes, key=lambda key: subtype_scores[key])
    nav_score = subtype_scores[nav_primary]
    explore_score = subtype_scores[explore_primary]

    if success:
        bucket, subtype, confidence = "success", "none", "high"
    elif nav_score <= 0.0 and explore_score <= 0.0:
        bucket, subtype, confidence = "insufficient_evidence", "unknown", "low"
    elif physical_chain_dominant or nav_score >= explore_score:
        bucket, subtype = "navigation_chain", nav_primary
        confidence = _confidence(nav_score, nav_score - explore_score)
    else:
        bucket, subtype = "exploration_policy", explore_primary
        confidence = _confidence(explore_score, explore_score - nav_score)

    ranked = sorted(subtype_scores, key=lambda key: subtype_scores[key], reverse=True)
    secondary = [
        {
            "subtype": key,
            "score": round(subtype_scores[key], 3),
            "evidence": evidence[key],
        }
        for key in ranked
        if subtype_scores[key] > 0.0 and key != subtype
    ][:3]
    primary_evidence = evidence.get(subtype, [])
    if success:
        primary_evidence = ["episode_end reports success=true"]
    elif bucket == "insufficient_evidence":
        if episode_exception:
            primary_evidence = [
                "episode terminated with exception: {}".format(
                    episode_exception.get("error", "unknown error")
                )
            ]
        else:
            primary_evidence = ["no structured signal crossed a triage threshold"]

    next_checks = {
        "localization_failure": "Inspect stage_map pose-fit spans and residuals; compare the first recovered moving-baseline stage before changing exploration policy.",
        "mapping_failure": "Inspect map-quality reasons and usable-map fields on the first bad stage; verify whether a later recovery stage becomes usable.",
        "collision_or_stall": "Inspect the listed forward-action steps and nearby simulator poses; distinguish obstacle contact from commands against an invalid local goal.",
        "planning_execution_failure": "Inspect A* failure stages and execution goal progress before tuning high-level exploration.",
        "repeated_exploration": "Compare repeated goal_sim_pose values with explore_direction; confirm the repeat persists on good-map stages.",
        "unproductive_direction": "Compare per-stage exploration direction and offline distance trend while keeping map/localization quality fixed.",
        "semantic_selection_or_verification_loop": "Inspect rejected candidates and verification evidence to separate false detections from an over-strict verifier.",
        "none": "No failure ablation is needed for this successful episode.",
        "unknown": "Collect a complete structured run or inspect result.txt and terminal output manually.",
    }

    return {
        "source": str(source),
        "episode": end.get("episode", start.get("episode")),
        "success": success,
        "steps": end.get("step"),
        "goal": start.get("goal"),
        "scene": start.get("scene"),
        "final_distance_to_goal": metrics.get("distance_to_goal"),
        "minimum_distance_to_goal": metrics.get("min_distance_to_goal"),
        "distance_improvement_m": round(distance_improvement, 6)
        if distance_improvement is not None
        else None,
        "bucket": bucket,
        "subtype": subtype,
        "confidence": confidence,
        "primary_score": round(
            nav_score
            if bucket == "navigation_chain"
            else explore_score
            if bucket == "exploration_policy"
            else 0.0,
            3,
        ),
        "navigation_chain_score": round(nav_score, 3),
        "exploration_policy_score": round(explore_score, 3),
        "evidence": primary_evidence,
        "next_check": next_checks.get(subtype),
        "secondary": secondary,
        "counters": {
            "stages": stage_count,
            "good_map_stages": good_maps,
            "bad_map_stages": bad_maps,
            "pose_fit_bad_stages": pose_bad,
            "forward_actions": len(forward),
            "collision_proxies": len(collisions),
            "collision_proxy_rate": round(collision_rate, 6),
            "max_collision_streak": max_collision_streak,
            "astar_failures": astar_failures,
            "explore_decisions": len(explore_decisions),
            "repeated_explore_goals": len(repeats),
            "verification_rejected": rejected_verify,
            "verification_inconclusive": inconclusive_verify,
        },
        "invalid_jsonl_lines": invalid_lines,
        "exception": episode_exception.get("error"),
        "recorder_summary": summary,
    }


def analyze_path(target: Path) -> List[Dict[str, Any]]:
    reports = []
    for path in discover_logs(target):
        events, invalid = load_events(path)
        reports.append(classify(events, path, invalid_lines=invalid))
    return reports


def render_markdown(reports: List[Dict[str, Any]]) -> str:
    blocks = []
    for report in reports:
        lines = [
            "## Episode {}".format(report.get("episode")),
            "",
            "- Classification: `{}` → `{}` (confidence: `{}`)".format(
                report["bucket"], report["subtype"], report["confidence"]
            ),
            "- Outcome: success={}, steps={}, final distance={} m, minimum distance={} m".format(
                report["success"],
                report.get("steps"),
                report.get("final_distance_to_goal"),
                report.get("minimum_distance_to_goal"),
            ),
            "- Scores: navigation_chain={}, exploration_policy={}".format(
                report["navigation_chain_score"],
                report["exploration_policy_score"],
            ),
        ]
        for item in report.get("evidence", []):
            lines.append("- Evidence: {}".format(item))
        if report.get("next_check"):
            lines.append("- Next check: {}".format(report["next_check"]))
        if report.get("secondary"):
            lines.extend(["", "Secondary signals:"])
            for item in report["secondary"]:
                lines.append(
                    "- `{}` score={}: {}".format(
                        item["subtype"],
                        item["score"],
                        "; ".join(item.get("evidence") or []),
                    )
                )
        lines.extend(["", "Source: `{}`".format(report["source"])])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def write_reports(target: Path, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    reports = analyze_path(target)
    if not reports:
        return []
    destination = output_dir or (target if target.is_dir() else target.parent)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "failure_summary.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "failure_summary.md").write_text(
        render_markdown(reports) + "\n", encoding="utf-8"
    )
    return reports


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="episode directory, run directory, or diagnostics JSONL"
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="write report instead of stdout")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    reports = analyze_path(args.path)
    if not reports:
        print(
            "No failure_diagnostics.jsonl found under {}. Use result.txt and terminal output for limited manual analysis.".format(
                args.path
            ),
            file=sys.stderr,
        )
        return 2
    output = (
        json.dumps(reports, ensure_ascii=False, indent=2)
        if args.format == "json"
        else render_markdown(reports)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0
