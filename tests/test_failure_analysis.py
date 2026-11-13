import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.failure_analysis import analyze_path, classify, write_reports


class FailureAnalysisTest(unittest.TestCase):
    def test_navigation_chain_has_causal_precedence(self):
        events = [
            {"event": "episode_start", "episode": 2, "goal": "chair"},
            {
                "event": "stage_map",
                "stage": 0,
                "map_quality": "bad",
                "pose_fit_bad": True,
            },
            {
                "event": "stage_map",
                "stage": 1,
                "map_quality": "bad",
                "pose_fit_bad": True,
            },
            {
                "event": "decision",
                "stage": 1,
                "nav_mode": "explore",
                "goal_type": "explore_point",
                "repeated_explore_goal_count": 2,
            },
            {"event": "episode_end", "episode": 2, "success": False},
        ]

        report = classify(events, Path("episode_2/failure_diagnostics.jsonl"))

        self.assertEqual(report["bucket"], "navigation_chain")
        self.assertEqual(report["subtype"], "localization_failure")

    def test_repeated_exploration_on_good_maps(self):
        events = [{"event": "episode_start", "episode": 3, "goal": "bed"}]
        for stage in range(3):
            events.append(
                {
                    "event": "stage_map",
                    "stage": stage,
                    "map_quality": "good",
                    "pose_fit_bad": False,
                }
            )
            events.append(
                {
                    "event": "decision",
                    "stage": stage,
                    "nav_mode": "explore",
                    "goal_type": "explore_point",
                    "explore_direction": "front",
                    "repeated_explore_goal_count": 0 if stage == 0 else 1,
                }
            )
        events.extend(
            [
                {
                    "event": "action",
                    "action_id": 1,
                    "distance_to_goal": 4.0,
                    "collision_proxy": False,
                },
                {
                    "event": "action",
                    "action_id": 1,
                    "distance_to_goal": 3.9,
                    "collision_proxy": False,
                },
                {"event": "episode_end", "episode": 3, "success": False},
            ]
        )

        report = classify(events, Path("episode_3/failure_diagnostics.jsonl"))

        self.assertEqual(report["bucket"], "exploration_policy")
        self.assertEqual(report["subtype"], "repeated_exploration")

    def test_writes_episode_reports(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_0"
            episode_dir.mkdir()
            log_path = episode_dir / "failure_diagnostics.jsonl"
            rows = [
                {"event": "episode_start", "episode": 0, "goal": "sofa"},
                {
                    "event": "episode_end",
                    "episode": 0,
                    "success": True,
                    "metrics": {"distance_to_goal": 0.2},
                },
            ]
            log_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            reports = write_reports(episode_dir)

            self.assertEqual(analyze_path(episode_dir)[0]["bucket"], "success")
            self.assertEqual(reports[0]["bucket"], "success")
            self.assertTrue((episode_dir / "failure_summary.md").is_file())
            self.assertTrue((episode_dir / "failure_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
