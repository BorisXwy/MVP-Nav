import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.core.failure_diagnostics import FailureDiagnostics
from src.core.navigation import _record_pose_fit_diagnostics


class FailureDiagnosticsTest(unittest.TestCase):
    def test_records_collision_map_and_repeated_goal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "failure_diagnostics.jsonl"
            recorder = FailureDiagnostics(str(path), 7, flush_every=2)
            recorder.set_initial_pose((0.0, 0.0, 0.0))
            recorder.emit("episode_start", step=0, goal="bed")
            recorder.record_action(
                step=1,
                action_id=1,
                sim_pose=(0.0, 0.0, 0.0),
                distance_to_goal=4.0,
            )
            recorder.record_action(
                step=2,
                action_id=1,
                sim_pose=(0.25, 0.0, 0.0),
                distance_to_goal=3.8,
            )
            recorder.record_stage_map(
                stage=0,
                step=2,
                map_info={
                    "stage_map_quality": "bad",
                    "stage_map_quality_metrics": {
                        "reasons": ["pose_fit_stationary_drift_high"]
                    },
                },
                sim_pose=(0.25, 0.0, 0.0),
                bev_pose=(10.0, 20.0, 0.0),
            )
            for stage, goal_x in ((0, 1.0), (1, 1.2)):
                recorder.record_decision(
                    stage=stage,
                    step=2 + stage,
                    nav_mode="explore",
                    goal={"type": "explore_point", "pose": [100.0, 200.0, 0.0]},
                    goal_sim_pose=(goal_x, 1.0, 0.0),
                    explore_direction="front",
                    gssl_size=1,
                    global_object_counts={"main": 0, "sub": 0},
                )
            summary = recorder.summary()
            recorder.emit(
                "episode_end", step=3, success=False, summary=summary, flush=True
            )
            recorder.close()

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            actions = [row for row in rows if row["event"] == "action"]
            decisions = [row for row in rows if row["event"] == "decision"]
            self.assertTrue(actions[0]["collision_proxy"])
            self.assertFalse(actions[1]["collision_proxy"])
            self.assertTrue(
                [row for row in rows if row["event"] == "stage_map"][0][
                    "pose_fit_bad"
                ]
            )
            self.assertEqual(decisions[1]["repeated_explore_goal_count"], 1)
            self.assertEqual(summary["collision_proxy_count"], 1)
            self.assertEqual(summary["repeated_explore_goals"], 1)

    def test_disabled_recorder_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "failure_diagnostics.jsonl"
            recorder = FailureDiagnostics(str(path), 0, enabled=False)
            recorder.emit("episode_start")
            recorder.close()
            self.assertFalse(path.exists())

    def test_pose_fit_diagnostic_detects_stationary_vggt_drift(self):
        map_info = {}
        map_inst = SimpleNamespace(
            frame_indices=[0, 1],
            trajectory_with_pixel=np.asarray(
                [[0.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float64
            ),
            transform_sim2bev={
                "scale": 1.0,
                "rotation": 0.0,
                "translation": np.zeros(2),
            },
        )
        pose_history = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0)}

        _record_pose_fit_diagnostics(
            map_inst, map_info, pose_history, SimpleNamespace()
        )

        fit = map_info["sim_bev_transform_fit"]
        self.assertEqual(fit["mode"], "diagnostic_existing_transform")
        self.assertTrue(
            any(
                reason.startswith("pose_fit_stationary_drift_high")
                for reason in fit["quality_reasons"]
            )
        )


if __name__ == "__main__":
    unittest.main()
