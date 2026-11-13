import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.core.navigation import run_episode


IDENTITY_TRANSFORM = {
    "scale": 1.0,
    "rotation": 0.0,
    "translation": np.zeros(2),
}


class FakeEnvironment:
    def __init__(self):
        self.pose = [0.0, 0.0, 0.0]
        self.info = {}
        self.goal_name = "chair"
        self.scene_path = "fake-scene.glb"

    def get_sim_location(self):
        return tuple(self.pose)

    def get_metrics(self):
        return 0.0, 0.0, 3.0, 0.0

    def step(self, action):
        return {}, False, self.info


class FakeAgent:
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        self.last_action = None
        self.obs_history = []
        self.node_segment_result = []
        self.main_segment_result = []
        self.sub_segment_result = []
        self.args = SimpleNamespace(save_episode_video=False)

    def control_step(self, action):
        self.last_action = action
        self.env.pose[2] += 0.1
        self.obs_history.append(np.zeros((3, 2, 2), dtype=np.float32))
        return [np.zeros((2, 2, 3), dtype=np.uint8)], False, self.env.info


class FakeMap:
    def __init__(self, set2, set1, step, stage, model_info, args, obs_provider):
        del set2, set1, step, model_info, args, obs_provider
        self.stage = stage
        self.bev = np.zeros((4, 4, 3), dtype=np.uint8)
        self.bev_step = 1.0
        self.transform_sim2bev = IDENTITY_TRANSFORM
        self.transform_bev2sim = IDENTITY_TRANSFORM
        self.stage_space_safety_field = None
        self.frame_indices = [0, 1]
        self.trajectory_with_pixel = np.zeros((2, 3), dtype=np.float64)

    def run_stage(self, loc_agent):
        del loc_agent
        return {
            "stage": self.stage,
            "stage_map_quality": "good",
            "transform_sim2bev": self.transform_sim2bev,
            "transform_bev2sim": self.transform_bev2sim,
        }


class FakeGraph:
    def __init__(self):
        self.track_main = ["chair"]
        self.track_sub = []
        self.GSSL = []
        self.global_objects = {"main": [], "sub": []}
        self.plan_result = {"explore_direction": "front"}

    def update_stage(self, map_info):
        self.map_info = map_info

    def GSSL_gen(self, loc_agent):
        del loc_agent
        self.GSSL = []

    def get_nav_mode_and_goal_via_fuse(self, loc_agent, vis=True):
        del loc_agent, vis
        return "recognition", None


class NavigationDiagnosticsIntegrationTest(unittest.TestCase):
    def test_episode_writes_complete_structured_log(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_0"
            obs_dir = episode_dir / "saved_images"
            depth_dir = episode_dir / "saved_depths"
            segment_dir = episode_dir / "saved_segment_results"
            midterm_dir = episode_dir / "saved_midterm_planned"
            shorterm_dir = episode_dir / "saved_shorterm_planned"
            for path in (obs_dir, depth_dir, segment_dir, midterm_dir, shorterm_dir):
                path.mkdir(parents=True, exist_ok=True)

            env = FakeEnvironment()
            agent = FakeAgent(env)
            args = SimpleNamespace(
                turn_angle=360,
                max_episode_steps=3,
                succeed_match_points=200,
                goal_type="object",
                save_episode_video=False,
                failure_diagnostics_enabled=True,
                failure_diagnostics_flush_every=4,
                failure_diagnostics_collision_motion_m=0.02,
                failure_diagnostics_repeat_goal_radius_m=0.75,
                failure_diagnostics_distance_interval=1,
                success_dist=1.0,
            )

            success, steps, _ = run_episode(
                0,
                env,
                agent,
                FakeGraph(),
                FakeMap,
                {},
                args,
                use_vlm_per_stage=False,
                obs_save_folder=str(obs_dir),
                depth_save_folder=str(depth_dir),
                segment_save_folder=str(segment_dir),
                midterm_save_folder=str(midterm_dir),
                shorterm_save_folder=str(shorterm_dir),
            )

            rows = [
                json.loads(line)
                for line in (episode_dir / "failure_diagnostics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            events = [row["event"] for row in rows]
            self.assertFalse(success)
            self.assertEqual(steps, 3)
            self.assertIn("episode_start", events)
            self.assertIn("stage_map", events)
            self.assertIn("decision", events)
            self.assertIn("execution_segment", events)
            self.assertEqual(events[-1], "episode_end")


if __name__ == "__main__":
    unittest.main()
