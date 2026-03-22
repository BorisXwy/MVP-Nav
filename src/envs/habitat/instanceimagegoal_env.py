import json
import gzip
import gym
import numpy as np
import quaternion
import habitat
import os
import torch
import numpy as np
import json
from PIL import Image

from configs.categories import name2index
from src.utils.fmm.pose_utils import get_l2_distance, get_rel_pose_change
from habitat.utils.visualizations import maps


class InstanceImageGoal_Env(habitat.RLEnv):
    def __init__(self, args, rank, config_env, dataset):
        super().__init__(config_env, dataset)
        self.args = args
        self.rank = rank
        self._task_config = config_env

        self.split = config_env.habitat.dataset.split
        self.device = torch.device("cuda",  \
            int(config_env.habitat.simulator.habitat_sim_v0.gpu_device_id))
        self.episodes_dir = os.path.join("data/datasets/instance_imagenav/hm3d/v3", self.split)
        self.episode_no = -1

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # Episode Dataset info
        self.eps_data = None
        self.eps_data_idx = None
        self.goal_idx = None
        self.goal_name = None

        # Episode tracking info
        self.timestep = None
        self.stopped = None
        self.path_length = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None

        # Top-down trajectory
        self.topdown_map = None          # Grayscale base map (uint8)
        self.traj_pixels = []            # [(row, col), ...]
        self.traj_saved = False          # Avoid duplicate save

        self.name2index = name2index
        self.index2name = {v: k for k, v in self.name2index.items()}
        if self.args.goal_type == 'text':
            with gzip.open(self.args.text_goal_dataset, 'rt') as f:
                self.text_goal_dataset = json.load(f)
            self.average_acc = 0

    def update_after_reset(self):
        args = self.args

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        if self.scene_path != self.last_scene_path:
            episodes_file = self.episodes_dir + \
                "/content/{}.json.gz".format(scene_name)

            print("Loading episodes from: {}".format(episodes_file))
            with gzip.open(episodes_file, 'r') as f:
                self.eps_data = json.loads(
                    f.read().decode('utf-8'))["episodes"]

            self.eps_data_idx = 0
            self.last_scene_path = self.scene_path

        # Load episode info
        episode = self.eps_data[self.eps_data_idx]
        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)

        self.episode_geo_distance = episode["info"]["geodesic_distance"]
        self.episode_euc_distance = episode["info"]["euclidean_distance"]

        goal_name = episode["object_category"]
        goal_idx = episode["goal_object_id"]

        self.goal_idx = 0
        self.goal_name = goal_name
        self.gt_goal_idx = self.name2index[goal_name]
        self.goal_object_id = int(self._env.current_episode.goal_object_id)

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args
        self.global_step = 0
        new_scene = self.episode_no % args.num_eval_episodes == 0

        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []
       
        if self.args.environment == 'habitat':
            obs = super().reset()
        self.update_after_reset()

        # Init top-down map and trajectory cache for this episode
        self._init_topdown_map_and_traj()

        if new_scene:
            self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id
            print("Changing scene: {}/{}".format(self.rank, self.scene_name))

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id

        rgb = obs['rgb'].astype(np.uint8)

        if self.args.environment == 'habitat':
            self.last_sim_location = self.get_sim_location()
        # upstair or downstair check
        # self.start_height = self._env.current_episode.start_position[1]
        agent_state = self._env.sim.get_agent_state(0).position
        self.start_height = agent_state[1]
        self.agent_height = self.args.camera_height

        self.start_position = self._env.sim.get_agent_state(0).position
        self.start_rotation = self._env.sim.get_agent_state(0).rotation
            
        torch.set_grad_enabled(False)

        self.info['goal_cat_id'] = self.gt_goal_idx
        if self.args.goal_type == 'ins-image' or self.args.goal_type == 'text':
            if self.args.self_designed == True:
                instance_imagegoal_file = input("Enter the path to the instance imagegoal file: ")
                self.info['instance_imagegoal'] = np.array(Image.open(instance_imagegoal_file))
            else:
                self.info['instance_imagegoal'] = obs['instance_imagegoal']
            self.instance_imagegoal = self.info['instance_imagegoal']
        if self.args.goal_type == 'text':
            if self.args.self_designed == True:
                text_goal = input("Enter the text goal: ")
                self.info['text_goal'] = text_goal
            else:
                self.info['text_goal'] = self.text_goal_dataset['attribute_data'][self.habitat_env.current_episode.goal_key]
            self.text_goal = self.info['text_goal']

        print(f"rank:{self.rank}, episode:{self.episode_no}, cat_id:{self.gt_goal_idx}, cat_name:{self.goal_name}")
        torch.set_grad_enabled(True)

        # Set info
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['goal_cat_id'] = self.gt_goal_idx
        self.info['goal_name'] = self.goal_name
        self.info['agent_height'] = self.agent_height
        self.info['goal_key'] = self.habitat_env.current_episode.goal_key
        self.info['episode_no'] = self.episode_no

        rotation_angle = 0 
       
        # Record initial position to trajectory
        self._record_traj_point()

        return obs, self.info
    
    def set_goal_cat_id(self, idx):
        self.gt_goal_idx = idx
        self.info['goal_cat_id'] = idx
        self.info['goal_name'] = self.index2name[idx]

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        if action == 0:
        # if action['action_args']['velocity_stop'] > 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            # action = 3

        if self.args.environment == 'habitat':
            obs, rew, done, _ = super().step(action)

        # Record one trajectory point after each step
        self._record_traj_point()

        agent_state = self._env.sim.get_agent_state(0).position
        self.agent_height = self.args.camera_height + agent_state[1] - self.start_height
        self.info['agent_height'] = self.agent_height



        # Get pose change
        dx, dy, do = self.get_pose_change(obs)
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += get_l2_distance(0, dx, 0, dy)

        if done:
            if self.args.self_designed == True:
                spl, success, dist, soft_spl = 0., 0., 0., 0.
            else:
                spl, success, dist, soft_spl = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success
            self.info['soft_spl'] = soft_spl
            self.info['geo_distance'] = self.episode_geo_distance
            self.info['euc_distance'] = self.episode_euc_distance

        rgb = obs['rgb'].astype(np.uint8)

        self.timestep += 1
        self.info['time'] = self.timestep

        return obs, done, self.info

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 10.0)

    def get_reward(self, observations):
        _, s, d, _ = self.get_metrics()
        if d > 6. :
            d = 6.
        if self.args.environment == 'habitat':
            curr_sim_pose = self.get_sim_location()
        dx, dy, do = get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        reward =  10. * s
        
        return reward

    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        metrics = self.habitat_env.get_metrics()
        spl, success, dist = metrics['spl'], metrics['success'], metrics['distance_to_goal']
        soft_spl = metrics['soft_spl']
        return spl, success, dist, soft_spl

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        return self.info

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self, obs):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        if self.args.environment == 'habitat':
            curr_sim_pose = self.get_sim_location()
        dx, dy, do = get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def _init_topdown_map_and_traj(self, map_resolution: int = 1024):
        """
        Build top-down grayscale map from current scene NavMesh and clear this episode's trajectory cache.
        """
        if getattr(self, "habitat_env", None) is None:
            self.topdown_map = None
            self.traj_pixels = []
            self.traj_saved = False
            return

        sim = self.habitat_env.sim

        try:
            self.topdown_map = maps.get_topdown_map_from_sim(
                sim=sim,
                map_resolution=map_resolution,
            )
        except Exception:
            # Map build failed; skip trajectory drawing only
            self.topdown_map = None

        self.traj_pixels = []
        self.traj_saved = False

    def _record_traj_point(self):
        """
        Get current agent world coords (x,y,z) from Habitat sim, map to top-down (row,col) via maps.to_grid, append to trajectory cache.
        """
        if self.topdown_map is None or getattr(self, "habitat_env", None) is None:
            return

        sim = self.habitat_env.sim
        try:
            agent_state = sim.get_agent_state(0)
        except Exception:
            return

        pos = agent_state.position  # [x, y, z] in meters
        real_x = float(pos[0])
        real_z = float(pos[2])

        try:
            row, col = maps.to_grid(
                real_z,
                real_x,
                self.topdown_map.shape[:2],
                sim=sim,
            )
        except Exception:
            return

        H, W = self.topdown_map.shape[:2]
        if 0 <= row < H and 0 <= col < W:
            self.traj_pixels.append((int(row), int(col)))

    def save_topdown_traj(self, save_dir: str, filename: str = "topdown_traj.png"):
        """
        Draw this episode's trajectory on the top-down map and save to save_dir/filename.
        Trajectory as polyline; start (green), agent end (red); env goal (yellow) if available.
        """
        if (
            self.topdown_map is None
            or not self.traj_pixels
            or self.traj_saved
            or save_dir is None
        ):
            return

        import matplotlib.pyplot as plt
        import numpy as np

        H, W = self.topdown_map.shape[:2]
        fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)
        ax.imshow(self.topdown_map, cmap="gray")
        traj = np.array(self.traj_pixels, dtype=np.int32)
        rows, cols = traj[:, 0], traj[:, 1]
        ax.plot(cols, rows, color="cyan", linewidth=2, alpha=0.8, label="Trajectory")
        start_r, start_c = rows[0], cols[0]
        end_r, end_c = rows[-1], cols[-1]
        ax.scatter(start_c, start_r, c="lime", s=80, marker="o", edgecolors="black", linewidths=1.5, label="Start")
        ax.scatter(end_c, end_r, c="red", s=80, marker="o", edgecolors="white", linewidths=1.5, label="Agent End")
        goal_pixel = None
        try:
            episode = self.habitat_env.current_episode
            if episode.goals and hasattr(episode.goals[0], "position"):
                goal_pos = episode.goals[0].position
                sim = self.habitat_env.sim
                goal_row, goal_col = maps.to_grid(
                    goal_pos[2],  # real_z
                    goal_pos[0],  # real_x
                    self.topdown_map.shape[:2],
                    sim=sim,
                )
                if 0 <= goal_row < H and 0 <= goal_col < W:
                    goal_pixel = (goal_row, goal_col)
        except Exception:
            goal_pixel = None

        if goal_pixel is not None:
            gr, gc = goal_pixel
            ax.scatter(gc, gr, c="yellow", s=90, marker="*", edgecolors="black", linewidths=1.5, label="Env Goal")

        ax.set_axis_off()
        ax.legend(loc="lower right")
        plt.tight_layout(pad=0)

        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        self.traj_saved = True

    def is_agent_at_goal_border(self, pixel_threshold: int = 5) -> bool:
        """
        Check whether at episode end the agent is at env's true goal 'border':
        project episode.goal.position to topdown_map with same maps.to_grid; success if pixel distance to last trajectory point < threshold.
        """
        if self.topdown_map is None or not self.traj_pixels or getattr(self, "habitat_env", None) is None:
            return False

        H, W = self.topdown_map.shape[:2]
        end_row, end_col = self.traj_pixels[-1]
        if not (0 <= end_row < H and 0 <= end_col < W):
            return False

        goal_pixel = None
        try:
            episode = self.habitat_env.current_episode
            if episode.goals and hasattr(episode.goals[0], "position"):
                goal_pos = episode.goals[0].position  # (x, y, z)
                sim = self.habitat_env.sim
                goal_row, goal_col = maps.to_grid(
                    goal_pos[2],  # real_z
                    goal_pos[0],  # real_x
                    self.topdown_map.shape[:2],
                    sim=sim,
                )
                if 0 <= goal_row < H and 0 <= goal_col < W:
                    goal_pixel = (goal_row, goal_col)
        except Exception:
            goal_pixel = None

        if goal_pixel is None:
            return False

        gr, gc = goal_pixel
        dr = float(end_row - gr)
        dc = float(end_col - gc)
        dist = (dr ** 2 + dc ** 2) ** 0.5
        return dist <= float(pixel_threshold)
