from omegaconf import OmegaConf
import numpy as np
from pathlib import Path

from habitat.config.default import get_config
from habitat import make_dataset

from src.data.nav_datasets import dataset_json_path, get_spec, scene_root

from .instanceimagegoal_env import InstanceImageGoal_Env


def _omega_update(config, key, value):
    OmegaConf.update(config, key, value, force_add=True)


def make_env_fn(args, config_env, rank=0):
    dataset = make_dataset(config_env.habitat.dataset.type, config=config_env.habitat.dataset)
    OmegaConf.set_readonly(config_env, False)
    if dataset.episodes:
        config_env.habitat.simulator.scene = dataset.episodes[0].scene_id
    OmegaConf.set_readonly(config_env, True)

    env = InstanceImageGoal_Env(args=args, rank=rank,
                            config_env=config_env,
                            dataset=dataset
                            )

    env.seed(rank)
    return env


def construct_envs(args):
    config_path = "src/envs/habitat/configs/" + args.task_config
    basic_config = get_config(config_path=config_path)
    OmegaConf.set_readonly(basic_config, False)
    basic_config.habitat.dataset.split = args.split
    _apply_dataset_overrides(args, basic_config)
    OmegaConf.set_readonly(basic_config, True)

    dataset = make_dataset(basic_config.habitat.dataset.type, config=basic_config.habitat.dataset)
    scenes = basic_config.habitat.dataset.content_scenes
    if "*" in basic_config.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(basic_config.habitat.dataset)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    env_idx = 0
    config_env = get_config(config_path=config_path)
    OmegaConf.set_readonly(config_env, False)
    _apply_dataset_overrides(args, config_env)

    if len(scenes) > 0:
        contentss = scenes[
            sum(scene_split_sizes[:env_idx]):
            sum(scene_split_sizes[:env_idx + 1])
        ]

        config_env.habitat.dataset.content_scenes = list(contentss)

    gpu_id = args.device.index

    config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

    config_env.habitat.environment.iterator_options.shuffle = False

    config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = args.env_frame_width
    config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = args.env_frame_height
    config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = args.hfov
    config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0, args.camera_height, 0]

    config_env.habitat.simulator.agents.main_agent.height = args.camera_height
    config_env.habitat.simulator.turn_angle = args.turn_angle

    config_env.habitat.dataset.split = args.split

    envs = make_env_fn(args, config_env)
    print(f"Habitat env ready: {envs.__class__.__name__}")
    return envs


def _apply_dataset_overrides(args, config_env):
    nav_dataset = getattr(args, "nav_dataset", None)
    if nav_dataset:
        spec = get_spec(nav_dataset)
        data_path = dataset_json_path(spec, args.split, getattr(args, "data_root", None))
        _omega_update(config_env, "habitat.dataset.type", {
            "objectnav": "ObjectNav-v1",
            "instance_imagenav": "InstanceImageNav-v1",
        }[spec.task])
        _omega_update(config_env, "habitat.dataset.data_path", str(data_path))
        _omega_update(config_env, "habitat.dataset.scenes_dir", str(scene_root(spec.dataset, getattr(args, "scene_data_root", None))))
        if spec.dataset == "hm3d":
            _omega_update(config_env, "habitat.dataset.content_scenes_path", "{data_path}/content/{scene}.json.gz")
        args.nav_task = spec.task
        args.nav_dataset_name = spec.dataset
        if spec.task == "instance_imagenav":
            args.instance_imagenav_dataset_dir = str(data_path.parent)
    elif getattr(args, "data_root", None):
        # Keep the YAML-selected dataset, but allow moving the data directory.
        data_path = str(config_env.habitat.dataset.data_path)
        if data_path.startswith("data/"):
            _omega_update(config_env, "habitat.dataset.data_path", str(resolve_relative_data_path(data_path, args.data_root)))


def resolve_relative_data_path(path, data_root):
    if path.startswith("data/"):
        return Path(data_root).expanduser().resolve() / path[5:]
    return path
