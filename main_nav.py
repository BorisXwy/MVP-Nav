"""
MVP-Nav navigation entry.

The entry is split into config, model loading, env construction and episode
running so dataset/env smoke tests do not pay the model-loading cost.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
for extra_path in [
    PROJECT_ROOT / "third_party" / "habitat-lab",
    PROJECT_ROOT / "LightGlue-main",
]:
    if extra_path.exists():
        sys.path.insert(0, str(extra_path))

from src.data.nav_datasets import get_spec, verify_dataset


DEFAULT_DEVICE_ID = 1


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    print(f"[time] {label}: {time.perf_counter() - start:.2f}s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/config_habitat.yaml", type=str)
    parser.add_argument("--goal_type", default=None, choices=["ins-image", "text", "object"])
    parser.add_argument("--self_designed", action="store_true")
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID, type=int)
    parser.add_argument("--nav-dataset", default=None, choices=["instance_imagenav_hm3d", "objectnav_hm3d", "objectnav_mp3d"])
    parser.add_argument("--data-root", default="sharedata", type=str)
    parser.add_argument("--scene-data-root", default="data/scene_datasets", type=str)
    parser.add_argument("--bert-path", default="data/models/bert-base-uncased", type=str)
    parser.add_argument("--split", default=None, type=str)
    parser.add_argument("--num-episodes", default=None, type=int)
    parser.add_argument("--max-episode-length", default=None, type=int)
    parser.add_argument("--task-config", default=None, type=str)
    parser.add_argument("--skip-dataset-check", action="store_true")
    parser.add_argument("--smoke-env-only", action="store_true")
    return parser.parse_args()


def build_config(cli_args):
    with open(cli_args.config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    merged = dict(config)
    merged.update({k: v for k, v in vars(cli_args).items() if v is not None})

    if cli_args.nav_dataset:
        spec = get_spec(cli_args.nav_dataset)
        merged["nav_task"] = spec.task
        merged["nav_dataset_name"] = spec.dataset
        if cli_args.goal_type is None:
            merged["goal_type"] = "object" if spec.task == "objectnav" else "ins-image"
        if cli_args.task_config is None:
            merged["task_config"] = {
                "objectnav_hm3d": "tasks/objectnav_hm3d.yaml",
                "objectnav_mp3d": "tasks/objectnav_mp3d.yaml",
                "instance_imagenav_hm3d": "tasks/hm3d.yaml",
            }[cli_args.nav_dataset]
    elif cli_args.goal_type is None:
        merged["goal_type"] = merged.get("goal_type", "ins-image")

    if cli_args.split is not None:
        merged["split"] = cli_args.split
    if cli_args.num_episodes is not None:
        merged["num_eval_episodes"] = cli_args.num_episodes
    if cli_args.max_episode_length is not None:
        merged["max_episode_length"] = cli_args.max_episode_length
        merged["max_episode_steps"] = cli_args.max_episode_length
    if cli_args.task_config is not None:
        merged["task_config"] = cli_args.task_config

    args = SimpleNamespace(**merged)
    args.is_debugging = sys.gettrace() is not None
    if args.is_debugging:
        args.experiment_id = "debug"

    args.log_dir = os.path.join(args.dump_location, args.experiment_id, "log")
    args.visualization_dir = os.path.join(args.dump_location, args.experiment_id, "visualization")
    args.map_size = args.map_size_cm // args.map_resolution
    args.global_width = args.map_size
    args.global_height = args.map_size
    args.local_width = int(args.global_width / args.global_downscaling)
    args.local_height = int(args.global_height / args.global_downscaling)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda:{cli_args.device_id}" if args.cuda else "cpu")
    args.num_scenes = args.num_processes
    args.num_episodes = int(args.num_eval_episodes)
    args.apply_leveling_transform = bool(getattr(args, "apply_leveling_transform", True))
    args.data_root = cli_args.data_root
    args.scene_data_root = cli_args.scene_data_root
    args.bert_path = cli_args.bert_path
    args.skip_dataset_check = cli_args.skip_dataset_check
    args.smoke_env_only = cli_args.smoke_env_only

    if torch.cuda.is_available():
        torch.cuda.set_device(cli_args.device_id)

    return args


def check_dataset(args):
    if not getattr(args, "nav_dataset", None) or args.skip_dataset_check:
        return
    spec = get_spec(args.nav_dataset)
    found = verify_dataset(spec, [args.split], args.data_root)
    print(f"Dataset ready: {spec.name} -> {found[args.split]}")


def add_lightglue_to_model_info(model_info):
    from lightglue import DISK, LightGlue

    if model_info.get("extractor") is not None and model_info.get("matcher") is not None:
        return
    model_info["extractor"] = DISK(max_num_keypoints=2048).eval().to(model_info["device"])
    model_info["matcher"] = LightGlue(features="disk").eval().to(model_info["device"])


def initialize_model(args):
    sys.path.append("third_party/Grounded-Segment-Anything/")
    from grounded_sam_demo import load_model
    from segment_anything import SamPredictor, sam_model_registry
    from src.map.vggt.models.vggt import VGGT

    device = args.device
    print(f"Using device: {device}")

    with timed("Grounded-SAM load"):
        groundingdino_config_file = "third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        groundingdino_checkpoint = "data/models/groundingdino_swint_ogc.pth"
        sam_checkpoint = "data/models/sam_vit_h_4b8939.pth"
        local_bert_path = args.bert_path
        groundingdino = load_model(
            groundingdino_config_file,
            groundingdino_checkpoint,
            local_bert_path,
            device,
        )
        sam_predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device))

    with timed("VGGT load"):
        vggt = VGGT()
        vggt.load_state_dict(torch.load("data/models/model.pt", map_location=device))
        vggt.eval()

    model_info = {
        "vggt": vggt,
        "groundingdino": groundingdino,
        "sam_predictor": sam_predictor,
        "device": device,
    }
    add_lightglue_to_model_info(model_info)
    return model_info


def make_episode_dirs(experiment_dir, episode_idx):
    episode_dir = Path(experiment_dir) / f"episode_{episode_idx}"
    dirs = {
        "episode": episode_dir,
        "obs": episode_dir / "saved_images",
        "segment": episode_dir / "saved_segment_results",
        "depth": episode_dir / "saved_depths",
        "midterm": episode_dir / "saved_midterm_planned",
        "shorterm": episode_dir / "saved_shorterm_planned",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def configure_goal(graph, envs, args):
    if getattr(args, "goal_type", None) == "object":
        goal_name = getattr(envs, "goal_name", None) or getattr(envs, "object_goal", None)
        graph.set_image_goal(None, {"main_objects": [goal_name], "sub_objects": []})
        return
    graph.set_image_goal(envs.instance_imagegoal)


def run_smoke_env(args):
    from src.envs.habitat import construct_envs

    with timed("Habitat env construction"):
        envs = construct_envs(args)
    with timed("Habitat reset"):
        obs, info = envs.reset()
    print(
        "Smoke reset ok: "
        f"scene={getattr(envs, 'scene_path', None)}, "
        f"goal={info.get('goal_name')}, "
        f"obs_keys={sorted(obs.keys())}"
    )


def run_navigation(args):
    from src.agent.unigoal.agent import UniGoal_Agent
    from src.envs.habitat import construct_envs
    from src.graph.graphv2 import Graph
    from src.map.spacev6 import Map
    from src.pipeline import run_episode
    from src.core.nav_metrics import summarize_nav_metrics, zero_nav_metrics

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)

    model_info = initialize_model(args)
    with timed("Habitat env construction"):
        envs = construct_envs(args)

    args.step_size = envs.config["simulator"]["forward_step_size"]
    agent = UniGoal_Agent(args, envs, model_info)

    current_time = datetime.now().strftime("%Y%m%d-%H-%M")
    experiment_dir = os.path.join("vis_log_nav", current_time)
    good_experiment_dir = os.path.join("vis_log_nav_good", current_time)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(good_experiment_dir, exist_ok=True)
    experiment_result_path = os.path.join(experiment_dir, "experiment_result.txt")

    with open(experiment_result_path, "w", encoding="utf-8") as exp_f:
        exp_f.write(f"Experiment start: {current_time}\n")
    with open("result_nav.txt", "w", encoding="utf-8") as result_f:
        result_f.write(f"Experiment start: {current_time}\n")
        result_f.write(f"Episode logs: {experiment_dir}\n")

    success_count = 0
    metric_rows = []
    for episode_idx in range(args.num_episodes):
        episode_start_time = time.time()
        dirs = make_episode_dirs(experiment_dir, episode_idx)
        episode_result_path = dirs["episode"] / "result.txt"

        try:
            args.midterm_save_folder = str(dirs["midterm"])
            graph = Graph(args=args, model_info=model_info)
            obs, infos = agent.reset()
            configure_goal(graph, envs, args)

            with open(episode_result_path, "w", encoding="utf-8") as result_f:
                result_f.write(f"=== Episode: {episode_idx} ===\n")
                result_f.write(f"Prep time: {time.time() - episode_start_time:.2f} s\n")
                nav_start_time = time.time()
                success_this, step_count, episode_metrics = run_episode(
                    episode_idx,
                    envs,
                    agent,
                    graph,
                    Map,
                    model_info,
                    args,
                    use_vlm_per_stage=getattr(args, "use_vlm_per_stage", True),
                    obs_save_folder=str(dirs["obs"]),
                    depth_save_folder=str(dirs["depth"]),
                    segment_save_folder=str(dirs["segment"]),
                    midterm_save_folder=str(dirs["midterm"]),
                    shorterm_save_folder=str(dirs["shorterm"]),
                    result_file_handle=result_f,
                )
                result_f.write(f"Nav time: {time.time() - nav_start_time:.2f} s\n")
                result_f.write(f"Steps this episode: {step_count}\n")
                result_f.write(
                    "Metrics this episode: "
                    f"success={episode_metrics['success']:.3f}, "
                    f"spl={episode_metrics['spl']:.3f}, "
                    f"soft_spl={episode_metrics['soft_spl']:.3f}, "
                    f"distance_to_goal={episode_metrics['distance_to_goal']:.3f}\n"
                )
                result_f.write(f"Total time this episode: {time.time() - episode_start_time:.2f} s\n")

            success_count += int(bool(success_this))
            metric_rows.append(episode_metrics)
            append_experiment_result(
                experiment_result_path,
                episode_idx,
                success_this,
                step_count,
                episode_start_time,
                episode_metrics,
            )
            save_good_episode_if_needed(envs, success_this, dirs["episode"], good_experiment_dir, episode_idx)
        except Exception as exc:
            metric_rows.append(zero_nav_metrics())
            log_episode_exception(exc, episode_idx, experiment_dir, experiment_result_path)

    summary_metrics = summarize_nav_metrics(metric_rows)
    summary = (
        f"SR: {summary_metrics['sr']:.4f}, "
        f"SPL: {summary_metrics['spl']:.4f}, "
        f"SoftSPL: {summary_metrics['soft_spl']:.4f}, "
        f"DTS: {summary_metrics['distance_to_goal']:.4f}"
    )
    print(summary)
    with open(experiment_result_path, "a", encoding="utf-8") as exp_f:
        exp_f.write(summary + "\n")
    with open("result_nav.txt", "a", encoding="utf-8") as f:
        f.write(summary + "\n")


def append_experiment_result(path, episode_idx, success, steps, episode_start_time, metrics):
    with open(path, "a", encoding="utf-8") as exp_f:
        exp_f.write(
            f"Episode {episode_idx}: success={int(bool(success))}, "
            f"spl={metrics['spl']:.4f}, "
            f"soft_spl={metrics['soft_spl']:.4f}, "
            f"distance_to_goal={metrics['distance_to_goal']:.4f}, "
            f"steps={steps}, total_time={time.time() - episode_start_time:.2f} s\n"
        )


def save_good_episode_if_needed(envs, success, episode_dir, good_experiment_dir, episode_idx):
    if not success or not hasattr(envs, "is_agent_at_goal_border"):
        return
    try:
        if envs.is_agent_at_goal_border(pixel_threshold=5):
            import shutil

            good_episode_dir = os.path.join(good_experiment_dir, f"episode_{episode_idx}")
            shutil.copytree(episode_dir, good_episode_dir, dirs_exist_ok=True)
            print(f"[GOOD] Episode {episode_idx} saved to {good_episode_dir}")
    except Exception as exc:
        print(f"Failed to save good episode: {exc}")


def log_episode_exception(exc, episode_idx, experiment_dir, experiment_result_path):
    import traceback

    msg = f"!!! Episode {episode_idx} exception: {exc}"
    print(msg)
    traceback.print_exc()
    episode_dir = Path(experiment_dir) / f"episode_{episode_idx}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    with open(episode_dir / "result.txt", "a", encoding="utf-8") as result_f:
        result_f.write(msg + "\n")
    with open(experiment_result_path, "a", encoding="utf-8") as exp_f:
        exp_f.write(msg + "\n")
    with open("result_nav.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    args = build_config(parse_args())
    check_dataset(args)
    if args.smoke_env_only:
        run_smoke_env(args)
    else:
        run_navigation(args)


if __name__ == "__main__":
    main()
