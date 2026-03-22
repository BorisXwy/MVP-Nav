"""
MVP-Nav main entry (paper-aligned version, Habitat instance image goal simulation).

Uses the src package: core flow matches RSS2026 MVP-Nav paper and appendix.
- No finding/judgement edge-repositioning, no pre-trigger; strict Algorithm 1 per-stage order.
- Judgement success: after rotation capture, decide by LightGlue match count (>= succeed_match_points) between goal image and observations.

Runs single-episode navigation loop via src.pipeline.run_episode.
"""

import os
import sys
import yaml
from types import SimpleNamespace
import numpy as np
import torch
import argparse
from datetime import datetime
import time

# Device can be set in config or here
target_device_id = 0
if torch.cuda.is_available():
    torch.cuda.set_device(target_device_id)

from src.envs.habitat import construct_envs
from src.agent.unigoal.agent import UniGoal_Agent
from src.graph.graphv2 import Graph
from src.map.spacev6 import Map
from src.pipeline import run_episode
from src.map.vggt.models.vggt import VGGT

sys.path.append("third_party/Grounded-Segment-Anything/")
from grounded_sam_demo import load_model
from segment_anything import sam_model_registry, SamPredictor
from lightglue import LightGlue, DISK


def get_config():
    """Load config: merge CLI args with YAML."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/config_habitat.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--goal_type", default="ins-image", type=str)
    parser.add_argument("--self_designed", action="store_true")

    args = parser.parse_args([])

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    args_dict = vars(args)
    args_dict.update(config)
    args = SimpleNamespace(**args_dict)

    args.is_debugging = sys.gettrace() is not None
    if args.is_debugging:
        args.experiment_id = "debug"

    args.log_dir = os.path.join(args.dump_location, args.experiment_id, "log")
    args.visualization_dir = os.path.join(args.dump_location, args.experiment_id, "visualization")
    args.map_size = args.map_size_cm // args.map_resolution
    args.global_width, args.global_height = args.map_size, args.map_size
    args.local_width = int(args.global_width / args.global_downscaling)
    args.local_height = int(args.global_height / args.global_downscaling)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda:{target_device_id}" if args.cuda else "cpu")
    args.num_scenes = args.num_processes
    args.num_episodes = int(args.num_eval_episodes)
    args.apply_leveling_transform = bool(getattr(args, "apply_leveling_transform", True))

    return args


def add_lightglue_to_model_info(model_info):
    """Inject LightGlue extractor/matcher into model_info for Graph and Judgement matching."""
    if model_info.get("extractor") is not None and model_info.get("matcher") is not None:
        return
    extractor = DISK(max_num_keypoints=2048).eval().to(model_info["device"])
    matcher = LightGlue(features="disk").eval().to(model_info["device"])
    model_info["extractor"] = extractor
    model_info["matcher"] = matcher


def initialize_model(args):
    """Load Grounded-SAM and VGGT."""
    device = args.device
    print(f"Using device: {device}")

    print("Initializing and loading Grounded_SAM model...")
    groundingdino_config_file = "third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    groundingdino_checkpoint = "data/models/groundingdino_swint_ogc.pth"
    sam_version = "vit_h"
    sam_checkpoint = "data/models/sam_vit_h_4b8939.pth"
    local_bert_path = "/mnt/pool1/sharehome/xiewenyuan/vlm/object_nav/bert-base-uncased"
    groundingdino = load_model(
        groundingdino_config_file, groundingdino_checkpoint, local_bert_path, device
    )
    sam_predictor = SamPredictor(
        sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
    )

    print("Initializing and loading VGGT model...")
    vggt = VGGT()
    vggt.load_state_dict(torch.load("data/models/model.pt"))
    vggt.eval()
    model_info = {
        "vggt": vggt,
        "groundingdino": groundingdino,
        "sam_predictor": sam_predictor,
        "device": device,
    }
    return model_info


def main():
    args = get_config()
    model_info = initialize_model(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)

    add_lightglue_to_model_info(model_info)
    envs = construct_envs(args)
    step_size = envs.config["simulator"]["forward_step_size"]
    args.step_size = step_size
    agent = UniGoal_Agent(args, envs, model_info)

    current_time = datetime.now().strftime("%Y%m%d-%H-%M")
    # Paper-version log dir (separate from main.py vis_log for comparison)
    experiment_dir = os.path.join("vis_log_nav", current_time)
    os.makedirs(experiment_dir, exist_ok=True)
    good_experiment_dir = os.path.join("vis_log_nav_good", current_time)
    os.makedirs(good_experiment_dir, exist_ok=True)
    experiment_result_path = os.path.join(experiment_dir, "experiment_result.txt")
    with open(experiment_result_path, "w", encoding="utf-8") as exp_f:
        exp_f.write(f"Experiment start (paper-aligned src): {current_time}\n")

    with open("result_nav.txt", "w", encoding="utf-8") as result_f:
        result_f.write(f"Experiment start (paper-aligned): {current_time}\n")
        result_f.write(f"Episode logs: {experiment_dir}\n")

    success_count = 0
    episode_idx = 0
    finished = False

    while not finished:
        try:
            episode_dir = os.path.join(experiment_dir, f"episode_{episode_idx}")
            obs_save_folder = os.path.join(episode_dir, "saved_images")
            segment_save_folder = os.path.join(episode_dir, "saved_segment_results")
            depth_save_folder = os.path.join(episode_dir, "saved_depths")
            midterm_save_folder = os.path.join(episode_dir, "saved_midterm_planned")
            shorterm_save_folder = os.path.join(episode_dir, "saved_shorterm_planned")
            for folder in [
                obs_save_folder,
                depth_save_folder,
                segment_save_folder,
                midterm_save_folder,
                shorterm_save_folder,
            ]:
                os.makedirs(folder, exist_ok=True)

            episode_result_path = os.path.join(episode_dir, "result.txt")
            episode_start_time = time.time()

            args.midterm_save_folder = midterm_save_folder
            graph = Graph(args=args, model_info=model_info)
            obs, infos = agent.reset()
            graph.set_image_goal(envs.instance_imagegoal)

            prep_end_time = time.time()
            prep_elapsed = prep_end_time - episode_start_time

            with open(episode_result_path, "w", encoding="utf-8") as result_f:
                result_f.write(f"=== Episode: {episode_idx} (paper-aligned) ===\n")
                result_f.write(f"Prep time: {prep_elapsed:.2f} s\n")

                nav_start_time = time.time()
                success_this, step_count = run_episode(
                    episode_idx,
                    envs,
                    agent,
                    graph,
                    Map,
                    model_info,
                    args,
                    use_vlm_per_stage=getattr(args, "use_vlm_per_stage", True),
                    obs_save_folder=obs_save_folder,
                    depth_save_folder=depth_save_folder,
                    segment_save_folder=segment_save_folder,
                    midterm_save_folder=midterm_save_folder,
                    shorterm_save_folder=shorterm_save_folder,
                    result_file_handle=result_f,
                )
                nav_elapsed = time.time() - nav_start_time
                total_elapsed = time.time() - episode_start_time

                result_f.write(f"Nav time: {nav_elapsed:.2f} s\n")
                result_f.write(f"Steps this episode: {step_count}\n")
                result_f.write(f"Total time this episode: {total_elapsed:.2f} s\n")

            with open(experiment_result_path, "a", encoding="utf-8") as exp_f:
                exp_f.write(
                    f"Episode {episode_idx}: success={int(bool(success_this))}, "
                    f"steps={step_count}, total_time={total_elapsed:.2f} s\n"
                )

            if success_this:
                success_count += 1
                try:
                    is_good = False
                    if hasattr(envs, "is_agent_at_goal_border"):
                        is_good = envs.is_agent_at_goal_border(pixel_threshold=5)
                    if is_good:
                        import shutil
                        good_episode_dir = os.path.join(good_experiment_dir, f"episode_{episode_idx}")
                        shutil.copytree(episode_dir, good_episode_dir, dirs_exist_ok=True)
                        print(f"[GOOD] Episode {episode_idx} saved to {good_episode_dir}")
                except Exception as copy_e:
                    print(f"Failed to save good episode: {copy_e}")

        except Exception as e:
            import traceback
            msg = f"!!! Episode {episode_idx} exception: {str(e)}"
            print(msg)
            traceback.print_exc()
            episode_dir = os.path.join(experiment_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)
            episode_result_path = os.path.join(episode_dir, "result.txt")
            with open(episode_result_path, "a", encoding="utf-8") as result_f:
                result_f.write(msg + "\n")
            with open(experiment_result_path, "a", encoding="utf-8") as exp_f:
                exp_f.write(msg + "\n")
            with open("result_nav.txt", "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        episode_idx += 1
        if episode_idx >= args.num_episodes:
            finished = True

    sr = success_count / episode_idx if episode_idx > 0 else 0
    summary = f"Success rate: {sr}"
    print(summary)
    with open(experiment_result_path, "a", encoding="utf-8") as exp_f:
        exp_f.write(summary + "\n")
    with open("result_nav.txt", "a", encoding="utf-8") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
