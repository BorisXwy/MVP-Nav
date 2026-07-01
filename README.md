# MVP-Nav

**MVP-Nav: Multi-layer Value Map Planner Navigator** is a physical-aware, RGB-only framework for zero-shot Object Goal Navigation (ZSON). MVP-Nav has been accepted to **RSS 2026**.

MVP-Nav reconstructs explicit physical occupancy from monocular RGB observations, lifts 2D semantic instances into 3D oriented bounding boxes, and plans through a Multi-layer Value Map (MVM) that combines semantic priorities, exploration guidance, and traversability constraints.

![MVP-Nav method overview](docs/assets/method_overview.png)

## Method Overview

The system is organized as a recursive navigation loop:

1. **Physical Perception**: RGB observations are processed by VGGT and Grounded-SAM to build point clouds, oriented bounding boxes, BEV maps, and a Global Spatial Semantic List (GSSL).
2. **VLM Reasoning**: the goal and GSSL are scored by an LLM/VLM module to choose the navigation mode and semantic priorities.
3. **MVM Planning**: semantic, direction, exploration, and safety fields are fused into a shared cost space to select a midterm goal.
4. **Low-level Execution**: A* and FMM generate short-term goals, with semantic floor re-projection used as a safety check.

Core mapping:

| Component | Code |
| --- | --- |
| Physical Perception | `src/core/physical_perception.py`, `src/map/spacev6.py` |
| VLM Reasoning | `src/core/vlm_reasoning.py`, `src/graph/graphv2.py` |
| MVM Planning | `src/core/mvm_planning.py`, `src/graph/graphv2.py` |
| Low-level Execution | `src/core/low_level_execution.py`, `src/agent/unigoal/agent.py` |
| Episode loop | `src/core/navigation.py`, `src/pipeline/navigation_loop.py` |
| Main entry | `main_nav.py` |

## Repository Layout

```text
.
├── main_nav.py                         # Main Habitat navigation entry
├── configs/                            # Runtime config
├── scripts/download_nav_datasets.py    # Episode dataset downloader
├── src/
│   ├── agent/                          # Agent control and perception wrapper
│   ├── core/                           # Navigation modules
│   ├── data/                           # Dataset path and validation helpers
│   ├── envs/                           # Habitat and real-world env adapters
│   ├── graph/                          # GSSL, VLM reasoning, MVM
│   ├── map/                            # VGGT, point cloud, BEV and OBB logic
│   └── pipeline/                       # run_episode wrapper
└── docs/assets/                        # README figures
```

Large datasets, checkpoints, logs, and local experiment artifacts are ignored by git.

## Installation

The environment follows the same dependency family as UniGoal, with additional MVP-Nav modules. Use Python 3.8 for best compatibility with Habitat 0.2.3.

### 1. Create Environment

```bash
conda create -n mvpnav python=3.8
conda activate mvpnav
```

Install Habitat:

```bash
conda install habitat-sim==0.2.3 -c conda-forge -c aihabitat
pip install -e third_party/habitat-lab
```

### 2. Install Third-party Packages

Install LightGlue, Detectron2, PyTorch3D, and Grounded-SAM:

```bash
pip install git+https://github.com/cvg/LightGlue.git
pip install git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/facebookresearch/pytorch3d.git

git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git third_party/Grounded-Segment-Anything
cd third_party/Grounded-Segment-Anything
git checkout 5cb813f
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd ../../
```

Install common runtime packages:

```bash
conda install pytorch::faiss-gpu
pip install hydra-core omegaconf gym numpy-quaternion opencv-python scikit-image matplotlib supervision transformers
```

If a `requirements.txt` is provided in your checkout, install it after the Habitat and third-party packages:

```bash
pip install -r requirements.txt
```

### 3. Model Checkpoints

Place model files under `data/models/`:

```text
data/models/
├── groundingdino_swint_ogc.pth
├── sam_vit_h_4b8939.pth
├── model.pt                         # VGGT checkpoint
└── bert-base-uncased/                # Local BERT tokenizer/model directory
```

Download the public SAM and GroundingDINO checkpoints:

```bash
mkdir -p data/models
wget -O data/models/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O data/models/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Pass a different BERT path with:

```bash
python main_nav.py --bert-path path/to/bert-base-uncased ...
```

### 4. LLM and VLM

Option 1: use a local Ollama model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2-vision
```

Option 2: use an OpenAI-compatible API. Set `llm_model`, `vlm_model`, `api_key`, and `base_url` in `configs/config_habitat.yaml`.

Do not commit private API keys.

## Datasets

By default, episode datasets are expected under:

```text
sharedata/datasets/
```

You can override this with `--data-root`.

### ObjectNav Episodes

Download Habitat ObjectNav episode files:

```bash
python scripts/download_nav_datasets.py --dataset objectnav_hm3d --data-root sharedata
python scripts/download_nav_datasets.py --dataset objectnav_mp3d --data-root sharedata
```

Expected layout:

```text
sharedata/
└── datasets/
    └── objectnav/
        ├── hm3d/
        │   └── v2/
        │       ├── train/
        │       ├── val/
        │       └── val_mini/
        └── mp3d/
            └── v1/
                ├── train/
                ├── val/
                └── val_mini/
```

### Scene Assets

HM3D and MP3D scene meshes require separate access/licensing and are not downloaded by `download_nav_datasets.py`.

Place or symlink scenes under:

```text
data/
└── scene_datasets/
    ├── hm3d_v0.2/
    │   └── val/
    │       ├── 00800-TEEsavR23oF/
    │       ├── ...
    │       └── 00899-58NLZxWBSpk/
    └── mp3d/
        ├── 17DRP5sb8fy/
        ├── ...
        └── zsNo4HB9uLZ/
```

### InstanceImageNav and TextNav

For instance-image-goal and text-goal runs, use the layout inherited from UniGoal:

```text
data/
└── datasets/
    ├── textnav/
    │   └── val/
    │       └── val_text.json.gz
    └── instance_imagenav/
        └── hm3d/
            └── v3/
                └── val/
                    ├── content/
                    │   ├── <scene>.json.gz
                    │   └── ...
                    └── val.json.gz
```

## Quick Checks

Validate syntax and dataset paths:

```bash
python -m py_compile main_nav.py src/data/nav_datasets.py scripts/download_nav_datasets.py
python main_nav.py --nav-dataset objectnav_hm3d --split val_mini --num-episodes 1 --smoke-env-only
python main_nav.py --nav-dataset objectnav_mp3d --split val_mini --num-episodes 1 --smoke-env-only
```

`--smoke-env-only` constructs Habitat and resets one episode without loading the heavy MVP-Nav perception models.

## Running

Run ObjectNav on HM3D:

```bash
python main_nav.py \
  --nav-dataset objectnav_hm3d \
  --split val_mini \
  --num-episodes 1
```

Run ObjectNav on MP3D:

```bash
python main_nav.py \
  --nav-dataset objectnav_mp3d \
  --split val_mini \
  --num-episodes 1
```

Run InstanceImageNav on HM3D:

```bash
python main_nav.py \
  --nav-dataset instance_imagenav_hm3d \
  --split val_mini \
  --num-episodes 1
```

Useful flags:

```text
--device-id 0
--data-root sharedata
--scene-data-root data/scene_datasets
--bert-path data/models/bert-base-uncased
--skip-dataset-check
--smoke-env-only
```

Outputs are written to:

```text
vis_log_nav/<timestamp>/
vis_log_nav_good/<timestamp>/
result_nav.txt
outputs/
```

These paths are ignored by git.

## Code Structure

Core:

- `main_nav.py`: entry point for Habitat evaluation.
- `src/agent/unigoal/agent.py`: observation handling, semantic segmentation, and low-level actions.
- `src/graph/graphv2.py`: goal parsing, GSSL, semantic scoring, and MVM planning.
- `src/map/spacev6.py`: VGGT-based reconstruction, object fusion, OBBs, and BEV map generation.
- `src/core/navigation.py`: recursive episode loop.

Environment and utilities:

- `src/envs/habitat/`: Habitat environment wrappers and task configs.
- `src/utils/fmm/`: Fast Marching Method utilities.
- `src/utils/visualization/`: visualization and video helpers.
- `src/utils/llm.py`: LLM/VLM wrappers.
- `src/data/nav_datasets.py`: dataset specs, paths, and validation helpers.

## Notes

- Keep datasets, checkpoints, generated logs, and private credentials out of git.
- The repository assumes Habitat scene assets are prepared locally under `data/scene_datasets/`.
- `sharedata/` is a convenient local/shared data root; use `--data-root` if your layout differs.
