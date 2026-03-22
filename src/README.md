# MVP-Nav source layout (src, paper-aligned)

This directory is the **RSS 2026 paper- and appendix-aligned** implementation: copied from src1 with paper-conflicting changes removed. Entry point: project root `main_nav.py`.

Differences from src1: no finding/judgement edge-repositioning; no pre-trigger; Judgement success is decided by LightGlue match count (>= succeed_match_points) between goal image and observations after rotation capture.

---

## 1. Paper §III and src module mapping

| Paper component | Description | src module | Main files |
|-----------------|-------------|------------|------------|
| **1. Physical Perception** | Monocular sequence -> GSSL (3D OBB, pseudo-depth, VGGT + Grounded-SAM) | `map/` | `map/spacev6.py`: `obs_load` -> `build_pcd` -> `merge_objects`; VGGT in `map/vggt/` |
| **2. VLM-based High-level Reasoning** | GSSL + goal -> semantic scores, nav mode (Explore/Find/Judge) | `graph/` | `graph/graphv2.py`: `GSSL_gen`, `analyze_navigation_status`; `utils/llm.py` for LLM/VLM |
| **3. Multi-layer Valuemap (MVM) Planning** | Φsem, Φdir, Φtrav -> Φtotal, gmid = argmax | `graph/` | `graph/graphv2.py`: `get_nav_mode_and_goal_via_fuse` (from `fuse_fields_and_extract_goal`) |
| **4. Low-level Execution Loop** | A* + sliding-window FMM -> short-term goal gst, semantic ground reprojection safety check | `agent/` + `utils/fmm/` | `agent/unigoal/agent.py`: `get_action`, `step`; `utils/fmm/my_fmm.py`: `compute_astar_path`, `get_local_goal_fmm` |

---

## 2. Directory overview

```
src/
├── README.md                 # This file (paper-aligned)
├── core/                     # Paper four-layer entry points
│   ├── physical_perception.py  # §III-C
│   ├── vlm_reasoning.py        # §III-D
│   ├── mvm_planning.py         # §III-E
│   ├── low_level_execution.py  # §III-F, check_goal_match_in_obs
│   └── navigation.py         # Algorithm 1 run_episode (no edge-repos / pre-trigger)
├── pipeline/                 # run_episode delegates to core.navigation
├── agent/                    # Agent: actions, observations, short-term goals
├── envs/                     # Habitat and real-world envs
├── graph/                    # GSSL, VLM, MVM, nav mode and gmid
├── map/                      # Point cloud, OBB, BEV, fields
└── utils/                    # FMM, camera, LLM, visualization
```

---

## 3. Data flow (paper Fig.2 / Algorithm 1)

1. **Physical Perception**  
   Observation sequence -> `Map.obs_load()` -> VGGT depth/pose -> `build_pcd()` -> `merge_objects()` -> BEV and GSSL -> `generate_map_and_fields()`, `convert_to_sim()` -> `map_info`.

2. **VLM + MVM**  
   `map_info` -> `Graph.update_stage()` -> `Graph.get_nav_mode_and_goal_via_fuse(loc_agent)` -> `(nav_mode, midterm_goal)`.

3. **Low-level Execution**  
   explore: A* path -> FMM short-term goal -> `Agent.step()`; finding/judgement: rotate and match goal image to decide success.

Entry: `main_nav.py` (uses `src.pipeline`); `main.py` uses `src1` (with edge-repos and pre-trigger).

---

## 4. Decoupling (Map / Graph and external deps)

- **Map ↔ Agent**: `Map` takes `obs_provider` (e.g. from agent) with `obs_history`, `node/main/sub_segment_result`, `segment_save_folder`; `core/physical_perception.py` builds it from agent.
- **Graph ↔ third_party**: extractor and matcher (DISK/LightGlue) are injected via `model_info` by the main script; call `add_lightglue_to_model_info(model_info)` before creating `Graph`.
