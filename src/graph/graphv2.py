import numpy as np 
import cv2 
import os
import sys  
from scipy.spatial import KDTree 
import json

from scipy.ndimage import binary_dilation, gaussian_filter 
from typing import List, Dict, Any
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from src.utils.my_tools import try_merge_obb,get_mode,apply_transform
from src.utils.llm import LLM, VLM
from src.graph.graphbuilder import GraphBuilder
from src.graph.goalgraphdecomposer import GoalGraphDecomposer
import math
from collections import Counter

from PIL import Image
# Decouple from third_party: extractor/matcher injected via model_info by main; fallback here
try:
    from lightglue import LightGlue, DISK
except ImportError:
    LightGlue = DISK = None


# Semantic field Gaussian decay: pixel distance and pixel size only
def gaussian_decay(distance, extent_size_pixel, score):
    """
    Potential field Gaussian decay: decay weight from distance (px), OBB size (px) and score.
    """
    sigma = np.linalg.norm(extent_size_pixel[[0, 2]]) * score / 3.0
    MIN_SIGMA_PIXELS = 3.0 
    sigma = max(sigma, MIN_SIGMA_PIXELS) 
    decay_weight = np.exp(-0.5 * (distance / sigma)**2)
    return decay_weight


class Graph:
    
    def __init__(self, args, model_info):
        self.args = args
        self.map_info = None
        self.global_objects = {
            'node': [],
            'main': [],
            'sub': [],
        }
        self.global_kd_tree = {
            'node': None,
            'main': None,
            'sub': None,
        }
        self.stage = None
        self.grid_resolution = None
        
        self.last_mode = "explore"
        self.last_goal = None
        self.last_high_interest_points = []
        
        self.stage_space_safety_field = None
        self.stage_space_explore_field = None
        self.stage_space_exploration_history_field = None
        self._exploration_history_trajectories = []
        self.gradient_fields = {} 
        
        self.stage_bev = None
        self.safe_ground_mask = None
        self.obstacle_mask = None    
        self.unknown_mask = None    
        # Init semantic / direction fields to avoid AttributeError on first use
        self.stage_semantic_field = None
        self.stage_direction_field = None
        
        self.BG_CLASSES = ["wall", "floor", "ceiling", "ground", "furniture", "rug", "kitchen"] 
        self.node_space = 'table. tv. chair. cabinet. sofa. bed. window. plant. door. doorframe'
        self.prompt_image2text = (
            "This image is a NAVIGATION GOAL (indoor scene). List only objects that typically appear indoors in a house or room.\n"
            "Output STRICTLY in this format: [<Primary Object>, <Secondary 1>, <Secondary 2>, ...]\n"
            "Rules:\n"
            "1. Primary (first item): THE single object the agent should navigate to or interact with—e.g. firebox, sofa, cat, cabinet. Only ONE noun.\n"
            "2. Secondary (optional, 0–4 items): Other discrete objects that help confirm the place (e.g. chair, table, door). Do NOT list floors, walls, ceilings, outlets, or surfaces (e.g. tiled floor, rug).\n"
            "2b. All items must be indoor/household objects (e.g. furniture, appliances, fixtures); do not list outdoor-only things (e.g. tree, car, sky).\n"
            "3. Use ONLY bare nouns: 'chair' not 'wooden chair', 'window' not 'window with blinds'. No adjectives, no extra phrases.\n"
            "4. Write each object as ONE word with no spaces: 'coffeecup' not 'coffee cup', 'firebox' not 'fire box'. Concatenate multi-word terms.\n"
            "5. Output ONLY the list, no other text. Example: [Firebox, Coffeecup, Chair, Table]"
        )
        self.grounded_sam = (model_info['groundingdino'], model_info['sam_predictor'])
        self.device = model_info['device']
        self.llm = LLM(self.args.base_url, self.args.api_key, self.args.llm_model)
        self.vlm = VLM(self.args.base_url, self.args.api_key, self.args.vlm_model)
        self.graphbuilder = GraphBuilder(self.llm)
        self.goalgraphdecomposer = GoalGraphDecomposer(self.llm)
        # Prefer main-injected extractor/matcher
        if model_info.get("extractor") is not None and model_info.get("matcher") is not None:
            self.extractor = model_info["extractor"]
            self.image_matcher = model_info["matcher"]
        elif LightGlue is not None and DISK is not None:
            self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
            self.image_matcher = LightGlue(features='disk').eval().to(self.device)
        else:
            raise RuntimeError("Graph requires model_info extractor/matcher or lightglue.DISK/LightGlue")
        self.previous_nav_mode = "explore"
        self.save_folder = None

    def _get_hw_shape(self):
        """Return (H, W) of stage_bev."""
        if self.stage_bev is not None and self.stage_bev.ndim >= 2:
            return self.stage_bev.shape[0], self.stage_bev.shape[1] 
        return 0, 0 

    def update_stage(self, map_info):
        """
        Update current stage data, compute traversability masks, load spatial fields from perception.
        """
        self.map_info = map_info
        self.stage = map_info['stage'] 
        self.stage_bev = map_info['stage_bev']
        self.save_folder = map_info['save_folder']
        self.bev_step = map_info['bev_step']
        
        # 使用 cv2.inRange 计算掩码
        if self.stage_bev is not None and self.stage_bev.ndim == 3:
            bev_np = self.stage_bev
            self.obstacle_mask = cv2.inRange(bev_np, np.array([40, 40, 40]), np.array([60, 60, 60]))
            self.safe_ground_mask = cv2.inRange(bev_np, np.array([190, 190, 190]), np.array([210, 210, 210]))
            self.unknown_mask = cv2.inRange(bev_np, np.array([240, 240, 240]), np.array([255, 255, 255]))
            # safe_areas = cv2.inRange(bev_np, np.array([190, 190, 190]), np.array([210, 210, 210]))
            # unknown_areas = cv2.inRange(bev_np, np.array([240, 240, 240]), np.array([255, 255, 255]))

            # # 将所有检测到的区域都合并为安全区域
            # self.safe_ground_mask = cv2.bitwise_or(safe_areas, unknown_areas)
        else:
            H, W = self._get_hw_shape()
            self.safe_ground_mask = np.zeros((H, W), dtype=np.uint8)
            self.obstacle_mask = np.zeros((H, W), dtype=np.uint8)
            self.unknown_mask = np.zeros((H, W), dtype=np.uint8)

        self.map_pixels_x, self.map_pixels_z = self._get_hw_shape()
            
        self.transform_sim2bev = map_info['transform_sim2bev']
        self.transform_bev2sim = map_info['transform_bev2sim']
        self.grid_resolution = map_info['grid_resolution']

        # 当前阶段物体列表与轨迹：对 None / 非预期类型做鲁棒处理，避免后续下标访问出错
        raw_stage_obj_bev = map_info.get('stage_object_list_bev')
        if isinstance(raw_stage_obj_bev, dict):
            self.stage_object_list_bev = raw_stage_obj_bev
        else:
            self.stage_object_list_bev = {"node": [], "main": [], "sub": []}

        raw_stage_obj_sim = map_info.get('stage_objects_list_sim')
        if isinstance(raw_stage_obj_sim, dict):
            self.stage_object_list_sim = raw_stage_obj_sim
        else:
            self.stage_object_list_sim = {"node": [], "main": [], "sub": []}

        traj_bev = map_info.get('stage_trajectory_bev')
        if traj_bev is None:
            self.stage_trajectory_bev = np.zeros((0, 2), dtype=np.float32)
        else:
            self.stage_trajectory_bev = traj_bev
        
        self.stage_space_safety_field = map_info.get('stage_space_safety_field', None)
        self.stage_space_explore_field = map_info.get('stage_space_explore_field', None)

        # 探索历史场：将当前阶段轨迹转为 sim 坐标，保留全部阶段（不截断）
        traj_bev = map_info.get('stage_trajectory_bev')
        transform_bev2sim = map_info.get('transform_bev2sim')
        if traj_bev is not None and transform_bev2sim is not None and getattr(traj_bev, 'shape', (0,))[0] > 0:
            traj_sim_list = []
            for i in range(traj_bev.shape[0]):
                px, pz = float(traj_bev[i, 0]), float(traj_bev[i, 1])
                x_sim, z_sim, _ = apply_transform((px, pz, 0.0), transform_bev2sim)
                traj_sim_list.append((x_sim, z_sim))
            self._exploration_history_trajectories.append((self.stage, traj_sim_list))

        if self.map_info['stage_map_quality'] == 'good':
            self.update_global_objects()

    def set_image_goal(self, image, description = None):
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self.instance_imagegoal = image
        if not description:
            text_goal = self.vlm(self.prompt_image2text, [self.instance_imagegoal])
            print("VLM 原版返回:", text_goal)
            main_objects = []
            sub_objects = []
            objects = self.graphbuilder.get_objects(text_goal)
            main_objects.append(objects[0])
            for obj in objects[1:]:
                sub_objects.append(obj)
            self.main_objects = main_objects
            self.sub_objects = sub_objects
        else: 
            self.main_objects = description['main_objects']
            self.sub_objects = description['sub_objects']

        self.track_main = self._process_goal_objects(self.main_objects)
        self.track_sub = self._process_goal_objects(self.sub_objects)
        print("主要目标物体:", self.track_main)
        print("次要目标物体:", self.track_sub)
        
    def _process_goal_objects(self, goal_objects):
        node_space = self.node_space
        # 1. 解析现有元素
        existing_elements = [item.strip() for item in node_space.split('.') if item.strip()]
        final_unique_list = list(existing_elements) 

        track_objects = set() 
        existing_set = set(final_unique_list) 


        temp_list_to_add = []
        
        for new_item in goal_objects:
            new_item = new_item.strip()
            if not new_item:
                continue
            if new_item in existing_set:
                track_objects.add(new_item)
                continue
            if 'room' in new_item or 'object' in new_item or 'hall' in new_item or 'office' in new_item :
                continue
            is_bg_item = False
            for bg_class in self.BG_CLASSES:
                if bg_class in new_item:
                    is_bg_item = True
                    break
            
            if is_bg_item:
                continue

            can_be_added = True

            for existing_item in final_unique_list:
                if (new_item in existing_item) or (existing_item in new_item):
                    track_objects.add(existing_item) 
                    can_be_added = False
                    break
            
            if not can_be_added:
                continue
                
            temp_list_to_add.append(new_item)
            existing_set.add(new_item) # 即使不立即添加到 final_unique_list，也更新集合以便后续检查
            track_objects.add(new_item) # 记录为跟踪对象

        temp_list_to_add = temp_list_to_add[:5]

        final_unique_list.extend(temp_list_to_add)


        formatted_elements = [
            item.replace(' ', '_') if ' ' in item else item
            for item in final_unique_list
        ]
        separator = '. '
        final_string_content = separator.join(formatted_elements)
        final_string = final_string_content + '.'
        
        self.node_space = final_string

        return list(track_objects)
    
    def update_global_objects(self):
        """仅在做融合/加入时赋予分数：加入时 main=1.0 / sub=0.8 / node=0（node 由 VLM 后处理到 0.4~0.8），不做全局统一设 0.9。"""
        for key in self.global_objects.keys():
    
            global_centers = [obj['center'][[0, 2]] for obj in self.global_objects[key] if 'center' in obj and len(obj['center']) >= 3]
            if global_centers:
                self.global_kd_tree[key] = KDTree(np.array(global_centers))
            else:
                self.global_kd_tree[key] = None

            FUSION_RADIUS = getattr(self.args, 'fusion_radius', 0.5) 
            
            # 2. 融合操作和探索度标注
            for i, stage_obj in enumerate(self.stage_object_list_sim[key]):
                stage_center_sim_2d = stage_obj['center'][[0, 2]]
                best_global_match_idx = None
                
                if self.global_kd_tree[key]:
                    nearby_indices = self.global_kd_tree[key].query_ball_point(stage_center_sim_2d, FUSION_RADIUS)
                    for global_idx in nearby_indices:
                        if global_idx >= len(self.global_objects[key]): continue 
                        global_obj = self.global_objects[key][global_idx]
                        can_merge, distance = try_merge_obb(global_obj, stage_obj, mode="strict")
                        if can_merge:
                            caption_condition = get_mode(global_obj.get('caption')) == get_mode(stage_obj.get('caption'))
                            if caption_condition:
                                best_global_match_idx = global_idx
                                break

                if best_global_match_idx is not None:

                    global_obj_to_update = self.global_objects[key][best_global_match_idx]
                    global_obj_to_update, _ = try_merge_obb(global_obj_to_update, stage_obj, mode="strict") 

                    exp_score = 0.9 # 【标记当前阶段物体】为已融合 (0.9)
                else:
                    new_obj = stage_obj.copy()
                    # 仅加入时赋分：main=1.0, sub=0.8, node=0（由 VLM 打分后重映射到 0.4~0.8）
                    if key == 'main':
                        new_obj['exploration_score'] = 1.0
                    elif key == 'sub':
                        new_obj['exploration_score'] = 0.8
                    else:
                        new_obj['exploration_score'] = 0.0  # node
                    self.global_objects[key].append(new_obj)

                    exp_score = 0.0 

                if key == 'node' and isinstance(self.stage_object_list_bev, dict):
                    bev_list_for_key = self.stage_object_list_bev.get(key) or []
                    if 0 <= i < len(bev_list_for_key):
                        bev_list_for_key[i]['exploration_score'] = exp_score
            
        # 3. 可视化
        self.visualize_fusion_results()

    def GSSL_gen(self, loc_agent):
        """
        简化物体列表并计算相对语义位置。
        为每个物体分配唯一 ID 以便后续精确回传分数。
        """
        object_list = self.global_objects
        agent_x, agent_z, theta = loc_agent
        simplified_list = []
        
        for key in self.global_objects.keys():
            # 使用 enumerate 确保同一类别的不同实例拥有唯一标识
            for idx, obj in enumerate(object_list[key]):
                center_bev = self.sim_to_bev_point(obj['center'])
                x_p, z_p = center_bev[0], center_bev[2]
                
                # 过滤视野范围内的物体
                if 0 <= x_p < self.map_pixels_x and 0 <= z_p < self.map_pixels_z:
                    # 1. 提取物体最频繁出现的标签
                    captions = obj.get('caption', [])
                    main_label = Counter(captions).most_common(1)[0][0] if captions else "unknown"

                    # 2. 计算相对角度和距离
                    dx = obj['center'][0] - agent_x
                    dz = obj['center'][2] - agent_z

                    global_angle = np.arctan2(dz, dx)
                    relative_angle = global_angle - theta
                    
                    # 标准化角度到 [-180, 180]
                    angle_deg = (np.degrees(relative_angle) + 180) % 360 - 180
                    distance = np.sqrt(dx**2 + dz**2)

                    # 3. 映射角度到英文方向描述
                    if -22.5 <= angle_deg < 22.5:
                        direction = "directly in front"
                    elif 22.5 <= angle_deg < 67.5:
                        direction = "to the front-right"
                    elif 67.5 <= angle_deg < 112.5:
                        direction = "to the right"
                    elif 112.5 <= angle_deg < 157.5:
                        direction = "to the back-right"
                    elif angle_deg >= 157.5 or angle_deg < -157.5:
                        direction = "directly behind"
                    elif -157.5 <= angle_deg < -112.5:
                        direction = "to the back-left"
                    elif -112.5 <= angle_deg < -67.5:
                        direction = "to the left"
                    else:
                        direction = "to the front-left"

                    # 4. 生成唯一 ID 并同步记录在原始对象中
                    obj_id = f"{key}_{idx}"
                    obj['temp_id'] = obj_id 

                    # 5. 构建简化对象（注意：这里的字典将保持对原始 obj 的引用）
                    simplified_list.append({
                        "id": obj_id,
                        "caption": main_label,
                        "position": f"{direction}, {distance:.2f}m away",
                        "explore": obj.get('merged_status', False),
                        "original_obj": obj  # 关键点：直接保存原始对象的引用以确保回传
                    })
                    
        self.GSSL = simplified_list
        return simplified_list

    def analyze_navigation_status(self, current_img, goal_img):
        """
        优化后的 VLM 分析函数：强化探索得分与方向的逻辑对齐。
        """
        GSSL = self.GSSL
        main_goal = self.track_main
        sub_goal = self.track_sub
        
        # 准备发送给 VLM 的数据，排除掉 Python 引用对象
        vlm_input_list = [
            {k: v for k, v in item.items() if k != 'original_obj'} 
            for item in GSSL
        ]
        gssl_text = json.dumps(vlm_input_list, indent=2)

        prompt = f"""
        You are an expert robotic navigation system. Analyze the Ego-view (Image 1) and the Goal (Image 2).
        
        [Context]
        - Main Target: {main_goal}
        - Current Sub-goal: {sub_goal}
        - Object List (GSSL): {gssl_text}

        [Task Requirements]
        1. **Semantic Scoring**: Assign "exploration_score" (0.5-1.0) to GSSL objects (where explore=false). 
           - High scores for objects resembling the target OR structural elements (doors, hallways) that likely lead to the target.
        2. **Directional Alignment**: Identify the open area (pathway) in Image 1 that is most promising. 
           - The suggested "explore_direction" MUST align with the sector containing your highest-scored objects.
        3. **Navigation Mode**: 
           - 'find': Target is clearly visible and reachable.
           - 'judge': An object looks like the target but needs closer verification.
           - 'explore': Target not visible; must move toward high-potential areas.

        [Output Constraints]
        - Your "explore_direction" should be: 'left' (if path is +30° to +90°), 'front' (-30° to +30°), or 'right' (-90° to -30°).
        - Ensure logical consistency: If the best object is "to the left", the direction must be "left".

        Return ONLY a JSON object:
        {{
          "reasoning": "Briefly explain the target location and path choice.",
          "scores": [{{"id": "object_id", "score": 0.9}}],
          "mode": "explore/find/judge",
          "explore_direction": "left/front/right"
        }}
        """

        # 调用 VLM (假设 self.vlm 支持图像列表)
        response_text = self.vlm(prompt, [current_img, goal_img])
        
        try:
            # 兼容处理 Markdown 格式的 JSON
            clean_json = response_text.replace('```json', '').replace('```', '').strip()
            decision = json.loads(clean_json)
        except Exception as e:
            print(f"VLM JSON Parsing Error: {e}")
            decision = {"scores": [], "mode": "explore", "explore_direction": "front"}

        # 提取 VLM 原始分数映射
        score_map = {str(item['id']): float(item['score']) for item in decision.get('scores', [])}

        # 1) 先把 VLM 分数写回 GSSL / original_obj
        for item in GSSL:
            # 已被标记为 -1 的目标：保持淘汰状态，不再参与 VLM 更新
            prev_score = float(item['original_obj'].get('exploration_score', 0.0))
            if prev_score < 0.0:
                item['exploration_score'] = -1.0
                item['original_obj']['exploration_score'] = -1.0
                continue

            obj_id = item['id']
            if obj_id in score_map:
                new_val = score_map[obj_id]
            else:
                # 保留上一时刻的分数，若无则初始化为 0
                new_val = prev_score
            item['exploration_score'] = new_val
            item['original_obj']['exploration_score'] = new_val

        # 2) 按照语义类别重写最终得分：
        #    - node 类: 仅在 0.4~0.8 区间内线性压缩
        #    - main 类: 自动 1.0
        #    - sub  类: 自动 0.8
        for item in GSSL:
            # 明确淘汰的目标：保持 -1，不参与 main/sub/node 重映射
            if float(item.get('exploration_score', 0.0)) < 0.0:
                item['cls_type'] = 'rejected'
                continue

            caption = str(item.get('caption', '')).lower()
            base_score = float(item['exploration_score'])

            # 判断是否属于 main / sub 目标集合
            is_main = any(m.lower() in caption for m in self.track_main)
            is_sub = any(s.lower() in caption for s in self.track_sub)

            if is_main:
                final_score = 1.0
                cls_type = 'main'
            elif is_sub:
                final_score = 0.8
                cls_type = 'sub'
            else:
                # 其他都视作 node 类：将原始 0~1 分数压缩到 [0.4, 0.8]
                clipped = max(0.0, min(1.0, base_score))
                final_score = 0.4 + 0.4 * clipped
                cls_type = 'node'

            item['exploration_score'] = final_score
            item['original_obj']['exploration_score'] = final_score
            item['cls_type'] = cls_type  # 方便后续调试或可视化

        # 3) 基于“有没有 1.0 / 0.8 分数”来重写导航模式（忽略已被淘汰的目标）：
        valid_items = [it for it in GSSL if float(it.get('exploration_score', 0.0)) >= 0.0]
        has_main_1 = any(abs(it['exploration_score'] - 1.0) < 1e-5 for it in valid_items)
        has_sub_08 = any(abs(it['exploration_score'] - 0.8) < 1e-5 for it in valid_items)

        if has_main_1:
            final_mode = "judgement"
        elif has_sub_08:
            final_mode = "finding"
        else:
            final_mode = "explore"

        self.plan_result = {
            "mode": final_mode,
            "explore_direction": decision.get("explore_direction", "front"),
            "reasoning": decision.get("reasoning", ""),
            "updated_gssl": GSSL
        }

        return self.plan_result

    def visualize_fusion_results(self):
        """可视化全局地图状态和当前阶段状态的双图（X-Z BEV），基于统一的 exploration_score 0.9/0.0。"""

        # --- 1. 初始化双子图和确定范围 ---
        node_has_any_center = False
        for key in self.global_objects.keys():
            all_objects_for_range = self.global_objects[key] if self.global_objects[key] else self.stage_object_list_sim[key]
            all_centers = np.array([
                obj['center'][[0, 2]]
                for obj in all_objects_for_range
                if 'center' in obj and len(obj['center']) >= 3
            ])

            # 如果当前类别没有任何中心点，则跳过该类别；
            # 只有在 node 类始终没有任何 center 时，才在循环结束后统一打印告警。
            if all_centers.size == 0:
                continue

            if key == "node":
                node_has_any_center = True

            min_x, max_x = all_centers[:, 0].min() - 1.0, all_centers[:, 0].max() + 1.0
            min_z, max_z = all_centers[:, 1].min() - 1.0, all_centers[:, 1].max() + 1.0
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 10)) 
            ax1 = axes[0] 
            ax2 = axes[1] 

            for ax in [ax1, ax2]:
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(min_x, max_x)
                ax.set_ylim(min_z, max_z)
                ax.set_xlabel("X-Axis (m)")
                ax.set_ylabel("Z-Axis (m)")
                ax.grid(True, linestyle='--', alpha=0.5)

            # --- 2. 定义分数映射表 ---
            # 全局图 (ax1) 和当前阶段图 (ax2) 共享相同的分数语义，但使用不同的颜色/标记。

            # 全局图 (Global Objects) 颜色/标记映射
            GLOBAL_SCORE_MAP = {
                0.9:   {'color': 'red', 'label': 'Historical/Fused (0.9)', 'marker': 'o', 'zorder': 1}, 
                0.0:   {'color': 'blue', 'label': 'Newly Added (0.0)', 'marker': 'o', 'zorder': 3},  
            }

            # 当前阶段图 (Stage Objects BEV) 颜色/标记映射
            STAGE_SCORE_MAP = {
                0.9:   {'color': 'blue', 'label': 'Fused Match (0.9 s)', 'marker': 's', 'zorder': 2},
                0.0:   {'color': 'green', 'label': 'New/Unmatched (0.0 ^)', 'marker': '^', 'zorder': 2},
            }

            # --- 3. 绘制【全局地图状态】(左图 ax1) ---
            ax1.set_title("Global Map State (X-Z BEV) - 0.9:Explored/Fused, 0.0:New Added")
            global_legend_handles = []

            for obj in self.global_objects[key]:
                if 'center' not in obj or len(obj['center']) < 3: continue
                
                x_p, z_p = obj['center'][0], obj['center'][2]
                class_name = get_mode(obj.get('caption'))
                
                # 从全局物体中读取 exploration_score
                score = obj.get('exploration_score', 0.9) # 默认 0.9 (历史/已探索)
                
                if np.isclose(score, 0.0): 
                    config = GLOBAL_SCORE_MAP[0.0]
                else: 
                    config = GLOBAL_SCORE_MAP[0.9] 
                
                # 绘制中心点和标签 (代码省略，与之前一致)
                ax1.scatter(x_p, z_p, color=config['color'], s=120, marker=config['marker'], alpha=0.8, zorder=config['zorder']) 
                ax1.text(x_p, z_p, f'{class_name}', fontsize=8, color='black', weight='bold',
                        ha='center', va='top', zorder=config['zorder'] + 1,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor=config['color']))

            # 构建全局图例 (代码省略，与之前一致)
            added_labels = set()
            for score in [0.9, 0.0]: 
                config = GLOBAL_SCORE_MAP[score]
                if config['label'] not in added_labels:
                    global_legend_handles.append(plt.Line2D([0], [0], marker=config['marker'], color='w', 
                                                        markerfacecolor=config['color'], markersize=10, label=config['label'], linewidth=0))
                    added_labels.add(config['label'])
            ax1.legend(handles=global_legend_handles, loc='best', fontsize='small')


            # --- 4. 绘制【当前阶段状态】(右图 ax2) ---
            ax2.set_title("Current Stage State (X-Z BEV) - 0.9:Fused, 0.0:New/Unmatched")
            stage_legend_handles = []
            
            # 遍历当前阶段 sim 列表 (假设坐标系一致，从 sim 取坐标)
            for i, obj in enumerate(self.stage_object_list_sim[key]):
                if 'center' not in obj or len(obj['center']) < 3: continue
                    
                x_p, z_p = obj['center'][0], obj['center'][2]
                class_name = get_mode(obj.get('caption'))
                
                # 从 stage_object_list_bev 中获取 exploration_score（对缺失/None 做鲁棒处理）
                bev_list_for_key = (self.stage_object_list_bev or {}).get(key, [])
                exp_data = bev_list_for_key[i] if 0 <= i < len(bev_list_for_key) else {}
                score = exp_data.get('exploration_score', 0.0) 
                
                if np.isclose(score, 0.9): 
                    config = STAGE_SCORE_MAP[0.9]
                else: 
                    config = STAGE_SCORE_MAP[0.0] 

                # 绘制中心点和标签 (代码省略，与之前一致)
                ax2.scatter(x_p, z_p, color=config['color'], s=150, marker=config['marker'], zorder=config['zorder']) 
                ax2.text(x_p, z_p, f'{class_name}', fontsize=8, color='black', weight='bold',
                        ha='center', va='top', zorder=config['zorder'] + 1,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=config['color']))

            # 构建当前阶段图例 (代码省略，与之前一致)
            added_labels = set()
            for score in [0.9, 0.0]: 
                config = STAGE_SCORE_MAP[score]
                if config['label'] not in added_labels:
                    stage_legend_handles.append(plt.Line2D([0], [0], marker=config['marker'], color='w', 
                                                        markerfacecolor=config['color'], markersize=10, label=config['label'], linewidth=0))
                    added_labels.add(config['label'])
            ax2.legend(handles=stage_legend_handles, loc='best', fontsize='small')

            # --- 5. 保存和清理 ---
            plt.tight_layout()
            save_path = os.path.join(self.save_folder, f'fusion_dual_{key}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True) 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        # 只有 node 类在全局和当前阶段都没有任何 center 时，才打印一次告警
        if not node_has_any_center:
            print("警告: 无有效物体中心点可供可视化。")

    def sim_to_bev_point(self, point):

        transform = self.transform_sim2bev

        x_sim, y_sim, z_sim = point[0], point[1], point[2]
        
        # 应用 2D 相似变换到 XZ 平面
        pose_bev = (x_sim, z_sim, 0)
        x_bev, z_bev, _ = apply_transform(pose_bev, transform)
        
        # y 轴使用相同的缩放比例，但独立处理
        scale = transform['scale']
        y_bev = y_sim*scale
        
        return np.array([x_bev, y_bev, z_bev])

    def sim_to_bev_matrix(self, matrix):

        """
        将 Sim 坐标系下的方向矩阵转换为 BEV 坐标系下的方向矩阵。
        为了提升鲁棒性，这里对输入的 shape 做“尽量容错”的处理，而不是直接抛出异常。
        """
        # 1. 归一化为 3x3 旋转矩阵
        if matrix is None:
            # 回退为单位矩阵
            matrix = np.eye(3, dtype=np.float32)
        else:
            matrix = np.asarray(matrix)
            if matrix.shape == (4, 4):
                # 典型的 4x4 齐次矩阵：取左上 3x3
                matrix = matrix[:3, :3]
            elif matrix.shape != (3, 3):
                # 其他未知形状：记录并回退为单位矩阵，避免打断导航流程
                print(f"[Graph.sim_to_bev_matrix] 非预期方向矩阵形状 {matrix.shape}，已回退为单位矩阵。")
                matrix = np.eye(3, dtype=np.float32)

        transform = self.transform_sim2bev

        # 从方向矩阵提取 yaw 角（绕 y 轴的旋转）
        yaw_sim = np.arctan2(matrix[2, 0], matrix[0, 0])

        # 应用相同的角度变换
        _, _, yaw_bev = apply_transform((0, 0, yaw_sim), transform)

        # 重建方向矩阵（创建绕 y 轴的旋转矩阵）
        cy, sy = np.cos(yaw_bev), np.sin(yaw_bev)

        bev_matrix = np.array([
            [cy, 0, sy],
            [0,  1,  0],
            [-sy, 0, cy]
        ], dtype=np.float32)

        return bev_matrix
    
    def bev_to_sim_point(self, point):
        """
        将 BEV 像素坐标 (x_pixel, y_pixel, z_pixel) 转换为 Sim 3D 坐标 (x_sim, y_sim, z_sim)。
        """
        transform = self.transform_bev2sim

        x_bev, y_bev, z_bev = point[0], point[1], point[2]
        
        # 应用 2D 相似变换到 XZ 平面
        pose_bev = (x_bev, z_bev, 0)
        x_sim, z_sim, _ = apply_transform(pose_bev, transform)
        
        # y 轴使用相同的缩放比例，但独立处理
        scale = transform['scale']
        y_sim = y_bev * scale
        
        return np.array([x_sim, y_sim, z_sim])
  
    def bev_to_sim_matrix(self, matrix):
        """
        [旧版逻辑] 将 BEV 坐标系下的方向矩阵 (3x3) 转换为 Sim 坐标系下的方向矩阵。
        """
        # 为保持与 sim_to_bev_matrix 一致，这里同样做形状容错，不再抛出异常。
        if matrix is None:
            matrix = np.eye(3, dtype=np.float32)
        else:
            matrix = np.asarray(matrix)
            if matrix.shape == (4, 4):
                matrix = matrix[:3, :3]
            elif matrix.shape != (3, 3):
                print(f"[Graph.bev_to_sim_matrix] 非预期方向矩阵形状 {matrix.shape}，已回退为单位矩阵。")
                matrix = np.eye(3, dtype=np.float32)

        transform = self.transform_bev2sim

        # 从方向矩阵提取 yaw 角（绕 y 轴的旋转）
        yaw_bev = np.arctan2(matrix[2, 0], matrix[0, 0])

        # 应用相同的角度变换
        _, _, yaw_sim = apply_transform((0, 0, yaw_bev), transform)

        # 重建方向矩阵（创建绕 y 轴的旋转矩阵）
        cy, sy = np.cos(yaw_sim), np.sin(yaw_sim)

        sim_matrix = np.array([
            [cy, 0, sy],
            [0,  1,  0],
            [-sy, 0, cy]
        ], dtype=np.float32)

        return sim_matrix

    def generate_semantic_field(self):
        """
        生成语义势能场（兴趣场）。完全在 BEV 像素域内工作。
        不再包含探索反转逻辑，仅保留兴趣叠加。
        """
        field_name = 'semantic_interest'
        

        map_pixels_x, map_pixels_z = self._get_hw_shape()
        if map_pixels_x == 0 or map_pixels_z == 0:
            return np.array([], dtype=np.float32) 

        # 准备网格坐标
        x_indices, z_indices = np.mgrid[0:map_pixels_x, 0:map_pixels_z]
        grid_coords_pixels = np.stack([x_indices.ravel(), z_indices.ravel()], axis=-1) 

        # 初始化势能场
        stage_semantic_field = np.zeros((map_pixels_x, map_pixels_z), dtype=np.float32)

        # 可视化准备
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.set_title('Object Distribution')
        ax1.set_xlim(0, map_pixels_x)
        ax1.set_ylim(0, map_pixels_z)
        ax1.set_aspect('equal')
        
        for key in self.global_objects.keys():
            if key not in self.global_objects:
                continue
                
            for obj in self.global_objects[key]:
                # 1. 坐标转换
                center_sim = obj['center'] 
                extent_sim = obj['extent'] 
                orientation_sim = obj['orientation']

                center_bev = self.sim_to_bev_point(center_sim)
                x_p, z_p = center_bev[0], center_bev[2]
                
                # 过滤地图外的物体
                if 0 <= x_p < self.map_pixels_x and 0 <= z_p < self.map_pixels_z:
                    center_coords_pixel = np.array([x_p, z_p])
                    orientation = self.sim_to_bev_matrix(orientation_sim)
                    scale = self.transform_sim2bev['scale']
                    extent = extent_sim * scale
                    caption = get_mode(obj.get('caption', 'object'))

                    # --- 绘制物体框逻辑 (用于可视化) ---
                    half_length, half_width = extent[0] / 2, extent[2] / 2
                    forward, right = orientation[:, 0], orientation[:, 2] 
                    pixel_corners = [
                        ((x_p + half_length * forward[0] + half_width * right[0]), (z_p + half_length * forward[2] + half_width * right[2])),
                        ((x_p + half_length * forward[0] - half_width * right[0]), (z_p + half_length * forward[2] - half_width * right[2])),
                        ((x_p - half_length * forward[0] - half_width * right[0]), (z_p - half_length * forward[2] - half_width * right[2])),
                        ((x_p - half_length * forward[0] + half_width * right[0]), (z_p - half_length * forward[2] + half_width * right[2]))
                    ]
                    for j in range(4):
                        start_p, end_p = pixel_corners[j], pixel_corners[(j + 1) % 4]
                        ax1.plot([start_p[0], end_p[0]], [start_p[1], end_p[1]], color='red', linewidth=1)
                    ax1.scatter(x_p, z_p, c='red', s=20, marker='x')
                    ax1.text(x_p, z_p - 10, caption, fontsize=7, color='red', ha='center', va='top')

                    # --- 核心得分逻辑 ---
                    # 状态系数：已探索(True)=0.5, 未探索(False)=1.0
                    multiplier = 0.5 if obj.get('merged_status', False) else 1.0
                    
                    # 使用 VLM 生成的得分（主程序与 post_process 均通过 run_vlm_reasoning 调用 VLM 后写入）
                    base_score = obj.get('exploration_score', 0.0)
                    final_score = base_score * multiplier

                    if final_score < 0.05:
                        continue

                    # 2. 计算高斯衰减并叠加
                    distances = np.linalg.norm(grid_coords_pixels - center_coords_pixel, axis=1) 
                    distances = distances.reshape(map_pixels_x, map_pixels_z)
                    
                    # 这里的 score 参数决定了高斯的峰值高度和覆盖范围
                    decay_weights = gaussian_decay(distances*0.5, extent, final_score)
                    
                    # 累加势能
                    stage_semantic_field += final_score * decay_weights

        # 整体限制在 [0, 1]
        stage_semantic_field = np.clip(stage_semantic_field, 0.0, 1.0)

        # 右侧热力图用于调试（可选）；保存的 semantic_interest.png 仅含场图，无坐标轴/网格/标题/colorbar
        im = ax2.imshow(stage_semantic_field.T, cmap='magma', origin='lower', alpha=0.9)
        ax2.set_title('Semantic Interest Heatmap')
        ax2.set_aspect('equal')
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        plt.close()

        # 仅保存场本身
        fig2, ax_only = plt.subplots(1, 1, figsize=(8, 6))
        ax_only.imshow(stage_semantic_field.T, cmap='magma', origin='lower')
        ax_only.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        save_path = os.path.join(self.save_folder, f"{field_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()
        self.stage_semantic_field = stage_semantic_field
        return stage_semantic_field
    
    def generate_direction_field(self, loc_agent):
        """
        生成探索方向场。
        修改点：值随距离递增（近处为0），扩大扇形覆盖范围。
        """
        command = self.plan_result["explore_direction"]
        field_name = f'direction_mask_{command}'
        x, z, theta = apply_transform(loc_agent, self.transform_sim2bev)
        
        # 1. 获取尺寸
        map_pixels_x, map_pixels_z = self._get_hw_shape()
        
        if map_pixels_x == 0 or map_pixels_z == 0:
            return np.array([], dtype=np.float32)

        # 2. 解析指令调整角度（正前 0°，左 30°，右 -30°）
        offset = 0.0
        if command == 'left':
            offset = np.deg2rad(30)
        elif command == 'right':
            offset = np.deg2rad(-30)
        
        target_theta = theta + offset

        # 3. 准备网格
        x_grid, z_grid = np.ogrid[:map_pixels_x, :map_pixels_z]

        # 4. 计算向量与距离
        dx = x_grid - x
        dz = z_grid - z
        dist = np.sqrt(dx**2 + dz**2)

        # 5. 计算极角
        phi = np.arctan2(dz, dx)

        # 6. 计算角度衰减 (扩大扇形角度)
        angle_diff = np.abs(np.arctan2(np.sin(phi - target_theta), np.cos(phi - target_theta)))
        
        # 扇形高斯衰减半角（与方向一致：正前/左30/右30）
        sigma_rad = np.deg2rad(30) 
        direction_mask = np.exp(-(angle_diff**2) / (2 * sigma_rad**2))

        # 7. 距离增强：近处快速增大，远处缓慢增长（指数趋近 1）
        # 计算当前地图内的最大可能距离用于归一化
        max_dist = np.sqrt(map_pixels_x**2 + map_pixels_z**2)
        dist_norm = np.clip(dist / max_dist, 0.0, 1.0)
        dist_weight = 1.0 - np.exp(-3.0 * dist_norm)  # 近处斜率大，远处逐渐饱和到 1
        
        # 组合：角度权重 * 距离权重
        direction_mask = direction_mask * dist_weight

        # 8. 视野截断 (角度大一点，允许看到侧后方一点点，从 pi/2 改为 2/3 pi)
        direction_mask[angle_diff > (np.pi * 0.66)] = 0.0
        
        direction_mask = direction_mask.astype(np.float32)

        # --- 仅保存场图，无坐标轴/网格/标题/colorbar/机器人标记 ---
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='white')
        vmax = direction_mask.max() if direction_mask.max() > 0 else 1
        ax.imshow(direction_mask.T, cmap='magma', origin='lower', vmin=0, vmax=vmax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        save_path = os.path.join(self.save_folder, f"{field_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()

        self.stage_direction_field = direction_mask
        return direction_mask

    def _compute_exploration_history_field(self):
        """
        探索历史场：上一阶段权重 0.5，再上一 0.3，更早阶段均为 0.1（不截断阶段数）。
        高斯模糊叠加后归一化到 [0,1]，再 1-field 使值越大表示越未探索。
        高斯核扩散半径为配置 sigma 的 3 倍。仅使用当前 BEV 范围内的点。无轨迹时返回 None。
        """
        map_pixels_x, map_pixels_z = self._get_hw_shape()
        if map_pixels_x == 0 or map_pixels_z == 0 or not getattr(self, '_exploration_history_trajectories', None):
            return None
        trajs = self._exploration_history_trajectories
        if len(trajs) == 0:
            return None
        transform_sim2bev = getattr(self, 'transform_sim2bev', None)
        if transform_sim2bev is None:
            return None
        sigma_base = float(getattr(self.args, 'exploration_history_sigma', 5.0))
        sigma = sigma_base * 3.0
        acc = np.zeros((map_pixels_x, map_pixels_z), dtype=np.float32)
        n = len(trajs)
        for j, (_, traj_sim) in enumerate(trajs):
            if j == n - 1:
                w = 0.7
            elif j == n - 2:
                w = 0.5
            else:
                w = 0.3
            for (x_sim, z_sim) in traj_sim:
                bx, bz, _ = apply_transform((x_sim, z_sim, 0.0), transform_sim2bev)
                ix = int(round(bx))
                iz = int(round(bz))
                if 0 <= ix < map_pixels_x and 0 <= iz < map_pixels_z:
                    acc[ix, iz] += w
        if acc.max() <= 0:
            return None
        acc = gaussian_filter(acc, sigma=sigma, mode='constant', cval=0)
        if acc.max() > 0:
            acc = acc / acc.max()
        exploration_history_field = 1.0 - acc
        return exploration_history_field.astype(np.float32)

    def fuse_fields_and_extract_goal(self, vis=True):
        """
        全相乘叠加场逻辑：Total = Semantic * Direction * Safety [* ExplorationHistory]，gmid = argmax。
        与论文 MVM 一致：Φsem、Φdir、Φtrav → Φtotal；可选探索历史场参与。
        """
        map_pixels_x, map_pixels_z = self._get_hw_shape()
        if map_pixels_x == 0 or map_pixels_z == 0:
            return np.array([0, 0, 0.0], dtype=np.float64), 0.0
        # 1. 获取三个场（形状统一为 (map_pixels_x, map_pixels_z)）
        sem = getattr(self, 'stage_semantic_field', np.full((map_pixels_x, map_pixels_z), 1e-6, dtype=np.float32))
        exp = getattr(self, 'stage_direction_field', np.full((map_pixels_x, map_pixels_z), 1e-6, dtype=np.float32))
        safe = getattr(self, 'stage_space_safety_field', np.zeros((map_pixels_x, map_pixels_z), dtype=np.float32))
        if sem.shape != (map_pixels_x, map_pixels_z):
            sem = np.full((map_pixels_x, map_pixels_z), 1e-6, dtype=np.float32)
        if exp.shape != (map_pixels_x, map_pixels_z):
            exp = np.full((map_pixels_x, map_pixels_z), 1e-6, dtype=np.float32)
        if safe.shape != (map_pixels_x, map_pixels_z):
            safe = np.zeros((map_pixels_x, map_pixels_z), dtype=np.float32)

        # 1.1 探索历史场：始终计算并可视化，仅当开关打开时参与融合
        expl_hist_raw = self._compute_exploration_history_field()
        use_expl_hist = getattr(self.args, 'use_exploration_history_field', False)
        if expl_hist_raw is not None:
            self.stage_space_exploration_history_field = expl_hist_raw
            if vis and self.save_folder:
                plt.figure(figsize=(8, 7))
                im = plt.imshow(expl_hist_raw.T, origin='lower', cmap='viridis')
                plt.colorbar(im, label='Exploration History (high=less explored)')
                plt.title(f"Stage {self.stage}: Exploration History Field")
                plt.xlabel("BEV X (pixels)")
                plt.ylabel("BEV Z (pixels)")
                save_path = os.path.join(self.save_folder, 'exploration_history_field.png')
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
            expl_hist = expl_hist_raw if use_expl_hist else np.ones((map_pixels_x, map_pixels_z), dtype=np.float32)
        else:
            self.stage_space_exploration_history_field = None
            expl_hist = np.ones((map_pixels_x, map_pixels_z), dtype=np.float32)

        # 2. 融合公式：(安全场*3 + 语义场) * 方向场 * 历史场
        plan = getattr(self, 'plan_result', {'mode': 'explore', 'explore_direction': 'front'})
        combined_safe_sem = safe * 3.0 + sem
        fused_field = combined_safe_sem * exp * expl_hist

        # 2.1 仅在可通行区域内选目标：与当前阶段的可通行地面掩码做与运算
        traversible_mask = getattr(self, "safe_ground_mask", None)
        if traversible_mask is not None and traversible_mask.shape == fused_field.shape:
            valid = traversible_mask > 0
            if np.any(valid):
                fused_field = fused_field * valid.astype(fused_field.dtype)

        # 3. 提取最高点 gmid = argmax Φtotal，并确定目标朝向
        max_idx = np.argmax(fused_field)
        x_idx, z_idx = np.unravel_index(max_idx, fused_field.shape)

        # 先用“最近 frontier”方案计算基础 yaw，再根据 explore_direction 约束到对应扇形
        base_yaw_frontier = self._calculate_yaw_towards_nearest_frontier([x_idx, z_idx])
        goal_yaw = self._clamp_yaw_to_explore_direction(base_yaw_frontier)
        max_score = fused_field[x_idx, z_idx]
        
        target_px = np.array([x_idx, z_idx, goal_yaw])

        # 4. 仅保存合成场图，无坐标轴/网格/标题/colorbar/散点/图例
        if vis:
            plt.figure(figsize=(8, 7))
            plt.imshow(fused_field.T, origin='lower', cmap='magma')
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            save_path = os.path.join(self.save_folder, f'final_fused_field_s{self.stage}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"🎯 Target selected at pixel ({x_idx}, {z_idx}) with fused score {max_score:.4e}")

            # 额外：多方案 yaw 可视化（不影响实际 goal_yaw）
            try:
                path_pixels = getattr(self, "last_astar_path", None)
            except Exception:
                path_pixels = None
            # 可视化中仍同时展示“最近 frontier”与其它方案，便于对比
            self._debug_visualize_goal_yaw_variants([x_idx, z_idx], base_yaw_frontier, path_pixels)

        # 存储到属性以便后续控制器调用
        self.final_fused_field = fused_field
        self.best_target_px = target_px
        
        return target_px, max_score

    def delete_high_interest_point(self):
        for key in ['main', 'sub']:
            index_list = self.associated_global_indices[key]
            sorted_indices = sorted(index_list, reverse=True) 
            for index in sorted_indices:
                try:
                    del self.global_objects[key][index] 
                    print(f"Deleted element at index {index} from self.global_objects['{key}']")
                except IndexError:
                    print(f"Error: Index {index} is out of bounds for self.global_objects['{key}']")
                except Exception as e:
                    print(f"An error occurred during deletion: {e}")

    def _calculate_yaw_towards_nearest_unknown(self, goal_pose):
        """
        计算从目标点到最近未知区域的朝向角 (弧度)。
        """
        if not hasattr(self, 'unknown_mask') or self.unknown_mask is None:
            # 如果未知区域掩码不存在，则不设置特定的朝向（例如，设置为0或返回None）
            return 0.0 
        
        # 1. 找到所有未知区域的坐标
        unknown_coords = np.argwhere(self.unknown_mask)
        
        if unknown_coords.size == 0:
            # 没有未知区域，朝向保持默认（例如，朝向前方或返回None）
            return 0.0

        # 2. 计算目标点到所有未知区域的距离
        # 目标点（已是像素坐标）
        target_point = goal_pose 
        
        # 计算距离
        distances = np.linalg.norm(unknown_coords - target_point, axis=1)
        
        # 3. 找到最近的未知区域坐标
        nearest_idx = np.argmin(distances)
        nearest_unknown_coord = unknown_coords[nearest_idx]
        
        # 4. 计算朝向角 (使用 atan2)
        # 朝向是目标点指向最近未知点的向量
        
        # dy 是 z 坐标（通常对应地图上的 Y 轴）
        # dx 是 x 坐标（通常对应地图上的 X 轴）
        dy = nearest_unknown_coord[1] - target_point[1] 
        dx = nearest_unknown_coord[0] - target_point[0]
        
        # atan2 返回的角度是从 x 轴正方向逆时针旋转到向量 (dx, dy) 的角度 (弧度)
        yaw = math.atan2(dy, dx)
        
        return yaw

    def _calculate_yaw_towards_nearest_frontier(self, goal_pose):
        """
        计算从目标点到最近的“未知区域边界点”（即邻域有未知区域的地面安全点）
        的朝向角 (弧度)。
        
        依赖于：self.unknown_mask (未知区域掩码)
                self.safe_ground_mask (安全地面区域掩码)
        """
        if (not hasattr(self, 'unknown_mask') or self.unknown_mask is None or 
                not hasattr(self, 'safe_ground_mask') or self.safe_ground_mask is None):
            # 缺少必要的掩码信息
            return 0.0 
        kernel = np.ones((3, 3), np.uint8)
        dilated_unknown = cv2.dilate(self.unknown_mask.astype(np.uint8), kernel, iterations=1)
        unknown_border_mask = (dilated_unknown > 0) & (self.safe_ground_mask > 0)
        border_coords = np.argwhere(unknown_border_mask)
        if border_coords.size == 0:
            return 0.0
        target_point = np.array(goal_pose) 
        distances = np.linalg.norm(border_coords - target_point, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_border_coord = border_coords[nearest_idx]
        dx = nearest_border_coord[0] - target_point[0] # row (Y)
        dy = nearest_border_coord[1] - target_point[1] # col (X)
        yaw = math.atan2(dy, dx)

        mask_to_display = unknown_border_mask.astype(float)
        
        # 绘制边界掩码
        plt.imshow(mask_to_display.T, cmap='Wistia', origin= 'lower', alpha=0.4)
        
        # 2. 绘制目标点 (goal_pose/target_point)
        # target_point[0] 是行 (Y)，target_point[1] 是列 (X)
        plt.plot(target_point[0], target_point[1], 
                 marker='*', markersize=12, color='red', 
                 label='Goal Pose (Target Point)', linestyle='')

        # 3. 绘制最近点 (nearest_border_coord)
        # nearest_border_coord[0] 是行 (Y)，nearest_border_coord[1] 是列 (X)
        plt.plot(nearest_border_coord[0], nearest_border_coord[1], 
                 marker='o', markersize=10, color='blue', fillstyle='none', linewidth=2,
                 label='Nearest Border Coord', linestyle='')

        # 4. 绘制 yaw 角度对应的箭头 (从 Target Point 指向 Nearest Border Coord)
        # 箭头起点 (X_start, Y_start) = (target_point[1], target_point[0])
        # 箭头长度 (dX, dY) = (dx, dy)
        plt.arrow(target_point[0], target_point[1], 
                  dx, dy, 
                  color='green', linewidth=2.5, head_width=5, head_length=5, 
                  label=f'Yaw Vector ({math.degrees(yaw):.2f}°)')


        save_path = os.path.join(self.save_folder, f"goal_pose_decision.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        return yaw

    def _calculate_yaw_unknown_area_weighted(self, goal_pose, num_bins: int = 12):
        """
        方案1：按未知边界规模 + 距离加权选方向。

        将 unknown_border_mask 上的边界点按角度划分到 num_bins 个扇形，
        每个扇形的权重为 边界点数量 / (1 + 平均距离)，选权重最大的扇形中心角作为 yaw。
        """
        if (not hasattr(self, "unknown_mask") or self.unknown_mask is None or
                not hasattr(self, "safe_ground_mask") or self.safe_ground_mask is None):
            return None

        kernel = np.ones((3, 3), np.uint8)
        dilated_unknown = cv2.dilate(self.unknown_mask.astype(np.uint8), kernel, iterations=1)
        unknown_border_mask = (dilated_unknown > 0) & (self.safe_ground_mask > 0)
        border_coords = np.argwhere(unknown_border_mask)
        if border_coords.size == 0:
            return None

        target_point = np.array(goal_pose)
        gx, gz = target_point[0], target_point[1]

        dx = border_coords[:, 0] - gx  # row (Y)
        dy = border_coords[:, 1] - gz  # col (X)
        dists = np.sqrt(dx**2 + dy**2) + 1e-6
        angles = np.arctan2(dy, dx)    # [-pi, pi]

        bin_edges = np.linspace(-math.pi, math.pi, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        weights = np.zeros(num_bins, dtype=np.float64)

        for i in range(num_bins):
            mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
            if not np.any(mask):
                continue
            count = np.sum(mask)
            mean_dist = np.mean(dists[mask])
            weights[i] = count / (1.0 + mean_dist)

        if np.all(weights <= 0):
            return None

        best_idx = int(np.argmax(weights))
        return float(bin_centers[best_idx])

    def _clamp_yaw_to_explore_direction(self, base_yaw: float) -> float:
        """
        方案2：将 base_yaw 限制在当前 explore_direction 扇形内。

        front: [-30°, +30°]
        left : [+30°, +90°]
        right: [-90°, -30°]
        """

        def _norm_angle(a: float) -> float:
            a = (a + math.pi) % (2 * math.pi) - math.pi
            return a

        command = getattr(self, "plan_result", {}).get("explore_direction", "front")
        base_yaw = _norm_angle(base_yaw)

        def deg(v):
            return math.radians(v)

        if command == "left":
            low, high = deg(30), deg(90)
        elif command == "right":
            low, high = deg(-90), deg(-30)
        else:  # "front" or others
            low, high = deg(-30), deg(30)

        if low <= base_yaw <= high:
            return base_yaw

        d_low = abs(_norm_angle(base_yaw - low))
        d_high = abs(_norm_angle(base_yaw - high))
        return low if d_low < d_high else high

    def _calculate_yaw_along_path_tail(self, path_pixels, tail_len: int = 10):
        """
        方案3：沿 A* 路径末端方向计算 yaw（若路径足够长）。
        path_pixels: [(x0, z0), (x1, z1), ...]
        """
        if path_pixels is None or len(path_pixels) < 2:
            return None
        pts = np.array(path_pixels[-tail_len:])
        x0, z0 = pts[0, 0], pts[0, 1]
        x1, z1 = pts[-1, 0], pts[-1, 1]
        dx = x1 - x0
        dy = z1 - z0
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return float(math.atan2(dy, dx))

    def _debug_visualize_goal_yaw_variants(self, goal_pose, base_yaw_frontier, path_pixels=None):
        """
        综合可视化多种 yaw 方案，仅用于调试对比，不影响实际控制：

        - nearest_frontier           : 当前使用的“最近边界点”方案
        - unknown_area_weighted      : 方案1，按未知区域规模 + 距离加权
        - frontier_clamped_to_dir    : 方案2，在方案0基础上限制在 explore_direction 扇形内
        - path_tail_direction        : 方案3，若提供路径，则用路径末端方向
        """
        if self.save_folder is None:
            return

        variants = {}
        variants["nearest_frontier"] = base_yaw_frontier

        yaw_area = self._calculate_yaw_unknown_area_weighted(goal_pose)
        if yaw_area is not None:
            variants["unknown_area_weighted"] = yaw_area

        yaw_clamped = self._clamp_yaw_to_explore_direction(base_yaw_frontier)
        variants["frontier_clamped_to_dir"] = yaw_clamped

        yaw_path = self._calculate_yaw_along_path_tail(path_pixels) if path_pixels is not None else None
        if yaw_path is not None:
            variants["path_tail_direction"] = yaw_path

        if (not hasattr(self, "unknown_mask") or self.unknown_mask is None or
                not hasattr(self, "safe_ground_mask") or self.safe_ground_mask is None):
            return

        kernel = np.ones((3, 3), np.uint8)
        dilated_unknown = cv2.dilate(self.unknown_mask.astype(np.uint8), kernel, iterations=1)
        unknown_border_mask = (dilated_unknown > 0) & (self.safe_ground_mask > 0)
        mask_to_display = unknown_border_mask.astype(float)

        num_vars = len(variants)
        if num_vars == 0:
            return

        fig, axes = plt.subplots(1, num_vars, figsize=(4 * num_vars, 4), squeeze=False)
        axes = axes[0]

        gx, gz = goal_pose[0], goal_pose[1]

        arrow_len = 30
        for ax, (name, yaw) in zip(axes, variants.items()):
            ax.imshow(mask_to_display.T, cmap="Wistia", origin="lower", alpha=0.4)
            ax.plot(gx, gz, marker="*", markersize=10, color="red", linestyle="", label="Goal")

            dx = arrow_len * math.cos(yaw)
            dy = arrow_len * math.sin(yaw)
            ax.arrow(gx, gz, dx, dy, color="green", linewidth=2.0, head_width=4, head_length=4)

            ax.set_title(f"{name}\n({math.degrees(yaw):.1f}°)")
            ax.set_aspect("equal")

        plt.tight_layout()
        save_path = os.path.join(self.save_folder, "goal_pose_decision_multi.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def get_nav_mode_and_goal_via_fuse(self, loc_agent, vis=True):
        """
        MVM 规划主入口：基于 fuse_fields_and_extract_goal 选 gmid。
        鲁棒性逻辑移植自 get_next_mode_and_goal（建图质量、高兴趣点判定/删除、explore 多场融合）。
        Returns:
            (nav_mode, midterm_goal): 与 get_next_mode_and_goal 相同格式。
        """
        d_thres = getattr(self.args, 'd_thres', self.bev_step)

        if self.stage_space_safety_field is None or self.stage_space_explore_field is None:
            self.last_mode = "recognition"
            self.last_goal = None
            print('建图结果差')
            self._visualize_decision("recognition", None, "Lack of useful map info")
            return "recognition", None

        def find_nearest_object_in_bev(key):
            """根据当前 BEV 内是否存在对应类别物体判断：若有则取离 agent 最近的一个，用于 judgement/finding。"""
            traj = getattr(self, 'stage_trajectory_bev', None)
            if traj is None or (hasattr(traj, '__len__') and len(traj) == 0):
                return None, float('inf'), {"main": [], "sub": []}
            obj_list = getattr(self, 'stage_object_list_bev', None) or {}
            objs = obj_list.get(key, []) if isinstance(obj_list, dict) else []
            if not objs:
                return None, float('inf'), {"main": [], "sub": []}
            last_pos = self.stage_trajectory_bev[-1]
            p_x, p_z = float(last_pos[0]), float(last_pos[1])
            agent_coord = np.array([p_x, p_z])
            best_dist = float('inf')
            goal_x_bev = goal_z_bev = 0.0
            for obj in objs:
                center = obj.get('center')
                if center is None or (hasattr(center, '__len__') and len(center) < 3):
                    continue
                cx = float(center[0])
                cz = float(center[2])
                d = np.linalg.norm(agent_coord - np.array([cx, cz]))
                if d < best_dist:
                    best_dist = d
                    goal_x_bev, goal_z_bev = cx, cz
            if best_dist == float('inf'):
                return None, float('inf'), {"main": [], "sub": []}
            goal = {
                'pose': np.array([goal_x_bev, goal_z_bev]),
                'type': 'interest_point',
                'distance': best_dist,
            }
            return goal, best_dist, {"main": [], "sub": []}

        self.associated_global_indices = None

        # judgement：当前 BEV 内有 main 目标且 agent 距离 < d_thres 则进入
        goal_main, dist_main, _ = find_nearest_object_in_bev('main')
        if self.last_mode == "judgement" or (self.last_mode == "finding" and goal_main is None):
            self.delete_high_interest_point()
            print("💡 高兴趣点删除")
        if goal_main is not None and dist_main < d_thres:
            self.last_mode = "judgement"
            self.last_goal = goal_main
            self._visualize_decision("judgement", goal_main, f"Main target in BEV, dist: {dist_main:.2f}px")
            return "judgement", goal_main

        # 若上一阶段是 finding，本阶段 BEV 内没有 main，则对 sub 打 -1 避免再次被选
        if self.last_mode == "finding" and getattr(self, "associated_global_indices", None):
            sub_indices = self.associated_global_indices.get("sub", [])
            for idx in sub_indices:
                if 0 <= idx < len(self.global_objects.get("sub", [])):
                    try:
                        self.global_objects["sub"][idx]["exploration_score"] = -1.0
                    except Exception:
                        pass
            self.associated_global_indices = None

        # finding：当前 BEV 内有 sub 目标且 agent 距离 < d_thres 则进入
        goal_sub, dist_sub, assoc_sub = find_nearest_object_in_bev('sub')
        if goal_sub is not None and dist_sub < d_thres:
            self.last_mode = "finding"
            self.last_goal = goal_sub
            self.associated_global_indices = assoc_sub
            self._visualize_decision("finding", goal_sub, f"Sub target in BEV, dist: {dist_sub:.2f}px")
            return "finding", goal_sub

        self.last_mode = "explore"
        if not hasattr(self, 'plan_result') or self.plan_result is None:
            self.plan_result = {'mode': 'explore', 'explore_direction': 'front', 'reasoning': '', 'updated_gssl': []}
        self.generate_semantic_field()
        self.generate_direction_field(loc_agent)
        target_px, max_score = self.fuse_fields_and_extract_goal(vis=vis)
        goal_explore = {
            'pose': target_px,
            'type': 'explore_point',
            'score': float(max_score),
        }
        self.last_goal = goal_explore
        return "explore", goal_explore

    def _visualize_decision(self, mode, goal, decision_reason=""):
        """可视化决策过程和目标点（原先函数体存在但未封装成方法，现补全为类内方法）。"""

        if self.safe_ground_mask is None or self.safe_ground_mask.size == 0:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 获取地图尺寸用于设置显示范围
        map_height, map_width = self.safe_ground_mask.shape
        
        # 0. 安全场 (Safety Field) - 修改为显示安全场
        if self.stage_space_safety_field is not None:
            im0 = axes[0].imshow(self.stage_space_safety_field.T, cmap='Greens', origin='lower', alpha=0.8)
            axes[0].set_title('Safety Field')
            plt.colorbar(im0, ax=axes[0])
        else:
            axes[0].text(0.5, 0.5, 'No safety field data', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Safety Field')
        

        # 4. 组合场
        if mode == "explore" and hasattr(self, 'exploration_scores'):
            im4 = axes[4].imshow(self.exploration_scores.T, cmap='cool', origin='lower', alpha=0.8)
            axes[4].set_title('Combined Exploration Field')
            plt.colorbar(im4, ax=axes[4])

            if goal is not None:
                pose = goal['pose']
                goal_x, goal_z = pose[0], pose[1]
                yaw = pose[2] if len(pose) > 2 else 0.0
                color_map = {
                    'judgement': 'red',
                    'approaching': 'orange', 
                    'finding': 'yellow',
                    'explore': 'green',
                    'recognition': 'purple'
                }
                color = color_map.get(mode, 'purple')
                
                # 确保目标点在显示范围内
                if 0 <= goal_x <= map_height and 0 <= goal_z <= map_width :
                    axes[4].scatter(goal_x, goal_z, c=color, s=600, marker='*', 
                                edgecolors='white', linewidth=3, zorder=6)
                    arrow_length = 5  
                    dx = arrow_length * np.cos(yaw)
                    dz = arrow_length * np.sin(yaw)
                    
                    axes[4].arrow(goal_x, goal_z, dx, dz, 
                                head_width=2, head_length=3, 
                                fc=color, ec='white', linewidth=2, 
                                zorder=7, alpha=0.8)
                                

        else:
            axes[4].text(0.5, 0.5, 'No combined field\n(Not in explore mode)', 
                        ha='center', va='center', transform=axes[4].transAxes)
            axes[4].set_title('Combined Field')

        
        # 5. 决策结果（主图）
        # 显示安全地面作为背景
        axes[5].imshow(self.safe_ground_mask.T, cmap='gray', alpha=0.3, origin='lower')
        
        # 显示轨迹和当前位置
        current_pos = None
        if hasattr(self, 'stage_trajectory_bev') and self.stage_trajectory_bev is not None and self.stage_trajectory_bev.size > 0:
            trajectory = self.stage_trajectory_bev
            
            # 确保轨迹是二维数组且包含有效数据
            if trajectory.ndim == 2 and trajectory.shape[0] > 0:
                # 只显示最近的一些轨迹点，避免过于密集
                display_trajectory = trajectory if trajectory.shape[0] <= 50 else trajectory[-50:]
                axes[5].plot(display_trajectory[:, 0], display_trajectory[:, 1], 'b-', 
                            linewidth=3, alpha=0.8)
                
                # 获取当前位置
                current_pos = trajectory[-1]
                axes[5].scatter(current_pos[0], current_pos[1], c='blue', s=200, marker='o', 
                            edgecolors='white', linewidth=2, zorder=5)
                print(f"当前位置: ({current_pos[0]:.1f}, {current_pos[1]:.1f})")
            else:
                print("轨迹数据格式异常")
        
        # 显示目标点
        if goal is not None:
            goal_x, goal_z = goal['pose'][0], goal['pose'][1]
            color_map = {
                'judgement': 'red',
                'approaching': 'orange', 
                'finding': 'yellow',
                'explore': 'green',
                'recognition': 'purple'
            }
            color = color_map.get(mode, 'purple')
            
            # 确保目标点在显示范围内
            if 0 <= goal_x <= map_height and 0 <= goal_z <= map_width :
                axes[5].scatter(goal_x, goal_z, c=color, s=600, marker='*', 
                            edgecolors='white', linewidth=3, zorder=6)
                print(f"目标点: ({goal_x:.1f}, {goal_z:.1f})")
                
            else:
                print(f"目标点超出范围: ({goal_x:.1f}, {goal_z:.1f}), 地图范围: [0,{map_height}]x[0,{map_width}]")
        
        # 设置坐标轴范围
        axes[5].set_xlim(0, map_height)
        axes[5].set_ylim(0, map_width)
        
        # 设置所有子图的属性
        for i, ax in enumerate(axes):
            ax.set_aspect('equal')
            ax.set_xlim(0, map_height)
            ax.set_ylim(0, map_width)
            # 为子图添加边框
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_folder, f"decision_{mode}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Mode: {mode}, Reason: {decision_reason}")