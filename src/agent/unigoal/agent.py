import warnings
warnings.filterwarnings('ignore')
import math
import os
import re
import cv2
import sys
from PIL import Image
import skimage.morphology
from skimage.draw import line_aa, line
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import supervision as sv

from src.utils.fmm.fmm_planner_policy import FMMPlanner
from src.utils.fmm.my_fmm import get_local_goal_fmm
import src.utils.fmm.pose_utils as pu
from src.utils.visualization.semantic_prediction import SemanticPredMaskRCNN
from src.utils.visualization.visualization import (
    init_vis_image,
    draw_line,
    get_contour_points,
    line_list,
    add_text_list
)
from src.utils.visualization.save import save_video
from src.utils.llm import LLM
from src.utils.my_tools import apply_transform

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch
sys.path.append('third_party/Grounded-Segment-Anything/')
from grounded_sam_demo import get_grounding_output
from torchvision import transforms as T

TARGET_SIZE = 518
DIVISOR = 14


def preprocess_image(np_array: np.ndarray) -> dict:
    """
    Take an HxWx3 NumPy array and run 'crop' preprocessing:
    1. Scale width to 518 pixels.
    2. Scale height proportionally and align to multiple of 14.
    3. If new height > 518, center-crop height to 518.

    Args:
        np_array (np.ndarray): Input array shape (H, W, 3), dtype typically uint8 (0-255).

    Returns:
        dict: Keys: 'rgb' (np.ndarray scaled/cropped (H', W', 3)), 'tensor' (torch.Tensor (3,H',W') in [0,1]),
              'final_shape' (H', W'), 'crop_range' (dict, only if height was cropped).

    Raises:
        ValueError: If input ndarray shape is invalid.
    """
    
    if np_array.ndim != 3 or np_array.shape[2] != 3:
        raise ValueError(f"Input array must be HxWx3. Got shape: {np_array.shape}")

    H_orig, W_orig, C = np_array.shape

    # --- 1. Size computation and adjustment ---
    new_width = TARGET_SIZE
    base_height = H_orig * (new_width / W_orig)
    new_height = round(base_height / DIVISOR) * DIVISOR
    
    if new_width == 0 or new_height == 0:
        raise ValueError("Calculated new dimensions are zero after scaling.")

    # --- 2. Image resize ---
    resized_array = cv2.resize(np_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    cropped_array = resized_array
    crop_range = {}
    
    # --- 3. Center crop height (if new height > 518) ---
    final_height = new_height
    if new_height > TARGET_SIZE:
        final_height = TARGET_SIZE
        
        # Crop start/end indices
        h_diff = new_height - TARGET_SIZE
        y_start = h_diff // 2
        y_end = y_start + TARGET_SIZE
        
        cropped_array = resized_array[y_start:y_end, :, :]
        
        crop_range = {
             'y_start': y_start, 
             'y_end': y_end
        }

    # --- 4. Convert to PyTorch tensor and normalize ---
    to_tensor = T.ToTensor()
    # cropped_array is uint8 (H, W, C)
    image_tensor = to_tensor(cropped_array) 

    # --- 5. Assemble output ---
    result = {
        'rgb': cropped_array,  # Return processed uint8 NumPy array
        'tensor': image_tensor,
        'final_shape': (final_height, new_width),
        **({'crop_range': crop_range} if crop_range else {})
    }
    
    return result

class UniGoal_Agent():

    def __init__(self, args, envs, model_info):
        print("--- UniGoal_Agent: init start ---")
        self.args = args
        self.envs = envs
        self.grounded_sam = (model_info['groundingdino'], model_info['sam_predictor'])
        # self.vggt = model_info['vggt']
        self.device = model_info['device']

        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])
        self.sem_pred = SemanticPredMaskRCNN(args)
        self.llm = LLM(self.args.base_url, self.args.api_key, self.args.llm_model)
        self.turn_angle = args.turn_angle

        self.selem = skimage.morphology.disk(3)
        self.obs_shape = None
        self.last_action = None
        self.instance_imagegoal = None
        self.text_goal = None
        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='disk').eval().to(self.device)
        self.obs_save_folder = None
        self.depth_save_folder = None
        self.segment_save_folder = None
        self.shorterm_save_folder = None
        self.obs_history = None
        self.global_segment_result = None

        self.node_space = None
        self.ground_space = "ground. floor. rug. carpet. tile"
        self.node_space = 'table. tv. chair. cabinet. sofa. bed. window. plant. door. doorframe'
        self.node_segment_result = []
        self.main_segment_result = []
        self.sub_segment_result = []


        self.track_main = None
        self.track_sub = None
        self.classes = ['item']
        self.track_main_seg = None
        self.track_sub_seg = None

        self.step_count = 0

        torch.set_grad_enabled(False)

    def reset(self):
        args = self.args
        obs, info =self.envs.reset()
        self.obs_save_folder = None
        self.depth_save_folder = None
        self.segment_save_folder = None
        self.shorterm_save_folder = None
        self.global_segment_result = []
        self.obs_history = []
        self.obs_ground = []
        self.track_main = None
        self.track_sub = None
        self.track_main_seg = []
        self.track_sub_seg = []
        self._last_ground_mask = None
        self._last_safety_ok = None
        self._last_st_proj_uv = None

        if self.args.goal_type == 'ins-image':
            self.instance_imagegoal = self.envs.instance_imagegoal
        elif self.args.goal_type == 'text':
            self.text_goal = self.envs.text_goal
        if args.environment == 'habitat':
            idx = self.get_goal_cat_id()
            if idx is not None:
                self.envs.set_goal_cat_id(idx)

        processed_result = preprocess_image(obs['rgb'])
        cropped_obs = processed_result['rgb']
        obs_tensor = processed_result['tensor']
        self.rgb = cropped_obs
        obs_list = [cropped_obs]
        
        return obs_list, info

    def control_step(self, external_action):
            """
            Function responsible for taking the action and preprocessing observations.
            Args:
                agent_input (dict):
                    dict with following keys:
                        'map_pred'  (ndarray): (M, M) map prediction
                        'goal'      (ndarray): (M, M) mat denoting goal locations
                        'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                    and planning window (gx1, gx2, gy1, gy2)
                        'found_goal' (bool): whether the goal object is found
                external_action (int):
                    An integer representing the action to be executed, 
                    received from an external source.
            Returns:
                obs (ndarray): preprocessed observations ((4+C) x H x W)
                reward (float): amount of reward returned after previous action
                done (bool): whether the episode has ended
                info (dict): contains timestep, pose, goal category and
                            evaluation metric info
            """
            # Use externally provided action and ensure validity
            action_to_take = external_action

            if action_to_take >= 0:
                action = {'action': action_to_take}
                obs, done, info = self.envs.step(action)
                processed_result = preprocess_image(obs['rgb'])
                cropped_obs = processed_result['rgb']
                obs_tensor = processed_result['tensor']
                self.last_action = action['action']
                self.rgb = cropped_obs

                obs_list, ground_mask = self.segment_single_image(obs_tensor)
                positive_pixels = np.sum(ground_mask > 0)
                if positive_pixels > 0:
                    self.obs_ground.append(1)
                else:
                    self.obs_ground.append(0)
                obs_list.append(cropped_obs)

                filename = f"image_{self.step_count}.png"
                filepath = os.path.join(self.obs_save_folder, filename)
                rgb_bgr = cv2.cvtColor(cropped_obs, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, rgb_bgr)

                self.obs_history.append(obs_tensor)   

                return obs_list, done, self.envs.info

    def step(self, agent_input, visualize=False):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        action = self.get_action(agent_input, visualize)

        if action is None:
            print("Error: action is None")

        if action >= 0:

            action = {'action': action}
            obs, done, info = self.envs.step(action)
            processed_result = preprocess_image(obs['rgb'])
            cropped_obs = processed_result['rgb']
            obs_tensor = processed_result['tensor']
            self.last_action = action['action']
            self.rgb = cropped_obs

            obs_list, ground_mask = self.segment_single_image(obs_tensor)
            positive_pixels = np.sum(ground_mask > 0)
            if positive_pixels > 0:
                self.obs_ground.append(1)
            else:
                self.obs_ground.append(0)
            obs_list.append(cropped_obs)

            if getattr(self.args, "save_episode_video", False):
                self._last_ground_mask = ground_mask

            filename = f"image_{self.step_count}.png"
            filepath = os.path.join(self.obs_save_folder, filename)
            rgb_bgr = cv2.cvtColor(cropped_obs, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, rgb_bgr)

            self.obs_history.append(obs_tensor)  

            return obs_list, done, self.envs.info
        
    def get_action(self, agent_input, visualize):
        """
        Compute next action from BEV map, robot start pose and global goal.

        Args:
            bev_map (np.ndarray): (H, W, 3) BEV map.
            start (tuple): (x, y, theta) robot pose.
            goal (tuple): (x, y) global or local goal.

        Returns:
            int: action ID: 0=stop, 1=forward, 2=left, 3=right.
        """
        # 1. Robot current pose
        start = agent_input['pose']
        goal = agent_input['midterm_goal']
        bev_map = agent_input['bev']
        bev_step = agent_input['bev_step']
        path = agent_input['planned_path']
        obs_with_ground = agent_input['obs'][0]
        start_x, start_y, start_o = start[0], start[1], start[2]

        dcor2goal = np.linalg.norm(np.array(start[:2]) - np.array(goal[:2]))
        dyaw2goal = goal[2] - start[2]

        # 2. Get short-term goal via get_local_goal
        if dcor2goal > bev_step:
            try:
                stg_x, stg_y = get_local_goal_fmm(
                    bev_map = bev_map.transpose(1, 0, 2).copy(),
                    max_limit = 500,
                    start = (start_x, start_y),
                    goal = goal,
                    step_size = 50,
                    visualize = visualize,
                    global_path= path,
                    save_path = os.path.join(self.shorterm_save_folder,f"step_{self.step_count}.png")
                )
                self.shorterm_goal = (stg_x, stg_y)

            except AttributeError:
                print("Warning: self.get_local_goal not found. Using dummy local goal.")
                stg_x, stg_y = start_x + 1.0, start_y  # Assume goal ahead
                stop = False


            local_stg_x = stg_x - start_x
            local_stg_y = stg_y - start_y
            
            # Angle from robot heading to short-term goal direction
            angle_st_goal = math.degrees(math.atan2(local_stg_y, local_stg_x))
        else:
            angle_st_goal = math.degrees(goal[2])
        angle_agent = math.degrees(start_o)          
            
        # Relative angle (Agent - Goal), keep in [-180, 180]
        relative_angle = angle_agent - angle_st_goal
        if relative_angle > 180:
                relative_angle -= 360
        elif relative_angle < -180:
                relative_angle += 360
        
        turn_angle_threshold = self.turn_angle/2
        
        if relative_angle < - turn_angle_threshold:
            final_action_id = 2  # Left
        elif relative_angle > turn_angle_threshold:
            final_action_id = 3  # Right
        else:
            final_action_id = 1  # Forward

        angle_to_goal = -relative_angle
        
        if abs(angle_to_goal) < turn_angle_threshold:
            direction_desc = "front"
        elif angle_to_goal >= turn_angle_threshold and angle_to_goal <= 165:
            direction_desc = "front-left"
        elif angle_to_goal > 165 and angle_to_goal <= 195:
            direction_desc = "back"
        elif angle_to_goal < -turn_angle_threshold and angle_to_goal >= -165:
            direction_desc = "front-right"
        elif angle_to_goal < -165 and angle_to_goal >= -195:
            direction_desc = "back"
        else:
            if angle_to_goal > 0:
                direction_desc = "back-left"
            else:
                direction_desc = "back-right"

        # 6. Optional: goal projection safety check (compute/visualize vs use in decision)
        self._last_safety_ok = None
        self._last_st_proj_uv = None
        compute_safety = getattr(self.args, "safety_verification_compute_and_visualize", False) or getattr(
            self.args, "safety_verification_use_in_decision", False
        )
        use_safety_in_decision = getattr(self.args, "safety_verification_use_in_decision", False)
        if (
            compute_safety
            and final_action_id == 1
            and hasattr(self, "shorterm_goal")
            and self.shorterm_goal is not None
        ):
            transform_bev2sim = agent_input.get("transform_bev2sim")
            if transform_bev2sim is not None:
                from src.core.safety_utils import bev_point_to_image, is_point_in_floor_mask
                uv = bev_point_to_image(
                    self.envs,
                    transform_bev2sim,
                    self.shorterm_goal,
                    getattr(self.args, "camera_height", 0.88),
                    getattr(self.args, "env_frame_width", 480),
                    getattr(self.args, "env_frame_height", 640),
                    getattr(self.args, "hfov", 79),
                )
                self._last_st_proj_uv = uv
                if uv is not None:
                    obs_list = agent_input.get("obs") or []
                    current_obs = obs_list[-1] if obs_list else None
                    ground_mask = np.zeros((1, 1), dtype=np.uint8)
                    if current_obs is not None:
                        try:
                            if isinstance(current_obs, np.ndarray):
                                current_tensor = torch.from_numpy(
                                    current_obs.astype(np.float32) / 255.0
                                ).permute(2, 0, 1).unsqueeze(0).to(self.device)
                                current_tensor = current_tensor.squeeze(0)
                            else:
                                current_tensor = current_obs.to(self.device) if hasattr(current_obs, "to") else current_obs
                            _, ground_mask = self.segment_single_image(current_tensor)
                        except Exception:
                            ground_mask = np.zeros((640, 480), dtype=np.uint8)
                    ew = max(1, getattr(self.args, "env_frame_width", 480))
                    eh = max(1, getattr(self.args, "env_frame_height", 640))
                    Hm, Wm = ground_mask.shape[0], ground_mask.shape[1]
                    u_m = uv[0] * Wm / ew
                    v_m = uv[1] * Hm / eh
                    self._last_safety_ok = is_point_in_floor_mask(u_m, v_m, ground_mask, margin=3)
                    if use_safety_in_decision and not self._last_safety_ok:
                        final_action_id = 3  # Turn right for reactive obstacle avoidance
                else:
                    self._last_safety_ok = False

        # 7. Return final action ID
        return final_action_id

    def preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        sem_seg_pred, seg_predictions = self.pred_sem(
            rgb.astype(np.uint8), use_seg=use_seg)

        if args.environment == 'habitat':
            depth = self.preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state, seg_predictions

    def preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[0]):
            depth[i, :][depth[i, :] == 0.] = depth[i, :].max() + 0.01

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def pred_sem(self, rgb, depth=None, use_seg=True, pred_bbox=False):
        if pred_bbox:
            semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
            return self.pred_box, seg_predictions
        else:
            if use_seg:
                semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
                semantic_pred = semantic_pred.astype(np.float32)
                if depth is not None:
                    normalize_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.rgb_vis = cv2.cvtColor(normalize_depth, cv2.COLOR_GRAY2BGR)
            else:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                self.rgb_vis = rgb[:, :, ::-1]
            return semantic_pred, seg_predictions
        
    def get_goal_cat_id(self):
        if self.args.goal_type == 'ins-image':
            instance_whwh, seg_predictions = self.pred_sem(self.instance_imagegoal.astype(np.uint8), None, pred_bbox=True)
            ins_whwh = [instance_whwh[i] for i in range(len(instance_whwh)) \
                if (instance_whwh[i][2][3]-instance_whwh[i][2][1])>1/6*self.instance_imagegoal.shape[0] or \
                    (instance_whwh[i][2][2]-instance_whwh[i][2][0])>1/6*self.instance_imagegoal.shape[1]]
            if ins_whwh != []:
                ins_whwh = sorted(ins_whwh,  \
                    key=lambda s: ((s[2][0]+s[2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((s[2][1]+s[2][3]-self.instance_imagegoal.shape[0])/2)**2 \
                    )
                if ((ins_whwh[0][2][0]+ins_whwh[0][2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((ins_whwh[0][2][1]+ins_whwh[0][2][3]-self.instance_imagegoal.shape[0])/2)**2 < \
                            ((self.instance_imagegoal.shape[1] / 6)**2 )*2:
                    return int(ins_whwh[0][0])
            return None
        elif self.args.goal_type == 'text':
            for i in range(10):
                text_goal_id = self.llm(self.prompt_text2object.replace('{text}', self.text_goal['intrinsic_attributes']))
                try:
                    text_goal_id = re.findall(r'\d+', text_goal_id)[0]
                    text_goal_id = int(text_goal_id)
                    if 0 <= text_goal_id < 6:
                        return text_goal_id
                except:
                    pass
            return 0
                

    def segment_single_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Run Grounding per four semantic spaces, then SAM segmentation, and store by category.
        Categories are independent; all detections kept. Returns final ground_mask.
        Also produces four annotated images (ground, node, main, sub) and saves a stitched image.
        """
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)
            
        groundingdino = self.grounded_sam[0]
        sam_predictor = self.grounded_sam[1]
        
        # --- 0. Image prep and SAM encode ---
        
        # Normalized image for DINO
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_resized = normalize(image_tensor.clone())
        
        # HWC uint8 image for SAM encode and size
        tensor_for_pil = image_tensor.cpu().float() 
        try:
            original_image_pil = T.ToPILImage()(tensor_for_pil)
            original_image_np = np.array(original_image_pil.convert("RGB")).astype(np.uint8)
        except Exception:
            # Fallback: PIL conversion failed, use NumPy
            original_image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
        H, W = original_image_np.shape[0], original_image_np.shape[1]

        ground_mask = np.zeros((H, W), dtype=bool)
        
        # SAM encode once
        sam_predictor.set_image(original_image_np)

        # Store all DINO results with source key
        all_dino_results = [] 

        caption_configs = [
            ("ground", self.ground_space, (0.3,0.25)),
            ("node", self.node_space, (0.3,0.3)),
            ("main", self.track_main, (0.5,0.5)),
            ("sub", self.track_sub, (0.5,0.5)),
        ]
        
        box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.6)
        mask_annotator = sv.MaskAnnotator(opacity=0.2)
        PADDING = 100
        BORDER_COLOR = [255, 255, 255]
        
        # Per-category annotation storage
        ground_annotations = []
        node_annotations = []
        main_annotations = []
        sub_annotations = []

        try:
            with torch.no_grad():
                # --- 1. Grounding DINO per category, store separately ---
                for key, caption_space, thresholds in caption_configs:

                    if isinstance(caption_space, list):
                        caption_space = '. '.join(caption_space) + '.'
                   
                    
                    boxes_filt, caption = get_grounding_output(
                        groundingdino, 
                        image_resized, 
                        caption=caption_space, 
                        box_threshold=thresholds[0], 
                        text_threshold=thresholds[1], 
                        with_logits=False, 
                        device=self.device
                    )
                    
                    if len(caption) > 0:
                        current_boxes = boxes_filt.clone() 
                        for i in range(current_boxes.size(0)):
                            current_boxes[i] = current_boxes[i] * torch.Tensor([W, H, W, H])
                            current_boxes[i][:2] -= current_boxes[i][2:] / 2
                            current_boxes[i][2:] += current_boxes[i][:2]
                        
                        for box, cap in zip(current_boxes.cpu(), caption):
                            all_dino_results.append({
                                "key": key,
                                "caption": cap,
                                "xyxy": box
                            })

                if all_dino_results:
                    # --- 2. Merge DINO boxes and run SAM segmentation ---
                    
                    xyxy_merged_list = [d["xyxy"] for d in all_dino_results]
                    xyxy_merged_tensor = torch.stack(xyxy_merged_list).to(self.device)
                    
                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                        xyxy_merged_tensor, original_image_np.shape[:2]
                    ).to(self.device)

                    mask_tensor, conf_tensor_sam, _ = sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )
                    
                    mask_final = mask_tensor.squeeze(1).cpu().numpy()
                    conf_final_sam = conf_tensor_sam.squeeze(1).cpu().numpy()
                    
                    # --- 3. Classify results and update ground mask ---
                    H_mid = H // 2
                    OVER_MID_THRESHOLD = 0.30 
                    
                    for i, mask in enumerate(mask_final):
                        result_meta = all_dino_results[i]
                        key = result_meta["key"]
                        caption = result_meta["caption"] 
                        
                        conf = conf_final_sam[i]
                        xyxy = result_meta["xyxy"].numpy()
                        
                        if key == "ground":
                            mask_area = np.sum(mask)
                            if mask_area > 0:
                                area_above_mid = np.sum(mask[:H_mid, :]) 
                                ratio_above_mid = area_above_mid / mask_area
                                if ratio_above_mid <= OVER_MID_THRESHOLD:
                                    ground_mask = np.logical_or(ground_mask, mask)
                        elif key == "node":
                            self.node_segment_result.append({
                                "mask": mask, "caption": caption, "confidence": conf, "xyxy": xyxy, "frame_index": self.step_count,
                            })
                            node_annotations.append((mask, xyxy, conf, caption))
                        elif key == "main":
                            self.main_segment_result.append({
                                "mask": mask, "caption": caption, "confidence": conf, "xyxy": xyxy, "frame_index": self.step_count,
                            })
                            main_annotations.append((mask, xyxy, conf, caption))
                        elif key == "sub":
                            self.sub_segment_result.append({
                                "mask": mask, "caption": caption, "confidence": conf, "xyxy": xyxy, "frame_index": self.step_count,
                            })
                            sub_annotations.append((mask, xyxy, conf, caption))
                        
                    if np.any(ground_mask):
                        ground_annotations.append((ground_mask, "ground"))
                        
                # --- 4. Build annotated image per category ---
                annotated_images_for_stitching = []

                def create_annotated_image(image_np, annotations_list, title=""):
                    if not annotations_list:
                        padded_img = cv2.copyMakeBorder(
                            image_np.copy(), PADDING, PADDING, PADDING, PADDING, 
                            cv2.BORDER_CONSTANT, value=BORDER_COLOR
                        )
                        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        text_x = (padded_img.shape[1] - text_size[0]) // 2
                        text_y = text_size[1] + 10
                        cv2.putText(padded_img, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                        return padded_img

                    if len(annotations_list[0]) == 4:
                        masks_to_draw, xyxy_to_draw, confs_to_draw, captions_to_draw = zip(*annotations_list)
                        
                        xyxy_final = np.stack(xyxy_to_draw)
                        masks_conf_final = np.array(confs_to_draw)
                        mask_final = np.stack(masks_to_draw)

                        padded_image = cv2.copyMakeBorder(
                            image_np.copy(), PADDING, PADDING, PADDING, PADDING, 
                            cv2.BORDER_CONSTANT, value=BORDER_COLOR
                        )
                        xyxy_padded = xyxy_final.copy()
                        xyxy_padded[:, [0, 2]] += PADDING
                        xyxy_padded[:, [1, 3]] += PADDING

                        masks_padded = np.pad(mask_final, 
                                            ((0, 0), (PADDING, PADDING), (PADDING, PADDING)), 
                                            mode='constant', constant_values=0)
                        detections_padded = sv.Detections(
                            xyxy=xyxy_padded, 
                            confidence=masks_conf_final,
                            class_id=np.arange(len(masks_padded)).astype(int), 
                            mask=masks_padded, 
                        )

                        labels = [
                            f"{cap} {conf:.2f}" 
                            for cap, conf in zip(captions_to_draw, masks_conf_final)
                        ]

                        annotated_img = mask_annotator.annotate(scene=padded_image.copy(), detections=detections_padded)
                        annotated_img = box_annotator.annotate(
                            scene=annotated_img, 
                            detections=detections_padded, 
                            labels=labels
                        )
                        
                    else:
                        masks_to_draw, captions_to_draw = zip(*annotations_list)
                        mask_final = np.stack(masks_to_draw)
                        padded_image = cv2.copyMakeBorder(
                            image_np.copy(), PADDING, PADDING, PADDING, PADDING, 
                            cv2.BORDER_CONSTANT, value=BORDER_COLOR
                        )

                        masks_padded = np.pad(mask_final, 
                                            ((0, 0), (PADDING, PADDING), (PADDING, PADDING)), 
                                            mode='constant', constant_values=0)
                        detections_padded = sv.Detections(
                            xyxy=np.zeros((len(masks_padded), 4)),
                            confidence=np.ones(len(masks_padded)),
                            class_id=np.arange(len(masks_padded)).astype(int), 
                            mask=masks_padded, 
                        )

                        annotated_img = mask_annotator.annotate(scene=padded_image.copy(), detections=detections_padded)
                        for i, mask in enumerate(masks_padded):
                            y_indices, x_indices = np.where(mask)
                            if len(x_indices) > 0 and len(y_indices) > 0:
                                center_x = int(np.mean(x_indices)) + PADDING
                                center_y = int(np.mean(y_indices)) + PADDING
                                cv2.putText(annotated_img, captions_to_draw[i], 
                                        (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Category title
                    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (annotated_img.shape[1] - text_size[0]) // 2
                    text_y = text_size[1] + 10 
                    cv2.putText(annotated_img, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    
                    return annotated_img
                annotated_images_for_stitching.append(create_annotated_image(original_image_np, ground_annotations, "Ground Objects"))
                annotated_images_for_stitching.append(create_annotated_image(original_image_np, node_annotations, "Node Objects"))
                annotated_images_for_stitching.append(create_annotated_image(original_image_np, main_annotations, "Main Track Objects"))
                annotated_images_for_stitching.append(create_annotated_image(original_image_np, sub_annotations, "Sub Track Objects"))

                # --- 5. Stitch and save ---
                target_h = max(img.shape[0] for img in annotated_images_for_stitching)
                target_w = max(img.shape[1] for img in annotated_images_for_stitching)
                resized_images = []
                for img in annotated_images_for_stitching:
                    scale = min(target_h / img.shape[0], target_w / img.shape[1])
                    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    pad_h = (target_h - new_h) // 2
                    pad_w = (target_w - new_w) // 2
                    padded = cv2.copyMakeBorder(
                        resized, pad_h, target_h - new_h - pad_h, 
                        pad_w, target_w - new_w - pad_w, 
                        cv2.BORDER_CONSTANT, value=BORDER_COLOR
                    )
                    resized_images.append(padded)

                top_row = np.hstack(resized_images[:2])
                bottom_row = np.hstack(resized_images[2:])
                four_grid_image = np.vstack([top_row, bottom_row])
                if self.segment_save_folder is not None:
                    output_filename = os.path.join(self.segment_save_folder, f"step{self.step_count}.png") 
                    cv2.imwrite(output_filename, cv2.cvtColor(four_grid_image, cv2.COLOR_RGB2BGR))
                return annotated_images_for_stitching, ground_mask
        
        finally:
            image_tensor.to('cpu')