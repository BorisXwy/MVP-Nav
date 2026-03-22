import os
import numpy as np
import torch
import cv2
from plyfile import PlyData, PlyElement
import shutil
from collections import deque, Counter
import matplotlib.pyplot as plt
import open3d as o3d # 用于 OBB 计算

from sklearn.cluster import DBSCAN

from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as SciPyRotation
from typing import List, Dict, Tuple, Any

import matplotlib.pyplot as plt

# NOTE: 请确保以下所有导入路径在您的环境中是有效的
from src.map.vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map, depth_to_world_coords_points
from src.map.vggt.utils.pose_enc import pose_encoding_to_extri_intri
# 假设这些是用户提供的辅助函数
from src.utils.my_tools import compute_similarity_transform, apply_transform, get_mode, try_merge_obb, add_obb_to_pointcloud, check_obb_intersection


import torchvision.transforms.functional as TF



# 3D 置信度阈值 (超参数)
CONF_THRESHOLD_3D = 0.5


def compute_camera_plane_tilt_angle(raw_trajectory: np.ndarray) -> float:
    """
    基于预测的相机外参（轨迹）拟合“相机平面”，并计算该平面与当前坐标系水平面（y=const）的夹角。
    """
    tilt_deg, _, _, _ = _compute_leveling_transform(raw_trajectory)
    return tilt_deg


def _compute_leveling_transform(raw_trajectory: np.ndarray):
    """
    拟合相机平面，计算使该平面变为水平（y=const）的 4x4 刚体变换 T：p_new = R @ p + t。
    绕相机位置质心旋转，使变换后相机平面与水平面平行。

    返回:
        tilt_angle_deg: 平面与水平面夹角（度）
        T_4x4: (4,4) 齐次变换矩阵
        R_3x3: 旋转矩阵
        t_3: 平移向量 (3,)
    """
    nan_ret = (float("nan"), np.eye(4), np.eye(3), np.zeros(3))
    if raw_trajectory is None or raw_trajectory.size == 0:
        return nan_ret
    positions = raw_trajectory[:, :3, 3]
    if len(positions) < 3:
        return nan_ret
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1].copy()
    if normal[1] < 0:
        normal = -normal
    cos_angle = np.clip(float(normal[1]), 0.0, 1.0)
    tilt_rad = np.arccos(cos_angle)
    tilt_angle_deg = float(np.degrees(tilt_rad))
    # 旋转 R：将法向量 n 转到 (0,1,0)，即相机平面变为水平
    r, _ = SciPyRotation.align_vectors([[0, 1, 0]], normal.reshape(1, 3))
    R = r.as_matrix()
    t = centroid - R @ centroid
    T_4x4 = np.eye(4)
    T_4x4[:3, :3] = R
    T_4x4[:3, 3] = t
    return tilt_angle_deg, T_4x4, R, t


def _apply_transform_4x4(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """points (N, 3)，返回 (N, 3)。"""
    R, t = T[:3, :3], T[:3, 3]
    return (points @ R.T) + t


class Map:
    """
    物理感知建图：观测序列 → GSSL（BEV、场、OBB 等）。
    通过 obs_provider 获取观测数据，与具体 Agent 类型解耦。
    """
    def __init__(self, last_set_2: int, last_set_1: int, step: int, stage: int, model_info, args, obs_provider):
        """
        obs_provider: 提供观测数据的对象，需有 obs_history, node_segment_result, main_segment_result,
                     sub_segment_result, segment_save_folder 属性（与 Agent 解耦）。
        """
        self.obs_provider = obs_provider

        # 阶段信息
        self.last_set_2 = last_set_2
        self.last_set_1 = last_set_1
        self.step = step
        self.stage = stage
        
        # 观测和点云数据
        self.recent_pcd = None
        self.predictions = None
        self.trajectory = None # PCD点云坐标系
        self.raw_trajectory = None
        self.trajectory_with_pixel = None # 用于 corrdinate_match
        self.camera_y_pcd = None
        
        # 地图和栅格化参数
        self.resolution_per_pixel = args.map_resolution
        self.step_size = args.step_size
        self.bev = None
        self.map_info = {}
        self.map_info['stage'] = stage
        
        self.objects_in_scene = {
            "node": [],
            "main": [],
            "sub": [],
        }
        self.stage_objects_list_pcd = {
            "node": [],
            "main": [],
            "sub": [],
        }
        self.stage_objects_list_sim = {
            "node": [],
            "main": [],
            "sub": [],
        }
        self.stage_object_list_bev = {
            "node": [],
            "main": [],
            "sub": [],
        }
        
        # 相似变换相关变量 (旧版逻辑的核心)
        self.transform_sim2bev = None
        self.transform_bev2sim = None

        # 模型和配置
        self.node_segment_results = []
        self.main_segment_results = []
        self.sub_segment_results = []
        self.save_folder = args.midterm_save_folder+f"/stage_{self.stage}"
        os.makedirs(self.save_folder, exist_ok=True)
        self.segment_save_folder= os.path.join(self.save_folder,"segment_result")
        os.makedirs(self.segment_save_folder, exist_ok=True)
        self.map_info['save_folder'] = self.save_folder
        self.vggt = model_info['vggt']
        self.device = model_info['device']
        self.args = args
        self.real_camera_height = args.camera_height
                
    def obs_load(self):

        a = self.last_set_2 + int((self.last_set_1 - self.last_set_2) * 0.9)
        b = self.last_set_1 - int((self.step - self.last_set_1) * 0.1)
        self.start_index = max(a, b)
        self.end_index = self.step
        
        obs_history = getattr(self.obs_provider, "obs_history", [])
        n_obs = len(obs_history)
        if n_obs == 0:
            print("Error: obs_history is empty.")
            return
        # 闭区间 [start_index, end_index] 必须在 [0, n_obs-1] 内，避免 IndexError
        self.end_index = min(self.end_index, n_obs - 1)
        if self.start_index > self.end_index:
            print("Error: Invalid image index range (start_index=%s, end_index=%s, len(obs_history)=%s)." % (self.start_index, self.end_index, n_obs))
            return
        list_of_tensors = []
        for index in range(self.start_index, self.end_index + 1):
            list_of_tensors.append(obs_history[index])

        node_seg = getattr(self.obs_provider, "node_segment_result", [])
        main_seg = getattr(self.obs_provider, "main_segment_result", [])
        sub_seg = getattr(self.obs_provider, "sub_segment_result", [])
        self.node_segment_results = [
            result for result in node_seg
            if self.start_index <= result['frame_index'] <= self.end_index 
        ]
        self.main_segment_results = [
            result for result in main_seg
            if self.start_index <= result['frame_index'] <= self.end_index 
        ]
        self.sub_segment_results = [
            result for result in sub_seg
            if self.start_index <= result['frame_index'] <= self.end_index 
        ]
        self.images = torch.stack(list_of_tensors, dim=0)

        segment_save_folder = getattr(self.obs_provider, "segment_save_folder", "")
        for i in range(self.start_index,self.end_index + 1):
            file_name = f"step{i}.png"
            source_file_path = os.path.join(segment_save_folder, file_name)
            target_file_path = os.path.join(self.segment_save_folder, file_name)    
            shutil.copy2(source_file_path, target_file_path)

    def build_pcd(self):

        self.obs_load()
        self.vggt.to(self.device)

        # 定义全局常量 (可根据需要调整)
        CONF_THRESHOLD_PERCENTILE = 10    # 深度置信度过滤的百分位数
        NB_NEIGHBORS = 30                 # 统计滤波：邻域点数
        STD_RATIO = 2.0                   # 统计滤波：标准差倍数阈值

        # --- VGG-T 推理部分 (保持不变) ---
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                self.predictions = self.vggt(self.images.to(self.device))
        predictions = self.predictions

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], self.images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        # ----------------------------------

        # --- 【新增：全局置信度动态阈值计算】 ---
        # 获取所有帧的深度置信度，并确保转换为 NumPy
        # .squeeze() 用于移除批次和通道维度
        raw_depth_conf = predictions["depth_conf"].cpu().numpy().squeeze() 
        
        # 扁平化所有置信度值
        conf_flat = raw_depth_conf.reshape(-1)
        
        # 计算动态阈值（使用 12.5 百分位数）
        CONF_THRESHOLD_3D = np.percentile(conf_flat, CONF_THRESHOLD_PERCENTILE)
        # -------------------------------------------------------------------

        for frame_index in range(self.end_index - self.start_index + 1):
            try:
                # 提取当前帧数据
                depth_map = predictions["depth"][0][frame_index]
                depth_conf = predictions["depth_conf"][0][frame_index]
                color_image = predictions["images"][0][frame_index] 
                extrinsic = predictions["extrinsic"][0][frame_index]
                intrinsic = predictions["intrinsic"][0][frame_index]
                detections_node = [
                    result for result in self.node_segment_results if result['frame_index'] == frame_index + self.start_index
                ]
                detections_main = [
                    result for result in self.main_segment_results if result['frame_index'] == frame_index + self.start_index
                ]
                detections_sub = [
                    result for result in self.sub_segment_results if result['frame_index'] == frame_index + self.start_index
                ]
                detections = {
                    "node": detections_node,
                    "main": detections_main,
                    "sub": detections_sub,
                }

                # 转换为 NumPy 并移除通道维度
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
                depth_map = np.squeeze(depth_map)
                
                if isinstance(depth_conf, torch.Tensor):
                    depth_conf = depth_conf.cpu().numpy()
                depth_conf = np.squeeze(depth_conf) 
                
                if isinstance(extrinsic, torch.Tensor):
                    extrinsic = extrinsic.cpu().numpy()
                if isinstance(intrinsic, torch.Tensor):
                    intrinsic = intrinsic.cpu().numpy()
                if isinstance(color_image, torch.Tensor):
                    color_image = color_image.cpu().numpy()
                else:
                    color_image = color_image

                if color_image.ndim == 3 and color_image.shape[0] in [3, 4]: 
                    color_image = color_image.transpose(1, 2, 0)
                else:
                    color_image = color_image

            except (KeyError, IndexError):
                print(f"Frame {frame_index} data (depth, image, pose, or intrinsic) missing.")
                continue

            
            # 深度图投影到pcd坐标系：点云空间的世界坐标系
            world_points_map, _, point_mask = depth_to_world_coords_points(depth_map, extrinsic, intrinsic)
            
            if world_points_map is None:
                continue
            
            # --- 【对象级筛选 1：基于动态 Depth Confidence】 ---
            # 保持了 > 1e-5 的保护，避免浮点数 0 带来的问题
            conf_mask = (depth_conf >= CONF_THRESHOLD_3D) & (depth_conf > 1e-5) 
            combined_valid_mask = point_mask & conf_mask
            # -------------------------------------------------
            target_h, target_w = combined_valid_mask.shape
            for key in detections.keys():
                for obj in detections[key]:

                    mask_2d = obj['mask']
                    # 确保 mask 与深度图尺寸一致（如 SAM 与 VGGT 分辨率不一致时）
                    if mask_2d.shape != (target_h, target_w):
                        mask_2d = cv2.resize(
                            mask_2d.astype(np.uint8), (target_w, target_h),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    caption = obj['caption']
                    final_mask = mask_2d & combined_valid_mask
                    # 若因深度置信度过严导致无有效点，退化为仅要求有效深度（不要求高置信度），避免 main 目标被整块过滤
                    if not np.any(final_mask) and np.any(mask_2d):
                        final_mask = mask_2d & point_mask
                    if not np.any(final_mask):
                        continue
                    points_w_original = world_points_map[final_mask] 
                    colors_original = color_image[final_mask].astype(np.float32) / 255.0 
                    points_to_use = points_w_original
                    colors_to_use = colors_original
                    if points_w_original.size > 0:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points_w_original)
                        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=NB_NEIGHBORS,
                                                                std_ratio=STD_RATIO)
                        points_to_use = np.asarray(pcd.select_by_index(ind).points)
                        colors_to_use = colors_original[ind]                     
                    # -----------------------------------------------------------------
                    if points_to_use.size > 0:
                        min_coords_aabb = np.min(points_to_use, axis=0)
                        max_coords_aabb = np.max(points_to_use, axis=0)
                        center_pcd = (min_coords_aabb + max_coords_aabb) / 2.0
                        dimensions_pcd = max_coords_aabb - min_coords_aabb
                        extent_pcd = dimensions_pcd / 2.0
                        volume_3d = np.prod(dimensions_pcd)
                        orientation_pcd = np.eye(3)
                    else:
                        center_pcd = np.array([0.0, 0.0, 0.0])
                        extent_pcd = np.array([0.0, 0.0, 0.0])
                        orientation_pcd = np.eye(3) 
                        volume_3d = 0.0
                    # ------------------------------------
                    # 保存单实例 OBB 以及其对应的点云，用于后续聚类/体素化融合
                    self.objects_in_scene[key].append({
                        'caption': [caption],
                        'center': center_pcd,
                        'frame_index': frame_index,
                        'extent': extent_pcd,
                        'volume': volume_3d,
                        'orientation': orientation_pcd,
                        'points': points_to_use,  # 用于簇内体素化与 OBB 重新估计
                    }) 
        # --- 轨迹计算部分 (保持不变) ---
        extrinsic_np = predictions["extrinsic"].cpu().numpy().squeeze(0) 
        cam_to_world_mat = closed_form_inverse_se3(extrinsic_np) 
        self.cam_to_world_mat = cam_to_world_mat
        trajectory = cam_to_world_mat[:, :3, :] # shape (N, 3, 4)
        self.raw_trajectory = trajectory
        tilt_deg, T_level, R_level, t_level = _compute_leveling_transform(trajectory)
        self.map_info["camera_plane_tilt_angle_deg"] = tilt_deg
        apply_leveling = getattr(self.args, "apply_leveling_transform", True)
        if apply_leveling and not np.isnan(tilt_deg) and tilt_deg >= 1e-4:
            # 对轨迹施加水平校正：位置与旋转
            trajectory = trajectory.copy()
            for i in range(len(trajectory)):
                trajectory[i, :3, 3] = _apply_transform_4x4(trajectory[i, :3, 3].reshape(1, 3), T_level).ravel()
                trajectory[i, :3, :3] = R_level @ trajectory[i, :3, :3]
            self.raw_trajectory = trajectory
            # 对 objects_in_scene 中的 center 与 points 施加同一变换
            for key in self.objects_in_scene:
                for obj in self.objects_in_scene[key]:
                    obj["center"] = _apply_transform_4x4(obj["center"].reshape(1, 3), T_level).ravel()
                    if obj.get("points") is not None and obj["points"].size > 0:
                        obj["points"] = _apply_transform_4x4(obj["points"], T_level)
        ty = trajectory[:, 1, 3]
        tx = trajectory[:, 0, 3]
        tz = trajectory[:, 2, 3]
        rotation_matrices = trajectory[:, :3, :3]
        forward_vecs_3d = rotation_matrices[:, 2, :]
        forward_vecs_2d = np.column_stack((forward_vecs_3d[:, 0], forward_vecs_3d[:, 2]))
        angles_rad = np.pi-np.arctan2(forward_vecs_2d[:, 1], forward_vecs_2d[:, 0]) 
        self.trajectory = np.column_stack((tx, tz, forward_vecs_2d ,angles_rad))
        if ty.size > 0:
            avg_y_pcd = np.mean(ty)
            self.camera_y_pcd = avg_y_pcd
        else:
            self.camera_y_pcd = None
        # --- 全局点云构建部分 ---
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        world_points = unproject_depth_map_to_point_map(
            predictions["depth"],
            predictions["extrinsic"],
            predictions["intrinsic"]
        )
        conf = predictions["depth_conf"]
        
        colors = self.images.cpu().numpy().transpose(0, 2, 3, 1)  # now (S, H, W, 3)
        colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
        points_flat = world_points.reshape(-1, 3)
        if apply_leveling and not np.isnan(tilt_deg) and tilt_deg >= 1e-4:
            points_flat = _apply_transform_4x4(points_flat, T_level)
        conf_flat = conf.reshape(-1)

        conf_mask_pcd = (conf_flat >= CONF_THRESHOLD_3D) & (conf_flat > 1e-5)
        filtered_points = points_flat[conf_mask_pcd]
        filtered_colors = colors_flat[conf_mask_pcd]

        # 将点云数据保存到实例变量中 (PlyData格式)
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        vertex_data = np.empty(filtered_points.shape[0], dtype=vertex_dtype)
        vertex_data['x'] = filtered_points[:, 0]
        vertex_data['y'] = filtered_points[:, 1]
        vertex_data['z'] = filtered_points[:, 2]
        vertex_data['red'] = filtered_colors[:, 0]
        vertex_data['green'] = filtered_colors[:, 1]
        vertex_data['blue'] = filtered_colors[:, 2]
        
        self.recent_pcd = PlyData([PlyElement.describe(vertex_data, name='vertex')], text=False)
        file_path = os.path.join(self.save_folder, 'raw_pcd.ply')
        self.recent_pcd.write(file_path)

        self.vggt.to("cpu")

    def pre_cluster_objects(self, objects, eps_ratio: float = 0.1, min_samples: int = 4, vis: bool = True, ):
        """
        在几何融合之前，根据物体的中心点进行基于密度的空间聚类（DBSCAN）。
        领域距离阈值 (eps) 与场景的整体空间尺度相关联。
        
        Args:
            eps_ratio (float): 领域距离阈值占场景最大尺度的比例 (例如 0.01 = 1%)。
            min_samples (int): 形成密集区域所需的最小样本数。
            vis (bool): 是否进行可视化。
        Returns:
            Dict[int, List[Dict]]: 聚类标签到对应物体列表的映射字典。
        """
        
        # 1. 数据提取
        # 假设每个物体对象都有一个 'center' 键，存储 [x, y, z] 坐标
        centers = np.array([obj['center'] for obj in objects])
        
        if len(centers) < 2:
            return {0: objects}

        # === 💥 动态计算 eps 的关键逻辑 💥 ===
        # np.ptp (Peak to Peak) 计算的是 (Max - Min)
        x_range = np.ptp(centers[:, 0]) 
        y_range = np.ptp(centers[:, 1])
        z_range = np.ptp(centers[:, 2])
        
        # 场景的最大几何尺度 (取三个维度中的最大跨度)
        max_scene_scale = max(x_range, y_range, z_range)
        
        # 计算动态 eps：eps = 场景最大尺度 * 比例因子
        calculated_eps = max_scene_scale * eps_ratio 
        
        
        # 2. 应用聚类
        dbscan = DBSCAN(eps=calculated_eps, min_samples=min_samples)
        labels = dbscan.fit_predict(centers)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # 3. 数据分组
        clustered_groups: Dict[int, List[Dict[str, Any]]] = {}
        for obj, label in zip(objects, labels):
            if label not in clustered_groups:
                clustered_groups[label] = []
            
            obj['cluster_label'] = label 
            clustered_groups[label].append(obj)

        return clustered_groups

    def merge_objects(self, 
                        eps_ratio: float = 0.1, 
                        min_samples: int = 4, 
                        iqr_k_threshold: float = 1.2, # 此处改为“体积中位数放大倍数”，默认 1.2
                        vis: bool = True):
        """
        基于“多帧 AABB + 统计约束”的简单稳健物体融合流程：

        1) 预处理与空间聚类 (Identification)
           - 输入：所有帧的 AABB（由 build_pcd 中的 objects_in_scene 给出）
           - 操作：提取中心点，按 3D 空间位置使用 DBSCAN 聚类，得到“同一物体”的簇
        2) 离群框体剔除 (Outlier Removal)
           - 在每个簇内，按体积做鲁棒过滤：保留体积 <= median_volume * iqr_k_threshold 的框
        3) 稳健边界提取 (Robust Boundary Estimation, 方案 A：分位数收缩)
           - 对每个簇内保留的 AABB 的 (xmin,ymin,zmin,xmax,ymax,zmax) 做分位数统计：
             xmin_final = Percentile(xmin_list, q_low)
             xmax_final = Percentile(xmax_list, q_high)
             等价地对 y,z 维做同样操作。
           - 得到收缩后的 AABB，并用其中心/半边长/单位旋转矩阵构造最终 OBB。

        Args:
            eps_ratio (float): DBSCAN 中 eps 与场景最大尺度的比例。
            min_samples (int): DBSCAN 的最小样本数，可理解为“至少多少帧”。
            iqr_k_threshold (float): 体积中位数的放大倍数，例如 1.2 表示过滤掉体积 > 1.2 * median_volume 的框。
            vis (bool): 是否进行可视化。
        Returns:
            List[Dict[str, Any]]: 融合后的物体列表。
        """
        # 结果容器：每个 key 是一个“稳定物体”的最终 AABB（cluster 级别）
        final_merged_objects = {
            "node": [],
            "main": [],
            "sub": [],
        }

        # 分位数参数（可以通过 args 覆盖）
        q_low = getattr(self.args, "aabb_percentile_low", 30.0)   # 内缩下边界
        q_high = getattr(self.args, "aabb_percentile_high", 70.0) # 内缩上边界
        volume_scale = iqr_k_threshold  # 复用该参数作为“中位数放大倍数”

        # 1. 按类别遍历所有帧上的 AABB
        for key in self.objects_in_scene.keys():
            objects_list = self.objects_in_scene[key]
            if len(objects_list) == 0:
                continue

            # main/sub 目标通常出现帧数少，用 min_samples=1 保留单帧检测；node 用原 min_samples 做多帧稳健融合
            min_samples_key = 1 if key in ("main", "sub") else min_samples

            # 1.1 使用已有的 pre_cluster_objects：按中心点做 DBSCAN 聚类
            clustered_groups = self.pre_cluster_objects(
                eps_ratio=eps_ratio,
                min_samples=min_samples_key,
                vis=vis,
                objects=objects_list,
            )

            # 1.2 遍历每个簇，按“多帧观测”构造稳健 AABB
            for label_id, objects_in_group_raw in clustered_groups.items():
                # -1 通常为噪声，直接忽略
                if label_id == -1:
                    continue

                if len(objects_in_group_raw) < min_samples_key:
                    # 出现帧数太少，视为不稳定检测（main/sub 已放宽为 1）
                    continue

                # 2. 离群框体剔除：体积过大的“脏数据”丢弃
                volumes = np.array([obj.get("volume", 0.0) for obj in objects_in_group_raw], dtype=np.float32)
                if len(volumes) == 0:
                    continue

                median_vol = float(np.median(volumes))
                if median_vol <= 0:
                    continue

                volume_threshold = median_vol * volume_scale
                kept_objs = [
                    obj for obj, v in zip(objects_in_group_raw, volumes)
                    if v > 0 and v <= volume_threshold
                ]
                if len(kept_objs) == 0:
                    continue

                # 3. 统计所有保留 AABB 的边界，做分位数收缩
                mins_list = []
                maxs_list = []
                all_captions = []

                for obj in kept_objs:
                    center = np.asarray(obj["center"], dtype=np.float32)
                    extent = np.asarray(obj["extent"], dtype=np.float32)  # half-extent
                    aabb_min = center - extent
                    aabb_max = center + extent
                    mins_list.append(aabb_min)
                    maxs_list.append(aabb_max)

                    caps = obj.get("caption", [])
                    if isinstance(caps, (list, tuple)):
                        all_captions.extend(list(caps))
                    else:
                        all_captions.append(caps)

                if len(mins_list) == 0:
                    continue

                mins_arr = np.stack(mins_list, axis=0)
                maxs_arr = np.stack(maxs_list, axis=0)

                # 分位数内缩边界
                aabb_min_final = np.percentile(mins_arr, q_low, axis=0)
                aabb_max_final = np.percentile(maxs_arr, q_high, axis=0)

                full_extent = aabb_max_final - aabb_min_final
                if np.any(full_extent <= 0):
                    continue

                center_final = (aabb_min_final + aabb_max_final) / 2.0
                half_extent = full_extent / 2.0
                orientation = np.eye(3, dtype=np.float32)  # 仍然使用世界轴对齐 AABB
                volume_final = float(np.prod(full_extent))

                if len(all_captions) > 0:
                    mode_label = get_mode(all_captions)
                    caption_field = [mode_label]
                else:
                    caption_field = []

                final_merged_objects[key].append(
                    {
                        "caption": caption_field,
                        "center": center_final.astype(np.float32),
                        "extent": half_extent.astype(np.float32),
                        "volume": volume_final,
                        "orientation": orientation,
                    }
                )

        # 最终用于下游的就是 final_merged_objects
        filtered_objects = final_merged_objects

        # 将最终结果存储到类属性或其他位置（下游统一使用这一版）
        self.stage_objects_list_pcd = filtered_objects

        # ====================================================================
        # --- 【新增】保存带OBB框的点云为PLY文件（多种 OBB 变体）---
        # ====================================================================
        if vis:
            if self.recent_pcd is not None:
                try:
                    # 1) 原始单实例 OBB（未聚类、未融合）：objects_in_scene
                    raw_obb_list = []
                    for _k, objs in self.objects_in_scene.items():
                        raw_obb_list.extend(objs)

                    if len(raw_obb_list) > 0:
                        pcd_raw = add_obb_to_pointcloud(self.recent_pcd, raw_obb_list)
                        output_path_raw = self.save_folder + f"/stage_{self.stage}_raw.ply"
                        pcd_raw.write(output_path_raw)

                    # 2) 聚类 + 融合后的 OBB（未做 caption 筛选）：final_merged_objects
                    merged_obb_list = []
                    for _k, objs in final_merged_objects.items():
                        merged_obb_list.extend(objs)

                    if len(merged_obb_list) > 0:
                        pcd_merged = add_obb_to_pointcloud(self.recent_pcd, merged_obb_list)
                        output_path_merged = self.save_folder + f"/stage_{self.stage}_merged.ply"
                        pcd_merged.write(output_path_merged)

                    # 3) 当前用于后续 pipeline 的 OBB（已融合 + caption 筛选）：filtered_objects
                    filtered_obb_list = []
                    for _k, objs in filtered_objects.items():
                        filtered_obb_list.extend(objs)

                    if len(filtered_obb_list) > 0:
                        pcd_filtered = add_obb_to_pointcloud(self.recent_pcd, filtered_obb_list)
                        # 保持原有命名不变，方便下游和已有脚本使用
                        output_path_filtered = self.save_folder + f"/stage_{self.stage}.ply"
                        pcd_filtered.write(output_path_filtered)
                except NameError:
                    print("\n⚠️ 无法执行可视化保存：缺少辅助函数或类属性。")
                except Exception as e:
                    print(f"\n⚠️ 可视化保存失败: {e}")

        return filtered_objects  # 返回融合后的列表
        
    def refine_map_with_morphology(self):
        """
        对 BEV 图执行形态学操作以消除散点并膨胀障碍物。
        """
        if self.bev is None:
            print("Error: BEV map not initialized.")
            return

        # 1. 提取各个区域的二值化掩码 (Masks)
        # 灰色区域 (地面/可通行) mask: [200, 200, 200]
        # 注意：由于颜色可能略有偏差，我们使用一个范围
        gray_mask = cv2.inRange(self.bev, np.array([190, 190, 190]), np.array([210, 210, 210]))
        
        # 黑色区域 (障碍物/墙壁) mask: [50, 50, 50]
        black_mask = cv2.inRange(self.bev, np.array([40, 40, 40]), np.array([60, 60, 60]))


        ground_kernel_size = getattr(self.args, "ground_close_kernel_size", 5)
        ground_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ground_kernel_size, ground_kernel_size))

        closed_gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, ground_kernel)

        obstacle_kernel_size = getattr(self.args, "obstacle_dilate_kernel_size", 7)
        obstacle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (obstacle_kernel_size, obstacle_kernel_size))
        obstacle_dilate_iterations = getattr(self.args, "obstacle_dilate_iterations", 0)

        dilated_black_mask = cv2.dilate(black_mask, obstacle_kernel, iterations=obstacle_dilate_iterations)
        combined_mask = cv2.bitwise_or(gray_mask, black_mask)
        self.bev[combined_mask > 0] = [255, 255, 255]

        # Step 4b: 先绘制膨胀后的黑色障碍物 (优先级最高)
        # 膨胀后的黑色区域会覆盖部分白色和（如果闭运算效果不好）部分灰色区域
        self.bev[dilated_black_mask > 0] = [50, 50, 50]

        # Step 4c: 绘制闭运算后的灰色地面
        # 只有在**不是**膨胀后的黑色障碍物的地方，才绘制灰色地面
        # 使用闭运算后的灰色区域 减去 膨胀后的黑色区域
        final_gray_mask = cv2.subtract(closed_gray_mask, dilated_black_mask)
        self.bev[final_gray_mask > 0] = [200, 200, 200]

    def generate_map_and_fields(self, loc_agent_sim):
        """
        投影点云、生成 BEV 地图、计算 OBB 投影、空间场。
        """
        if self.recent_pcd is None or self.trajectory is None:
            print("Error: Point cloud or trajectory not found.")
            return

        R = self.resolution_per_pixel
        points = self.recent_pcd['vertex']
        
        x = points['x']
        y =-points['y']
        z = points['z']
        
        # 1. 动态计算地图尺寸和偏移量 (PCD点云 -> BEV 像素)
        min_x, max_x = x.min(), x.max()
        min_z, max_z = z.min(), z.max()
        min_y, max_y = y.min(), y.max()
        
        margin_pixels = 10
        x_offset = -min_x + margin_pixels * R
        z_offset = -min_z + margin_pixels * R
        y_offset = -min_y + margin_pixels * R
        
        x_pixels = ((x + x_offset) / R).astype(int)
        z_pixels = ((z + z_offset) / R).astype(int)
        y_pixels = ((y + y_offset) / R).astype(int)
        # 相机在bev中的y值
        camera_y_bev = ((-self.camera_y_pcd + y_offset) / R).astype(int)
        
        map_pixels_x = x_pixels.max() + margin_pixels
        map_pixels_z = z_pixels.max() + margin_pixels
        
        map_pixels_x = max(map_pixels_x, 100)
        map_pixels_z = max(map_pixels_z, 100)
        
        self.bev = np.full((map_pixels_x, map_pixels_z, 3), 255, dtype=np.uint8)
        
        valid_mask = (x_pixels >= 0) & (x_pixels < map_pixels_x) & \
                        (z_pixels >= 0) & (z_pixels < map_pixels_z)
                    
        x_final = x_pixels[valid_mask]
        z_final = z_pixels[valid_mask]
        y_final = y_pixels[valid_mask]
        
        # 2. 地面/障碍物分类 (略)
        def find_ceiling_floor_peaks(y_final: np.ndarray, num_bins: int = 256, peak_height_ratio: float = 0.05, peak_distance: int = 10):
            hist, bin_edges = np.histogram(y_final, bins=num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            min_peak_height = hist.max() * peak_height_ratio
            peaks, _ = find_peaks(hist, height=min_peak_height, distance=peak_distance)
            if peaks.size == 0:
                return y_final.max(), y_final.min(), np.array([])
            peak_y_values = bin_centers[peaks]
            y_ceiling = peak_y_values.max()
            y_floor = peak_y_values.min()
            return y_ceiling, y_floor, peak_y_values
        
        y_min = y_final.min()
        y_max = y_final.max()
        y_peak_high, y_peak_low, _ = find_ceiling_floor_peaks(y_final)

        # 地面/障碍物高度比例：地面为 [y_peak_low, y_ground)，比例越大则越多点被判为地面
        ground_ratio = getattr(self.args, "ground_height_ratio", 0.3)
        obstacle_ratio = getattr(self.args, "obstacle_height_ratio", 0.8)

        ground_threshold = y_min + (y_max - y_min) * ground_ratio
        obstacle_threshold = y_min + (y_max - y_min) * obstacle_ratio

        ground_threshold_1 = y_peak_low + (y_peak_high - y_peak_low) * ground_ratio
        obstacle_threshold_1 = y_peak_low + (y_peak_high - y_peak_low) * obstacle_ratio

        self.y_ground = ground_threshold_1
        self.y_ceiling = obstacle_threshold_1

        self.camera_height_bev = camera_y_bev - y_peak_low

        ground_mask = y_final < self.y_ground
        obstacle_mask = (y_final >= self.y_ground) & (y_final < self.y_ceiling)
        
        # 创建未经过形态学处理的原始BEV地图
        self.bev_raw = np.full((map_pixels_x, map_pixels_z, 3), 255, dtype=np.uint8)                                         
        self.bev_raw[x_final[ground_mask], z_final[ground_mask]] = [200, 200, 200]  # 浅灰色地面
        self.bev_raw[x_final[obstacle_mask], z_final[obstacle_mask]] = [50, 50, 50]   # 深灰色障碍物
        
        # 应用形态学处理（原始版本）
        self.bev = self.bev_raw.copy()
        
        self.refine_map_with_morphology()
        self.map_info['stage_bev'] = self.bev
        
        # 3. 计算带像素的轨迹 (用于 corrdinate_match)
        # self.trajectory 结构: (X_pcd, Z__pcd, forward_x, forward_z, Angle_rad)
        traj_pcd_x, traj_pcd_z, traj_forward_x, traj_forward_z, traj_angle = self.trajectory.T
        
        traj_pix_x = ((traj_pcd_x + x_offset) / R).astype(int)
        traj_pix_z = ((traj_pcd_z + z_offset) / R).astype(int)
        
        # 结构: (X_pixel, Z_pixel, forward_x, forward_z, Angle_rad)
        self.trajectory_with_pixel = np.column_stack([
            traj_pix_x,           # X_pixel (新)
            traj_pix_z,           # Z_pixel (新)
            traj_forward_x,       # forward_x (保留)
            traj_forward_z,       # forward_z (保留)
            traj_angle            # Angle_rad (保留)
        ])
        
        self.map_info['grid_resolution'] = R
        self.map_info['map_pixels_x'] = map_pixels_x
        self.map_info['map_pixels_z'] = map_pixels_z
        
        # OBB 方向修正：旧版逻辑要求 Y 轴翻转
        flip_y = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

        def pcd_to_bev_point(point_pcd):
            # 投影到像素坐标 (x, z)，y 轴高度信息保留
            bev_x = int((point_pcd[0] + x_offset) / R)
            bev_z = int((point_pcd[2] + z_offset) / R)
            bev_y = int((-point_pcd[1] + y_offset) / R)
            return np.array([bev_x, bev_y, bev_z])
        
        for key in self.stage_objects_list_pcd.keys():
            for obj in self.stage_objects_list_pcd[key]:
                center_pcd = np.array(obj['center'])
                extent_pcd = np.array(obj['extent'])
                orientation_pcd = np.array(obj['orientation'])
                
                center_bev = pcd_to_bev_point(center_pcd)
                extent_bev = extent_pcd / R
                
                # OBB 方向 Y 轴翻转修正
                orientation_bev = orientation_pcd @ flip_y 
                
                self.stage_object_list_bev[key].append({
                    'caption': obj['caption'],
                    'center': center_bev,
                    'extent': extent_bev,
                    'orientation': orientation_bev,
                    'volume': obj['volume'],
                    # 'score': obj['score'],
                })

        self.map_info['stage_object_list_bev'] = self.stage_object_list_bev

        
        # 5. 最终信息 (Graph 需要的位姿和轨迹)
        self.map_info['stage_trajectory_bev'] = self.trajectory_with_pixel


        # ============================================================================
        # 第零部分：保存原始未经过形态学处理的BEV图
        # ============================================================================
        
        plt.figure(figsize=(12, 10))
        plt.imshow(self.bev_raw.transpose(1, 0, 2), origin="lower")
        plt.title('Raw BEV Map (Before Morphological Processing)')
        plt.axis('equal')
        plt.colorbar(label='Pixel Intensity', shrink=0.8)
        plt.grid(True, alpha=0.3)
        
        # 保存原始BEV图像
        raw_bev_path = os.path.join(self.save_folder, 'bev_raw_before_morphology.png')
        plt.savefig(raw_bev_path, dpi=300, bbox_inches='tight')
        plt.close()

            
        # ============================================================================
        # 第一部分：保存处理过的BEV图和轨迹，以及最后一帧的朝向
        # ============================================================================
        
        plt.figure(figsize=(12, 10))
        
        # 显示BEV地图
        plt.imshow(self.bev.transpose(1, 0, 2), origin="lower")  # 转置以正确显示方向
        
        # 绘制轨迹
        traj_pix_x = self.trajectory_with_pixel[:, 0].astype(int)
        traj_pix_z = self.trajectory_with_pixel[:, 1].astype(int)
        plt.plot(traj_pix_x, traj_pix_z, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        plt.scatter(traj_pix_x, traj_pix_z, c='blue', s=20, alpha=0.5)
        
        # 绘制最后一帧的位置和朝向
        last_point_x = traj_pix_x[-1]
        last_point_z = traj_pix_z[-1]
        last_angle = self.trajectory_with_pixel[-1, 4]
        
        # 计算朝向向量
        arrow_length = 30
        end_x = last_point_x + arrow_length * np.cos(last_angle)
        end_z = last_point_z + arrow_length * np.sin(last_angle)
        
        plt.scatter([last_point_x], [last_point_z], c='red', s=100, marker='o', label='Last Position')
        plt.arrow(last_point_x, last_point_z, 
                end_x - last_point_x, end_z - last_point_z,
                head_width=8, head_length=10, fc='red', ec='red', linewidth=3)
        
        plt.title('BEV Map with Trajectory and Orientation')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        bev_traj_path = os.path.join(self.save_folder, 'bev_trajectory_orientation.png')
        plt.savefig(bev_traj_path, dpi=300, bbox_inches='tight')
        plt.close()


        # ============================================================================
        # 第二部分：保存BEV图和OBB的中心点、框、语义
        # ============================================================================
        for key in self.stage_object_list_bev.keys():
            plt.figure(figsize=(12, 10))
            
            # 显示BEV地图
            plt.imshow(self.bev.transpose(1, 0, 2), origin="lower")
            
            # 绘制OBB框
            for obj in self.stage_object_list_bev[key]:
                center = obj['center']
                extent = obj['extent']
                orientation = obj['orientation']
                caption = get_mode(obj['caption'])
                

                x_p, z_p = center[0], center[2]
                
                half_length, half_width = extent[0] / 2, extent[2] / 2
                forward, right = orientation[:, 0], orientation[:, 2] 
                
                pixel_corners = [
                    ((x_p + half_length * forward[0] + half_width * right[0]), (z_p + half_length * forward[2] + half_width * right[2])),
                    ((x_p + half_length * forward[0] - half_width * right[0]), (z_p + half_length * forward[2] - half_width * right[2])),
                    ((x_p - half_length * forward[0] - half_width * right[0]), (z_p - half_length * forward[2] - half_width * right[2])),
                    ((x_p - half_length * forward[0] + half_width * right[0]), (z_p - half_length * forward[2] + half_width * right[2]))
                ]
                
                
                # for j in range(4):
                #     start_point, end_point = pixel_corners[j], pixel_corners[(j + 1) % 4]
                #     plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                #             color='red', linewidth=2, linestyle='-')
                
                # 绘制中心点
                plt.scatter(x_p, z_p, c='red', s=50,)
                
                # 添加语义标签
                plt.text(x_p, z_p - 15, caption, 
                        fontsize=30, color='red', ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            bev_obb_path = os.path.join(self.save_folder, f'bev_{key}_objects.png')
            plt.savefig(bev_obb_path, dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()


        # ============================================================================
        # 第三部分：计算并保存空间场
        # ============================================================================

        self.corrdinate_match(loc_agent_sim)
        self.calculate_space_fields(self.bev_step)

        if self.stage_space_safety_field is not None:
            # 1. 安全场单独保存（仅场图，无坐标轴/网格/标题/colorbar）
            plt.figure(figsize=(10, 8))
            plt.imshow(self.stage_space_safety_field.transpose(1, 0), cmap='magma', origin="lower")
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            safety_path = os.path.join(self.save_folder, 'safety_field.png')
            plt.savefig(safety_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        # ============================================================================
        # 第四部分：对比显示原始和处理后的BEV图
        # ============================================================================
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 原始BEV图
        ax1.imshow(self.bev_raw.transpose(1, 0, 2), origin="lower")
        ax1.set_title('Raw BEV Map (Before Morphology)')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        
        # 处理后的BEV图
        ax2.imshow(self.bev.transpose(1, 0, 2), origin="lower")
        ax2.set_title('Processed BEV Map (After Morphology)')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存对比图像
        comparison_path = os.path.join(self.save_folder, 'bev_comparison_raw_vs_processed.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

        return



    def calculate_space_fields(self, safety_max_dist_pixels: int = 10):
        """
        计算离散化空间安全场 (stage_space_safety_field) 和空间探索场 (stage_space_explore_field)。
        
        Args:
            safety_max_dist_pixels (int): 安全场成本从 1.0 降到 0.0 的最大像素距离。
        """
        if self.bev is None:
            print("Error: BEV map not initialized for field calculation.")
            # 初始化空场以避免后续错误
            H, W = 100, 100 # 假设一个默认尺寸或从其他地方获取
            self.stage_space_safety_field = np.zeros((H, W), dtype=np.float32)
            self.stage_space_explore_field = np.zeros((H, W), dtype=np.float32)
            self.map_info['stage_space_safety_field'] = self.stage_space_safety_field
            self.map_info['stage_space_explore_field'] = self.stage_space_explore_field
            return

        H, W, _ = self.bev.shape
 
        # ------------------------------------------------------------------------------------------------------------
        # 提取区域掩码 (保持不变)
        obstacle_mask = cv2.inRange(self.bev, np.array([40, 40, 40]), np.array([60, 60, 60]))
        safe_ground_mask = cv2.inRange(self.bev, np.array([190, 190, 190]), np.array([210, 210, 210]))
        unknown_mask = cv2.inRange(self.bev, np.array([240, 240, 240]), np.array([260, 260, 260]))

        safe_ground_count = np.sum(safe_ground_mask / 255)
        obstacle_count = np.sum(obstacle_mask / 255)  

        if safe_ground_mask is None or safe_ground_mask.size == 0 or (safe_ground_count > 0 and obstacle_count / safe_ground_count > 2):
            self.stage_space_safety_field = None
            self.map_info['stage_space_safety_field'] = self.stage_space_safety_field
            self.stage_space_explore_field = None
            self.map_info['stage_space_explore_field'] = self.stage_space_explore_field
            self.map_info['stage_map_quality'] = 'bad'
            return

        self.map_info['stage_map_quality'] = 'good'

        # ----------------------------------------------------
        # 构建安全场
        # ----------------------------------------------------
        dist_to_obstacle = distance_transform_edt(~obstacle_mask)
        dist_to_unknown = distance_transform_edt(~unknown_mask)
        dist_to_safe = np.minimum(dist_to_obstacle, 0.5 * dist_to_unknown)
        dist_to_safe[obstacle_mask.astype(bool)] = 0.0
        dist_to_safe[unknown_mask.astype(bool)] = 0.0
        # 1. 归一化安全场
        if dist_to_safe.max() > 0:  # 避免除以0
            safety_field = dist_to_safe / dist_to_safe.max()
        else:
            safety_field = dist_to_safe  # 全零场

        # 4. 保存结果
        self.stage_space_safety_field = safety_field
        self.map_info['stage_space_safety_field'] = self.stage_space_safety_field

        # ------------------------------------------------------------------------------------------------------------
        # --- B. 空间探索场 (stage_space_explore_field) - 最新点权重 1.0 ---
        
        # 1. 提取和截取轨迹像素坐标 (只考虑最后 N_MAX=10 步)
        full_traj_points = self.trajectory_with_pixel[:, :2]
        N_MAX = 10
        traj_points = full_traj_points[-N_MAX:] 
        
        N = len(traj_points)
        x_coords, z_coords = np.indices((H, W))
        all_pixels = np.column_stack([x_coords.ravel(), z_coords.ravel()])
        
        if N == 0:
            # 轨迹为空，所有安全地面都是最高探索倾向 1.0
            explore_field = np.zeros((H, W), dtype=np.float32)
            explore_field[safe_ground_mask > 0] = 1.0
            self.stage_space_explore_field = explore_field
            self.map_info['stage_space_explore_field'] = self.stage_space_explore_field
            return
            
        # 2. 计算每个像素点到所有轨迹点的距离 (H*W, N)
        distance_matrix = cdist(all_pixels, traj_points)
        
        # 3. 计算线性上升权重 (最新点权重 1.0，倒数第 N 点权重 0.0)
        if N > 1:
            weights = np.arange(N) / (N - 1)
        else:
            weights = np.array([1.0]) # N=1 时，权重直接为 1.0
            
        weighted_distance = distance_matrix * weights[np.newaxis, :]
        sum_weighted_distance = np.sum(weighted_distance, axis=1)
        sum_weights = np.sum(weights)

        explore_field_base = (sum_weighted_distance / sum_weights).reshape(H, W)

        # 创建归一化探索场
        normalized_explore = np.zeros((H, W), dtype=np.float32)

        # 只在安全地面上计算归一化值
        safe_ground_indices = safe_ground_mask > 0
        explore_ground_values = explore_field_base[safe_ground_indices]

        if explore_ground_values.size > 0 and explore_ground_values.max() > 1e-6:
            normalized_values = explore_ground_values / explore_ground_values.max()
            normalized_explore[safe_ground_indices] = normalized_values

        # ----------------------------------------------------------------------
        # 引入未知区域吸引力 (Frontier Field) 
        # ----------------------------------------------------------------------

        # --- 计算前沿吸引力 (使用距离变换加速) ---
        frontier_field = np.zeros((H, W), dtype=np.float32)
        
        # 1. 对未知区域进行距离变换
        # unknown_mask > 0 是我们要找的前景。distance_transform_edt 默认计算到前景的距离。
        # 如果 unknown_mask 全是 0，结果就是 Inf (或最大值)，需要处理
        if np.any(unknown_mask):
            # distance_to_unknown 的形状为 (H, W)，每个像素值为它到最近未知像素的距离
            distance_to_unknown = distance_transform_edt(unknown_mask == 0) # 0 是未知，非 0 是已知
            
            # 我们只关心安全地面上的距离
            min_dist_to_unknown = distance_to_unknown[safe_ground_mask > 0]
            
            safe_ground_indices = np.where(safe_ground_mask.ravel() > 0)[0] # 仍然需要索引来赋值
            
            if min_dist_to_unknown.max() > 1e-6:
                # 归一化距离：0.0（最近）到 1.0（最远）
                normalized_dist = min_dist_to_unknown/min_dist_to_unknown.max()
                # 吸引力：1.0（最近）到 0.0（最远）
                frontier_value = 1.0 - normalized_dist
            else:
                frontier_value = np.ones_like(min_dist_to_unknown)
                
            frontier_field.ravel()[safe_ground_indices] = frontier_value    
        

        # --- 综合加权求和 ---

                
        EXPLORE_WEIGHT = 0.6  # 远离轨迹的权重
        FRONTIER_WEIGHT = 0.4 # 靠近前沿的权重
        
        explore_field = (normalized_explore * EXPLORE_WEIGHT) + (frontier_field * FRONTIER_WEIGHT)

        # 6. 存储结果
        self.stage_space_explore_field = explore_field
        self.map_info['stage_space_explore_field'] = self.stage_space_explore_field
        return # 结束函数

    def corrdinate_match(self, loc_agent):
        """
        计算 Sim 坐标系 (全局) 和 BEV 像素坐标系 (局部) 之间的相似变换。
        loc_agent: Agent 在全局 Sim 坐标系中的位姿 (x, z, angle_rad)。
        """

        # 获取相机高度参数
        camera_height_bev = self.camera_height_bev  # BEV相机高度
        camera_height_sim = self.real_camera_height  # 实际相机高度
        resolution_per_pixel = self.resolution_per_pixel  # 每像素分辨率
        
        # 计算 Sim 到 BEV 的缩放因子
        sim2bev_scale = (camera_height_sim / camera_height_bev)
        
        self.bev_step = self.step_size / sim2bev_scale
        self.map_info['step_size'] = self.step_size
        self.map_info['bev_step'] = self.bev_step

        # 获取 Agent 的全局 Sim 坐标
        loc_agent = np.array(loc_agent)
        
        # 从 self.trajectory_with_pixel 提取 BEV 像素位姿 (X_pixel, Z_pixel, Angle_rad)
        # 结构: (X_bev, Z_bev, forward_x, forward_z, Angle_rad, X_pixel, Z_pixel)
        last_pose = self.trajectory_with_pixel[-1]
        loc_bev = (last_pose[0], last_pose[1], last_pose[4])

        # 计算相似变换矩阵 (包含旋转、平移、缩放)
        self.transform_sim2bev, self.transform_bev2sim = compute_similarity_transform(loc_agent, loc_bev, sim2bev_scale)
        
        self.map_info['transform_sim2bev'] = self.transform_sim2bev
        self.map_info['transform_bev2sim'] = self.transform_bev2sim


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
        y_ground_sim = self.y_ground * scale
        y_sim = y_bev * scale - y_ground_sim
        
        return np.array([x_sim, y_sim, z_sim])

    
    def bev_to_sim_matrix(self, matrix):
        """
        [旧版逻辑] 将 BEV 坐标系下的方向矩阵 (3x3) 转换为 Sim 坐标系下的方向矩阵。
        """
        if matrix.shape != (3, 3):
            raise ValueError("输入必须是3x3方向矩阵")
        
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
        ])
        
        return sim_matrix


    def convert_to_sim(self):
        """
        将 BEV 坐标系下的 OBBs (stage_object_list_bev) 转换为全局 Sim 坐标系的 OBBs (stage_objects_list_sim)。
        """
        if not self.transform_bev2sim:
            print("Warning: 相似变换矩阵未建立。跳过 convert_to_sim 转换。")
            return
        
        for key in self.stage_object_list_bev.keys():
            for obj in self.stage_object_list_bev[key]:
                # 确保 OBB 方向已从 list 转换为 np.array
                center_bev = obj['center']
                extent_bev = obj['extent']
                orientation_bev = np.array(obj['orientation']) 

                center_sim = self.bev_to_sim_point(center_bev)
                orientation_sim = self.bev_to_sim_matrix(orientation_bev)
                
                scale = self.transform_bev2sim['scale']
                extent_sim = extent_bev * scale
                dimensions_sim = extent_sim * 2.0
                volume_sim = np.prod(dimensions_sim)

                self.stage_objects_list_sim[key].append({
                    'caption': obj['caption'],
                    'center': center_sim,
                    'extent': extent_sim,
                    'volume': volume_sim,
                    'orientation': orientation_sim,
                    'merged_status': False,
                    # 'score': obj['score'],
                })
                
        self.map_info['stage_objects_list_sim'] = self.stage_objects_list_sim

    def run_stage(self, loc_agent_sim):
        """
        论文 §III-C Physical Perception：单阶段完整流程。
        obs_load → build_pcd → merge_objects → generate_map_and_fields → convert_to_sim，
        返回 map_info 供 VLM/MVM 使用。
        """
        self.obs_load()
        self.build_pcd()
        self.merge_objects()
        self.generate_map_and_fields(loc_agent_sim=loc_agent_sim)
        self.convert_to_sim()
        return self.map_info



