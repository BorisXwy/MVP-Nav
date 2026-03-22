import numpy as np
from typing import List, Dict, Tuple, Any, Union
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import math
import random
import time
from IPython.display import display, clear_output
import numpy as np
from plyfile import PlyData, PlyElement
import os


def compute_similarity_transform(pose_a, pose_b, alpha):
    """
    计算 A <-> B 的相似变换参数。
    pose_a, pose_b: (x, y, theta)
    alpha: 比例关系 alpha = length_in_A / length_in_B
    返回: (a_to_b, b_to_a) 两个 dict，每个包含:
        {
        'scale': s,
        'rotation': angle_radians,
        'translation': np.array([tx, ty])
        }
    变换定义（以 b_to_a 为例）:
    p_A = s * R(r) @ p_B + t
    theta_A = theta_B + r
    """
    xa, ya, theta_a = pose_a
    xb, yb, theta_b = pose_b

    # 旋转角 (从 B 到 A)
    phi = theta_a - theta_b

    # scale: B -> A 的缩放因子为 alpha (因为 length_A = alpha * length_B)
    s_b_to_a = float(alpha)
    s_a_to_b = 1.0 / float(alpha)

    # 旋转矩阵 R(phi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    R_phi = np.array([[cos_phi, -sin_phi],
                    [sin_phi,  cos_phi]])

    p_a = np.array([xa, ya])
    p_b = np.array([xb, yb])

    # 平移 t (从 B 到 A)
    t_b_to_a = p_a - s_b_to_a * (R_phi @ p_b)

    # 反方向的平移 (从 A 到 B)： t_a_to_b 满足 p_B = s_a_to_b * R(-phi) @ p_A + t_a_to_b
    # 推导得： t_a_to_b = - s_a_to_b * R(-phi) @ t_b_to_a
    cos_mphi = np.cos(-phi)
    sin_mphi = np.sin(-phi)
    R_mphi = np.array([[cos_mphi, -sin_mphi],
                    [sin_mphi,  cos_mphi]])
    t_a_to_b = - s_a_to_b * (R_mphi @ t_b_to_a)

    b_to_a = {
        'scale': s_b_to_a,
        'rotation': float(phi),
        'translation': t_b_to_a.astype(float)
    }
    a_to_b = {
        'scale': s_a_to_b,
        'rotation': float(-phi),
        'translation': t_a_to_b.astype(float)
    }

    transform_agent2bev = a_to_b
    transform_bev2agent = b_to_a

    return transform_agent2bev, transform_bev2agent

def apply_transform(pose, transform):
    """
    使用 transform 将 pose 从源坐标系变换到目标坐标系。
    pose: (x, y, theta)
    transform: dict 如上 {'scale','rotation','translation'}
    变换规则:
    p_out = s * R(r) @ p_in + t
    theta_out = theta_in + r
    返回 (x_out, y_out, theta_out)
    """
    x, y, theta = pose
    s = float(transform['scale'])
    r = float(transform['rotation'])
    t = np.asarray(transform['translation'], dtype=float).reshape(2)

    cos_r = np.cos(r)
    sin_r = np.sin(r)
    R = np.array([[cos_r, -sin_r],
                [sin_r,  cos_r]])

    p_in = np.array([x, y])
    p_out = s * (R @ p_in) + t
    theta_out = theta + r

    return float(p_out[0]), float(p_out[1]), float(theta_out)


def get_box_data(min_corner, max_corner, color=(255, 0, 0), current_vertex_count=0):
    """
    根据 Min/Max Corner 生成 3D AABB 的 NumPy 顶点数组和 PLY 边数据。
    """
    x_min, y_min, z_min = min_corner
    x_max, y_max, z_max = max_corner

    # 8 个顶点 (Corner)
    vertices_coords = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],  # 0, 1
        [x_min, y_max, z_min], [x_max, y_max, z_min],  # 2, 3
        [x_min, y_min, z_max], [x_max, y_min, z_max],  # 4, 5
        [x_min, y_max, z_max], [x_max, y_max, z_max]   # 6, 7
    ], dtype=np.float32)

    # 12 条边连接
    edges_indices = np.array([
        [0, 1], [0, 2], [1, 3], [2, 3], # Z=min
        [4, 5], [4, 6], [5, 7], [6, 7], # Z=max
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接边
    ], dtype=np.int32)
    
    # 构造顶点数组，赋予指定颜色
    vertices_color = np.tile(np.array(color, dtype=np.uint8), (8, 1))

    # 构造 PLY 边数据：需要将索引相对于总点云进行偏移
    edge_dtype = [('vertex1', 'i4'), ('vertex2', 'i4')]
    edges_data = np.empty(12, dtype=edge_dtype)
    edges_data['vertex1'] = edges_indices[:, 0] + current_vertex_count
    edges_data['vertex2'] = edges_indices[:, 1] + current_vertex_count
    
    return vertices_coords, vertices_color, edges_data

def box_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> Tuple[float, float]:
    """
    计算两个 3D AABB 框之间的 IoU。
    为了确保完全包含时能够合并，返回 IoU 和 IoMinV 的最大值。
    box_a, box_b 格式: [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    # 1. 计算交集体积 (Intersection Volume)
    inter_min = np.maximum(box_a[:3], box_b[:3])
    inter_max = np.minimum(box_a[3:], box_b[3:])
    inter_sides = inter_max - inter_min
    
    if np.any(inter_sides <= 0):
        # 无交集，直接返回 (0.0, 0.0)
        return 0.0, 0.0 

    inter_vol = np.prod(inter_sides)

    # 2. 计算两个框的体积 (Volume)
    # 假设 box_a[3:] > box_a[:3]
    vol_a = np.prod(box_a[3:] - box_a[:3])
    vol_b = np.prod(box_b[3:] - box_b[:3])

    # 3. 计算 IoU 和 IoMinV
    
    # 避免体积为零
    if vol_a <= 1e-6 and vol_b <= 1e-6:
        return 0.0, 0.0
        
    union_vol = vol_a + vol_b - inter_vol
    min_vol = min(vol_a, vol_b)

    # IoU
    iou = inter_vol / union_vol
    
    # IoMinV (Intersection over Minimum Volume)
    iominv = 0.0
    if min_vol > 1e-6:
        iominv = inter_vol / min_vol
    return iou, iominv


def generate_box_mesh(min_corner, max_corner, color=(255, 0, 0)):
    """
    根据 Min Corner 和 Max Corner 生成一个 3D AABB 的顶点和边数据。
    
    Args:
        min_corner (np.ndarray): [min_x, min_y, min_z]
        max_corner (np.ndarray): [max_x, max_y, max_z]
        color (tuple): 边界框的颜色 (R, G, B)
        
    Returns:
        tuple: (vertices_data, faces_data)
    """
    x_min, y_min, z_min = min_corner
    x_max, y_max, z_max = max_corner

    # 8 个顶点 (Corner)
    # 索引: 0=---, 1=+-+, 2=-++, 3=+++, 4=-+-, 5=++-, 6=--+, 7=+-+
    # 重新定义标准索引以便于连接：
    vertices = np.array([
        [x_min, y_min, z_min],  # 0
        [x_max, y_min, z_min],  # 1
        [x_min, y_max, z_min],  # 2
        [x_max, y_max, z_min],  # 3
        [x_min, y_min, z_max],  # 4
        [x_max, y_min, z_max],  # 5
        [x_min, y_max, z_max],  # 6
        [x_max, y_max, z_max]   # 7
    ], dtype=np.float32)

    # 12 条边（线框）连接
    edges = np.array([
        [0, 1], [0, 2], [1, 3], [2, 3], # Z=min 平面
        [4, 5], [4, 6], [5, 7], [6, 7], # Z=max 平面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接两个平面的边
    ], dtype=np.int32)
    
    # 转换为 PLY 格式的顶点数据
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices_data = np.empty(8, dtype=vertex_dtype)
    vertices_data['x'] = vertices[:, 0]
    vertices_data['y'] = vertices[:, 1]
    vertices_data['z'] = vertices[:, 2]
    vertices_data['red'] = color[0]
    vertices_data['green'] = color[1]
    vertices_data['blue'] = color[2]

    # 转换为 PLY 格式的边数据 (线段列表)
    edge_dtype = [('vertex1', 'i4'), ('vertex2', 'i4')]
    edges_data = np.empty(12, dtype=edge_dtype)
    edges_data['vertex1'] = edges[:, 0]
    edges_data['vertex2'] = edges[:, 1]
    
    return vertices_data, edges_data


def get_box_center(box: np.ndarray) -> np.ndarray:
    """计算 AABB 框的几何中心 [cx, cy, cz]"""
    return (box[:3] + box[3:]) / 2.0

def get_mode(labels: List[Any]) -> Any:
    """计算列表中元素的众数"""
    if not labels:
        return None
    data = Counter(labels)
    # 返回出现次数最多的元素的值
    return data.most_common(1)[0][0]


def check_obb_intersection(obj1: Dict[str, Any], obj2: Dict[str, Any], mode: str,k_factor: float = 1.5) -> bool:
    """
    精确判断两个 OBB 框体是否相交。
    利用“所有框体都和 xz 平面平行，旋转只围绕垂直的 y 轴”的特性：
    - Y 轴（垂直）方向：使用 AABB 方式检测相交。
    - XZ 平面（水平）方向：使用 2D 分离轴定理 (SAT) 检测相交。
    """
    C1 = np.array(obj1['center'])
    E1 = np.array(obj1['extent'])
    R1 = np.array(obj1['orientation'])
    C2 = np.array(obj2['center'])
    E2 = np.array(obj2['extent'])
    R2 = np.array(obj2['orientation'])

    if mode == 'loose':
        # 计算 OBB 的最长半长
        max_E1 = np.max(E1)
        max_E2 = np.max(E2)

        # 计算中心距离
        center_dist = np.linalg.norm(C1 - C2)

        # 条件 1：C1 在 OBB2 的扩展范围内
        condition_1 = center_dist <= k_factor * max_E2

        # 条件 2：C2 在 OBB1 的扩展范围内
        condition_2 = center_dist <= k_factor * max_E1

        # 如果满足任一条件，则认为匹配成功
        return condition_1 or condition_2


    if mode == 'strict':
        # Part 1: Y 轴 (垂直) AABB 检测
        min_y1, max_y1 = C1[1] - E1[1], C1[1] + E1[1]
        min_y2, max_y2 = C2[1] - E2[1], C2[1] + E2[1]
        y_overlap = (max_y1 >= min_y2) and (max_y2 >= min_y1)
        if not y_overlap:
            return False

        # Part 2: XZ 平面 2D OBB 相交检测 (SAT)
        P1, P2 = C1[[0, 2]], C2[[0, 2]]
        E_proj1, E_proj2 = E1[[0, 2]], E2[[0, 2]]
        A1 = R1[[0, 2], :][:, [0, 2]] # R1 的 X, Z 轴在 XZ 平面的投影
        A2 = R2[[0, 2], :][:, [0, 2]] # R2 的 X, Z 轴在 XZ 平面的投影
        
        axes = np.hstack([A1, A2]).T
        unique_axes = []
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-6: continue
            axis = axis / norm
            is_unique = True
            for u_axis in unique_axes:
                if np.abs(np.dot(axis, u_axis)) > 0.999:
                    is_unique = False
                    break
            if is_unique: unique_axes.append(axis)
        
        for axis in unique_axes:
            r1 = E_proj1[0] * np.abs(np.dot(axis, A1[:, 0])) + E_proj1[1] * np.abs(np.dot(axis, A1[:, 1]))
            r2 = E_proj2[0] * np.abs(np.dot(axis, A2[:, 0])) + E_proj2[1] * np.abs(np.dot(axis, A2[:, 1]))
            dist_on_axis = np.abs(np.dot(P1 - P2, axis))

            if dist_on_axis > r1 + r2 + 1e-6:
                return False 
                
        return True




def try_merge_obb(
    obj1: Dict[str, Any], 
    obj2: Dict[str, Any], 
    V_check: bool = False,
    mode: str = "loose"
) -> Union[Dict[str, Any], bool]:
    """
    尝试融合两个 OBB 字典。使用基于最长 Extent 距离的几何判断；
    如果满足，则执行 OBB 核心属性的体积加权平均融合。
    """

    def weighted_average_rotation(R_old: np.ndarray, V_old: float, R_new: np.ndarray, V_new: float) -> np.ndarray:
        """使用 SVD 投影计算两个旋转矩阵的体积加权平均。"""
        # ... (代码与前一个回复中定义的一致)
        if V_old + V_new == 0:
            return np.eye(3)
            
        total_V = V_old + V_new
        W_old = V_old / total_V
        W_new = V_new / total_V
        
        M = R_old * W_old + R_new * W_new
        
        U, _, VT = np.linalg.svd(M)
        R_merged = U @ VT
        
        if np.linalg.det(R_merged) < 0:
            VT[-1, :] *= -1
            # U[:, 2] *= -1
            R_merged = U @ VT

            
        return R_merged

    
    # --- 0. 数据准备 ---
    C1, E1, R1 = np.array(obj1['center']), np.array(obj1['extent']), np.array(obj1['orientation'])
    V1 = obj1['volume']
    C2, E2, R2 = np.array(obj2['center']), np.array(obj2['extent']), np.array(obj2['orientation'])
    V2 = obj2['volume']


    center_dist = np.linalg.norm(C1 - C2)

    # --- 1. 匹配判断：基于体积相似度 ---
    if V_check:
        vol_ratio = max(V1, V2) / (min(V1, V2) + 1e-6)
        if vol_ratio > 2.0: # 体积差距过大，不融合
            return False, center_dist
    
    # --- 1. 匹配判断：基于最长 Extent 距离 ---
    
    should_merge = check_obb_intersection(obj1, obj2, mode = mode)

    if not should_merge:
        return False, center_dist
        
    # --- 2. 执行 OBB 核心属性的体积加权融合 ---
    
    if V1 + V2 <= 1e-6:
        return False, center_dist # 两个都是空框，不融合
        
    total_V = V1 + V2
    W1 = V1 / total_V
    W2 = V2 / total_V

    C_merged = C1 * W1 + C2 * W2
    V_merged = V1 * W1 + V2 * W2 
    
    rho = np.array([
        (E1[0]**3 * W1 + E2[0]**3 * W2) ** (1/3),
        (E1[1]**3 * W1 + E2[1]**3 * W2) ** (1/3),
        (E1[2]**3 * W1 + E2[2]**3 * W2) ** (1/3)
    ])

    rho_V = rho[0] * rho[1] * rho[2]
    S = (V_merged / (rho_V + 1e-6)) ** (1/3)

    E_merged = rho * S
    
    # d. 方向 R 的加权平均
    R_merged = weighted_average_rotation(R1, V1, R2, V2)

    # e. 合并 Caption 列表
    caption_merged = obj1.get('caption', []) + obj2.get('caption', [])
    
    # --- 3. 构造并返回新的融合字典 (OBB 格式) ---
    if mode == 'loose':
        return {
            'caption': caption_merged,
            'center': C_merged,
            'extent': E_merged,
            'volume': V_merged,
            'orientation': R_merged,
        }, center_dist
    elif mode == 'strict':
        return {
            'caption': caption_merged,
            'center': C_merged,
            'extent': E_merged,
            'volume': V_merged,
            'orientation': R_merged,
        }, center_dist



def add_obb_to_pointcloud(pcd_ply_data, obb_list):
    """
    将OBB框添加到点云并保存为PLY文件
    
    Args:
        pcd_ply_data: PlyData对象 (从self.recent_pcd获取)
        obb_list: OBB列表，每个OBB包含:
            - center: 中心点 [x, y, z]
            - extent: 半长 [dx, dy, dz] 
            - orientation: 方向矩阵 [3x3]
        output_path: 输出PLY文件路径
    """
    # 从PlyData提取点云数据
    vertex_element = pcd_ply_data['vertex']
    pcd_x, pcd_y, pcd_z = vertex_element['x'], vertex_element['y'], vertex_element['z']
    pcd_red, pcd_green, pcd_blue = vertex_element['red'], vertex_element['green'], vertex_element['blue']
    num_pcd_points = len(pcd_x)
    
    all_box_vertices_coords = []
    all_box_vertices_colors = []
    all_box_edges = []
    current_vertex_count = num_pcd_points
    
    # OBB 框颜色：改为显眼的亮绿色
    obb_color = [0, 255, 0]
    
    # 为每个OBB生成顶点和边
    for obb in obb_list:
        center = obb['center']
        extent = obb['extent']
        orientation = obb['orientation']
        
        # 生成 OBB 的 8 个顶点
        vertices = _compute_obb_vertices(center, extent, orientation)

        # 为了让框在点云中更“粗”，我们在每个角点附近复制一些轻微偏移的点，
        # 形成一小团绿色点云，而不是单一细线。
        jitter_scale = np.linalg.norm(extent) * 0.02 + 1e-4  # 相对尺寸的小抖动
        num_jitter = 3
        jittered_vertices = [vertices]
        rng = np.random.default_rng(seed=42)
        for _ in range(num_jitter):
            jitter = (rng.standard_normal(vertices.shape) * jitter_scale).astype(np.float32)
            jittered_vertices.append(vertices + jitter)
        vertices_thick = np.concatenate(jittered_vertices, axis=0)

        # 顶点颜色（所有复制点同色）
        vertex_colors = np.tile(obb_color, (vertices_thick.shape[0], 1))

        # OBB 的 12 条边（索引仍基于原始 8 个角点）
        edges_indices = [
            [0,1], [1,2], [2,3], [3,0],  # 底面
            [4,5], [5,6], [6,7], [7,4],  # 顶面  
            [0,4], [1,5], [2,6], [3,7]   # 侧面
        ]
        
        # 调整边索引
        adjusted_edges = []
        for edge in edges_indices:
            adjusted_edges.append((current_vertex_count + edge[0], 
                                 current_vertex_count + edge[1]))
        
        all_box_vertices_coords.append(vertices_thick)
        all_box_vertices_colors.append(vertex_colors)
        all_box_edges.extend(adjusted_edges)
        current_vertex_count += 8

    # 合并点云和OBB数据
    pcd_coords = np.column_stack((pcd_x, pcd_y, pcd_z))
    pcd_colors = np.column_stack((pcd_red, pcd_green, pcd_blue))

    if all_box_vertices_coords:
        box_coords = np.concatenate(all_box_vertices_coords)
        box_colors = np.concatenate(all_box_vertices_colors)
        final_coords = np.concatenate([pcd_coords, box_coords])
        final_colors = np.concatenate([pcd_colors, box_colors])
        
        # 创建边数据
        edge_dtype = [('vertex1', 'i4'), ('vertex2', 'i4')]
        final_edges = np.array(all_box_edges, dtype=edge_dtype)
    else:
        final_coords = pcd_coords
        final_colors = pcd_colors
        final_edges = np.empty(0, dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])

    # 创建顶点数据
    final_vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    final_vertices_data = np.empty(len(final_coords), dtype=final_vertex_dtype)
    final_vertices_data['x'] = final_coords[:, 0].astype(np.float32)
    final_vertices_data['y'] = final_coords[:, 1].astype(np.float32)
    final_vertices_data['z'] = final_coords[:, 2].astype(np.float32)
    final_vertices_data['red'] = final_colors[:, 0].astype(np.uint8)
    final_vertices_data['green'] = final_colors[:, 1].astype(np.uint8)
    final_vertices_data['blue'] = final_colors[:, 2].astype(np.uint8)

    # 创建PLY元素
    ply_elements = [PlyElement.describe(final_vertices_data, name='vertex')] 
    if final_edges.size > 0:
        ply_elements.append(PlyElement.describe(final_edges, name='edge'))

    # 保存PLY文件
    pcd_with_obb = PlyData(ply_elements, text=False)
    return pcd_with_obb

def _compute_obb_vertices(center, extent, orientation):
    """
    计算OBB的8个顶点坐标
    """
    # 局部坐标系的8个角点
    local_corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # 底面
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]      # 顶面
    ], dtype=np.float32)
    
    # 缩放局部角点
    local_corners = local_corners * extent
    
    # 旋转并平移到世界坐标系
    vertices = (orientation @ local_corners.T).T + center
    
    return vertices



def draw_bev_circle(bev,pixel_coords, radius = 3, color = (0, 0, 255), fill = -1):
    """
    统一在 BEV 图上绘制圆点的方法。

    Args:
        pixel_coords (tuple): 像素坐标 (x_pixel, z_pixel)。
        radius (int): 圆点的半径。
        color (tuple): BGR 颜色值 (B, G, R)。
        fill (int): 绘制类型，-1 表示填充，>0 表示边框厚度。
    """
    cv2.circle(bev, pixel_coords, radius, color, fill)

def draw_bev_pose_arrow(bev, 
                            pose, 
                            arrow_length=40, 
                            color=(0, 0, 255), # 默认蓝色 (B, G, R)
                            thickness=1, 
                            tip_length=0.3,
                            theta_is_degrees=False):
        """
        统一在 BEV 图上绘制位姿箭头的方法。

        Args:
            pixel_coords (tuple): 像素坐标 (x_pixel, z_pixel)，即箭头起点。
            theta (float): 车辆的朝向角。默认假设为弧度制，且 0 弧度朝向右 (正 X)。
            arrow_length (int): 箭头的像素长度。
            color (tuple): BGR 颜色值 (B, G, R)。
            thickness (int): 箭头的粗细。
            tip_length (float): 箭头尖端的长度比例 (0.0 到 1.0)。
            theta_is_degrees (bool): 如果为 True，则输入的 theta 是角度制，需要转换。
        """
        x_p, z_p, theta = pose
        
        # 1. 角度处理：确保使用弧度
        if theta_is_degrees:
            theta_rad = math.radians(theta)
        else:
            theta_rad = theta

        # 2. 计算方向向量 (cos(theta), sin(theta))
        # 假设：0 弧度朝向右 (X)，正 Z 轴（图像的 Y 轴）向下。
        # 图像坐标系中，x = 列，z = 行。
        # cos(theta) 对应 x 轴分量
        # sin(theta) 对应 z 轴分量
        
        # 注意：在很多 BEV 视图中，Z 轴（向前）可能对应图像的 Y 轴（向下）。
        # 如果您的 theta = 0 是朝向前方的（-Z轴，即图像Y轴向上），则需要调整三角函数。
        # **根据您的示例代码：**
        # 原始代码使用了 pos[0] (cos) 对应 x 轴，pos[1] (sin) 对应 z 轴。
        # end_z = int(z_p + pos[1] * arrow_length) 表明：正弦分量与 z 轴（向下）正相关。
        # 我们沿用这个约定：
        dir_x = math.cos(theta_rad) # X 分量
        dir_z = math.sin(theta_rad) # Z (图像Y) 分量

        # 3. 计算箭头的末端点
        end_x = int(x_p + dir_x * arrow_length)
        end_z = int(z_p + dir_z * arrow_length) # Z 轴向下为正，所以用加法

        # 4. 绘制箭头
        cv2.arrowedLine(
            bev, 
            (x_p, z_p),         # 起点
            (end_x, end_z),     # 终点
            color,              # 颜色
            thickness,          # 粗细
            cv2.LINE_AA,        # 抗锯齿
            tipLength=tip_length # 箭头尖端长度
        )

def save_map(bev = None, filename="bev_map.png"):
    """
    简单保存当前的 BEV 图像。
    """
    # 优先使用传入的 bev 图像，如果未传入则使用实例的 self.bev
    image_to_save = bev
    
    if image_to_save is None:
        print("Error: No BEV map to save.")
        return

    # 旋转180度
    rotated_bev = cv2.rotate(image_to_save, cv2.ROTATE_180)

    # 保存镜像对称后的 BEV 图像
    cv2.imwrite(filename, image_to_save)

def save_mirrored_map(bev = None , flipcode = 1, filename="bev_map_mirrored.png"):
    """
    简单保存当前的 BEV 图像，保存前对其进行水平镜像对称操作。
    """
    # 优先使用传入的 bev 图像，如果未传入则使用实例的 self.bev
    image_to_save = bev
    
    if image_to_save is None:
        print("Error: No BEV map to save.")
        return


    # 对图像进行镜像对称操作
    mirrored_bev = cv2.flip(image_to_save, flipcode)

    # 旋转180度
    rotated_bev = cv2.rotate(mirrored_bev, cv2.ROTATE_180)

    # 保存镜像对称后的 BEV 图像
    cv2.imwrite(filename, rotated_bev)

def draw_temp_point_and_save(bev, bev_pixel_coords: tuple, filename: str = "bev_map_temp.png", 
                                color: tuple = (0, 255, 255), radius: int = 3):
        """
        在一个临时的 BEV 副本上绘制一个点，然后将该副本保存到文件。
        原始的 self.bev 不会被修改，实现“临时”画点的效果。

        Args:
            bev_pixel_coords (tuple): BEV地图上的像素坐标 (x_pixel, z_pixel)。
            filename (str): 临时保存的文件名。
            color (tuple): BGR 颜色值 (B, G, R)。默认为黄色 (0, 255, 255)。
            radius (int): 绘制圆点的半径。
        """
        # 1. 创建 BEV 图像的副本 (核心步骤：实现临时效果)
        temp_bev = bev.copy()
        
        # 2. 在副本上绘制点
        cv2.circle(temp_bev, bev_pixel_coords, radius, color, -1) # -1 表示填充圆

        # 3.保存镜像版
        save_mirrored_map(flipcode=1,filename=filename, bev=temp_bev)


def plot_navigation_status(agent_input, current_pose_sim, history_poses, shorterm_goal):
    """
    在左右两个子图上绘制导航状态和当前观察结果，并在图表下方打印当前姿态文字。

    Args:
    """
    current_obs = agent_input['obs']
    current_pose_bev = agent_input['pose']
    midterm_goal_bev = agent_input['midterm_goal']
    bev_map = agent_input['bev']

    arrow_len = 15 
    # 1. 创建包含两个子图（1行2列）的 Figure 对象
    fig, axes = plt.subplots(1, 2, figsize=(14, 7)) 
    
    # ----------------------------------------------------
    # 左图：BEV 地图和导航状态 (axes[0])
    # ----------------------------------------------------
    ax_map = axes[0]
    
    bev_map_transposed = np.transpose(bev_map, (1, 0, 2))
    ax_map.imshow(bev_map_transposed, origin='lower')
    if midterm_goal_bev is not None:
        goal_x, goal_z = midterm_goal_bev[0], midterm_goal_bev[1]
        ax_map.plot(goal_x, goal_z, 'o', color='red', markersize=5, label='Midterm Goal')
        if len(midterm_goal_bev) ==3:
            yaw = midterm_goal_bev[2]
            dx_goal = arrow_len*np.cos(yaw)
            dz_goal = arrow_len*np.sin(yaw)
            ax_map.arrow(goal_x, goal_z, dx_goal, dz_goal, head_width=5, head_length=5, fc='red', ec='red')
    if shorterm_goal is not None:
        shorterm_goal_x, shorterm_goal_z = shorterm_goal[0], shorterm_goal[1]
        ax_map.plot(shorterm_goal_x, shorterm_goal_z, 'P', color='purple', markersize=5, label='Shorterm Goal')
    if history_poses is not None:
        hist_x = [p[0] for p in history_poses]
        hist_z = [p[1] for p in history_poses]
        ax_map.plot(hist_x, hist_z, ':', color='blue', linewidth=1, label='History Path')
    curr_x, curr_z, curr_yaw = current_pose_bev[0], current_pose_bev[1], current_pose_bev[2]
    ax_map.plot(curr_x, curr_z, 'o', color='lime', markersize=8, label='Current Pose')
    dx = arrow_len * np.cos(curr_yaw) 
    dz = arrow_len * np.sin(curr_yaw)
    ax_map.arrow(curr_x, curr_z, dx, dz, head_width=5, head_length=5, fc='lime', ec='lime')
    ax_map.set_title(f"BEV Map & Navigation (Step: {len(history_poses)})")
    ax_map.set_xticks([]) 
    ax_map.set_yticks([])
    ax_map.legend()

    # ----------------------------------------------------
    # 右图：当前观察结果 (axes[1])
    # ----------------------------------------------------
    ax_obs = axes[1]
    if len(current_obs) > 1:
        images = current_obs[:-1] # N-1 张图片
        is_single_image = False
    else:
        images = current_obs[-1] # 1 张图片
        is_single_image = True
        
    ax_obs.set_title("Current Observation (Agent View)")
    ax_obs.set_xticks([]) 
    ax_obs.set_yticks([])

    # --- 补充部分开始 ---
    
    # 假设图片数据是 NumPy 数组或 PIL Image 对象，并且可以使用 ax.imshow()
    
    if is_single_image:
        # **情况二：一张图片 (len(current_obs) == 1)**
        # 直接在 ax_obs 上显示唯一的图片
        ax_obs.imshow(images)
        
    elif len(images) == 4:
        # **情况一：四张图片 (len(current_obs) == 5)**
        
        # 隐藏 ax_obs 本身，因为我们将使用一个 2x2 的 Gridspec 覆盖它
        ax_obs.axis('off')
        
        # 创建一个 2x2 的子图网格，并将其放置在 ax_obs 的位置
        # 'gs' 是 grid spec，它允许我们在一个更大的子图区域内定义子子图
        from matplotlib.gridspec import GridSpec
        
        # 获取 ax_obs 在整个 figure 中的范围
        bbox = ax_obs.get_position()
        
        # 在 ax_obs 的位置上创建 GridSpec
        gs = GridSpec(2, 2, 
                      left=bbox.x0, right=bbox.x1, 
                      bottom=bbox.y0, top=bbox.y1,
                      hspace=0, wspace=0) # 调整间距

        for i in range(4):
            # 在 2x2 网格的相应位置创建新的子图
            # fig 是整个图形对象，可以通过 axes[0].figure 获取
            ax_sub = ax_obs.figure.add_subplot(gs[i]) 
            
            # 显示图片
            ax_sub.imshow(images[i])
            
            # 清除子图的刻度
            ax_sub.set_xticks([]) 
            ax_sub.set_yticks([])
            
    else:
        # 如果不是 1 张或 4 张，可以选择显示列表中的第一张图片作为默认处理
        print(f"Warning: Unexpected number of images ({len(images)}). Showing the first one.")
        ax_obs.imshow(images[0])

    plt.tight_layout() 

    # 3. 显示图表
    display(fig)
    
    # 4. 打印文字信息
    # print(pose_text)

    # 2. 清除上一次的图和文字输出
    clear_output(wait=True) 
    time.sleep(0.1)
    
    plt.close(fig) # 关闭图表对象
