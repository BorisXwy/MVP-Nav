import numpy as np
import skfmm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq
import time # 导入 time 模块

# =====================================================================
# 核心规划函数：get_local_goal_fmm (已修改，将可视化提前)
# =====================================================================

# def get_local_goal_fmm(bev_map, max_limit, start, goal, step_size, visualize, global_path, save_path):

#     def compute_fmm_field(cost_map, goal_pos_local):
#         H, W = cost_map.shape
#         goal_x, goal_y = int(goal_pos_local[0]), int(goal_pos_local[1])
        
#         cost_obstacle = cost_map.max()
#         obstacle_mask = (cost_map == cost_obstacle) 

#         # 1. 设置边界条件 (phi)：
#         phi = np.ones(cost_map.shape)
#         phi[goal_y, goal_x] = -1 
        
#         # 3. FMM 的速度场 (Speed Map)
#         speed = 1.0 / cost_map
#         speed[obstacle_mask] = 1e-10 
        
#         try:
#             T_field = skfmm.travel_time(phi, speed, dx=1.0) 
#         except ValueError as e:
#             T_field = np.full(cost_map.shape, np.inf, dtype=np.float32)
        
#         return T_field.astype(np.float32)
    
#     start_x, start_y = int(start[0]), int(start[1])
#     goal_x_global, goal_y_global = int(goal[0]), int(goal[1])
        
#     H_global, W_global = bev_map.shape[:2]

#     min_side_half = min(H_global, W_global) // 2

#     window_size = min(min_side_half, max_limit)

#     if window_size % 2 != 0:
#         window_size += 1
#     half_w = window_size // 2
    
#     # --- 1. 地图裁剪 ---
#     y_min_global = max(0, start_y - half_w)
#     y_max_global = min(H_global, start_y + half_w)
#     x_min_global = max(0, start_x - half_w)
#     x_max_global = min(W_global, start_x + half_w)
    
#     cropped_bev = bev_map[y_min_global:y_max_global, x_min_global:x_max_global].copy()
#     start_x_local = start_x - x_min_global
#     start_y_local = start_y - y_min_global

    
#     H_local, W_local = cropped_bev.shape[:2]

#     # --- 2. 目标投影 ---
#     # --- 2/3. 目标投影 (基于 Global Path) --- (移除障碍物回退)
#     # ======================================================
    
#     local_goal_global = (goal_x_global, goal_y_global) # 默认：全局终点
#     path_found_in_window = False
    
#     if global_path and len(global_path) > 1:
        
#         # 路径顺序：[Goal, ..., Start]。目标：找到从 Goal 向 Start 方向搜索时，第一个进入窗口的点。
        
#         for px, py in global_path:
            
#             is_in_window = (x_min_global <= px < x_max_global and 
#                             y_min_global <= py < y_max_global)
            
#             if is_in_window:
#                 # 找到了第一个进入窗口的点。
#                 local_goal_global = (px, py)
#                 path_found_in_window = True
#                 break
#         else:
#             pass # 路径都在窗口外或整个路径都在窗口内

#     if not path_found_in_window:
#         # 如果 global_path 无效/为空，回退到原有的直线投影逻辑
#         dx_g = float(goal_x_global) - float(start_x)
#         dy_g = float(goal_y_global) - float(start_y)
#         R_window = np.sqrt(W_local**2 + H_local**2) / 2.0
#         dist_g = np.sqrt(dx_g**2 + dy_g**2)
#         ratio = min(R_window, dist_g) / dist_g if dist_g > 0 else 0
#         px_global = start_x + dx_g * ratio
#         py_global = start_y + dy_g * ratio
#         local_goal_global = (int(round(px_global)), int(round(py_global)))


#     # --- 转换为局部坐标 ---
#     goal_x_local_pre = local_goal_global[0] - x_min_global
#     goal_y_local_pre = local_goal_global[1] - y_min_global
    
#     # 确保目标点在局部地图内
#     goal_x_local = np.clip(goal_x_local_pre, 0, W_local - 1)
#     goal_y_local = np.clip(goal_y_local_pre, 0, H_local - 1)
    
#     goal_pos_local = (goal_x_local, goal_y_local) # <---
#     # --- 3. 成本图转换 ---
#     cost_traversible = 1.0
#     cost_obstacle = 100.0

#     if cropped_bev.ndim == 3:
#         channel_0 = cropped_bev[:, :, 0]
#     else:
#         channel_0 = cropped_bev
#     traversible_mask = (channel_0 == 200) | (channel_0 == 255)
#     cost_map_local = np.full(channel_0.shape, cost_obstacle, dtype=np.float32)
#     cost_map_local[traversible_mask] = cost_traversible
#     cost_map_local[traversible_mask] += 1e-6

#     # --- 4. 计算 FMM 场 ---
#     T_field_local = compute_fmm_field(cost_map_local, goal_pos_local)

#     # --- 5. 规划短期目标前的检查 ---
#     cy, cx = start_y_local, start_x_local 
#     current_cost = T_field_local[cy, cx]

#     # 初始化规划数据 (包含 FMM 结果，用于失败时可视化)
#     grad_y, grad_x = np.gradient(T_field_local)
#     planning_data = {
#         "cropped_bev": cropped_bev, 
#         "cost_map_local": cost_map_local, 
#         "T_field_local": T_field_local, 
#         "grad_x_local": grad_x,
#         "grad_y_local": grad_y,
#         "start_local": (start_x_local, start_y_local),
#         "goal_local": goal_pos_local,
#         "stg_local": (start_x_local, start_y_local), # 默认STG为起点
#         "local_window_bounds_global": (x_min_global, y_min_global, x_max_global, y_max_global)
#     }
    
#     stg_x_local, stg_y_local = cx, cy
    
#     if current_cost == np.inf:
#         # --- 可视化 (规划失败时) ---
#         if visualize:
#             visualize_fmm_planning_internal(bev_map, start, goal, start, planning_data,save_path)
#         return start
        
#     if not (0 < cx < W_local - 1 and 0 < cy < H_local - 1):
#         # --- 可视化 (规划失败时) ---
#         if visualize:
#             visualize_fmm_planning_internal(bev_map, start, goal, start, planning_data, save_path)
#         return start

#     # --- 6. 规划短期目标 (STG) ---
#     gx = grad_x[cy, cx]
#     gy = grad_y[cy, cx]
#     grad_norm = np.sqrt(gx**2 + gy**2)

#     if grad_norm == 0.0 or current_cost <= 0.0:
#         stg_x_local, stg_y_local = cx, cy
#     else:

#         direction_x = -gx / grad_norm 
#         direction_y = -gy / grad_norm 
        
#         stg_x_local_float = float(cx) + direction_x * step_size
#         stg_y_local_float = float(cy) + direction_y * step_size
        
#         stg_x_local = int(round(stg_x_local_float))
#         stg_y_local = int(round(stg_y_local_float))
        
#         stg_x_local = np.clip(stg_x_local, 0, W_local - 1)
#         stg_y_local = np.clip(stg_y_local, 0, H_local - 1)

#     # --- 7. 将短期目标点转换回全局坐标 ---
#     stg_x_global = stg_x_local + x_min_global
#     stg_y_global = stg_y_local + y_min_global
    
#     # 更新 planning_data 中的 STG (用于成功规划的可视化)
#     planning_data["stg_local"] = (stg_x_local, stg_y_local)

#     # --- 可视化阶段 (成功规划时) ---
#     if visualize:
#         visualize_fmm_planning_internal(bev_map, start, goal, (stg_x_global, stg_y_global), planning_data, save_path)

#     return (stg_x_global, stg_y_global)

def get_local_goal_fmm(bev_map, max_limit, start, goal, step_size, visualize, global_path, save_path):
    """
    基于 FMM 计算局部短期目标点 (STG)。
    
    参数:
        bev_map: 鸟瞰图 (BEV map)，numpy 数组，形状为 (H, W) 或 (H, W, C)。
        max_limit: 局部窗口的最大尺寸限制。
        start: 当前位置的全局坐标 (x, y)。
        goal: 全局终点的全局坐标 (x, y)。
        step_size: FMM 梯度下降的步长，用于确定 STG 的距离。
        visualize: 是否进行可视化。
        global_path: 全局路径，从全局终点到起点（或附近）的坐标列表 [(x, y), ...]。
        save_path: 可视化保存路径。
        
    返回值:
        短期目标点的全局坐标 (stg_x_global, stg_y_global)，或在规划失败时返回起点坐标。
    """

    def compute_fmm_field(cost_map, goal_pos_local):
        """计算 FMM Travel Time 场。"""
        try:
            H, W = cost_map.shape
            # 确保目标点在成本图范围内
            goal_x, goal_y = int(goal_pos_local[0]), int(goal_pos_local[1])
            if not (0 <= goal_y < H and 0 <= goal_x < W):
                # logging.warning(f"目标点 {goal_pos_local} 超出局部地图范围，返回无穷大的 T_field。")
                return np.full(cost_map.shape, np.inf, dtype=np.float32)

            cost_obstacle = cost_map.max() # 假设成本图中的最大值代表障碍物
            obstacle_mask = (cost_map == cost_obstacle) 

            # 1. 设置边界条件 (phi)：
            phi = np.ones(cost_map.shape, dtype=np.float32)
            phi[goal_y, goal_x] = -1.0 # 目标点为负值
            
            # 3. FMM 的速度场 (Speed Map)：V = 1 / Cost
            # 避免除以零：在计算速度之前，需要确保 cost_map 中的所有值都 > 0。
            # 在主函数中已经通过 `traversible_mask` 处理，这里再次检查以防万一。
            if np.any(cost_map <= 0):
                 # logging.warning("成本图中包含非正值，可能导致除以零。将非正值成本设为很高的值。")
                 cost_map[cost_map <= 0] = cost_obstacle 

            speed = 1.0 / cost_map
            # 对于障碍物，速度设为极小值（但不能为零，否则 FMM 可能陷入僵局）
            # 极小值设置为 1e-10 避免除以零错误
            speed[obstacle_mask] = 1e-10 
            
            # 使用 skfmm 计算 Travel Time 场
            T_field = skfmm.travel_time(phi, speed, dx=1.0) 
            
            return T_field.astype(np.float32)
            
        except ValueError as e:
            # skfmm 内部可能因为输入数据问题（如 NaN/Inf）抛出 ValueError
            # logging.error(f"skfmm.travel_time 计算失败: {e}")
            return np.full(cost_map.shape, np.inf, dtype=np.float32)
        except Exception as e:
            # 捕获其他可能的异常 (如内存错误、维度不匹配等)
            # logging.error(f"compute_fmm_field 发生未知错误: {e}")
            return np.full(cost_map.shape, np.inf, dtype=np.float32)

    # --- 0. 输入参数检查 ---
    # 确保 bev_map 是一个有效的 numpy 数组
    if not isinstance(bev_map, np.ndarray) or bev_map.ndim < 2:
        # logging.error("输入 bev_map 无效或维度不足。")
        return start
    
    H_global, W_global = bev_map.shape[:2]

    try:
        start_x, start_y = int(start[0]), int(start[1])
        goal_x_global, goal_y_global = int(goal[0]), int(goal[1])
    except (TypeError, IndexError):
        # logging.error(f"起点或终点坐标格式错误: start={start}, goal={goal}")
        return start
    
    # 确保起点在全局地图内
    if not (0 <= start_y < H_global and 0 <= start_x < W_global):
        # logging.warning(f"起点 {start} 超出全局地图范围 {H_global}x{W_global}。")
        return start

    # --- 1. 地图裁剪 ---
    try:
        min_side_half = min(H_global, W_global) // 2
        window_size = min(min_side_half, max_limit)
        # 确保窗口大小至少为 2，并且是偶数，以保证半宽计算正确
        if window_size < 2:
            window_size = 2
        if window_size % 2 != 0:
            window_size += 1
        half_w = window_size // 2
        
        y_min_global = max(0, start_y - half_w)
        y_max_global = min(H_global, start_y + half_w)
        x_min_global = max(0, start_x - half_w)
        x_max_global = min(W_global, start_x + half_w)
        
        # 裁剪操作
        cropped_bev = bev_map[y_min_global:y_max_global, x_min_global:x_max_global].copy()
        
        # 裁剪后的地图如果为空，则无法规划
        if cropped_bev.size == 0:
             # logging.warning(f"局部地图裁剪结果为空，坐标范围: x={x_min_global}:{x_max_global}, y={y_min_global}:{y_max_global}。")
             return start
             
        start_x_local = start_x - x_min_global
        start_y_local = start_y - y_min_global

        H_local, W_local = cropped_bev.shape[:2]
        
    except Exception as e:
        # logging.error(f"地图裁剪步骤发生错误: {e}")
        return start

    # --- 2. 目标投影 (基于 Global Path 或直线) ---
    local_goal_global = (goal_x_global, goal_y_global) # 默认：全局终点
    path_found_in_window = False
    
    # 检查 global_path 是否有效
    if global_path and isinstance(global_path, (list, tuple)) and len(global_path) > 1:
        try:
            for px, py in global_path:
                px, py = int(px), int(py)
                is_in_window = (x_min_global <= px < x_max_global and 
                                y_min_global <= py < y_max_global)
                
                if is_in_window:
                    local_goal_global = (px, py)
                    path_found_in_window = True
                    break
        except Exception as e:
            # logging.warning(f"处理 global_path 时发生错误，回退到直线投影: {e}")
            pass # 发生错误则回退到直线投影逻辑

    if not path_found_in_window:
        # 直线投影逻辑（如果 global_path 无效或路径都在窗口外）
        try:
            dx_g = float(goal_x_global) - float(start_x)
            dy_g = float(goal_y_global) - float(start_y)
            R_window = np.sqrt(W_local**2 + H_local**2) / 2.0
            dist_g = np.sqrt(dx_g**2 + dy_g**2)
            
            # 避免除以零
            if dist_g > 1e-6:
                ratio = min(R_window, dist_g) / dist_g
            else:
                ratio = 0
            
            px_global = start_x + dx_g * ratio
            py_global = start_y + dy_g * ratio
            local_goal_global = (int(round(px_global)), int(round(py_global)))
            
        except Exception as e:
            # logging.error(f"直线投影计算发生错误: {e}")
            # 如果投影计算失败，则将局部目标点设置为全局目标点（如果它在窗口内的话）
            # 或者直接返回起点，避免在后续计算中出错
            local_goal_global = (goal_x_global, goal_y_global)

    # --- 转换为局部坐标 ---
    goal_x_local_pre = local_goal_global[0] - x_min_global
    goal_y_local_pre = local_goal_global[1] - y_min_global
    
    # 确保目标点在局部地图内 (裁剪)
    goal_x_local = np.clip(goal_x_local_pre, 0, W_local - 1)
    goal_y_local = np.clip(goal_y_local_pre, 0, H_local - 1)
    
    goal_pos_local = (goal_x_local, goal_y_local)
    
    # --- 3. 成本图转换 ---
    try:
        cost_traversible = 1.0
        cost_obstacle = 100.0

        if cropped_bev.ndim == 3:
            channel_0 = cropped_bev[:, :, 0]
        else:
            channel_0 = cropped_bev
            
        # 确保数据类型兼容，如果不是整数类型，可能需要转换
        channel_0 = channel_0.astype(np.int32)
        
        # 检查可通行区域的颜色值 (200, 255)
        traversible_mask = (channel_0 == 200) | (channel_0 == 255)
        
        cost_map_local = np.full(channel_0.shape, cost_obstacle, dtype=np.float32)
        cost_map_local[traversible_mask] = cost_traversible
        # 加上一个极小值确保所有可通行区域的成本都 > 0
        cost_map_local[traversible_mask] += 1e-6
        
    except Exception as e:
        # logging.error(f"成本图转换步骤发生错误: {e}")
        return start

    # --- 4. 计算 FMM 场 ---
    T_field_local = compute_fmm_field(cost_map_local, goal_pos_local)

    # --- 5. 规划短期目标前的检查与数据准备 ---
    try:
        cy, cx = start_y_local, start_x_local 
        
        # 确保起点在局部地图内
        if not (0 <= cx < W_local and 0 <= cy < H_local):
            # logging.warning(f"起点 {cx, cy} 不在局部地图内 {W_local, H_local}。")
            return start

        current_cost = T_field_local[cy, cx]
        
        # 梯度计算
        grad_y, grad_x = np.gradient(T_field_local)
        
        # 初始化规划数据
        planning_data = {
            "cropped_bev": cropped_bev, 
            "cost_map_local": cost_map_local, 
            "T_field_local": T_field_local, 
            "grad_x_local": grad_x,
            "grad_y_local": grad_y,
            "start_local": (start_x_local, start_y_local),
            "goal_local": goal_pos_local,
            "stg_local": (start_x_local, start_y_local), # 默认STG为起点
            "local_window_bounds_global": (x_min_global, y_min_global, x_max_global, y_max_global)
        }
        
    except Exception as e:
        # logging.error(f"FMM 结果处理或梯度计算发生错误: {e}")
        return start
        
    stg_x_local, stg_y_local = cx, cy # 默认 STG 为当前位置
    
    # 检查 FMM 结果，如果当前位置不可达 (np.inf)
    if current_cost == np.inf:
        # logging.warning(f"FMM 场值在起点处为无穷大 ({current_cost})，目标点不可达。")
        if visualize:
            # 假设 visualize_fmm_planning_internal 存在
            # 传入的 STG 仍为起点 (start)
            visualize_fmm_planning_internal(bev_map, start, goal, start, planning_data, save_path)
        return start

    # --- 6. 规划短期目标 (STG) ---
    try:
        # 从 FMM 场梯度中提取当前位置的梯度
        gx = grad_x[cy, cx]
        gy = grad_y[cy, cx]
        grad_norm = np.sqrt(gx**2 + gy**2)

        if grad_norm == 0.0 or current_cost <= 0.0:
            # 梯度为零或成本为零（可能已经在目标点）
            stg_x_local, stg_y_local = cx, cy
        else:
            # 沿着负梯度方向（最陡峭下降方向）移动
            direction_x = -gx / grad_norm 
            direction_y = -gy / grad_norm 
            
            # 计算新的 STG 坐标（浮点）
            stg_x_local_float = float(cx) + direction_x * step_size
            stg_y_local_float = float(cy) + direction_y * step_size
            
            # 四舍五入到最近的整数像素坐标
            stg_x_local = int(round(stg_x_local_float))
            stg_y_local = int(round(stg_y_local_float))
            
            # 裁剪 STG 坐标，确保其仍在局部地图内
            stg_x_local = np.clip(stg_x_local, 0, W_local - 1)
            stg_y_local = np.clip(stg_y_local, 0, H_local - 1)

    except Exception as e:
        # logging.error(f"短期目标 (STG) 计算步骤发生错误: {e}")
        # 失败时返回当前位置作为 STG
        stg_x_local, stg_y_local = cx, cy


    # --- 7. 将短期目标点转换回全局坐标 ---
    stg_x_global = stg_x_local + x_min_global
    stg_y_global = stg_y_local + y_min_global
    
    # 更新 planning_data 中的 STG
    planning_data["stg_local"] = (stg_x_local, stg_y_local)

    # --- 可视化阶段 (成功规划时) ---
    if visualize:
        # 假设 visualize_fmm_planning_internal 存在
        visualize_fmm_planning_internal(bev_map, start, goal, (stg_x_global, stg_y_global), planning_data, save_path)

    return (stg_x_global, stg_y_global)

# =====================================================================
# 辅助函数：创建默认规划数据 (用于提前返回时)
# =====================================================================

def create_default_planning_data(cropped_bev, start_x_local, start_y_local, bounds):
    H_local, W_local = cropped_bev.shape[:2]
    cost_map_local = np.ones((H_local, W_local), dtype=np.float32)
    T_field_local = np.full((H_local, W_local), np.inf, dtype=np.float32)
    grad_zeros = np.zeros((H_local, W_local), dtype=np.float32)
    
    return {
        "cropped_bev": cropped_bev, 
        "cost_map_local": cost_map_local, 
        "T_field_local": T_field_local, 
        "grad_x_local": grad_zeros,
        "grad_y_local": grad_zeros,
        "start_local": (start_x_local, start_y_local),
        "goal_local": (start_x_local, start_y_local), # 默认目标点
        "stg_local": (start_x_local, start_y_local), # 默认 STG 为起点
        "local_window_bounds_global": bounds
    }

# =====================================================================
# 内部可视化函数 (无需修改，与原代码一致)
# =====================================================================

def visualize_fmm_planning_internal(bev_map_orig, start_global, global_goal_global, local_goal_global, planning_data, save_path):
    # 此函数逻辑保持不变，确保了在接收到 planning_data 后能正常绘图
    # ... (使用原代码中的可视化函数内容)
    cropped_bev = planning_data["cropped_bev"]
    cost_map_local = planning_data["cost_map_local"]
    T_field_local = planning_data["T_field_local"]
    grad_x_local = planning_data["grad_x_local"]
    grad_y_local = planning_data["grad_y_local"]
    start_local = planning_data["start_local"]
    goal_local = planning_data["goal_local"]
    stg_local = planning_data["stg_local"]
    local_window_bounds_global = planning_data["local_window_bounds_global"]
    
    x_min_global, y_min_global, x_max_global, y_max_global = local_window_bounds_global

    fig = plt.figure(figsize=(18, 12))
    
    ax0 = fig.add_subplot(2, 3, 1) # 1. 原始 BEV
    ax1 = fig.add_subplot(2, 3, 2) # 2. 裁剪后的 BEV
    ax2 = fig.add_subplot(2, 3, 3) # 3. 成本图
    ax3 = fig.add_subplot(2, 3, 4) # 4. T-Field
    ax4 = fig.add_subplot(2, 3, 5) # 5. 梯度场
    ax5 = fig.add_subplot(2, 3, 6, projection='3d') # 6. 3D FMM
    
    axes = [ax0, ax1, ax2, ax3, ax4, ax5]

    # --- 1. 原始 BEV 地图与全局/局部标记 ---
    ax0.imshow(bev_map_orig[:, :, 0] if bev_map_orig.ndim == 3 else bev_map_orig, origin='lower', cmap='gray', vmin=0, vmax=255)
    ax0.scatter(start_global[0], start_global[1], color='blue', marker='o', s=100, label='Start (Global)')
    ax0.scatter(global_goal_global[0], global_goal_global[1], color='red', marker='x', s=100, label='Global Goal')
    ax0.scatter(local_goal_global[0], local_goal_global[1], color='green', marker='*', s=150, label='Local Goal (STG)')
    rect = plt.Rectangle((x_min_global, y_min_global), x_max_global - x_min_global, y_max_global - y_min_global,
                         linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    ax0.add_patch(rect)
    ax0.set_title('1. Original BEV Map & Planning Elements')
    ax0.legend()
    ax0.set_xlim(0, bev_map_orig.shape[1])
    ax0.set_ylim(0, bev_map_orig.shape[0])


    # --- 2. 裁剪后的 BEV (局部窗口) ---
    ax1.imshow(cropped_bev[:, :, 0] if cropped_bev.ndim == 3 else cropped_bev, origin='lower', cmap='gray', vmin=0, vmax=255)
    ax1.scatter(start_local[0], start_local[1], color='blue', marker='o', s=100, label='Start (Local)')
    ax1.scatter(goal_local[0], goal_local[1], color='red', marker='x', s=100, label='Projected Local Goal')
    ax1.scatter(stg_local[0], stg_local[1], color='green', marker='*', s=150, label='Short-Term Goal (STG)')
    ax1.set_title('2. Cropped Local BEV Map')
    ax1.legend()
    ax1.set_xlim(0, cropped_bev.shape[1])
    ax1.set_ylim(0, cropped_bev.shape[0])


    # --- 3. 局部成本图 (Cost Map) ---
    ax2.clear()
    cmap_cost = mcolors.LinearSegmentedColormap.from_list("mycmap", ["green", "yellow", "red", "darkred"])
    im_cost = ax2.imshow(cost_map_local, origin='lower', cmap=cmap_cost)
    fig.colorbar(im_cost, ax=ax2, label='Cost')
    ax2.scatter(start_local[0], start_local[1], color='blue', marker='o', s=100, label='Start (Local)')
    ax2.scatter(goal_local[0], goal_local[1], color='red', marker='x', s=100, label='Projected Local Goal')
    ax2.set_title('3. Local Cost Map')
    ax2.legend()
    ax2.set_xlim(0, cost_map_local.shape[1])
    ax2.set_ylim(0, cost_map_local.shape[0])


    # --- 4. FMM 成本场 (T-Field) ---
    ax3.clear()
    T_field_vis = T_field_local.copy()
    T_field_vis[T_field_vis == np.inf] = np.nan
    
    im_tfield = ax3.imshow(T_field_vis, origin='lower', cmap='viridis_r', vmin=0, vmax=np.nanmax(T_field_vis))
    fig.colorbar(im_tfield, ax=ax3, label='FMM Cost (T)')
    ax3.scatter(start_local[0], start_local[1], color='blue', marker='o', s=100, label='Start (Local)')
    ax3.scatter(goal_local[0], goal_local[1], color='red', marker='x', s=100, label='Projected Local Goal')
    ax3.set_title('4. FMM Cost Field (T-Field) - Corrected')
    ax3.legend()
    ax3.set_xlim(0, T_field_local.shape[1])
    ax3.set_ylim(0, T_field_local.shape[0])


    # --- 5. 梯度向量场 & STG ---
    ax4.clear()
    ax4.imshow(cost_map_local, origin='lower', cmap=cmap_cost, alpha=0.5)
    
    step_vec = max(1, min(T_field_local.shape) // 10)
    y_coords, x_coords = np.mgrid[0:T_field_local.shape[0]:step_vec, 0:T_field_local.shape[1]:step_vec]
    
    # 关键：绘制负梯度（寻路方向）
    grad_x_sparse = -grad_x_local[y_coords, x_coords]
    grad_y_sparse = -grad_y_local[y_coords, x_coords]
    
    valid_mask = np.isfinite(grad_x_sparse) & np.isfinite(grad_y_sparse)
    grad_x_sparse = grad_x_sparse[valid_mask]
    grad_y_sparse = grad_y_sparse[valid_mask]
    x_coords_valid = x_coords[valid_mask]
    y_coords_valid = y_coords[valid_mask]

    norm_grads = np.sqrt(grad_x_sparse**2 + grad_y_sparse**2)
    valid_norm_mask = norm_grads > 0
    grad_x_sparse[valid_norm_mask] /= norm_grads[valid_norm_mask]
    grad_y_sparse[valid_norm_mask] /= norm_grads[valid_norm_mask]

    ax4.quiver(x_coords_valid, y_coords_valid, grad_x_sparse, grad_y_sparse,
               color='white', scale=20, width=0.005, headwidth=5, headlength=5, headaxislength=4)
    
    ax4.scatter(start_local[0], start_local[1], color='blue', marker='o', s=100, label='Start (Local)')
    ax4.scatter(goal_local[0], goal_local[1], color='red', marker='x', s=100, label='Projected Local Goal')
    ax4.scatter(stg_local[0], stg_local[1], color='green', marker='*', s=150, label='STG (Trace from Start)')
    
    ax4.arrow(start_local[0], start_local[1], 
              (stg_local[0]-start_local[0]), (stg_local[1]-start_local[1]), 
              color='magenta', head_width=2, head_length=2, linewidth=2, label='STG Direction')
    
    ax4.set_title('5. Negative Gradient Vector Field (Path Direction)')
    ax4.legend()
    ax4.set_xlim(0, T_field_local.shape[1])
    ax4.set_ylim(0, T_field_local.shape[0])


    # --- 6. 3D FMM 成本场 (使用 ax5) ---
    ax5.clear() 
    X_local, Y_local = np.meshgrid(np.arange(T_field_local.shape[1]), np.arange(T_field_local.shape[0]))
    
    T_field_3d = T_field_local.copy()
    max_finite_T = np.nanmax(T_field_3d[np.isfinite(T_field_3d)]) if np.any(np.isfinite(T_field_3d)) else 100
    T_field_3d[T_field_3d == np.inf] = max_finite_T * 1.5
    
    surf = ax5.plot_surface(X_local, Y_local, T_field_3d, cmap='viridis_r', edgecolor='none', alpha=0.8)
    ax5.set_title('6. 3D FMM Cost Field (T-Field) - Corrected')
    ax5.set_xlabel('X Local')
    ax5.set_ylabel('Y Local')
    ax5.set_zlabel('Cost (T)')
    fig.colorbar(surf, ax=ax5, label='Cost (T)', shrink=0.5, aspect=5)
    ax5.view_init(elev=30, azim=210)
    
    plt.tight_layout()
    # 确保保存路径存在且可写入
    plt.savefig(save_path)
    plt.close()



def visualize_astar_planning_internal(bev_map_orig, start_global, goal_global, g_costs, path, save_path):
    """
    可视化全局 A* 规划结果。
    """
    H, W = bev_map_orig.shape[:2]
    
    fig = plt.figure(figsize=(15, 6))
    
    # --- 1. G-Cost Field (实际成本场) ---
    ax1 = fig.add_subplot(1, 2, 1) 
    
    # 可视化 G_cost，将 Inf 设为 NaN，以便 plt.imshow 忽略
    g_costs_vis = g_costs.copy()
    g_costs_vis[g_costs_vis == np.inf] = np.nan
    
    im_cost = ax1.imshow(g_costs_vis, origin='lower', cmap='plasma_r', 
                         vmin=0, vmax=np.nanpercentile(g_costs_vis, 95))
    ax1.axis('off')

    # --- 2. 原始 BEV 地图与路径 ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(bev_map_orig[:, :, 0] if bev_map_orig.ndim == 3 else bev_map_orig, 
               origin='lower', cmap='gray', vmin=0, vmax=255)
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax2.plot(path_x, path_y, color='lime', linewidth=3)
    ax2.scatter(start_global[0], start_global[1], color='blue', marker='o', s=150)
    ax2.scatter(goal_global[0], goal_global[1], color='red', marker='x', s=150)
    ax2.set_xlim(0, W)
    ax2.set_ylim(0, H)
    ax2.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# =====================================================================
# 全局 A* 路径规划函数 (已修改：新增可视化参数和逻辑)
# =====================================================================

def compute_astar_path(bev_map, start, goal, visualize=False, save_path='/tmp/astar_path.png', safety_field=None):
    """
    使用 A* 算法计算从起点到终点的最短路径。这是一个全局规划器。
    若提供 safety_field，则作为 costmap：安全场越低（贴墙）的格子代价越高，路径会尽量远离墙。

    输入:
        bev_map (np.ndarray): 鸟瞰图/成本图。假设非 200/255 的值为障碍物。
        start (tuple): 起点全局坐标 (x, y)。
        goal (tuple): 终点全局坐标 (x, y)。
        visualize (bool): 是否启用可视化。
        save_path (str): 可视化图像的保存路径。
        safety_field (np.ndarray, optional): 形状 (W, H) 与 bev_map 的 (H,W) 对应，即 (map_pixels_x, map_pixels_z)。
            值域 [0,1]，越高越安全。用于加权移动代价，避免贴墙。
    输出:
        list of tuples: 路径点的列表 [(x1, y1), (x2, y2), ...]，如果路径不存在则返回空列表。
    """

    # --- 1. 数据准备 ---
    H, W = bev_map.shape[:2]
    
    # 将 start 和 goal 转换为 (y, x) 格式以便于 NumPy 索引
    start_yx = (int(start[1]), int(start[0]))
    goal_yx = (int(goal[1]), int(goal[0]))
    
    # 确定可通行区域 (与 FMM 逻辑一致：200 或 255 可通行)
    if bev_map.ndim == 3:
        cost_channel = bev_map[:, :, 0]
    else:
        cost_channel = bev_map
        
    # True 表示可通行
    is_traversible = (cost_channel == 200) | (cost_channel == 255)

    # 校验 safety_field 形状：(map_pixels_x, map_pixels_z) 即 (W, H)
    use_safety = safety_field is not None and isinstance(safety_field, np.ndarray) and safety_field.shape == (W, H)

    # 检查起点和终点是否可通行且在地图范围内
    if not (0 <= start_yx[0] < H and 0 <= start_yx[1] < W and is_traversible[start_yx]):
        print("A* 规划失败：起点不可通行或超出地图范围。")
        return []
    if not (0 <= goal_yx[0] < H and 0 <= goal_yx[1] < W and is_traversible[goal_yx]):
        print("A* 规划失败：终点不可通行或超出地图范围。")
        return []

    # --- 2. A* 算法初始化 ---
    open_list = [(0.0, start_yx)] 
    
    # 存储 G_cost (从起点到当前点的实际成本)
    g_costs = np.full((H, W), np.inf, dtype=np.float32) # <-- 收集 G_costs 用于可视化
    g_costs[start_yx] = 0.0
    
    came_from = {}
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def heuristic(p1_yx, p2_yx):
            """
            启发式函数：使用欧几里得距离 (Euclidean Distance)。
            在八向移动的网格中，这是最准确且可接受的启发式。
            """
            dy = p1_yx[0] - p2_yx[0]
            dx = p1_yx[1] - p2_yx[1]
            return abs(dy)+abs(dx)
    
    max_iter = max(10000, 4 * H * W)
    closed_set = set()
    cnt = 0
    # --- 3. 核心循环（safety 场加权：低安全区代价更高，避免贴墙）---
    while open_list and cnt < max_iter:
        current_f, current_yx = heapq.heappop(open_list)
        if current_yx in closed_set:
            continue
        closed_set.add(current_yx)
        current_y, current_x = current_yx
        
        if current_yx == goal_yx:
            break
        
        current_g = g_costs[current_y, current_x]
        
        for dy, dx in neighbors:
            neighbor_y, neighbor_x = current_y + dy, current_x + dx
            neighbor_yx = (neighbor_y, neighbor_x)
            
            if not (0 <= neighbor_y < H and 0 <= neighbor_x < W and is_traversible[neighbor_yx]):
                continue
            
            if abs(dy) == 1 and abs(dx) == 1:
                move_cost = np.sqrt(2)
            else:
                move_cost = 1.0

            if use_safety:
                s = float(safety_field[neighbor_x, neighbor_y])
                s = np.clip(s, 1e-6, 1.0)
                move_cost = move_cost * (1.0 + 2.0 * (1.0 - s))
            new_g_cost = current_g + move_cost
            
            if new_g_cost < g_costs[neighbor_yx]:
                g_costs[neighbor_yx] = new_g_cost
                f_cost = new_g_cost + heuristic(neighbor_yx, goal_yx)
                
                heapq.heappush(open_list, (f_cost, neighbor_yx))
                came_from[neighbor_yx] = current_yx
        cnt+=1
    print(f"A* 迭代次数: {cnt} (上限 {max_iter})")

    # --- 4. 路径回溯与格式化 ---
    path = []
    if goal_yx in came_from:
        current = goal_yx
        while current != start_yx:
            path.append((current[1], current[0])) 
            current = came_from[current]
        path.append((start_yx[1], start_yx[0]))
        print(f"A* 路径找到。路径长度: {len(path)} 像素点。")
    else:
        if cnt >= max_iter:
            print("A* 规划失败：达到迭代上限未找到路径。")
        else:
            print("A* 规划失败：无法找到从起点到终点的路径。")

    # --- 5. 可视化 (新增逻辑) ---
    if visualize:
        visualize_astar_planning_internal(bev_map, start, goal, g_costs, path, save_path)
        
    return path