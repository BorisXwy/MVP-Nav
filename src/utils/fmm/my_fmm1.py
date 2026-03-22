import numpy as np
import skfmm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq

# =====================================================================
# 核心规划函数：get_local_goal_fmm (已修改，将可视化提前)
# =====================================================================

def get_local_goal_fmm(bev_map, max_limit, start, goal, step_size, visualize, global_path, save_path):

    def compute_fmm_field(cost_map, goal_pos_local):
        H, W = cost_map.shape
        goal_x, goal_y = int(goal_pos_local[0]), int(goal_pos_local[1])
        
        cost_obstacle = cost_map.max()
        obstacle_mask = (cost_map == cost_obstacle) 

        # 1. 设置边界条件 (phi)：
        phi = np.ones(cost_map.shape)
        phi[goal_y, goal_x] = -1 
        
        # 3. FMM 的速度场 (Speed Map)
        speed = 1.0 / cost_map
        speed[obstacle_mask] = 1e-10 
        
        try:
            T_field = skfmm.travel_time(phi, speed, dx=1.0) 
        except ValueError as e:
            T_field = np.full(cost_map.shape, np.inf, dtype=np.float32)
        
        return T_field.astype(np.float32)
    
    start_x, start_y = int(start[0]), int(start[1])
    goal_x_global, goal_y_global = int(goal[0]), int(goal[1])
        
    H_global, W_global = bev_map.shape[:2]

    min_side_half = min(H_global, W_global) // 2

    window_size = min(min_side_half, max_limit)

    if window_size % 2 != 0:
        window_size += 1
    half_w = window_size // 2
    
    # --- 1. 地图裁剪 ---
    y_min_global = max(0, start_y - half_w)
    y_max_global = min(H_global, start_y + half_w)
    x_min_global = max(0, start_x - half_w)
    x_max_global = min(W_global, start_x + half_w)
    
    cropped_bev = bev_map[y_min_global:y_max_global, x_min_global:x_max_global].copy()
    start_x_local = start_x - x_min_global
    start_y_local = start_y - y_min_global

    
    H_local, W_local = cropped_bev.shape[:2]

    # --- 2. 目标投影 ---
    goal_x_local_init = goal_x_global - x_min_global
    goal_y_local_init = goal_y_global - y_min_global
    
    stg_x_global = start_x # 提前设置默认STG为起点
    stg_y_global = start_y 

    if (0 <= goal_x_local_init < W_local and 0 <= goal_y_local_init < H_local):
        goal_x_local, goal_y_local = goal_x_local_init, goal_y_local_init
    else:
        dx_g = float(goal_x_global) - float(start_x)
        dy_g = float(goal_y_global) - float(start_y)
        
        if dx_g == 0.0 and dy_g == 0.0:
            # *提前返回点 1：起点等于终点*
            print("--- [决策] 起点等于终点。返回起点。")
            # --- 可视化 (规划失败/结束时) ---
            if visualize:
                planning_data = create_default_planning_data(cropped_bev, start_x_local, start_y_local, (x_min_global, y_min_global, x_max_global, y_max_global))
                visualize_fmm_planning_internal(bev_map, start, goal, start, planning_data,save_path)
            return start
            
        
        R_window = np.sqrt(W_local**2 + H_local**2) / 2.0
        dist_g = np.sqrt(dx_g**2 + dy_g**2)
        ratio = min(R_window, dist_g) / dist_g if dist_g > 0 else 0
        px_global = start_x + dx_g * ratio
        py_global = start_y + dy_g * ratio
        goal_x_local = int(round(px_global - x_min_global))
        goal_y_local = int(round(py_global - y_min_global))
        
        goal_x_local = np.clip(goal_x_local, 0, W_local - 1)
        goal_y_local = np.clip(goal_y_local, 0, H_local - 1)
            
    goal_pos_local = (goal_x_local, goal_y_local)

    # -----------------------------------------------------
    # 【新增逻辑 A：检查投影目标点是否在障碍物上】
    # -----------------------------------------------------
    
    # 提前确定 cost_map_local 的转换逻辑 (简化检查)
    # 这部分代码最好移到 cost_map_local 转换之前，但为清晰展示，我们在此处引入检查逻辑
    
    is_obstacle_projection = False
    
    # 注意：这里需要确保 goal_y_local 和 goal_x_local 已经被计算
    if (0 <= goal_x_local < W_local and 0 <= goal_y_local < H_local):
        # 简单地检查 cropped_bev 的像素值
        # 假设：非 200/255 的点为障碍物 (与步骤 3 的逻辑保持一致)
        check_value = cropped_bev[goal_y_local, goal_x_local]
        if cropped_bev.ndim == 3:
             check_value = check_value[0] # 取第一个通道
             
        if not ((check_value == 200) or (check_value == 255)):
            is_obstacle_projection = True
    
    # -----------------------------------------------------
    # 【新增逻辑 B：执行回退搜索】
    # -----------------------------------------------------
    if is_obstacle_projection:
        
        # 定义一个搜索核半径 (例如 5 个像素)
        search_radius = 5
        found_stg = False
        
        # 搜索目标投影点周围的局部区域
        # 迭代顺序：从近到远
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                new_gx = goal_x_local + dx
                new_gy = goal_y_local + dy
                
                # 检查新点是否在局部地图内
                if (0 <= new_gx < W_local and 0 <= new_gy < H_local):
                    
                    # 检查新点是否可通行
                    check_value = cropped_bev[new_gy, new_gx]
                    if cropped_bev.ndim == 3:
                         check_value = check_value[0]
                         
                    if (check_value == 200) or (check_value == 255):
                        # 找到第一个可通行点，作为替代局部目标
                        goal_x_local, goal_y_local = new_gx, new_gy
                        goal_pos_local = (goal_x_local, goal_y_local)
                        found_stg = True
                        print(f"--- [决策] 目标投影点为障碍物，回退到最近的可通行点: ({dx}, {dy})")
                        break # 跳出内层循环
            if found_stg:
                break # 跳出外层循环

        if not found_stg:
            # *极端情况：投影点周围 search_radius 范围内全是障碍物*
            # 此时保持原投影点，FMM 场仍会返回 Inf，由后面的提前返回点 2 处理。
            # 或者可以返回起点，取决于您的规划策略。
            print("--- [决策] 目标投影点及其附近全是障碍物。返回起点。")
            
            # 为了避免计算 FMM 场，直接返回起点 (可选，取决于您的可视化需求)
            # return start 
            # 我们保持原代码流，让 FMM 计算 Inf 并触发返回点 2，以利用现有可视化逻辑
            pass # 保持原 goal_pos_local

    # --- 3. 成本图转换 ---
    cost_traversible = 1.0
    cost_obstacle = 100.0

    if cropped_bev.ndim == 3:
        channel_0 = cropped_bev[:, :, 0]
    else:
        channel_0 = cropped_bev
    traversible_mask = (channel_0 == 200) | (channel_0 == 255)
    cost_map_local = np.full(channel_0.shape, cost_obstacle, dtype=np.float32)
    cost_map_local[traversible_mask] = cost_traversible
    cost_map_local[traversible_mask] += 1e-6

    # --- 4. 计算 FMM 场 ---
    T_field_local = compute_fmm_field(cost_map_local, goal_pos_local)

    # --- 5. 规划短期目标前的检查 ---
    cy, cx = start_y_local, start_x_local 
    current_cost = T_field_local[cy, cx]

    # 初始化规划数据 (包含 FMM 结果，用于失败时可视化)
    grad_y, grad_x = np.gradient(T_field_local)
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
    
    stg_x_local, stg_y_local = cx, cy
    
    if current_cost == np.inf:
        # *提前返回点 2：起点 FMM 成本为 Inf，不可达*
        print("--- [决策] 起点 FMM 成本为 Inf，不可达。返回起点。")
        # --- 可视化 (规划失败时) ---
        if visualize:
            visualize_fmm_planning_internal(bev_map, start, goal, start, planning_data,save_path)
        return start
        
    if not (0 < cx < W_local - 1 and 0 < cy < H_local - 1):
        # *提前返回点 3：起点在局部地图边界上*
        print("--- [决策] 起点在局部地图边界上。返回起点。")
        # --- 可视化 (规划失败时) ---
        if visualize:
            visualize_fmm_planning_internal(bev_map, start, goal, start, planning_data, save_path)
        return start

    # --- 6. 规划短期目标 (STG) ---
    gx = grad_x[cy, cx]
    gy = grad_y[cy, cx]
    grad_norm = np.sqrt(gx**2 + gy**2)

    if grad_norm == 0.0 or current_cost <= 0.0:
        stg_x_local, stg_y_local = cx, cy
    else:

        direction_x = -gx / grad_norm 
        direction_y = -gy / grad_norm 
        
        stg_x_local_float = float(cx) + direction_x * step_size
        stg_y_local_float = float(cy) + direction_y * step_size
        
        stg_x_local = int(round(stg_x_local_float))
        stg_y_local = int(round(stg_y_local_float))
        
        stg_x_local = np.clip(stg_x_local, 0, W_local - 1)
        stg_y_local = np.clip(stg_y_local, 0, H_local - 1)

    # --- 7. 将短期目标点转换回全局坐标 ---
    stg_x_global = stg_x_local + x_min_global
    stg_y_global = stg_y_local + y_min_global
    
    # 更新 planning_data 中的 STG (用于成功规划的可视化)
    planning_data["stg_local"] = (stg_x_local, stg_y_local)

    # --- 可视化阶段 (成功规划时) ---
    if visualize:
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
                         vmin=0, vmax=np.nanpercentile(g_costs_vis, 95)) # 限制最大值以突出路径
    
    fig.colorbar(im_cost, ax=ax1, label='A* G-Cost (Actual Distance)')
    ax1.set_title('1. A* G-Cost Field')

    # --- 2. 原始 BEV 地图与路径 ---
    ax2 = fig.add_subplot(1, 2, 2) 

    # 显示原始地图
    ax2.imshow(bev_map_orig[:, :, 0] if bev_map_orig.ndim == 3 else bev_map_orig, 
               origin='lower', cmap='gray', vmin=0, vmax=255)
    
    # 绘制路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax2.plot(path_x, path_y, color='lime', linewidth=3, label='A* Global Path')
        
    # 绘制起点和终点
    ax2.scatter(start_global[0], start_global[1], color='blue', marker='o', s=150, label='Start')
    ax2.scatter(goal_global[0], goal_global[1], color='red', marker='x', s=150, label='Goal')
    
    ax2.set_title('2. Global Map with A* Path')
    ax2.legend()
    ax2.set_xlim(0, W)
    ax2.set_ylim(0, H)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =====================================================================
# 全局 A* 路径规划函数 (已修改：新增可视化参数和逻辑)
# =====================================================================

def compute_astar_path(bev_map, start, goal, visualize=False, save_path='/tmp/astar_path.png'):
    """
    使用 A* 算法计算从起点到终点的最短路径。这是一个全局规划器。
    
    输入:
        bev_map (np.ndarray): 鸟瞰图/成本图。假设非 200/255 的值为障碍物。
        start (tuple): 起点全局坐标 (x, y)。
        goal (tuple): 终点全局坐标 (x, y)。
        visualize (bool): 是否启用可视化。
        save_path (str): 可视化图像的保存路径。
        
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
        return abs(p1_yx[0] - p2_yx[0]) + abs(p1_yx[1] - p2_yx[1])

    # --- 3. 核心循环 ---
    while open_list:
        current_f, current_yx = heapq.heappop(open_list)
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

            new_g_cost = current_g + move_cost
            
            if new_g_cost < g_costs[neighbor_yx]:
                g_costs[neighbor_yx] = new_g_cost
                f_cost = new_g_cost + heuristic(neighbor_yx, goal_yx)
                
                heapq.heappush(open_list, (f_cost, neighbor_yx))
                came_from[neighbor_yx] = current_yx

    # --- 4. 路径回溯与格式化 ---
    path = []
    if goal_yx in came_from:
        current = goal_yx
        while current != start_yx:
            path.append((current[1], current[0])) 
            current = came_from[current]
        path.append((start_yx[1], start_yx[0]))
        path.reverse() 
        print(f"A* 路径找到。路径长度: {len(path)} 像素点。")
    else:
        print("A* 规划失败：无法找到从起点到终点的路径。")

    # --- 5. 可视化 (新增逻辑) ---
    if visualize:
        visualize_astar_planning_internal(bev_map, start, goal, g_costs, path, save_path)
        
    return path