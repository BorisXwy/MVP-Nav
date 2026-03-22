import cv2
import numpy as np
import skfmm
import skimage
from numpy import ma
from skimage.draw import line, circle_perimeter
import numba
import matplotlib.pyplot as plt

# @numba.jit(nopython=True)
def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    # for i in range(size):
    #     for j in range(size):
    #         if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
    #            ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
    #                 step_size ** 2 \
    #            and ((i + 0.5) - (size // 2 + sx)) ** 2 + \
    #            ((j + 0.5) - (size // 2 + sy)) ** 2 > \
    #                 (step_size - 1) ** 2:
    #             mask[i, j] = 1
    rr, cc = circle_perimeter(size // 2, size // 2, step_size)
    mask[rr, cc] = 1

    mask[size // 2, size // 2] = 1
    return mask

# @numba.jit(nopython=True)
def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    # for i in range(size):
    #     for j in range(size):
    #         if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
    #            ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
    #                 step_size ** 2:
    #             mask[i, j] = max(5,
    #                              (((i + 0.5) - (size // 2 + sx)) ** 2 +
    #                               ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    for i in range(step_size):
        rr, cc = circle_perimeter(size // 2, size // 2, i+1)
        mask[rr, cc] = i+1
    return mask


class FMMPlanner():
    def __init__(self, traversible, scale=1, step_size=5):
        self.scale = scale
        self.step_size = step_size
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.))
        self.fmm_dist = None

    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.)), \
            int(goal[1] / (self.scale * 1.))

        if self.traversible[goal_x, goal_y] == 0. and auto_improve:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return

    def set_multi_goal(self, goal_map, visited=None):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        if visited is not None:
            dd += visited * 0.5
        self.fmm_dist = dd
        return

    # def get_short_term_goal(self, state, step_size=5):
    #     self.step_size = step_size
    #     self.du = int(self.step_size / (self.scale * 1.))
    #     scale = self.scale * 1.
    #     state = [x / scale for x in state]
    #     dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
    #     mask = get_mask(dx, dy, scale, self.step_size)
    #     dist_mask = get_dist(dx, dy, scale, self.step_size)

    #     state = [int(x) for x in state]

    #     dist = np.pad(self.fmm_dist, self.du,
    #                   'constant', constant_values=self.fmm_dist.shape[0] ** 2)
    #     subset = dist[state[0]:state[0] + 2 * self.du + 1,
    #                   state[1]:state[1] + 2 * self.du + 1]

    #     assert subset.shape[0] == 2 * self.du + 1 and \
    #         subset.shape[1] == 2 * self.du + 1, \
    #         "Planning error: unexpected subset shape {}".format(subset.shape)

    #     subset *= mask
    #     subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

    #     if subset[self.du, self.du] < 0.25 * 100 / 5.:  # 25cm
    #         stop = True
    #     else:
    #         stop = False

    #     subset -= subset[self.du, self.du]
    #     ratio1 = subset / dist_mask
    #     subset[ratio1 < -1.5] = 1

    #     subset[self.du, self.du] = (self.fmm_dist.shape[0] ** 2 - subset[self.du, self.du]) / 5

    #     (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

    #     if subset[stg_x, stg_y] > -0.0001:
    #         replan = True
    #     else:
    #         replan = False

    #     return (stg_x + state[0] - self.du) * scale, \
    #            (stg_y + state[1] - self.du) * scale, replan, stop

    def get_short_term_goal_v1(self, state, k):
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]

        assert subset.shape[0] == 2 * self.du + 1 and \
            subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

        if subset[self.du, self.du] < 0.25 * 100 / 5.:  # 25cm
            stop = True
        else:
            stop = False

        subset -= subset[self.du, self.du]
        # ratio1 = subset / dist_mask
        # subset[ratio1 < -1.5] = 1

        subset[self.du, self.du] = (self.fmm_dist.shape[0] ** 2 - subset[self.du, self.du]) / 5

        top_k_min_indices_flat = np.argsort(subset.flatten())[:k]
        top_k_min_indices = np.unravel_index(top_k_min_indices_flat, subset.shape)
        stg_x, stg_y = top_k_min_indices[0], top_k_min_indices[1]

        # (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        if np.any(subset[stg_x, stg_y]) > -0.0001:
            replan = True
        else:
            replan = False

        return (stg_x + state[0] - self.du) * scale, \
               (stg_y + state[1] - self.du) * scale, replan, stop


    def get_short_term_goal(self, state, step_size=5):
        # ------------------- 检查代码 1: 输入参数 -------------------
        print("="*50)
        print("--- 规划开始 ---")
        print(f"输入原始状态 (物理坐标): {state}")
        print(f"步长 step_size (物理距离): {step_size}")
        print(f"地图比例尺 scale (物理/网格): {self.scale}")

        self.step_size = step_size
        self.du = int(self.step_size / (self.scale * 1.))
        scale = self.scale * 1.

        # ------------------- 检查代码 2: 坐标转换 -------------------
        state_float = [x / scale for x in state]
        print(f"网格浮点坐标: {state_float}")
        self.du = int(self.step_size / (self.scale * 1.))

        dx, dy = state_float[0] - int(state_float[0]), state_float[1] - int(state_float[1])
        print(f"亚网格偏移 (dx, dy): ({dx:.4f}, {dy:.4f})")
        
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state_int = [int(x) for x in state_float]
        print(f"网格整数坐标 (row, col): {state_int}")
        print(f"搜索半径 du (网格步长): {self.du}")


        # ------------------- 检查代码 3: 提取局部地图 -------------------
        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        
        # 注意：这里 state[0] 是行 (row)，state[1] 是列 (col)
        subset = dist[state_int[0]:state_int[0] + 2 * self.du + 1,
                      state_int[1]:state_int[1] + 2 * self.du + 1]

        print(f"FMM 距离图 'subset' 形状: {subset.shape}")
        
        # 确保 subset 提取正确，这是你代码中原有的断言
        assert subset.shape[0] == 2 * self.du + 1 and \
            subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        # ------------------- 可视化 1: 原始局部 FMM 距离 -------------------
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(subset, origin='lower', cmap='plasma')
        plt.colorbar(label='FMM Distance to Goal')
        plt.title('1. 原始局部 FMM 距离')
        plt.scatter(self.du, self.du, color='red', marker='x', label='Current Pos')
        plt.legend()


        # ------------------- 4. 局部势场修正 -------------------
        subset_masked = subset * mask
        subset_cost = subset_masked + (1 - mask) * self.fmm_dist.shape[0] ** 2

        # ------------------- 检查代码 4: 停止判断 -------------------
        current_fmm_dist = subset_cost[self.du, self.du]
        stop_threshold = 0.25 * 100 / 5. # 假设 25cm 换算后的网格值
        if current_fmm_dist < stop_threshold:
            stop = True
        else:
            stop = False
        print(f"\n当前位置 FMM 距离: {current_fmm_dist:.4f}")
        print(f"停止阈值: {stop_threshold:.4f}，是否停止 (stop): {stop}")

        # 势场差值 (当前位置的 FMM 距离减去)
        subset_diff = subset_cost - subset_cost[self.du, self.du]
        
        # 计算 ratio1
        # 避免除以零，将 dist_mask 中为 0 的点（即中心点）设置为一个很小的非零值
        dist_mask_safe = np.where(dist_mask == 0, 1e-6, dist_mask) 
        ratio1 = subset_diff / dist_mask_safe
        
        # 应用 ratio1 修正
        subset_final = subset_diff.copy()
        subset_final[ratio1 < -1.5] = 1
        
        # 确保中心点不是 STG
        subset_final[self.du, self.du] = (self.fmm_dist.shape[0] ** 2 - subset_cost[self.du, self.du]) / 5

        # ------------------- 可视化 2: 修正后的势场 (代价图) -------------------
        plt.subplot(1, 3, 2)
        im = plt.imshow(subset_final, origin='lower', cmap='RdYlGn')
        plt.colorbar(im, label='Final Cost (Negative is Better)')
        plt.title('2. 最终代价图 (subset_final)')
        plt.scatter(self.du, self.du, color='red', marker='x', label='Current Pos')
        plt.legend()


        # ------------------- 5. 确定短期目标 -------------------
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset_final), subset_final.shape)

        # ------------------- 检查代码 5: STG 结果 -------------------
        stg_cost = subset_final[stg_x, stg_y]
        print(f"\n短期目标 (STG) 局部网格坐标 (row, col): ({stg_x}, {stg_y})")
        print(f"STG 的最终代价: {stg_cost:.4f}")

        if stg_cost > -0.0001:
            replan = True
        else:
            replan = False
        print(f"是否需要重新规划 (replan): {replan}")

        # ------------------- 可视化 3: 最终选择 -------------------
        plt.subplot(1, 3, 3)
        plt.imshow(subset_final, origin='lower', cmap='RdYlGn')
        plt.colorbar(label='Final Cost')
        plt.title('3. 选定的 STG')
        plt.scatter(self.du, self.du, color='red', marker='x', s=100, label='Current Pos')
        plt.scatter(stg_y, stg_x, color='blue', marker='o', s=100, label='STG')
        plt.legend()
        plt.show() # 显示所有图

        
        # ------------------- 6. 返回结果 -------------------
        stg_global_x = (stg_x + state_int[0] - self.du) * scale
        stg_global_y = (stg_y + state_int[1] - self.du) * scale

        print(f"\n短期目标 (STG) 全局物理坐标 (X, Y): ({stg_global_x:.2f}, {stg_global_y:.2f})")
        print("--- 规划结束 ---")
        print("="*50)
        
        # 强制暂停 2 秒，方便观察可视化结果
        # time.sleep(2) 

        return stg_global_x, stg_global_y, replan, stop

    def _find_nearest_goal(self, goal):
        traversible = skimage.morphology.binary_dilation(
            np.zeros(self.traversible.shape),
            skimage.morphology.disk(2)) != True
        traversible = traversible * 1.
        planner = FMMPlanner(traversible)
        planner.set_goal(goal)

        mask = self.traversible

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal
