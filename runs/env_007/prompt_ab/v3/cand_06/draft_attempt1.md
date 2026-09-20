```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 模块级状态：回合边界检测、连续计数、一次性发放标记 ----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---- 几何常量（来自环境事实） ----
    HALF_W = 5.0
    HALF_H = 4.0
    DOCK_TOL_X = 0.024   # |obs[12]| <= 0.024
    DOCK_TOL_Y = 0.030   # |obs[13]| <= 0.030
    SPEED_TOL = 0.05     # 货箱速度 < 0.05 m/s
    ANGLE_TOL = 30.0 * 3.141592653589793 / 180.0

    # ---- 货箱到泊位的有符号偏移（归一化）与米制距离 ----
    dx = float(next_obs[12]) * HALF_W
    dy = float(next_obs[13]) * HALF_H
    dist = (dx * dx + dy * dy) ** 0.5

    # ---- 货箱速度（世界系，m/s） ----
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- 货箱朝向误差 ----
    crate_ang = _atan2(float(next_obs[11]), float(next_obs[10]))
    ang_err = crate_ang if crate_ang >= 0.0 else -crate_ang

    components = {}

    # =====================================================================
    # 1) 主进展信号：增量式（只在"这一帧更接近泊位"时给分，避免悬停收割）
    # =====================================================================
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    progress = _PREV_DIST[0] - dist
    _PREV_DIST[0] = dist
    if progress < -0.5:
        progress = -0.5
    if progress > 0.5:
        progress = 0.5
    components["crate_to_dock_progress"] = 1.0 * progress

    # =====================================================================
    # 2) 接触轻柔度：接触中且正在接近时惩罚（唯一能教"提前松手"的信号）
    # =====================================================================
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact_penalty"] = gentleness

    # =====================================================================
    # 3) 接近泊位时的速度抑制（门控：仅在已经进入泊位容差区附近才激活）
    # =====================================================================
    near_gate = 1.0 - min(1.0, dist / 0.6)
    if near_gate < 0.0:
        near_gate = 0.0
    if crate_speed > SPEED_TOL:
        excess = crate_speed - SPEED_TOL
        components["crate_speed_near_dock"] = -0.3 * near_gate * (excess ** 2)
    else:
        components["crate_speed_near_dock"] = 0.0

    # =====================================================================
    # 4) 朝向对齐 shaping（门控：仅在货箱已接近泊位时激活，不阻碍推进）
    # =====================================================================
    align_gate = 1.0 - min(1.0, dist / 0.8)
    if align_gate < 0.0:
        align_gate = 0.0
    align_factor = 1.0 - min(1.0, ang_err / ANGLE_TOL)
    if align_factor < 0.0:
        align_factor = 0.0
    components["crate_orientation_align"] = 0.3 * align_gate * align_factor

    # =====================================================================
    # 5) 越界防护（hinge：仅在小车/货箱接近仓库边界时生效）
    # =====================================================================
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    ax = cart_x if cart_x >= 0.0 else -cart_x
    ay = cart_y if cart_y >= 0.0 else -cart_y
    if ax > 0.9:
        oob += (ax - 0.9) ** 2
    if ay > 0.9:
        oob += (ay - 0.9) ** 2
    components["out_of_bounds_penalty"] = -2.0 * oob

    # =====================================================================
    # 6) 完成事件：一次性发放（连续 10 步满足 完全进入 + 对齐 + 静止）
    # =====================================================================
    inside = 1.0 if (abs(float(next_obs[12])) <= DOCK_TOL_X and abs(float(next_obs[13])) <= DOCK_TOL_Y) else 0.0
    aligned = 1.0 if ang_err < ANGLE_TOL else 0.0
    still = 1.0 if crate_speed < SPEED_TOL else 0.0
    cond = 1.0 if (inside > 0.5 and aligned > 0.5 and still > 0.5) else 0.0

    if cond > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 首次进入泊位的稀疏奖励（整局只发一次，帮助跨过"进得去但停不住"）
    enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 20.0
    components["first_dock_entry"] = enter_bonus

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["docked_success"] = success_event

    total_reward = 0.0
    for k in components:
        total_reward += components[k]

    return (float(total_reward), components)


# ---- 模块级可变状态容器（回合边界重置） ----
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]


def _atan2(y, x):
    # 无 import 的 atan2 近似实现（足够用于朝向误差判断）
    if x == 0.0 and y == 0.0:
        return 0.0
    ax = x if x >= 0.0 else -x
    ay = y if y >= 0.0 else -y
    if ax >= ay:
        r = ay / (ax + 1e-12)
        base = r - r * r * r / 3.0 + r * r * r * r * r / 5.0
        ang = base
    else:
        r = ax / (ay + 1e-12)
        base = r - r * r * r / 3.0 + r * r * r * r * r / 5.0
        ang = 1.5707963267948966 - base
    if x < 0.0:
        ang = 3.141592653589793 - ang
    if y < 0.0:
        ang = -ang
    return ang
```