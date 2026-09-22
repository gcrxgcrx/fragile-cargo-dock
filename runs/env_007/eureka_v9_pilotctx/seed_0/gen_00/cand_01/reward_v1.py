# =====================================================================
# 模块级状态（一次性事件 + 回合边界检测）
# =====================================================================
_PREV_T = [-1.0]         # 上一帧的 obs[18]（时间比例，回合内单调递增，重置时回落）
_STREAK = [0]            # "泊位内 + 对齐 + 慢" 的连续步计数
_SUCCESS_PAID = [False]  # 一次性完成事件是否已发放（同一 episode 只发一次）
_ENTERED = [False]       # 是否曾经完全进入泊位（dock_enter 一次性）
_HARD_HITS = [0]         # 本回合累计硬冲击次数（可观测代理）
_FAIL_PAID = [False]     # 一次性失败事件是否已发放

# 环境事实给出的尺度常量
_HALF_W = 5.0            # 仓库半宽（米）：obs[0] = cart_x / 5.0
_HALF_H = 4.0            # 仓库半高（米）：obs[1] = cart_y / 4.0
_REL_SCALE = 3.0         # obs[6]/obs[7] 与 obs[4]/obs[8]/obs[9] 的米制尺度
_DOCK_TOL_X = 0.024      # |obs[12]| 容差（完全进入泊位）
_DOCK_TOL_Y = 0.030      # |obs[13]| 容差
_ALIGN_COS = 0.8660254   # cos(30°)：货箱朝向对齐阈值
_SPEED_TOL = 0.05        # m/s：货箱几乎静止阈值
_HARD_HIT_CLOSING = 1.0  # 接触接近速度 >= 1.0 m/s 记一次硬冲击
_SOFT_EDGE = 0.95        # 边界守卫起点（归一化，1.0 是墙，>1.05 出界）
_ROUGH_K = 0.3           # roughness 代理系数（接触接近速度惩罚）

# =====================================================================
# 自检记录（每步量级估计；dt≈0.1s，正常推箱速度≈1 m/s）
# ---------------------------------------------------------------------
# R_idle    : 什么都不做、货箱静止在初始位置：
#             approach≈0, progress≈0, roughness=0, action_cost≈0,
#             time_cost=-0.002                      →  R_idle  ≈ -0.002
# R_push    : 正常推箱（接触、closing≈0、每步把货箱向泊位推进约 0.10 m）：
#             progress=+0.10, approach≈0（车-箱间距不变）, roughness≈0,
#             罚项≈-0.0025                          →  R_push  ≈ +0.098
#             自检①：R_push(-) > R_idle  ✓（差 ≈0.10，不小于任何罚项量级）
# R_settled : 停稳在泊位内（|obs[12]|<=0.024, |obs[13]|<=0.030, 对齐, 速度<0.05）：
#             settled_hold=+2.0, progress≈0, 罚项≈-0.002
#                                                     →  R_settled ≈ +2.0 > R_push ✓
# 自检②：④(真正入泊停稳 10 步) ≈ 10*2.0 + 300(一次性) + 5 ≈ 325
#        远高于 ③(泊位外 0.3 m 悬停 400 步 ≈ -0.8)                    ✓
# 自检④：同一停稳状态连续调用 12 次，每步增量恒为 +1.98 左右（线性增长），
#        第 10 次额外叠加一次性 +300（整回合仅一次），其后每步增量不变（不被事件关掉）
# 自检⑤(轻柔度)：contact 且 closing=1.0 m/s → roughness=-0.30 且 hard_hit=-0.5；
#        contact 且 closing=0.05 m/s → roughness=-0.015。
#        差值 ≈ 0.785 ≫ 正常推箱单步推进值 ≈0.10 → 高速撞击明确不划算 ✓
# 自检⑤(边界)：obs[0]=obs[1]=0 → 守卫 0；|obs[0]|=1.05 → 守卫 -1.2，
#        并触发一次性 terminal_failure(-100)                                ✓
# =====================================================================
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 0) 回合边界检测（obs[18] 重置时回落） ----------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _SUCCESS_PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAIL_PAID[0] = False
    _PREV_T[0] = t

    # ---------------- 1) approach_cargo：+1.0/米（有符号、对称） ----------------
    # 小车->货箱距离 = 车体系相对位置模长（旋转不变）× 3.0 米
    rx_p = obs[6] * _REL_SCALE
    ry_p = obs[7] * _REL_SCALE
    rx_n = next_obs[6] * _REL_SCALE
    ry_n = next_obs[7] * _REL_SCALE
    d_cc_p = (rx_p * rx_p + ry_p * ry_p) ** 0.5
    d_cc_n = (rx_n * rx_n + ry_n * ry_n) ** 0.5
    approach_cargo = 1.0 * (d_cc_p - d_cc_n)

    # ---------------- 2) progress：+1.0/米（有符号、对称） ----------------
    # 货箱->泊位距离 = hypot(obs[12]*半宽, obs[13]*半高)
    dx_p = obs[12] * _HALF_W
    dy_p = obs[13] * _HALF_H
    dx_n = next_obs[12] * _HALF_W
    dy_n = next_obs[13] * _HALF_H
    d_dock_p = (dx_p * dx_p + dy_p * dy_p) ** 0.5
    d_dock_n = (dx_n * dx_n + dy_n * dy_n) ** 0.5
    progress = 1.0 * (d_dock_p - d_dock_n)

    # ---------------- 3) dock_enter：+5.0 一次性（首次完全进入容差） ----------------
    inside_x = abs(next_obs[12]) <= _DOCK_TOL_X
    inside_y = abs(next_obs[13]) <= _DOCK_TOL_Y
    inside_dock = inside_x and inside_y
    dock_enter = 0.0
    if inside_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # ---------------- 4) roughness / hard_hit：接触接近速度代理 ----------------
    crate_vx = next_obs[8] * _REL_SCALE
    crate_vy = next_obs[9] * _REL_SCALE
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * _REL_SCALE - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # 仅"接触中且正在接近"时惩罚；匀速推箱（closing≈0）不判为撞击
    roughness = -_ROUGH_K * contact * closing

    hard_hit = 0.0
    if contact > 0.5 and closing >= _HARD_HIT_CLOSING:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # ---------------- 5) action_cost / time_cost ----------------
    a0 = float(action[0])
    a1 = float(action[1])
    action_cost = -0.0005 * (a0 * a0 + a1 * a1)
    time_cost = -0.002

    # ---------------- 6) 停稳期每步收益 + 一次性完成事件 ----------------
    hn = (next_obs[10] * next_obs[10] + next_obs[11] * next_obs[11]) ** 0.5
    if hn > 0.000001:
        heading_cos = next_obs[10] / hn
    else:
        heading_cos = 0.0
    aligned = heading_cos >= _ALIGN_COS                      # 朝向误差 < 30°
    crate_speed = ((next_obs[8] * _REL_SCALE) ** 2 + (next_obs[9] * _REL_SCALE) ** 2) ** 0.5
    slow = crate_speed < _SPEED_TOL                          # 几乎静止

    settled = inside_dock and aligned and slow

    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 停稳期每步收益：谓词成立就每步发放，不被任何其它逻辑关掉
    settled_hold = 2.0 if settled else 0.0

    # 一次性完成事件（连续 10 步谓词成立时发一次；本环境此刻立即终止 episode）
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _SUCCESS_PAID[0]:
        _SUCCESS_PAID[0] = True
        terminal_success = 300.0

    # ---------------- 7) 边界守卫（随越界接近单调下降） ----------------
    boundary_guard = 0.0
    ox = abs(next_obs[0]) - _SOFT_EDGE
    if ox > 0.0:
        boundary_guard -= 12.0 * ox
    oy = abs(next_obs[1]) - _SOFT_EDGE
    if oy > 0.0:
        boundary_guard -= 12.0 * oy

    # 货箱世界坐标（由小车位置 + 车体系相对向量经朝向旋转还原）
    hx = obs[2]
    hy = obs[3]
    cart_x = obs[0] * _HALF_W
    cart_y = obs[1] * _HALF_H
    crate_x = cart_x + (rx_n * hx - ry_n * hy)
    crate_y = cart_y + (rx_n * hy + ry_n * hx)

    cx = abs(crate_x) / _HALF_W - _SOFT_EDGE
    if cx > 0.0:
        boundary_guard -= 12.0 * cx
    cy = abs(crate_y) / _HALF_H - _SOFT_EDGE
    if cy > 0.0:
        boundary_guard -= 12.0 * cy

    # ---------------- 8) 一次性失败事件 ----------------
    cart_out = (abs(next_obs[0]) > 1.05) or (abs(next_obs[1]) > 1.05)
    crate_out = (abs(crate_x) > _HALF_W * 1.05) or (abs(crate_y) > _HALF_H * 1.05)
    terminal_failure = 0.0
    if (cart_out or crate_out or _HARD_HITS[0] >= 3) and not _FAIL_PAID[0]:
        _FAIL_PAID[0] = True
        terminal_failure = -100.0

    # ---------------- 9) 汇总 ----------------
    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
        "settled_hold": settled_hold,
        "boundary_guard": boundary_guard,
    }

    total_reward = (approach_cargo + progress + dock_enter + roughness + action_cost
                    + time_cost + hard_hit + terminal_success + terminal_failure
                    + settled_hold + boundary_guard)

    return float(total_reward), components