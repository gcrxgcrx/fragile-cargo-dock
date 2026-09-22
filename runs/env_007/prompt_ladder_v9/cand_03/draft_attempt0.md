```python
# ================================================================
# 模块级状态与常量（必须声明在函数之前）
# ================================================================
_PREV_T = [-1.0]       # 上一步 obs[18]（时间比例），用于检测回合边界
_STREAK = [0]          # "泊位内 + 对齐 + 慢" 的连续步数
_PAID = [False]        # terminal_success 是否已发放（整局只发一次）
_ENTERED = [False]     # dock_enter 是否已发放（整局只发一次）
_FAILED = [False]      # terminal_failure 是否已发放（整局只发一次）
_HARD_HITS = [0]       # 硬冲击代理累计计数（>=3 判失败）

_HALF_W = 5.0          # 仓库半宽：obs[0] / obs[12] 的还原尺度（米）
_HALF_H = 4.0          # 仓库半高：obs[1] / obs[13] 的还原尺度（米）

_TOL_X = 0.024         # 泊位 x 容差（归一化，= 0.12 m）
_TOL_Y = 0.030         # 泊位 y 容差（归一化，= 0.12 m）
_ALIGN_COS = 0.8660254 # cos(30°)：朝向对齐阈值
_SPEED_OK = 0.05       # 停靠速度阈值（m/s）

_GENTLE_K = 0.30       # roughness 代理系数（接触冲量的可观测替代）
_HARD_HIT_CLOSING = 1.0
_HARD_HIT_PENALTY = 0.5
_HARD_HIT_LIMIT = 3

_SETTLE_BONUS = 0.5    # 停稳期每步收益（仅在完成谓词成立时发放）
_BOUND_TH = 0.90       # 越界守卫起始阈值
_BOUND_K = 200.0       # 越界守卫二次系数


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # 0. 回合边界检测（obs[18] 单调递增，重置时回落）
    # ------------------------------------------------------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _FAILED[0] = False
        _HARD_HITS[0] = 0
    _PREV_T[0] = t

    # ------------------------------------------------------------
    # 1. approach_cargo  (+1.0 / 米)：小车->货箱距离的本帧缩短量（有符号）
    #    obs[6], obs[7] 为车体系相对位置 / 3.0 (m)
    # ------------------------------------------------------------
    rcx = float(obs[6]) * 3.0
    rcy = float(obs[7]) * 3.0
    d_cc_prev = (rcx * rcx + rcy * rcy) ** 0.5
    nrcx = float(next_obs[6]) * 3.0
    nrcy = float(next_obs[7]) * 3.0
    d_cc_next = (nrcx * nrcx + nrcy * nrcy) ** 0.5
    approach_cargo = (d_cc_prev - d_cc_next) * 1.0   # 靠近为正，远离为负

    # ------------------------------------------------------------
    # 2. progress  (+1.0 / 米)：货箱->泊位距离的本帧缩短量（有符号）
    #    obs[12], obs[13] 为归一化有符号偏移，乘半宽/半高还原为米
    # ------------------------------------------------------------
    pdx = float(obs[12]) * _HALF_W
    pdy = float(obs[13]) * _HALF_H
    d_pd_prev = (pdx * pdx + pdy * pdy) ** 0.5
    ndx = float(next_obs[12]) * _HALF_W
    ndy = float(next_obs[13]) * _HALF_H
    d_pd_next = (ndx * ndx + ndy * ndy) ** 0.5
    progress = (d_pd_prev - d_pd_next) * 1.0         # 靠近为正，远离为负

    # ------------------------------------------------------------
    # 3. dock_enter  (+5.0, 整局一次)：货箱首次完全进入泊位容差
    # ------------------------------------------------------------
    in_dock = (abs(float(next_obs[12])) <= _TOL_X) and (abs(float(next_obs[13])) <= _TOL_Y)
    dock_enter = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # ------------------------------------------------------------
    # 4. roughness  (接触冲量代理，负项)：只在"接触且正在接近"时生效
    #    低速/匀速推箱 closing≈0 -> 不罚，绝不压制正常推进
    # ------------------------------------------------------------
    cart_fwd = float(obs[4]) * 3.0                       # 小车前向速度 m/s
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    ch = float(obs[2])
    sh = float(obs[3])
    crate_along = crate_vx * ch + crate_vy * sh          # 货箱速度沿车头方向分量
    closing = cart_fwd - crate_along                     # 接近速度
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    roughness = -_GENTLE_K * contact * closing           # 越撞越罚

    # ------------------------------------------------------------
    # 5. hard_hit  (-0.5 / 次)：单步硬冲击代理（高速接触）
    # ------------------------------------------------------------
    hard_hit = 0.0
    if contact > 0.5 and closing > _HARD_HIT_CLOSING:
        hard_hit = -_HARD_HIT_PENALTY
        if _HARD_HITS[0] < 100:
            _HARD_HITS[0] += 1

    # ------------------------------------------------------------
    # 6. action_cost (-0.0005 * ||a||^2)、time_cost (-0.002 / 步)
    # ------------------------------------------------------------
    a0 = float(action[0])
    a1 = float(action[1])
    action_cost = -0.0005 * (a0 * a0 + a1 * a1)
    time_cost = -0.002

    # ------------------------------------------------------------
    # 7. 越界守卫（对小车位置单调下降的惩罚；|obs[0]|,|obs[1]| -> 1.05 即出界）
    # ------------------------------------------------------------
    bx = abs(float(next_obs[0]))
    by = abs(float(next_obs[1]))
    guard = 0.0
    if bx > _BOUND_TH:
        guard -= _BOUND_K * (bx - _BOUND_TH) ** 2
    if by > _BOUND_TH:
        guard -= _BOUND_K * (by - _BOUND_TH) ** 2

    # ------------------------------------------------------------
    # 8. 完成谓词：泊位内（容差）+ 对齐（<30°）+ 慢（<0.05 m/s）
    #    连续 10 步 -> terminal_success 一次性 +300
    #    谓词成立期间每步给停稳收益（不得被任何其它逻辑关掉）
    # ------------------------------------------------------------
    crate_spd = ((float(next_obs[8]) * 3.0) ** 2 + (float(next_obs[9]) * 3.0) ** 2) ** 0.5
    aligned = float(next_obs[10]) > _ALIGN_COS     # 朝向误差 < 30°（目标朝向为 0）
    slow = crate_spd < _SPEED_OK
    settled = in_dock and aligned and slow

    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    settle_bonus = _SETTLE_BONUS if settled else 0.0

    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # ------------------------------------------------------------
    # 9. terminal_failure (-100, 整局一次)：
    #    小车/货箱出界（|归一化坐标| > 1.05）或累计硬冲击 >= 3
    #    货箱世界坐标 = 小车位置 + R(heading) * 车体系相对位置
    # ------------------------------------------------------------
    cart_x = float(next_obs[0]) * _HALF_W
    cart_y = float(next_obs[1]) * _HALF_H
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    crate_wx = cart_x + rel_x * ch - rel_y * sh
    crate_wy = cart_y + rel_x * sh + rel_y * ch
    crate_nx = crate_wx / _HALF_W
    crate_ny = crate_wy / _HALF_H

    out_of_bounds = (bx > 1.05) or (by > 1.05) \
        or (abs(crate_nx) > 1.05) or (abs(crate_ny) > 1.05) \
        or (_HARD_HITS[0] >= _HARD_HIT_LIMIT)

    terminal_failure = 0.0
    if out_of_bounds and not _FAILED[0]:
        _FAILED[0] = True
        terminal_failure = -100.0

    # ------------------------------------------------------------
    # 10. 汇总
    # ------------------------------------------------------------
    total = (approach_cargo + progress + dock_enter + roughness
             + hard_hit +