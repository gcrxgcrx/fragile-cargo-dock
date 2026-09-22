```python
# =====================================================================
# v9 reward: 用 obs + action 重写已知奖励结构（分项语义 + 权重）
# 模块级状态（必须写在 compute_reward 之前）
# =====================================================================
_PREV_T = [-1.0]     # 上一步的 obs[18]，用于检测 episode 边界
_STREAK = [0]        # 完成谓词连续成立的步数
_PAID = [False]      # terminal_success 是否已发放（整局一次）
_ENTERED = [False]   # 货箱是否曾首次完全进入泊位（dock_enter 整局一次）
_HARD_HITS = [0]     # 累计硬冲击次数（可观测代理）
_HIT_FLAG = [0]      # 上一步是否处于硬冲击（上升沿去抖）
_FAILED = [False]    # terminal_failure 是否已发放（整局一次）


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -----------------------------------------------------------------
    # 0. episode 边界检测：obs[18] 在一个回合内单调递增，重置时回落
    # -----------------------------------------------------------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 0.0025:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _HIT_FLAG[0] = 0
        _FAILED[0] = False
    _PREV_T[0] = t

    # -----------------------------------------------------------------
    # 1. 几何量还原（只用已声明的 obs 维度）
    # -----------------------------------------------------------------
    # 小车 -> 货箱距离（米）：obs[6],obs[7] = 货箱相对小车车体系位置 / 3.0 m
    d_cc_prev = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    d_cc_next = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5

    # 货箱 -> 泊位距离（米）：obs[12] / 仓库半宽(5.0 m)，obs[13] / 仓库半高(4.0 m)
    d_cd_prev = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    d_cd_next = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5

    # -----------------------------------------------------------------
    # 2. 分项实现
    # -----------------------------------------------------------------
    # (1) approach_cargo: +1.0 / 米，本帧小车->货箱距离缩短量，有符号对称
    approach_cargo = 1.0 * (d_cc_prev - d_cc_next)

    # (2) progress: +1.0 / 米，本帧货箱->泊位距离缩短量，有符号对称
    progress = 1.0 * (d_cd_prev - d_cd_next)

    # (3) dock_enter: 货箱首次完全进入泊位容差，一次性 +5.0
    #     容差取自环境事实：|obs[12]| <= 0.024 且 |obs[13]| <= 0.030
    inside_dock = (abs(next_obs[12]) <= 0.024) and (abs(next_obs[13]) <= 0.030)
    dock_enter = 0.0
    if inside_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # (4) roughness: -0.02 /(N·s) 接触冲量比例惩罚的“可观测代理”
    #     观测里没有冲量维，用 接近速度 × 接触 构造：
    #     closing = 小车前向速度 - 货箱速度沿车头方向的分量（仅取正，即“正在接近”）
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # 系数 0.05 = 0.02/(N·s) × 2.5 (代理冲量 ≈ 2.5·closing)：
    # 匀速推箱时 closing≈0 -> 该项≈0，不惩罚正常推进；
    # 1.0 m/s 撞上去时才显著（自检③）。
    roughness = -0.05 * contact * closing

    # (5) action_cost: -0.0005 × 动作平方和
    action_cost = -0.0005 * (action[0] ** 2 + action[1] ** 2)

    # (6) time_cost: -0.002 / 步
    time_cost = -0.002

    # (7) hard_hit: 单步内发生一次硬冲击 -> -0.5（用“接触 + 高接近速度”代理，上升沿去抖）
    hard = (contact > 0.5) and (closing > 0.5)
    hard_hit = 0.0
    if hard and _HIT_FLAG[0] == 0:
        _HARD_HITS[0] += 1
        hard_hit = -0.5
    _HIT_FLAG[0] = 1 if hard else 0

    # (8) 货箱世界坐标还原（小车世界位置 obs[0]*5.0, obs[1]*4.0 + 车体系偏移旋转）
    cx_body = next_obs[6] * 3.0 * next_obs[2] - next_obs[7] * 3.0 * next_obs[3]
    cy_body = next_obs[6] * 3.0 * next_obs[3] + next_obs[7] * 3.0 * next_obs[2]
    crate_nx = (next_obs[0] * 5.0 + cx_body) / 5.0
    crate_ny = (next_obs[1] * 4.0 + cy_body) / 4.0

    # (9) 停稳期每步收益（完成谓词：泊位内 + 对齐 + 慢），每步发放，不被任何开关关掉
    aligned = next_obs[10] >= 0.8660          # 朝向误差 < 30°
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    slow = crate_speed < 0.05                 # 货箱几乎静止
    settled = inside_dock and aligned and slow
    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0
    settled_bonus = 0.3 if settled else 0.0

    # (10) terminal_success: 连续 10 步成立时一次性 +300（环境随后立即终止 episode）
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # (11) 越界守卫：随接近边界单调下降，0.90 以后明显生效（1.05 即出界）
    b = abs(next_obs[0])
    if abs(next_obs[1]) > b:
        b = abs(next_obs[1])
    if abs(crate_nx) > b:
        b = abs(crate_nx)
    if abs(crate_ny) > b:
        b = abs(crate_ny)
    over = b - 0.90
    if over < 0.0:
        over = 0.0
    boundary_penalty = -(30.0 * over * over + 5.0 * over)

    # (12) terminal_failure: 小车/货箱越界，或累计硬冲击 >= 3，一次性 -100
    oob = (abs(next_obs[0]) > 1.05) or (abs(next_obs[1]) > 1.05) \
          or (abs(crate_nx) > 1.05) or (abs(crate_ny) > 1.05)
    terminal_failure = 0.0
    if (oob or _HARD_HITS[0] >= 3)