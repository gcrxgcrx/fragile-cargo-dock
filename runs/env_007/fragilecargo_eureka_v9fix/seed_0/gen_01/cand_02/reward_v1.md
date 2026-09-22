分析：当前代理与任务分基本对齐（末期成功率约 93%，task score 294），主要问题有三点：①`boundary_guard` 是无上界的线性惩罚，出现 −255 的离群回合，破坏价值学习稳定性；②`hard_hit` 全程 0 触发（阈值不现实），是死组件，而 `roughness` 用连续 closing 速度惩罚，会在"起步顶住货箱"这一必要动作上扣分；③泊位内只有一次性 `dock_enter` 与稀疏 `settle_bonus`，末段（包含∧低速∧对齐）缺少连续梯度，1/20 回合因超时截断。改进：把边界守卫改为饱和形式（单源上限 7.5），把 roughness 改为 hinge（只有 closing 超过 0.3 m/s 的硬接触才罚），新增"泊位包含度"与"包含×低速×对齐"的连续几何联合塑形（仅当货箱已在容差内才非零，且量级远低于 300 的一次性成功，不构成悬停陷阱），并轻微提高时间成本。所有新增项均为非负，自检 ①静止 ≈ −0.003，②稳定推箱 ≈ +0.01 以上，②>① 严格成立。

_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_FAILED = [False]
_HARD_HITS = [0]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _FAILED[0] = False
        _HARD_HITS[0] = 0
    _PREV_T[0] = t

    # 1. 主信号：货箱→泊位距离本帧缩短量（米）
    old_dock = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    new_dock = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = (old_dock - new_dock) * 1.2

    # 2. 接触建立：小车→货箱距离本帧缩短量（米）
    old_rel = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    new_rel = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5
    approach_cargo = (old_rel - new_rel) * 0.8

    # 3. 泊位包含度（只在容差内非零，中心最大），非负，不与推进冲突
    dx_n = abs(next_obs[12])
    dy_n = abs(next_obs[13])
    fx = 1.0 - dx_n / 0.024
    if fx < 0.0:
        fx = 0.0
    fy = 1.0 - dy_n / 0.030
    if fy < 0.0:
        fy = 0.0
    contain_factor = (fx * fy) ** 0.5
    in_dock = (dx_n <= 0.024 and dy_n <= 0.030)
    containment = 0.3 * contain_factor

    # 4. 首次完全进入泊位一次性奖励
    dock_enter = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 5. 末段联合塑形：包含 ∧ 低速 ∧ 对齐（连续几何平均，仅 dock 内非零）
    speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align = max(abs(next_obs[10]), abs(next_obs[11]))
    align_f = (align - 0.7071) / 0.2929
    if align_f < 0.0:
        align_f = 0.0
    if align_f > 1.0:
        align_f = 1.0
    slow_f = 1.0 / (1.0 + 8.0 * speed)
    joint = (contain_factor * slow_f * align_f) ** (1.0 / 3.0)
    settle_shape = 0.4 * joint

    # 6. 停稳谓词与连续 10 步计数
    slow = speed < 0.05
    aligned = align >= 0.866
    settled = in_dock and slow and aligned
    settle_bonus = 1.0 if settled else 0.0
    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # 7. 安全：只惩罚超出阈值的闭接近速度（hinge，正常推箱不触发）
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along = crate_vx * next_obs[2] + crate_vy * next_obs[3]
    closing = next_obs[4] * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    excess = closing - 0.3
    if excess < 0.0:
        excess = 0.0
    roughness = -0.15 * contact * excess

    hard_hit = 0.0
    if contact > 0.5 and closing > 1.2:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # 8. 轻量动作与时间成本
    action_cost = -0.0005 * (action[0] ** 2 + action[1] ** 2)
    time_cost = -0.003

    # 9. 货箱世界坐标恢复（用于边界）
    cart_x = next_obs[0] * 5.0
    cart_y = next_obs[1] * 4.0
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    cos_h = next_obs[2]
    sin_h = next_obs[3]
    crate_x = cart_x + rel_x * cos_h - rel_y * sin_h
    crate_y = cart_y + rel_x * sin_h + rel_y * cos_h
    crate_nx = crate_x / 5.0
    crate_ny = crate_y / 4.0

    # 10. 边界守卫：饱和形式，单源上限 7.5，避免离群爆炸
    boundary_guard = 0.0
    cart_margin = max(abs(next_obs[0]), abs(next_obs[1]))
    if cart_margin > 0.9:
        over_c = cart_margin - 0.9
        if over_c > 0.3:
            over_c = 0.3
        boundary_guard += -25.0 * over_c
    crate_margin = max(abs(crate_nx), abs(crate_ny))
    if crate_margin > 0.9:
        over_k = crate_margin - 0.9
        if over_k > 0.3:
            over_k = 0.3
        boundary_guard += -25.0 * over_k

    terminal_failure = 0.0
    cart_out = (abs(next_obs[0]) > 1.05 or abs(next_obs[1]) > 1.05)
    crate_out = (abs(crate_nx) > 1.05 or abs(crate_ny) > 1.05)
    if (cart_out or crate_out or _HARD_HITS[0] >= 3) and not _FAILED[0]:
        _FAILED[0] = True
        terminal_failure = -100.0

    total = (progress + approach_cargo + containment + dock_enter + settle_shape +
             settle_bonus + terminal_success + roughness + hard_hit +
             action_cost + time_cost + boundary_guard + terminal_failure)

    # 自检①（什么都不做）：total ≈ -0.003（仅时间成本）
    # 自检②（稳定把货箱推向泊位，closing<0.3）：progress>0、containment/settle_shape=0 或非负，
    #         罚项仅 action_cost(~0)+time_cost(-0.003)+roughness(0)，total ≈ +0.01 以上 > ①
    # 自检③：containment/settle_shape 均非负，低速对齐只在 dock 内激活，不会惩罚推动动作
    # 自检④：边界守卫单源被截断在 7.5，消除了旧版 −255 的离群回合

    components = {
        "progress": progress,
        "approach_cargo": approach_cargo,
        "containment": containment,
        "dock_enter": dock_enter,
        "settle_shape": settle_shape,
        "settle_bonus": settle_bonus,
        "terminal_success": terminal_success,
        "roughness": roughness,
        "hard_hit": hard_hit,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "boundary_guard": boundary_guard,
        "terminal_failure": terminal_failure,
    }
    return float(total), components