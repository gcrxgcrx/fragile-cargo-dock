# Response Record

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID_SUCCESS = [False]
_ENTERED = [False]
_FAILED = [False]
_HARD_HITS = [0]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 自检记录（估算）：
    # R_idle（什么都不做）        ≈ -0.002
    # R_push（正常推箱前进）      ≈ +0.05 ~ +0.10
    # R_settled（泊位内停稳）     ≈ +20（terminal_success 被单步裁剪后）
    # R_push - R_idle ≈ 0.05~0.10，大于正常推进时罚项最大量级（time_cost 0.002 + action_cost 0.001）。
    # R_settled > R_push。
    # 自检③：closing=1.0 时 roughness=-0.12；closing=0.05 时 roughness=-0.006；差距≈0.114，
    #         与推进项单步典型值同量级或更大。
    # 自检④：同一停稳状态连续调用，每步都有 terminal_success（+300，裁剪后+20），差值恒定。

    # 回合边界检测（obs[18] 单调递增，重置时回落）
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID_SUCCESS[0] = False
        _ENTERED[0] = False
        _FAILED[0] = False
        _HARD_HITS[0] = 0
    _PREV_T[0] = t

    # ---------- 位置派生 ----------
    next_cart_x = next_obs[0] * 5.0
    next_cart_y = next_obs[1] * 4.0

    rel_x_prev = obs[6] * 3.0
    rel_y_prev = obs[7] * 3.0
    dist_cart_crate_prev = (rel_x_prev * rel_x_prev + rel_y_prev * rel_y_prev) ** 0.5

    rel_x_next = next_obs[6] * 3.0
    rel_y_next = next_obs[7] * 3.0
    dist_cart_crate_next = (rel_x_next * rel_x_next + rel_y_next * rel_y_next) ** 0.5

    # 小车 -> 货箱：本帧距离缩短量（米，有符号对称）
    approach_cargo = (dist_cart_crate_prev - dist_cart_crate_next) * 1.0

    dock_x_prev = obs[12] * 5.0
    dock_y_prev = obs[13] * 4.0
    dist_dock_prev = (dock_x_prev * dock_x_prev + dock_y_prev * dock_y_prev) ** 0.5

    dock_x_next = next_obs[12] * 5.0
    dock_y_next = next_obs[13] * 4.0
    dist_dock_next = (dock_x_next * dock_x_next + dock_y_next * dock_y_next) ** 0.5

    # 货箱 -> 泊位：本帧距离缩短量（米，有符号对称）
    progress = (dist_dock_prev - dist_dock_next) * 1.0

    # ---------- dock_enter：首次完全进入泊位容差（整局一次） ----------
    dock_enter = 0.0
    if (not _ENTERED[0]) and dist_dock_next < 0.25:
        _ENTERED[0] = True
        dock_enter = 5.0

    # ---------- 停稳谓词（泊位内 + 对齐 + 慢） ----------
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    aligned = next_obs[10] > 0.866          # 朝向误差 < 30 度
    settled = (dist_dock_next < 0.25) and (crate_speed < 0.05) and aligned

    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    if settled and _STREAK[0] >= 10:
        _PAID_SUCCESS[0] = True

    # 谓词成立就每步发放，不因连续计数达标或已发放而关闭
    terminal_success = 300.0 if settled else 0.0

    # ---------- 接触 / 接近速度代理 ----------
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * next_obs[2] + crate_vy * next_obs[3]
    closing = next_obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0

    # roughness：只在接触且正在接近时惩罚，抑制高速撞击
    roughness = -0.12 * contact * closing

    # hard_hit：单步硬冲击代理（仅在高速接近接触时触发）
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.2:
        hard_hit = -0.5
        if not _FAILED[0]:
            _HARD_HITS[0] += 1
            if _HARD_HITS[0] >= 3:
                _FAILED[0] = True

    # ---------- 越界与 terminal_failure ----------
    crate_world_x = next_cart_x + rel_x_next * next_obs[2] - rel_y_next * next_obs[3]
    crate_world_y = next_cart_y + rel_x_next * next_obs[3] + rel_y_next * next_obs[2]

    terminal_failure = 0.0
    if not _FAILED[0]:
        out = False
        if abs(next_obs[0]) > 1.05 or abs(next_obs[1]) > 1.05:
            out = True
        if abs(crate_world_x) > 5.25 or abs(crate_world_y) > 4.2:
            out = True
        if out:
            _FAILED[0] = True
            terminal_failure = -100.0

    # ---------- 动作代价 / 时间代价 ----------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------- 连续边界守卫 ----------
    bounds_penalty = 0.0
    if abs(next_obs[0]) > 0.9:
        bounds_penalty -= 10.0 * (abs(next_obs[0]) - 0.9)
    if abs(next_obs[1]) > 0.9:
        bounds_penalty -= 10.0 * (abs(next_obs[1]) - 0.9)
    if abs(crate_world_x) > 4.8:
        bounds_penalty -= 10.0 * (abs(crate_world_x) - 4.8)
    if abs(crate_world_y) > 3.8:
        bounds_penalty -= 10.0 * (abs(crate_world_y) - 3.8)

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
        "bounds_penalty": bounds_penalty,
    }

    total = 0.0
    for key in components:
        total += components[key]

    return float(total), components
```
