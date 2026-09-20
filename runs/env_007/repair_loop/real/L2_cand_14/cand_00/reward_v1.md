分析：任务分数为负且 20/20 全部截断，说明货箱从未真正进入泊位——`enter_event`、`dock_quality_gate`、`success_event` 全程为 0，唯一有效信号 `crate_dock_progress` 又只在 0.3% 步数触发（增量被 `progress<0` 截断为 0），信号过于稀疏且量级太小。`bounds_guard` 早期主导（-9.2）但后期归零，说明它只是噪声而非有效约束。核心问题：接近信号被"只奖励前进"的截断抹平，且没有把货箱推向泊位的持续性引导；同时缺少朝向/速度的稠密 shaping。改进方向：① 用稠密的距离势能（bounded）替代稀疏 delta，保证每步有梯度；② 用几何平均的联合条件软代理持续给分；③ 把朝向对齐做成稠密 shaping；④ 保留边界 hinge、轻柔门控但不压制推动；⑤ 速度惩罚仅在泊位附近且用门控，避免惩罚推进。

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = 1.0
    _PREV_T[0] = t

    # ---------- 几何量 ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # 货箱朝向误差（rad）
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    ang_err = (sh * sh + (ch - 1.0) * (ch - 1.0)) ** 0.5
    if ang_err > 2.0:
        ang_err = 2.0
    # 对齐度：误差 0 -> 1，误差 >= 0.5236(30deg) -> 0
    align = 1.0 - ang_err / 0.5236
    if align < 0.0:
        align = 0.0

    # 货箱速度（m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 静止度：速度 0 -> 1，速度 >= 0.25 -> 0（宽区间，避免过早惩罚推进）
    slow_soft = 1.0 - crate_speed / 0.25
    if slow_soft < 0.0:
        slow_soft = 0.0
    # 严格静止（用于完成判定）
    slow_strict = 1.0 - crate_speed / 0.05
    if slow_strict < 0.0:
        slow_strict = 0.0

    dock_state = 1.0 if (inside > 0.5 and align > 0.0 and crate_speed < 0.05) else 0.0

    # ---------- 完成事件（连续 10 步）----------
    if dock_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 40.0

    # ---------- 到达泊位后的停靠质量奖励（进入泊位后持续给分，鼓励静止+对齐）----------
    if dock_state > 0.5:
        # 完成态：仅保留一次性事件 + 少量停靠保持奖励
        hold_bonus = 1.0 * align * slow_strict
        components = {
            "crate_dock_progress": 0.0,
            "dock_quality_gate": hold_bonus,
            "gentleness": 0.0,
            "bounds_guard": 0.0,
            "success_event": success_event,
            "enter_event": enter_event,
        }
        total = success_event + enter_event + hold_bonus
        return (float(total), components)

    # ---------- 稠密接近势能（每步有梯度，不截断）----------
    # 距离归一化：dist 通常在 [0, ~1.5]，用 bounded 压缩
    near_energy = 1.0 / (1.0 + 6.0 * dist)  # dist=0 -> 1, dist 大 -> 0
    # 用对齐门控（对齐越好，接近越有价值），但不为 0
    crate_dock_progress = 1.2 * near_energy * (0.4 + 0.6 * align)

    # ---------- 增量接近奖励（保留，鼓励持续靠近）----------
    progress = _PREV_DIST[0] - dist
    if progress < 0.0:
        progress = 0.0
    _PREV_DIST[0] = dist
    progress_bonus = 3.0 * progress * (0.4 + 0.6 * align)

    # ---------- 联合条件软代理（几何平均，避免塌缩）----------
    # 三个连续因子：接近、对齐、静止
    prox_f = 1.0 - dist / 0.20
    if prox_f < 0.0:
        prox_f = 0.0
    # 几何平均，任一项非零即有梯度
    joint = (prox_f * align * slow_soft) ** (1.0 / 3.0)
    dock_quality_gate = 1.5 * joint

    # ---------- 朝向对齐稠密 shaping（货箱朝向误差越小越好）----------
    align_shaping = 0.3 * align

    # ---------- 轻柔度：接触中且快速接近时惩罚（门控，不压制正常推进）----------
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    # 只在接近泊位时启用轻柔约束，避免阻碍远距离推进
    near_gate = 1.0 - dist / 0.30
    if near_gate < 0.0:
        near_gate = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.3 * contact * closing * near_gate

    # ---------- 越界守卫（hinge，仅边界附近生效）----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    bounds_guard = 0.0
    for pos in (abs(cx), abs(cy)):
        if pos > 0.95:
            bounds_guard -= 30.0 * (pos - 0.95)
    crate_world_x = float(next_obs[6]) * 3.0 * float(obs[2]) - float(next_obs[7]) * 3.0 * float(obs[3]) + cx * 5.0
    crate_world_y = float(next_obs[6]) * 3.0 * float(obs[3]) + float(next_obs[7]) * 3.0 * float(obs[2]) + cy * 4.0
    for pos in (abs(crate_world_x) / 5.0, abs(crate_world_y) / 4.0):
        if pos > 0.95:
            bounds_guard -= 30.0 * (pos - 0.95)
    if bounds_guard < -60.0:
        bounds_guard = -60.0

    components = {
        "crate_dock_progress": crate_dock_progress + progress_bonus + align_shaping,
        "dock_quality_gate": dock_quality_gate,
        "gentleness": gentleness,
        "bounds_guard": bounds_guard,
        "success_event": success_event,
        "enter_event": enter_event,
    }
    total = crate_dock_progress + progress_bonus + align_shaping + dock_quality_gate + gentleness + bounds_guard + success_event + enter_event
    return (float(total), components)
```