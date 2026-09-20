# Response Record

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测（obs[18] 单调递增，重置时回落） ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- 几何量（全部来自声明维度） ----------
    # 货箱中心到泊位中心的有符号偏移（归一化）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # 泊位几何阈值（环境事实）：|dx|<=0.024 且 |dy|<=0.030
    # 用连续因子表示"进入容差区"的程度
    fx = 1.0 - abs(dx) / 0.024
    if fx < 0.0:
        fx = 0.0
    if fx > 1.0:
        fx = 1.0
    fy = 1.0 - abs(dy) / 0.030
    if fy < 0.0:
        fy = 0.0
    if fy > 1.0:
        fy = 1.0
    inside_factor = fx * fy  # 1 表示完全进入

    # 货箱朝向误差（cos 越大越对齐）
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 朝向对齐因子：30° 对应 cos(30°)=0.866
    align_factor = (crate_cos - 0.866) / (1.0 - 0.866)
    if align_factor < 0.0:
        align_factor = 0.0
    if align_factor > 1.0:
        align_factor = 1.0

    # 货箱速度（世界系，m/s）
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    # 速度达标因子：< 0.05 m/s 为达标，0.5 m/s 以上为 0
    speed_factor = (0.5 - crate_speed) / (0.5 - 0.05)
    if speed_factor < 0.0:
        speed_factor = 0.0
    if speed_factor > 1.0:
        speed_factor = 1.0

    # ---------- 组件 1：货箱向泊位的增量进展（主信号，增量形式） ----------
    # 用归一化距离的减少量作为进展，避免"停在附近一直收分"
    if _PREV_DIST[0] < 0.0:
        progress = 0.0
    else:
        progress = _PREV_DIST[0] - dist
    _PREV_DIST[0] = dist
    # 限幅，避免单步跳变主导
    if progress > 0.05:
        progress = 0.05
    if progress < -0.05:
        progress = -0.05
    crate_progress = 40.0 * progress

    # ---------- 组件 2：进入泊位容差区的联合质量（门控式，仅接近时激活） ----------
    # 只在货箱已经相当接近泊位时才给质量信号，避免远处刷分
    near_gate = 1.0 - dist / 0.2
    if near_gate < 0.0:
        near_gate = 0.0
    if near_gate > 1.0:
        near_gate = 1.0
    # 联合条件：进入 + 对齐 + 慢速（几何平均，避免塌缩）
    jc = (inside_factor * align_factor * speed_factor) ** (1.0 / 3.0)
    docking_quality = 6.0 * near_gate * jc

    # ---------- 组件 3：接触轻柔度（核心技能信号） ----------
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 组件 4：接近泊位时的货箱速度抑制（仅接近时激活） ----------
    # 注意：这是"接近泊位且货箱还快"的惩罚，不惩罚正常推箱
    overspeed = crate_speed - 0.05
    if overspeed < 0.0:
        overspeed = 0.0
    if overspeed > 1.0:
        overspeed = 1.0
    slowdown = -2.0 * near_gate * overspeed

    # ---------- 组件 5：越界软惩罚（hinge，仅在接近边界时） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.9:
        oob += (abs(cart_x) - 0.9) ** 2
    if abs(cart_y) > 0.9:
        oob += (abs(cart_y) - 0.9) ** 2
    out_of_bounds = -20.0 * oob

    # ---------- 组件 6：动作平滑（轻量，不压制推进） ----------
    a0 = float(action[0])
    a1 = float(action[1])
    smoothness = -0.02 * (a0 * a0 + a1 * a1)

    # ---------- 完成条件（从 obs 显式推断） ----------
    inside = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = crate_cos >= 0.866
    slow = crate_speed < 0.05
    if inside and aligned and slow:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 一次性"首次进入泊位"奖励
    enter_bonus = 0.0
    if inside and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 40.0

    # 一次性完成事件奖励（连续 10 步满足）
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    components = {
        "crate_progress": crate_progress,
        "docking_quality": docking_quality,
        "gentleness": gentleness,
        "slowdown_near_dock": slowdown,
        "out_of_bounds": out_of_bounds,
        "smoothness": smoothness,
        "enter_bonus": enter_bonus,
        "success_event": success_event,
    }
    total = (
        crate_progress
        + docking_quality
        + gentleness
        + slowdown
        + out_of_bounds
        + smoothness
        + enter_bonus
        + success_event
    )
    return float(total), components
```
