# Response Record

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- 泊位几何（来自环境事实的容差） ----------
    dx = float(next_obs[12])   # 有符号 x 偏移 / 半宽
    dy = float(next_obs[13])   # 有符号 y 偏移 / 半高
    tol_x = 0.024
    tol_y = 0.030

    # 归一化距离度量（0 = 泊位中心）
    dist = (dx * dx + dy * dy) ** 0.5

    # ---------- 主进度：货箱向泊位靠近（增量形式） ----------
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    delta = _PREV_DIST[0] - dist          # 正 = 更接近泊位
    _PREV_DIST[0] = dist

    # 悬停陷阱防护：只在"更接近"时给分，且做有界压缩
    if delta > 0.0:
        progress = 8.0 * delta / (1.0 + 2.0 * delta)
    else:
        progress = 0.0

    # ---------- 完成条件（从 obs 显式推断） ----------
    inside = 1.0 if (abs(dx) <= tol_x and abs(dy) <= tol_y) else 0.0

    # 货箱朝向误差
    cos_h = float(next_obs[10])
    sin_h = float(next_obs[11])
    heading_err = (sin_h * sin_h + (cos_h - 1.0) * (cos_h - 1.0)) ** 0.5
    aligned = 1.0 if heading_err < 0.5 else 0.0   # < ~30°

    # 货箱速度
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    if inside > 0.5 and aligned > 0.5 and slow > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 一次性成功事件
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 一次性首次进入泊位奖励
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 5.0

    # ---------- 接近泊位时的停稳引导（门控，仅靠近时激活） ----------
    # 只有货箱已经比较接近泊位时才抑制速度，避免阻碍推进
    near_gate = max(0.0, 1.0 - dist / 0.25)
    if near_gate > 0.0:
        dock_speed_pen = -0.6 * near_gate * crate_speed
    else:
        dock_speed_pen = 0.0

    # 朝向对齐 shaping（仅靠近泊位时）
    if near_gate > 0.0:
        align_bonus = 0.3 * near_gate * (1.0 - heading_err)
    else:
        align_bonus = 0.0

    # ---------- 接触轻柔度（核心技能信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 越界惩罚（hinge，仅在靠近边界时） ----------
    cart_x = abs(float(next_obs[0]))
    cart_y = abs(float(next_obs[1]))
    oob = 0.0
    if cart_x > 0.9:
        oob -= 2.0 * (cart_x - 0.9)
    if cart_y > 0.9:
        oob -= 2.0 * (cart_y - 0.9)

    # 货箱越界风险（用其到泊位偏移的极端值间接判断，泊位在仓库内）
    crate_edge = abs(dx) + abs(dy)
    if crate_edge > 1.6:
        oob -= 1.0 * (crate_edge - 1.6)

    # ---------- 动作平滑（轻量，避免压制推动） ----------
    smooth = -0.02 * (float(action[0]) ** 2) - 0.02 * (float(action[1]) ** 2)

    components = {
        "progress": progress,
        "enter_event": enter_event,
        "success_event": success_event,
        "dock_speed_pen": dock_speed_pen,
        "align_bonus": align_bonus,
        "gentleness": gentleness,
        "oob": oob,
        "smooth": smooth,
    }

    total = (
        progress
        + enter_event
        + success_event
        + dock_speed_pen
        + align_bonus
        + gentleness
        + oob
        + smooth
    )

    return (float(total), components)
```
