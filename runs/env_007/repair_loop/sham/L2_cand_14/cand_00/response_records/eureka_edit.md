# Response Record

**分析**：训练反馈显示任务分数极低（-1.29）、20/20 全部 truncation，说明策略从未成功停靠；`success_event`/`enter_event`/`crate_dock_progress` 的激活率几乎为 0，主进度信号太稀疏（只有"更接近"才给分，导致一旦停滞或后退就完全没有梯度）；`bounds_guard` 逻辑写反（越界时反而为正，且激活率异常），`gentleness` 偶发巨大负值（-366）压过一切，说明接触惩罚尺度爆炸；`dock_quality_gate` 依赖 `progress` 导致大部分时间为 0。改进方向：把主进度改为**每步都有的稠密势能差信号**（对距离做 bounded 势能，保证靠近即有正梯度、远离有负梯度而非零），朝向与静止做成**连续门控因子**乘到主信号上而非独立稀疏事件；`gentleness` 改为有界、仅在接近泊位且高速接近时生效的软惩罚；修正 `bounds_guard` 符号与尺度；保留一次性进入/成功事件但降低权重避免主导。核心自检：推箱使 dist 下降 → 势能差为正 → 单步总奖励 > 静止不动（势能差≈0）。

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
    if dist > 2.0:
        dist = 2.0

    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # 货箱朝向误差（rad），<30deg 视为对齐
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    ang_err = (sh * sh + (ch - 1.0) * (ch - 1.0)) ** 0.5
    if ang_err > 2.0:
        ang_err = 2.0
    align = 1.0 - ang_err / 0.5236
    if align < 0.0:
        align = 0.0

    # 货箱速度（m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 - crate_speed / 0.05
    if slow < 0.0:
        slow = 0.0

    dock_state = 1.0 if (inside > 0.5 and align > 0.0 and crate_speed < 0.05) else 0.0

    # ---------- 完成事件（连续 10 步）----------
    if dock_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 60.0

    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 15.0

    # ---------- 稠密势能差（每步都有梯度，靠近为正、远离为负）----------
    # Phi = 1/(1+k*dist)，dist 越小 Phi 越大；用 (Phi_next - Phi_prev) 作为进度
    phi_next = 1.0 / (1.0 + 4.0 * dist)
    phi_prev = 1.0 / (1.0 + 4.0 * _PREV_DIST[0])
    _PREV_DIST[0] = dist
    progress = phi_next - phi_prev  # 靠近为正，远离为负

    # 朝向门控：对齐越好，进度信号越强（但不对齐时仍保留基础梯度）
    align_gate = 0.35 + 0.65 * align
    crate_dock_progress = 40.0 * progress * align_gate

    # ---------- 联合停靠质量（连续门控乘到主信号，不单独按步计分）----------
    # 越接近泊位、越对齐、越慢，门控越高；作为对"进入并稳定"的额外引导
    near = 1.0 - dist / 0.12
    if near < 0.0:
        near = 0.0
    dock_quality_gate = 30.0 * near * align * slow * progress

    # ---------- 轻柔度：接近时对高速接近做有界软惩罚 ----------
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    # 只在接近泊位时启用，且用有界形式避免尺度爆炸
    near_soft = 1.0 - dist / 0.25
    if near_soft < 0.0:
        near_soft = 0.0
    close_ratio = closing / (1.0 + closing)
    gentleness = -3.0 * contact * near_soft * close_ratio

    # ---------- 越界守卫（越界为负）----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    bounds_guard = 0.0
    for pos in (abs(cx), abs(cy)):
        if pos > 0.95:
            bounds_guard -= 40.0 * (pos - 0.95)
    crate_world_x = float(next_obs[6]) * 3.0 * float(obs[2]) - float(next_obs[7]) * 3.0 * float(obs[3]) + cx * 5.0
    crate_world_y = float(next_obs[6]) * 3.0 * float(obs[3]) + float(next_obs[7]) * 3.0 * float(obs[2]) + cy * 4.0
    for pos in (abs(crate_world_x) / 5.0, abs(crate_world_y) / 4.0):
        if pos > 0.95:
            bounds_guard -= 40.0 * (pos - 0.95)
    if bounds_guard < -60.0:
        bounds_guard = -60.0

    components = {
        "crate_dock_progress": crate_dock_progress,
        "dock_quality_gate": dock_quality_gate,
        "gentleness": gentleness,
        "bounds_guard": bounds_guard,
        "success_event": success_event,
        "enter_event": enter_event,
    }
    total = crate_dock_progress + dock_quality_gate + gentleness + bounds_guard + success_event + enter_event
    return (float(total), components)
```
