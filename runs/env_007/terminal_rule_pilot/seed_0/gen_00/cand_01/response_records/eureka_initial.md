# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 泊位几何阈值（来自环境事实）----
    # |obs[12]| <= 0.024 且 |obs[13]| <= 0.030 视为货箱完全进入泊位
    # 朝向误差 < 30deg -> cos(err) > cos(30deg) = 0.8660254
    # 速度 < 0.05 m/s -> 归一化后 (0.05/3.0) = 0.0166667

    dock_x_ok = 1.0 - min(1.0, abs(next_obs[12]) / 0.024)
    dock_y_ok = 1.0 - min(1.0, abs(next_obs[13]) / 0.030)

    # 货箱朝向与泊位对齐程度：泊位朝向假定为 0（cos=1, sin=0）
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    # 归一化朝向误差因子：cos(err) 映射到 [0,1]
    heading_factor = max(0.0, min(1.0, (crate_cos + 1.0) * 0.5))

    # 货箱速度（归一化到 m/s 再压缩）
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    speed_norm = crate_speed / 3.0
    speed_factor = 1.0 / (1.0 + 40.0 * speed_norm)

    # ---- 组件 1：货箱向泊位的增量进展（主进度信号）----
    prev_dist = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    next_dist = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress_delta = prev_dist - next_dist
    # 只奖励靠近，远离给轻微惩罚（同向于任务）
    crate_progress = 6.0 * progress_delta

    # ---- 组件 2：泊位内停靠质量（位置 + 朝向 + 速度联合门控，仅在接近泊位时激活）----
    # 接近度门：距离泊位中心越近，门越开
    near_gate = max(0.0, 1.0 - next_dist / 0.5)
    dock_quality = 3.0 * near_gate * (dock_x_ok * dock_y_ok) ** 0.5 * heading_factor * speed_factor

    # ---- 组件 3：完成事件奖励（一次性大额）----
    # 完成条件：货箱完全在泊位内 + 朝向对齐 + 速度足够低
    in_dock = 1.0 if (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030) else 0.0
    aligned = 1.0 if crate_cos >= 0.8660254 else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0
    docked_success = 1.0 if (in_dock > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0
    completion_bonus = 2000.0 * docked_success

    # ---- 组件 4：软接触/硬碰撞抑制（仅在接触时且速度突变明显时轻微惩罚）----
    contact = next_obs[14]
    cart_speed = abs(next_obs[4]) * 3.0
    # 接触且速度较大时视为潜在硬碰撞风险，给轻微惩罚
    contact_risk = 0.0
    if contact > 0.5:
        contact_risk = -0.5 * min(1.0, crate_speed / 1.0) * min(1.0, cart_speed / 1.5)

    # ---- 组件 5：越界防护（hinge，仅接近边界时生效）----
    # 小车位置归一化到 [-1,1]，边界在 |obs| = 1.0
    cart_edge = max(abs(next_obs[0]), abs(next_obs[1]))
    out_penalty = 0.0
    if cart_edge > 0.85:
        out_penalty = -2.0 * (cart_edge - 0.85) / 0.15

    # ---- 组件 6：动作平滑（轻量，避免抑制必要推动）----
    smooth_penalty = -0.05 * (action[0] ** 2 + action[1] ** 2)

    components = {
        "crate_progress": crate_progress,
        "dock_quality": dock_quality,
        "completion_bonus": completion_bonus,
        "contact_risk": contact_risk,
        "out_of_bounds": out_penalty,
        "action_smoothness": smooth_penalty,
    }

    total_reward = (
        crate_progress
        + dock_quality
        + completion_bonus
        + contact_risk
        + out_penalty
        + smooth_penalty
    )

    return float(total_reward), components
```
