# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 回合边界检测（obs[18] 单调递增，重置时回落）----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---- 泊位几何（环境事实给出的容差）----
    # 泊位半宽 0.30 m -> 归一化 0.30/12.5? 卡片给出 |obs[12]|<=0.024 且 |obs[13]|<=0.030 为完全进入
    dock_x = float(next_obs[12])
    dock_y = float(next_obs[13])
    dist = (dock_x * dock_x + dock_y * dock_y) ** 0.5

    old_dock_x = float(obs[12])
    old_dock_y = float(obs[13])
    old_dist = (old_dock_x * old_dock_x + old_dock_y * old_dock_y) ** 0.5

    # ---- 货箱速度（世界系，恢复为 m/s）----
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- 朝向误差（货箱朝向 vs 泊位朝向，泊位朝向视为 0 轴）----
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    ang_err = 0.0
    if ch != 0.0 or sh != 0.0:
        # 朝向误差的绝对值近似：|sin| 反映偏离 0 轴的程度
        ang_err = abs(sh)
    align_factor = 1.0 - ang_err
    if align_factor < 0.0:
        align_factor = 0.0

    # ---- 组件 1：货箱向泊位的增量进展（increment，避免悬停收割）----
    progress = old_dist - dist
    if progress < 0.0:
        progress = 0.0
    crate_to_dock_progress = 2.0 * progress

    # ---- 组件 2：接近泊位时的对齐/减速质量（门控，只在接近泊位时激活）----
    # 门控：距离越近门越大，但只在未完成区域生效
    near_gate = 1.0 / (1.0 + 20.0 * dist)
    # 速度因子：接近泊位时低速有利（仅在 near_gate 激活下，不构成全局低速奖励）
    speed_factor = 1.0 / (1.0 + 8.0 * crate_speed)
    crate_docking_quality = 0.6 * near_gate * align_factor * speed_factor

    # ---- 组件 3：接触轻柔度（接触中且在接近时惩罚）----
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- 组件 4：越界 hinge 惩罚（小车与货箱靠近边界时）----
    out_of_bounds = 0.0
    ax = abs(float(next_obs[0]))
    ay = abs(float(next_obs[1]))
    if ax > 0.85:
        out_of_bounds -= 0.5 * (ax - 0.85)
    if ay > 0.85:
        out_of_bounds -= 0.5 * (ay - 0.85)
    # 货箱相对小车位置恢复（车体系）越界近似
    crx = abs(float(next_obs[6]))
    cry = abs(float(next_obs[7]))
    if crx > 0.9:
        out_of_bounds -= 0.3 * (crx - 0.9)
    if cry > 0.9:
        out_of_bounds -= 0.3 * (cry - 0.9)

    # ---- 组件 5：动作平滑（轻量）----
    action_smoothness = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---- 完成条件（显式从 obs 推断，容差取自环境事实）----
    inside = (abs(dock_x) <= 0.024) and (abs(dock_y) <= 0.030)
    aligned = ang_err < 0.5  # 30° -> |sin30| = 0.5
    slow = crate_speed < 0.05
    complete_now = inside and aligned and slow

    if complete_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---- 一次性事件：首次进入泊位 ----
    enter_bonus = 0.0
    if inside and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 20.0

    # ---- 一次性事件：完成（连续 10 步满足）----
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    components = {}
    components["crate_to_dock_progress"] = crate_to_dock_progress
    components["crate_docking_quality"] = crate_docking_quality
    components["soft_contact_penalty"] = gentleness
    components["out_of_bounds_penalty"] = out_of_bounds
    components["action_smoothness"] = action_smoothness
    components["enter_dock_bonus"] = enter_bonus
    components["docked_success_event"] = success_event

    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + gentleness
        + out_of_bounds
        + action_smoothness
        + enter_bonus
        + success_event
    )
    return float(total_reward), components


_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
```
