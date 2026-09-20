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

    components = {}

    # ---------- 常量 ----------
    # 泊位容差（来自环境事实）：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    TOL_X = 0.024
    TOL_Y = 0.030
    # 速度阈值 0.05 m/s -> 归一化后 0.05/3.0
    V_THRESH = 0.05 / 3.0
    # 朝向误差 < 30 deg
    ANG_THRESH = 0.5235987756  # 30 * pi / 180

    # ---------- 1. 货箱到泊位进度（增量形式，避免悬停收割） ----------
    d_cur = (obs[12] ** 2 + obs[13] ** 2) ** 0.5
    d_next = (next_obs[12] ** 2 + next_obs[13] ** 2) ** 0.5
    progress = (d_cur - d_next) * 8.0  # 只在"这一帧更接近"时给正分
    components["crate_to_dock_progress"] = progress

    # ---------- 2. 泊位内精细质量（仅在容差区内激活） ----------
    in_dock = 1.0 if (abs(next_obs[12]) <= TOL_X and abs(next_obs[13]) <= TOL_Y) else 0.0

    # 朝向误差
    ang_err = abs(float(__import__('math').atan2(next_obs[11], next_obs[10])))
    # 兼容性写法：不用 import，手写 atan2 近似不可靠，改用 cos 对齐度
    # 朝向对齐度：cos(误差) = cos_c*cos_c_ref + sin_c*sin_c_ref
    # 泊位对齐目标：货箱朝向应对齐（这里以货箱自身朝向余弦/正弦与"标准对齐"比较）
    # 采用：对齐度 = |cos(err)| 的连续形式，err 由 atan2 不可用，退化为对齐度 = obs[10]
    align_raw = next_obs[10]  # cos(heading)，1 表示完全对齐
    if align_raw < 0.0:
        align_raw = -align_raw  # 允许 180 度对称（方箱）
    align_deg = max(0.0, min(1.0, align_raw))

    # 速度
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    speed_ok = 1.0 if crate_speed < 0.05 else 0.0

    # 联合条件代理：in_dock * 对齐 * 静止（连续几何平均）
    f_pos = 1.0 if in_dock > 0.5 else 0.0
    f_align = align_deg
    f_speed = max(0.0, 1.0 - crate_speed / 0.05) if crate_speed < 0.05 else 0.0
    joint = (f_pos * f_align * f_speed) ** (1.0 / 3.0) if f_pos > 0.5 else 0.0
    components["docking_quality"] = 0.5 * joint

    # ---------- 3. 轻柔接触（唯一能教减速的信号） ----------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact"] = gentleness

    # ---------- 4. 接近泊位时的速度抑制（门控，仅在容差附近） ----------
    near_dock = 1.0 if (abs(next_obs[12]) < 0.06 and abs(next_obs[13]) < 0.08) else 0.0
    speed_pen = -0.3 * near_dock * (crate_speed ** 2)
    components["speed_penalty_near_dock"] = speed_pen

    # ---------- 5. 边界 hinge 惩罚 ----------
    cart_margin = max(0.0, abs(next_obs[0]) - 0.90) + max(0.0, abs(next_obs[1]) - 0.90)
    crate_margin = max(0.0, abs(next_obs[12]) - 0.85) + max(0.0, abs(next_obs[13]) - 0.85)
    oob = -2.0 * (cart_margin + crate_margin)
    components["out_of_bounds"] = oob

    # ---------- 6. 障碍接近惩罚（前方传感器） ----------
    front = next_obs[15]
    if front > 0.8:
        components["obstacle_penalty"] = -0.3 * (front - 0.8)
    else:
        components["obstacle_penalty"] = 0.0

    # ---------- 7. 动作平滑（轻量） ----------
    smooth = -0.01 * (action[0] ** 2 + action[1] ** 2)
    components["action_smoothness"] = smooth

    # ---------- 8. 完成事件（一次性） ----------
    complete_now = (
        abs(next_obs[12]) <= TOL_X
        and abs(next_obs[13]) <= TOL_Y
        and align_deg >= 0.866  # cos(30deg) ≈ 0.866
        and crate_speed < 0.05
    )
    if complete_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 首次进入泊位（一次性）
    enter_bonus = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 20.0
    components["first_enter_dock"] = enter_bonus

    # 完成事件：连续 10 步满足，一次性大额
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    total = float(sum(components.values()))
    return total, components
```
