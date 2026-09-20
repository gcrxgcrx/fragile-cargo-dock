# Response Record

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 几何与状态量 ----------
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dx_new = float(next_obs[12])
    dy_new = float(next_obs[13])
    dx_old = float(obs[12])
    dy_old = float(obs[13])

    # 归一化距离（用容差归一，便于门控）
    dist_new = (dx_new * dx_new + dy_new * dy_new) ** 0.5
    dist_old = (dx_old * dx_old + dy_old * dy_old) ** 0.5

    # 货箱速度（世界系，恢复为 m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差
    crate_ang = 0.0
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    crate_ang = (sh * sh + ch * ch) ** 0.5
    # 朝向对齐度（cos 误差），货箱朝向与泊位朝向一致（泊位朝向视为 +x 轴）
    cos_err = ch  # 归一化朝向的余弦；对齐时接近 1
    if cos_err < 0.0:
        cos_err = 0.0
    # 30° 对应 cos ≈ 0.866；用线性门控
    align_gate = (cos_err - 0.866) / (1.0 - 0.866)
    if align_gate < 0.0:
        align_gate = 0.0
    if align_gate > 1.0:
        align_gate = 1.0

    # 完成条件：完全进入 + 对齐 + 慢
    inside = 1.0 if (abs(dx_new) <= 0.024 and abs(dy_new) <= 0.030) else 0.0
    aligned = 1.0 if cos_err >= 0.866 else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0
    complete_state = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if complete_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 轻柔度信号 ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 组件 ----------
    components = {}

    # 1) 主进度：增量形式，仅当"这一帧更接近"时给分
    progress = 0.0
    if dist_new < dist_old:
        progress = 2.0 * (dist_old - dist_new)
    # 门控：对齐度越高，靠近收益越大（对齐是推进的必要条件，不单独给分）
    progress = progress * (0.3 + 0.7 * align_gate)
    components["crate_to_dock_progress"] = progress

    # 2) 轻柔接触惩罚（唯一教减速的信号）
    components["soft_contact"] = gentleness

    # 3) 接近泊位时的速度抑制（仅在容差外、接近泊位时生效，作为惩罚）
    near_dock = 0.0
    if dist_new < 0.15 and inside < 0.5:
        near_dock = 1.0
    speed_pen = -0.5 * near_dock * crate_speed
    components["crate_speed_penalty_near_dock"] = speed_pen

    # 4) 越界惩罚（hinge，仅接近边界时生效）
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    lim_x = 0.95
    lim_y = 0.95
    if abs(cart_x) > lim_x:
        oob -= (abs(cart_x) - lim_x) * 5.0
    if abs(cart_y) > lim_y:
        oob -= (abs(cart_y) - lim_y) * 5.0
    # 货箱越界（用恢复的货箱世界坐标近似：小车位置 + 车体系相对位置旋转）
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    cc = float(next_obs[2])
    ss = float(next_obs[3])
    crate_wx = cart_x * 5.0 + rel_x * cc - rel_y * ss
    crate_wy = cart_y * 4.0 + rel_x * ss + rel_y * cc
    if abs(crate_wx) > 4.8:
        oob -= (abs(crate_wx) - 4.8) * 2.0
    if abs(crate_wy) > 3.8:
        oob -= (abs(crate_wy) - 3.8) * 2.0
    components["out_of_bounds"] = oob

    # 5) 首次进入泊位：一次性事件
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 40.0
    components["first_enter_dock"] = enter_event

    # 6) 完成事件：一次性
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["docked_success"] = success_event

    # ---------- 完成状态下，除一次性事件外全部归零 ----------
    if complete_state > 0.5:
        components["crate_to_dock_progress"] = 0.0
        components["soft_contact"] = 0.0
        components["crate_speed_penalty_near_dock"] = 0.0
        components["out_of_bounds"] = 0.0

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)
```
