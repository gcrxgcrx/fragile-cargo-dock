# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 读取观测（只使用已声明索引）----
    # 货箱到泊位的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱速度（归一化，*3.0 恢复 m/s）
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差
    crate_ang = 0.0
    if (obs[10] != 0.0) or (obs[11] != 0.0):
        crate_ang = (obs[11] * obs[11]) ** 0.5
        if obs[10] < 0.0:
            crate_ang = 2.0 - crate_ang
    cos_err = obs[10]
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    heading_error = (2.0 - 2.0 * cos_err) ** 0.5  # 弧度近似

    # 接触标志
    contact = obs[14]

    # 障碍接近度
    sf = obs[15]
    sl = obs[16]
    sr = obs[17]

    # 时间比例
    time_frac = obs[18]

    # ---- 距离度量（归一化坐标下的欧氏距离）----
    dist_now = (dx * dx + dy * dy) ** 0.5
    dist_next = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_now - dist_next  # 正 = 更接近泊位

    # ---- 完成判据（严格取自环境事实）----
    # |obs[12]| <= 0.024 且 |obs[13]| <= 0.030 且 朝向误差 < 30deg 且 速度 < 0.05 m/s
    inside = 1.0 if ((ndx <= 0.024 and ndx >= -0.024) and (ndy <= 0.030 and ndy >= -0.030)) else 0.0
    aligned = 1.0 if heading_error < 0.5235987756 else 0.0
    still = 1.0 if crate_speed < 0.05 else 0.0
    docked_now = inside * aligned * still

    # ---- 组件 1：货箱向泊位推进（增量形式，只奖励"这一帧更接近"）----
    progress_reward = 40.0 * progress

    # ---- 组件 2：接近泊位时的对齐/静止联合门控 shaping（仅在接近时激活）----
    near_gate = 0.0
    if dist_now < 0.35:
        near_gate = (0.35 - dist_now) / 0.35
    align_factor = 0.0
    if heading_error < 0.7853981634:
        align_factor = (0.7853981634 - heading_error) / 0.7853981634
    still_factor = 0.0
    if crate_speed < 0.30:
        still_factor = (0.30 - crate_speed) / 0.30
    dock_quality = near_gate * (0.5 * align_factor + 0.5 * still_factor)
    dock_quality_reward = 2.0 * dock_quality

    # ---- 组件 3：完成事件奖励（一次性、主导）----
    complete_reward = 0.0
    if docked_now > 0.5:
        complete_reward = 3000.0

    # ---- 组件 4：货箱速度惩罚（仅在接近泊位时启用，避免阻碍推进）----
    speed_penalty = 0.0
    if dist_now < 0.25 and crate_speed > 0.05:
        excess = crate_speed - 0.05
        speed_penalty = -1.5 * excess * excess

    # ---- 组件 5：软接触惩罚（接触且货箱速度高，间接推断硬碰撞）----
    contact_penalty = 0.0
    if contact > 0.5 and crate_speed > 0.6:
        excess_v = crate_speed - 0.6
        contact_penalty = -0.5 * excess_v * excess_v

    # ---- 组件 6：越界惩罚（小车与货箱接近仓库边界）----
    oob_penalty = 0.0
    cart_x = obs[0]
    cart_y = obs[1]
    if cart_x > 0.95:
        oob_penalty = oob_penalty - 2.0 * (cart_x - 0.95) * (cart_x - 0.95)
    if cart_x < -0.95:
        oob_penalty = oob_penalty - 2.0 * (cart_x + 0.95) * (cart_x + 0.95)
    if cart_y > 0.95:
        oob_penalty = oob_penalty - 2.0 * (cart_y - 0.95) * (cart_y - 0.95)
    if cart_y < -0.95:
        oob_penalty = oob_penalty - 2.0 * (cart_y + 0.95) * (cart_y + 0.95)

    # ---- 组件 7：障碍接近惩罚（前方障碍接近时轻微抑制）----
    obstacle_penalty = 0.0
    if sf > 0.7:
        obstacle_penalty = obstacle_penalty - 0.5 * (sf - 0.7) * (sf - 0.7)

    # ---- 组件 8：动作平滑（轻量，可选）----
    smooth_penalty = -0.02 * (action[0] * action[0] + action[1] * action[1])

    total_reward = (
        progress_reward
        + dock_quality_reward
        + complete_reward
        + speed_penalty
        + contact_penalty
        + oob_penalty
        + obstacle_penalty
        + smooth_penalty
    )

    components = {
        "crate_progress": progress_reward,
        "dock_quality": dock_quality_reward,
        "dock_complete": complete_reward,
        "near_dock_speed_penalty": speed_penalty,
        "soft_contact_penalty": contact_penalty,
        "out_of_bounds_penalty": oob_penalty,
        "obstacle_penalty": obstacle_penalty,
        "action_smoothness": smooth_penalty,
    }

    return (float(total_reward), components)
```
