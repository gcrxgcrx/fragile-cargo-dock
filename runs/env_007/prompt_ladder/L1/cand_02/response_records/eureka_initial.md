# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 货箱到泊位的有符号偏移（归一化）----
    cur_dx = obs[12]
    cur_dy = obs[13]
    nxt_dx = next_obs[12]
    nxt_dy = next_obs[13]

    # ---- 主信号 1：货箱向泊位中心推进（delta 形式，避免悬停陷阱）----
    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5
    progress = cur_dist - nxt_dist
    # 有界压缩，防止单步极端值
    progress_bounded = progress / (1.0 + abs(progress))
    crate_to_dock_progress = 6.0 * progress_bounded

    # ---- 主信号 2：泊位内几何质量（位置 + 朝向），仅在接近泊位时激活 ----
    # 接近度门控：距离越近越接近 1
    near_gate = 1.0 / (1.0 + 12.0 * nxt_dist)

    # 位置精度（完全进入阈值 |dx|<=0.024, |dy|<=0.030）
    pos_err = (nxt_dx / 0.024) ** 2 + (nxt_dy / 0.030) ** 2
    pos_factor = 1.0 / (1.0 + pos_err)

    # 朝向对齐：货箱朝向误差（cos 分量），误差 < 30° 时 cos > 0.866
    crate_cos = next_obs[10]
    # 归一化到 [0,1]：cos=1 -> 1, cos=0.866 -> 0.866, cos<0 -> 0
    heading_factor = max(0.0, (crate_cos - 0.5) / 0.5)

    # 静止因子：货箱速度越低越好（仅在接近泊位时才有意义）
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    # 速度 < 0.05 时接近 1，速度大时接近 0
    speed_factor = 1.0 / (1.0 + (crate_speed / 0.05) ** 2)

    # 联合条件近似（几何平均，避免塌缩）
    dock_quality = (pos_factor * heading_factor * speed_factor) ** (1.0 / 3.0)
    crate_docking_quality = 4.0 * near_gate * dock_quality

    # ---- 条件信号 3：接近泊位时的货箱速度抑制（hinge，仅近距离生效）----
    near_dock = 1.0 / (1.0 + 20.0 * nxt_dist)
    speed_excess = max(0.0, crate_speed - 0.05)
    crate_speed_penalty_near_dock = -1.5 * near_dock * speed_excess

    # ---- 条件信号 4：软接触惩罚（接触 + 速度突变间接推断）----
    # 仅当接触且货箱速度高时惩罚，避免误罚正常推动
    contact = next_obs[14]
    contact_speed = crate_speed
    soft_contact_penalty = -0.5 * contact * max(0.0, contact_speed - 0.6)

    # ---- 条件信号 5：越界惩罚（小车/货箱接近仓库边界，hinge）----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    cart_margin = 0.90
    cart_oob = max(0.0, abs(cart_x) - cart_margin) + max(0.0, abs(cart_y) - cart_margin)

    # 货箱世界坐标恢复
    cart_heading_cos = next_obs[2]
    cart_heading_sin = next_obs[3]
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    crate_wx = (cart_x * 5.0) + rel_x * cart_heading_cos - rel_y * cart_heading_sin
    crate_wy = (cart_y * 4.0) + rel_x * cart_heading_sin + rel_y * cart_heading_cos
    crate_nx = crate_wx / 5.0
    crate_ny = crate_wy / 4.0
    crate_margin = 0.95
    crate_oob = max(0.0, abs(crate_nx) - crate_margin) + max(0.0, abs(crate_ny) - crate_margin)

    out_of_bounds_penalty = -3.0 * (cart_oob + crate_oob)

    # ---- 条件信号 6：动作平滑（轻量，不压制推动）----
    action_smoothness = -0.05 * (action[0] ** 2 + action[1] ** 2)

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_docking_quality": float(crate_docking_quality),
        "crate_speed_penalty_near_dock": float(crate_speed_penalty_near_dock),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)
```
