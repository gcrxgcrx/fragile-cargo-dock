# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 从 next_obs 提取货箱到泊位的偏移（有符号，归一化） ----
    dock_dx = next_obs[12]
    dock_dy = next_obs[13]
    dock_dist = (dock_dx * dock_dx + dock_dy * dock_dy) ** 0.5

    obs_dock_dx = obs[12]
    obs_dock_dy = obs[13]
    obs_dock_dist = (obs_dock_dx * obs_dock_dx + obs_dock_dy * obs_dock_dy) ** 0.5

    # ---- 货箱朝向误差（弧度） ----
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    heading_err = abs((crate_sin * crate_sin) ** 0.5)
    # 用 sin 分量近似朝向偏差幅度（|sin| 在 [-1,1]，对齐时 ~0）
    heading_err = heading_err

    # ---- 货箱速度（归一化单位） ----
    vx = next_obs[8]
    vy = next_obs[9]
    crate_speed = (vx * vx + vy * vy) ** 0.5

    # ---- 接触标志 ----
    contact = next_obs[14]

    # =========================================================
    # 组件 1: 货箱向泊位推进（主信号，improvement_delta）
    #   用 delta(distance) 避免悬停陷阱
    # =========================================================
    progress = obs_dock_dist - dock_dist
    r_progress = 3.0 * progress

    # =========================================================
    # 组件 2: 泊位接近度门控（仅在货箱接近泊位时激活的稠密信号）
    #   用 bounded 形式，且仅当接近时提供，避免全局悬停
    # =========================================================
    near_gate = max(0.0, 1.0 - dock_dist / 0.35)
    r_near = 0.6 * near_gate

    # =========================================================
    # 组件 3: 泊位内对齐 + 静止联合条件代理
    #   位置因子 * 朝向因子 * 速度因子 的几何平均
    #   仅在接近泊位时生效（门控），不惩罚推进动作
    # =========================================================
    pos_factor = max(0.0, 1.0 - dock_dist / 0.20)
    head_factor = max(0.0, 1.0 - heading_err / 0.50)
    speed_factor = max(0.0, 1.0 - crate_speed / 0.30)
    dock_quality = (pos_factor * head_factor * speed_factor) ** (1.0 / 3.0)
    r_dock_quality = 2.0 * near_gate * dock_quality

    # =========================================================
    # 组件 4: 接近泊位时的货箱速度抑制（hinge，仅接近时激活）
    #   只在接近泊位时惩罚速度，不影响推进过程
    # =========================================================
    speed_excess = max(0.0, crate_speed - 0.08)
    r_speed_pen = -1.2 * near_gate * (speed_excess * speed_excess)

    # =========================================================
    # 组件 5: 软接触惩罚（间接推断：接触 + 货箱速度突变）
    #   仅在接触且货箱速度较大时给轻惩罚，避免误伤正常推动
    # =========================================================
    if contact > 0.5:
        contact_speed_excess = max(0.0, crate_speed - 0.25)
        r_contact_pen = -0.5 * (contact_speed_excess * contact_speed_excess)
    else:
        r_contact_pen = 0.0

    # =========================================================
    # 组件 6: 越界惩罚（小车位置 hinge，接近边界时）
    #   小车位置归一化到约 [-1,1]，超过 0.92 视为接近边界
    # =========================================================
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    cart_edge = max(abs(cart_x), abs(cart_y))
    r_bounds = -1.0 * max(0.0, cart_edge - 0.92)

    # 货箱越界：由 obs[6], obs[7] 恢复货箱相对小车位置，判断是否远离
    # 用 obs[12], obs[13] 泊位偏移的绝对值过大作为货箱远离泊位/边界的代理
    crate_edge = max(abs(dock_dx), abs(dock_dy))
    r_crate_bounds = -0.8 * max(0.0, crate_edge - 1.6)

    # =========================================================
    # 组件 7: 障碍接近惩罚（前方/侧方传感器 hinge）
    #   仅在非常接近障碍时惩罚，避免压制正常移动
    # =========================================================
    obs_front = next_obs[15]
    obs_left = next_obs[16]
    obs_right = next_obs[17]
    obstacle_max = obs_front
    if obs_left > obstacle_max:
        obstacle_max = obs_left
    if obs_right > obstacle_max:
        obstacle_max = obs_right
    r_obstacle = -0.6 * max(0.0, obstacle_max - 0.85)

    # =========================================================
    # 组件 8: 动作平滑（轻量二次惩罚，避免抖动）
    # =========================================================
    a_drive = action[0]
    a_steer = action[1]
    r_smooth = -0.05 * (a_drive * a_drive + a_steer * a_steer)

    # =========================================================
    # 组件 9: 时间压力（轻微，避免拖延）
    # =========================================================
    time_frac = next_obs[18]
    r_time = -0.15 * time_frac

    # ---- 汇总 ----
    components = {}
    components["crate_to_dock_progress"] = r_progress
    components["dock_near_gate"] = r_near
    components["docking_quality"] = r_dock_quality
    components["crate_speed_near_dock_penalty"] = r_speed_pen
    components["soft_contact_penalty"] = r_contact_pen
    components["cart_out_of_bounds_penalty"] = r_bounds
    components["crate_out_of_bounds_penalty"] = r_crate_bounds
    components["obstacle_penalty"] = r_obstacle
    components["action_smoothness"] = r_smooth
    components["time_pressure"] = r_time

    total_reward = (
        r_progress
        + r_near
        + r_dock_quality
        + r_speed_pen
        + r_contact_pen
        + r_bounds
        + r_crate_bounds
        + r_obstacle
        + r_smooth
        + r_time
    )

    return (float(total_reward), components)
```
