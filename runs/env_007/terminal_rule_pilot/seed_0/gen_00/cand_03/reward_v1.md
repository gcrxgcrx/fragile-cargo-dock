```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 几何常量（来自环境事实） ----------
    dock_x_tol = 0.024   # |obs[12]| 阈值
    dock_y_tol = 0.030   # |obs[13]| 阈值

    # ---------- 主信号 1：货箱到泊位的增量进展（米制） ----------
    # obs[12]/obs[13] 是归一化有符号偏移，半宽/半高由事实推断为 5.0 / 4.0
    cur_dx = obs[12] * 5.0
    cur_dy = obs[13] * 4.0
    nxt_dx = next_obs[12] * 5.0
    nxt_dy = next_obs[13] * 4.0

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # 增量形式：只有"这一帧更接近"才给分，避免悬停收割
    progress = cur_dist - nxt_dist
    if progress > 0.05:
        progress = 0.05
    if progress < -0.05:
        progress = -0.05
    crate_progress = 12.0 * progress

    # ---------- 主信号 2：货箱朝向对齐（门控式增量） ----------
    # 货箱朝向误差（弧度）
    crate_ang = 0.0
    if next_obs[10] != 0.0 or next_obs[11] != 0.0:
        crate_ang = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
        crate_ang = crate_ang  # 单位化，仅用于判断朝向有效
    # 朝向误差的余弦相似度（对齐度），范围 [-1, 1]
    align_cos = next_obs[10]  # cos(heading)，泊位朝向假定为 0 度，cos 越大越对齐
    if align_cos > 1.0:
        align_cos = 1.0
    if align_cos < -1.0:
        align_cos = -1.0
    # 对齐度映射到 [0,1]
    align_score = (align_cos + 1.0) * 0.5

    # 只在货箱接近泊位时激活朝向 shaping（门控），避免全局收分
    near_gate_x = 1.0 - abs(next_obs[12]) / (dock_x_tol * 8.0)
    if near_gate_x < 0.0:
        near_gate_x = 0.0
    if near_gate_x > 1.0:
        near_gate_x = 1.0
    near_gate_y = 1.0 - abs(next_obs[13]) / (dock_y_tol * 8.0)
    if near_gate_y < 0.0:
        near_gate_y = 0.0
    if near_gate_y > 1.0:
        near_gate_y = 1.0
    near_gate = near_gate_x * near_gate_y

    crate_align = 2.0 * near_gate * align_score

    # ---------- 主信号 3：接近泊位时的速度抑制（门控，仅惩罚过快） ----------
    crate_speed = (next_obs[8] * next_obs[8] * 9.0 + next_obs[9] * next_obs[9] * 9.0) ** 0.5
    # 速度阈值：0.05 m/s 为成功静止阈值，此处允许一定余量
    speed_excess = crate_speed - 0.15
    if speed_excess < 0.0:
        speed_excess = 0.0
    # 仅在接近泊位时惩罚速度
    crate_speed_pen = -3.0 * near_gate * speed_excess

    # ---------- 完成事件奖励（一次性大额，主导过程信号） ----------
    # 完成条件：货箱完全在泊位内 + 朝向对齐（<30°）+ 速度 <0.05 m/s
    inside_x = 1.0 if abs(next_obs[12]) <= dock_x_tol else 0.0
    inside_y = 1.0 if abs(next_obs[13]) <= dock_y_tol else 0.0
    # 朝向误差 < 30° => cos(误差) > cos(30°) ≈ 0.866
    align_ok = 1.0 if next_obs[10] >= 0.866 else 0.0
    speed_ok = 1.0 if crate_speed < 0.05 else 0.0

    dock_event = 0.0
    if inside_x >= 1.0 and inside_y >= 1.0 and align_ok >= 1.0 and speed_ok >= 1.0:
        # 完成事件奖励：需满足 10*B > 3*(过程上限*400)
        # 过程上限估计：12*0.05 + 2 + 3*~0.5 ≈ 0.6+2+1.5 = 4.1 每步，*400 = 1640
        # 10*B > 3*1640 => B > 492，取 600 留余量
        dock_event = 600.0

    # ---------- 约束 1：软接触惩罚（间接推断，仅接触且速度突变时） ----------
    contact = next_obs[14]
    soft_contact = 0.0
    if contact > 0.5:
        # 接触时若货箱速度很高，说明可能是硬碰撞
        if crate_speed > 0.4:
            soft_contact = -2.0 * (crate_speed - 0.4)

    # ---------- 约束 2：越界惩罚（hinge，仅边界附近生效） ----------
    # 小车位置 obs[0], obs[1] 归一化到 [-2,2]，正常应在 [-1,1] 附近
    oob_pen = 0.0
    cart_ax = abs(next_obs[0])
    cart_ay = abs(next_obs[1])
    if cart_ax > 0.95:
        oob_pen -= 5.0 * (cart_ax - 0.95)
    if cart_ay > 0.95:
        oob_pen -= 5.0 * (cart_ay - 0.95)

    # ---------- 约束 3：动作平滑（轻量，防止抖振） ----------
    act_smooth = -0.05 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 汇总 ----------
    components = {}
    components["crate_progress"] = float(crate_progress)
    components["crate_align"] = float(crate_align)
    components["crate_speed_pen"] = float(crate_speed_pen)
    components["dock_event"] = float(dock_event)
    components["soft_contact"] = float(soft_contact)
    components["out_of_bounds"] = float(oob_pen)
    components["action_smooth"] = float(act_smooth)

    total_reward = (
        crate_progress
        + crate_align
        + crate_speed_pen
        + dock_event
        + soft_contact
        + oob_pen
        + act_smooth
    )

    return float(total_reward), components
```