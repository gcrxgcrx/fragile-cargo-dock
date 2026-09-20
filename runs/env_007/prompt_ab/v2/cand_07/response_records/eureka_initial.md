# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 几何与阈值常量（来自环境事实） ----
    # 泊位容差：|crate_to_dock_x| <= 0.024, |crate_to_dock_y| <= 0.030
    # 朝向误差 < 30 deg，速度 < 0.05 m/s，连续保持 10 步
    # 观测缩放：obs[12], obs[13] 为相对泊位偏移 / 仓库半宽(5.0)、半高(4.0)

    # ---- 当前帧货箱到泊位偏移 ----
    dx = obs[12] * 5.0
    dy = obs[13] * 4.0
    ndx = next_obs[12] * 5.0
    ndy = next_obs[13] * 4.0

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # ---- 组件 1：货箱向泊位推进（增量形式，避免悬停收割） ----
    progress = dist - next_dist  # >0 表示这一帧更接近泊位
    crate_progress = 6.0 * progress

    # ---- 组件 2：货箱朝向对齐（连续 bounded，越对齐越高） ----
    crate_heading = obs[10] * obs[10] + obs[11] * obs[11]
    if crate_heading > 1e-8:
        ch_cos = obs[10] / (crate_heading ** 0.5)
        ch_sin = obs[11] / (crate_heading ** 0.5)
    else:
        ch_cos = 1.0
        ch_sin = 0.0
    # 泊位朝向假定与仓库 x 轴一致，误差角由货箱朝向给出
    align = ch_cos  # cos(误差角)
    if align < 0.0:
        align = 0.0
    # 仅在接近泊位时激活对齐信号（门控），避免远离时过早约束
    near_gate = 1.0 / (1.0 + 2.0 * dist)
    heading_align = 0.4 * align * near_gate

    # ---- 组件 3：接近泊位时的速度抑制（门控，仅近泊位激活） ----
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    # 门控：距离越近，速度惩罚越强；远离时几乎为 0
    speed_gate = 1.0 / (1.0 + 8.0 * dist)
    speed_penalty = -0.5 * speed_gate * (crate_speed ** 2)

    # ---- 组件 4：软接触惩罚（仅在接触且速度突变大时轻罚） ----
    contact = obs[14]
    if contact > 0.5:
        # 用货箱速度近似推动强度，过大速度视为硬碰撞风险
        excess = crate_speed - 0.8
        if excess > 0.0:
            soft_contact = -0.3 * (excess ** 2)
        else:
            soft_contact = 0.0
    else:
        soft_contact = 0.0

    # ---- 组件 5：越界 hinge 惩罚（小车与货箱接近边界时） ----
    cart_x = obs[0]
    cart_y = obs[1]
    # obs[0] 归一化到仓库半宽，|.|<=1 为界内；留 0.9 安全边
    ob = 0.0
    ax = cart_x if cart_x >= 0.0 else -cart_x
    ay = cart_y if cart_y >= 0.0 else -cart_y
    if ax > 0.9:
        ob -= 2.0 * (ax - 0.9)
    if ay > 0.9:
        ob -= 2.0 * (ay - 0.9)
    # 货箱越界：由 obs[12]/obs[13] 恢复货箱世界位置
    # 货箱世界坐标 = 小车位置 + 旋转(车身系偏移)
    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    cr = obs[2]
    sr = obs[3]
    cw_x = cart_x * 5.0 + (cr * rel_x - sr * rel_y)
    cw_y = cart_y * 4.0 + (sr * rel_x + cr * rel_y)
    # 仓库半宽 5.0，半高 4.0
    cbx = cw_x / 5.0
    cby = cw_y / 4.0
    cbx = cbx if cbx >= 0.0 else -cbx
    cby = cby if cby >= 0.0 else -cby
    if cbx > 0.95:
        ob -= 3.0 * (cbx - 0.95)
    if cby > 0.95:
        ob -= 3.0 * (cby - 0.95)
    out_of_bounds = ob

    # ---- 组件 6：完成事件奖励（一次性大额，主导过程信号） ----
    # 完成条件：|obs[12]|<=0.024 且 |obs[13]|<=0.030 且 朝向误差<30deg 且 速度<0.05
    # 用 next_obs 判定；连续保持由环境内部处理（我们只在满足时给大额奖励）
    inside_x = 1.0 if (ndx if ndx >= 0.0 else -ndx) <= 0.024 else 0.0
    inside_y = 1.0 if (ndy if ndy >= 0.0 else -ndy) <= 0.030 else 0.0
    # 朝向误差 < 30 deg -> cos > cos(30)=0.866
    nc = next_obs[10] * next_obs[10] + next_obs[11] * next_obs[11]
    if nc > 1e-8:
        ncos = next_obs[10] / (nc ** 0.5)
    else:
        ncos = 1.0
    align_ok = 1.0 if ncos >= 0.866 else 0.0
    # 速度 < 0.05
    nvx = next_obs[8] * 3.0
    nvy = next_obs[9] * 3.0
    nspeed = (nvx * nvx + nvy * nvy) ** 0.5
    speed_ok = 1.0 if nspeed < 0.05 else 0.0
    # 联合完成条件
    docked = inside_x * inside_y * align_ok * speed_ok
    # 大额完成奖励：B = 2000，满足 10*B > 3*(过程上限*400)
    # 过程单步上限约 6*推进(<=3) + 0.4 + 0.5*速度^2(<=0.5) ≈ 19
    # 3*(19*400) = 22800；10*2000 = 20000 略低，加大到 3000
    completion_bonus = 3000.0 * docked

    # ---- 组件 7：动作平滑（轻量，避免抑制必要推动） ----
    smooth = -0.01 * (action[0] * action[0] + action[1] * action[1])

    total_reward = (
        crate_progress
        + heading_align
        + speed_penalty
        + soft_contact
        + out_of_bounds
        + completion_bonus
        + smooth
    )

    components = {
        "crate_progress": crate_progress,
        "heading_align": heading_align,
        "speed_penalty": speed_penalty,
        "soft_contact": soft_contact,
        "out_of_bounds": out_of_bounds,
        "completion_bonus": completion_bonus,
        "action_smooth": smooth,
    }

    return (float(total_reward), components)
```
