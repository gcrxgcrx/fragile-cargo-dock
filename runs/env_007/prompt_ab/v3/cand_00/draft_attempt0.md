```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 模块级状态（回合边界由 obs[18] 单调递增/重置检测） ----
    _PREV_T = compute_reward._PREV_T
    _STREAK = compute_reward._STREAK
    _PAID = compute_reward._PAID
    _ENTERED = compute_reward._ENTERED
    _PAID_ENTER = compute_reward._PAID_ENTER

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PAID_ENTER[0] = False
    _PREV_T[0] = t

    # ---- 基础量 ----
    # 泊位几何（环境事实）：半宽 0.6 m? 使用卡片给定容差换算
    # crate_to_dock 已归一化：|obs[12]|<=0.024, |obs[13]|<=0.030 视为完全进入
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist_norm = (dx * dx + dy * dy) ** 0.5

    pdx = float(obs[12])
    pdy = float(obs[13])
    prev_dist_norm = (pdx * pdx + pdy * pdy) ** 0.5

    # 货箱速度（世界系，m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度）
    crate_ang = 0.0
    ccos = float(next_obs[10])
    csin = float(next_obs[11])
    # 泊位朝向假定与仓库轴对齐，误差以货箱朝向相对 0 的最小角距近似
    crate_ang = abs(csin)  # sin 近似小角；|sin| <= sin(30°)=0.5 视为对齐
    align_ok = crate_ang < 0.5

    # ---- 组件 1: 货箱向泊位推进（增量形式，避免悬停收割） ----
    progress = prev_dist_norm - dist_norm
    if progress < 0.0:
        progress = progress * 0.5  # 远离时轻罚，不压制必要绕行
    crate_progress = 4.0 * progress

    # ---- 组件 2: 轻柔接触（唯一能教减速的信号） ----
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- 组件 3: 接近泊位时的速度抑制（门控：仅在接近泊位时激活） ----
    # 仅当货箱已经比较接近泊位（dist_norm < 0.15）才启用
    near_gate = 0.0
    if dist_norm < 0.15:
        near_gate = 1.0 - dist_norm / 0.15
    speed_penalty_near = -0.8 * near_gate * crate_speed

    # ---- 组件 4: 朝向对齐 shaping（仅在接近泊位时激活） ----
    align_penalty = -0.5 * near_gate * (crate_ang ** 2)

    # ---- 组件 5: 越界 hinge 惩罚 ----
    # 小车位置 obs[0],obs[1] 归一化到 [-2,2]，边界按 |x|>0.95 触发
    cart_x = abs(float(next_obs[0]))
    cart_y = abs(float(next_obs[1]))
    oob = 0.0
    if cart_x > 0.90:
        oob += (cart_x - 0.90)
    if cart_y > 0.90:
        oob += (cart_y - 0.90)
    out_of_bounds = -3.0 * oob

    # ---- 组件 6: 前方障碍接近惩罚（hinge，仅在很近时） ----
    sf = float(next_obs[15])
    obstacle_pen = 0.0
    if sf > 0.8:
        obstacle_pen = -1.0 * (sf - 0.8)
    # 左右两侧仅在极近时轻罚
    sl = float(next_obs[16])
    sr = float(next_obs[17])
    if sl > 0.9:
        obstacle_pen += -0.5 * (sl - 0.9)
    if sr > 0.9:
        obstacle_pen += -0.5 * (sr - 0.9)

    # ---- 完成条件（显式从 obs 推断，容差取自环境事实） ----
    inside = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    slow = crate_speed < 0.05
    done_cond = inside and slow and align_ok

    if done_cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 一次性"首次进入泊位"奖励
    enter_bonus = 0.0
    if inside and not _PAID_ENTER[0]:
        _PAID_ENTER[0] = True
        enter_bonus = 30.0

    # 一次性完成事件奖励
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    components = {
        "crate_progress": crate_progress,
        "gentleness": gentleness,
        "speed_near_dock": speed_penalty_near,
        "align_near_dock": align_penalty,
        "out_of_bounds": out_of_bounds,
        "obstacle_penalty": obstacle_pen,
        "enter_bonus": enter_bonus,
        "success_event": success_event,
    }

    total = 0.0
    for key in components:
        total += components[key]

    return float(total), components


# 模块级可变状态容器（在函数对象上挂载，避免 global）
compute_reward._PREV_T = [-1.0]
compute_reward._STREAK = [0]
compute_reward._PAID = [False]
compute_reward._ENTERED = [False]
compute_reward._PAID_ENTER = [False]
```