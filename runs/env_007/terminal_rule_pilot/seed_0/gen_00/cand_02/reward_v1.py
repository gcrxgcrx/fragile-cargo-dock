def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 几何常量（来自环境事实） ----------
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    DOCK_TOL_X = 0.024
    DOCK_TOL_Y = 0.030
    # 朝向误差阈值 30 度
    HEAD_TOL = 0.5235987755982988
    # 静止速度阈值 0.05 m/s（货箱速度分量归一化因子 3.0）
    SPEED_TOL = 0.05 / 3.0
    # 泊位外 0.3 m 参考（用于自检，不用于判据放宽）
    # 半宽 5.0, 半高 4.0

    # ---------- 观测读取 ----------
    # 货箱到泊位的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱世界系速度（归一化，因子 3.0 m/s）
    cvx = obs[8]
    cvy = obs[9]
    ncvx = next_obs[8]
    ncvy = next_obs[9]

    # 货箱朝向
    ccos = obs[10]
    csin = obs[11]

    # 接触标志
    contact = obs[14]
    ncontact = next_obs[14]

    # 障碍接近度
    s_front = obs[15]
    s_left = obs[16]
    s_right = obs[17]

    # 时间比例
    time_frac = obs[18]

    # 小车位置（归一化，半宽 5.0，半高 4.0）
    cart_x = obs[0]
    cart_y = obs[1]

    # ---------- 组件 1：货箱向泊位的增量进展（主信号，增量形式） ----------
    # 用归一化距离的减少量作为增量，避免悬停收割
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new  # 正=更接近
    # 增量项，量级控制
    crate_progress = 4.0 * progress

    # ---------- 组件 2：货箱接近泊位的联合质量（门控式，仅在接近时激活） ----------
    # 位置因子：越接近泊位中心越大，但用容差归一化
    pos_factor_x = max(0.0, 1.0 - abs(dx) / (DOCK_TOL_X * 20.0))
    pos_factor_y = max(0.0, 1.0 - abs(dy) / (DOCK_TOL_Y * 20.0))
    pos_factor = pos_factor_x * pos_factor_y

    # 朝向因子：|cos| 越接近 1 越好（对齐）
    head_factor = abs(ccos)

    # 速度因子：货箱速度越小越好（仅在接近泊位时作为门控）
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    speed_factor = max(0.0, 1.0 - crate_speed / (SPEED_TOL * 10.0))

    # 联合质量：几何平均，避免乘积塌缩
    dock_quality = (pos_factor * head_factor * speed_factor) ** (1.0 / 3.0)
    # 仅在接近泊位时激活（pos_factor 本身已随距离衰减，无需额外硬门控）
    crate_dock_quality = 1.5 * dock_quality

    # ---------- 组件 3：完成事件奖励（一次性大额，必须主导） ----------
    # 从 obs 显式推断完成条件：完全进入 + 朝向对齐 + 静止
    inside_x = abs(ndx) <= DOCK_TOL_X
    inside_y = abs(ndy) <= DOCK_TOL_Y
    head_ok = abs(next_obs[10]) >= 0.8660254037844387  # cos(30°)
    speed_ok = ((ncvx * ncvx + ncvy * ncvy) ** 0.5) <= SPEED_TOL

    # 完成事件：满足全部条件时给大额奖励
    # 注意：环境内部会连续保持 10 步才终止，这里每步满足都给完成奖励
    # 但 B 的量级需满足 10*B > 3*(过程组件上限*400)
    # 过程组件单步上限：crate_progress 最多约 4.0*0.1=0.4（距离减少有限）
    #                  crate_dock_quality 最多 1.5
    # 合计约 1.9，*400 = 760，*3 = 2280，10*B > 2280 → B > 228
    # 取 B = 250.0
    completion_bonus = 0.0
    if inside_x and inside_y and head_ok and speed_ok:
        completion_bonus = 250.0

    # ---------- 组件 4：轻柔操作（软接触间接推断，仅在接近泊位时轻罚） ----------
    # 接触时若货箱速度大，可能是硬碰撞；但正常推动也会有速度
    # 仅在接近泊位时抑制高速接触
    soft_contact_penalty = 0.0
    if contact > 0.5 and pos_factor > 0.05:
        # 接近泊位时，接触且速度大则轻罚
        if crate_speed > SPEED_TOL * 5.0:
            soft_contact_penalty = -0.05 * (crate_speed - SPEED_TOL * 5.0)

    # ---------- 组件 5：越界防护（hinge，仅在接近边界时生效） ----------
    # 小车位置归一化，边界约 |x|<=1, |y|<=1（仓库半宽/半高）
    # 用 hinge 在 0.85 之后才开始罚
    oob_penalty = 0.0
    cart_margin = 0.85
    if abs(cart_x) > cart_margin:
        oob_penalty -= 0.5 * (abs(cart_x) - cart_margin)
    if abs(cart_y) > cart_margin:
        oob_penalty -= 0.5 * (abs(cart_y) - cart_margin)
    # 货箱越界：由 obs[12], obs[13] 恢复货箱相对泊位偏移，泊位在仓库内
    # 货箱到泊位偏移归一化，若绝对值过大（>1.5）可能越界
    crate_margin = 1.5
    crate_dist = (dx * dx + dy * dy) ** 0.5
    if crate_dist > crate_margin:
        oob_penalty -= 0.3 * (crate_dist - crate_margin)

    # ---------- 组件 6：时间惩罚（轻量，避免拖延） ----------
    time_penalty = -0.01 * time_frac

    # ---------- 组件 7：动作平滑（轻量，可选） ----------
    # 避免剧烈动作，但不得压制必要推动
    action_smoothness = -0.02 * (action[0] ** 2 + action[1] ** 2)

    # ---------- 汇总 ----------
    components = {
        "crate_progress": float(crate_progress),
        "crate_dock_quality": float(crate_dock_quality),
        "completion_bonus": float(completion_bonus),
        "soft_contact_penalty": float(soft_contact_penalty),
        "oob_penalty": float(oob_penalty),
        "time_penalty": float(time_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = (
        crate_progress
        + crate_dock_quality
        + completion_bonus
        + soft_contact_penalty
        + oob_penalty
        + time_penalty
        + action_smoothness
    )

    return float(total_reward), components