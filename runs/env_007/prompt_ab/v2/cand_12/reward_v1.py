def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 几何恢复 ----
    # 泊位容差（来自环境事实）：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dock_tol_x = 0.024
    dock_tol_y = 0.030

    dx = next_obs[12]
    dy = next_obs[13]
    dx_prev = obs[12]
    dy_prev = obs[13]

    # 距离度量（归一化坐标下的欧氏距离）
    dist = (dx * dx + dy * dy) ** 0.5
    dist_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5

    # 货箱速度（归一化 -> 近似 m/s 尺度）
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度）
    angle_err = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    # 用 atan2 语义近似：cos 分量给出夹角余弦
    cos_err = next_obs[10]
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    # 朝向误差角（0 表示对齐），用 acos 的近似展开避免 import
    # acos(c) ≈ sqrt(2*(1-c)) 在 c 接近 1 时较准，这里做保守估计
    one_minus_c = 1.0 - cos_err
    if one_minus_c < 0.0:
        one_minus_c = 0.0
    heading_err = (2.0 * one_minus_c) ** 0.5

    contact = next_obs[14]
    time_frac = next_obs[18]

    components = {}

    # ---- 1. 主进程信号：货箱向泊位推进（增量形式，避免悬停收割） ----
    # 只在"这一帧更接近"时给分，且用 bounded 形式
    progress_raw = dist_prev - dist
    if progress_raw > 0.0:
        progress = progress_raw / (1.0 + progress_raw)
    else:
        progress = progress_raw / (1.0 - progress_raw) if progress_raw > -1.0 else -1.0
    components["crate_to_dock_progress"] = 6.0 * progress

    # ---- 2. 完成事件：货箱完全进入泊位 + 朝向对齐 + 几乎静止 ----
    # 位置门（连续）
    in_x = max(0.0, 1.0 - abs(dx) / dock_tol_x)
    in_y = max(0.0, 1.0 - abs(dy) / dock_tol_y)
    pos_gate = in_x * in_y

    # 朝向门（< 30° ≈ 0.5236 rad）
    heading_gate = max(0.0, 1.0 - heading_err / 0.5236)

    # 速度门（< 0.05 m/s）
    speed_gate = max(0.0, 1.0 - crate_speed / 0.05)

    # 联合完成代理（几何平均，避免塌缩）
    joint = (pos_gate * heading_gate * speed_gate) ** (1.0 / 3.0)

    # 完成事件大额奖励：仅在真正满足容差时给
    # 位置完全在容差内 => pos_gate 接近 1；用 hinge 强化
    fully_in = 0.0
    if abs(dx) <= dock_tol_x and abs(dy) <= dock_tol_y:
        fully_in = 1.0
    aligned = 0.0
    if heading_err < 0.5236:
        aligned = 1.0
    still = 0.0
    if crate_speed < 0.05:
        still = 1.0

    if fully_in > 0.0 and aligned > 0.0 and still > 0.0:
        # 完成事件奖励：B 需满足 10*B > 3*(过程上限*400)
        # 过程上限每步约 6.0 + 2.0 = 8.0，400 步约 3200，3 倍约 9600
        # 10*B > 9600 => B > 960；取 B = 5000
        components["docking_complete_event"] = 5000.0
    else:
        components["docking_complete_event"] = 0.0

    # 完成代理（连续引导，量级远小于完成事件）
    components["docking_quality_proxy"] = 2.0 * joint

    # ---- 3. 接近泊位时的速度抑制（门控形式，仅在接近时激活） ----
    # 接近度：距离越近门越大；只在 dist < 0.15 时激活
    if dist < 0.15:
        near_gate = max(0.0, 1.0 - dist / 0.15)
        # 速度抑制：速度越大惩罚越大，但仅在接近泊位时
        speed_pen = near_gate * min(crate_speed, 3.0)
        components["crate_speed_penalty_near_dock"] = -1.0 * speed_pen
    else:
        components["crate_speed_penalty_near_dock"] = 0.0

    # ---- 4. 轻柔接触：接触时速度突变大则惩罚（间接推断） ----
    # 用货箱速度与小车速度差近似冲击强度
    cart_v = next_obs[4] * 3.0
    if contact > 0.5:
        rel_speed = crate_speed - abs(cart_v)
        if rel_speed < 0.0:
            rel_speed = 0.0
        # 只在相对速度较大时惩罚
        if rel_speed > 0.5:
            components["soft_contact_penalty"] = -0.5 * (rel_speed - 0.5)
        else:
            components["soft_contact_penalty"] = 0.0
    else:
        components["soft_contact_penalty"] = 0.0

    # ---- 5. 越界惩罚（小车与货箱接近边界） ----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    # 归一化坐标，边界约 ±1.0（裁剪到 ±2.0），留安全余量 0.85
    oob = 0.0
    if abs(cart_x) > 0.85:
        oob += (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        oob += (abs(cart_y) - 0.85)
    # 货箱位置：由相对位置恢复（近似）
    crate_rel_x = next_obs[6] * 3.0
    crate_rel_y = next_obs[7] * 3.0
    # 用泊位偏移与小车位置近似货箱世界位置
    # 货箱到泊位偏移已知，泊位中心近似在仓库中线附近
    # 保守：用相对距离判断是否远离小车
    crate_dist_from_cart = (crate_rel_x * crate_rel_x + crate_rel_y * crate_rel_y) ** 0.5
    if crate_dist_from_cart > 4.0:
        oob += (crate_dist_from_cart - 4.0) * 0.5
    components["out_of_bounds_penalty"] = -3.0 * oob

    # ---- 6. 动作平滑（轻量，避免压制推动） ----
    a0 = action[0]
    a1 = action[1]
    components["action_smoothness"] = -0.02 * (a0 * a0 + a1 * a1)

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)