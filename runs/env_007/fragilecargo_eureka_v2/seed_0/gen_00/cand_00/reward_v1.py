def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置与几何常量
    half_w = 5.0
    half_h = 4.0

    # ---------- 主信号 1：货箱到 dock 的进度（delta 距离） ----------
    dx_cur = obs[12] * half_w
    dy_cur = obs[13] * half_h
    dx_nxt = next_obs[12] * half_w
    dy_nxt = next_obs[13] * half_h
    dist_cur = (dx_cur * dx_cur + dy_cur * dy_cur) ** 0.5
    dist_nxt = (dx_nxt * dx_nxt + dy_nxt * dy_nxt) ** 0.5
    progress = dist_cur - dist_nxt
    w_progress = 6.0
    r_progress = w_progress * progress

    # ---------- 主信号 2：货箱朝向对齐（门控 + 联合 proxy） ----------
    # 货箱朝向余弦/正弦，dock 朝向假设与世界 x 轴对齐（cos=1, sin=0）
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    # 对齐度 = |cos(theta)|，范围 [0,1]，1 表示与 dock 轴平行
    align = crate_cos * crate_cos
    align = align ** 0.5  # |cos|
    # 只在货箱接近 dock 时才给对齐分（门控）
    near_gate = max(0.0, 1.0 - dist_nxt / 2.5)
    w_align = 1.5
    r_align = w_align * align * near_gate

    # ---------- 主信号 3：货箱低速停靠（门控 + 联合 proxy） ----------
    vx_n = next_obs[8] * 3.0
    vy_n = next_obs[9] * 3.0
    speed_n = (vx_n * vx_n + vy_n * vy_n) ** 0.5
    # 速度门：速度 < 0.05 满分，> 0.6 为 0
    speed_gate = max(0.0, min(1.0, (0.6 - speed_n) / 0.55))
    # 在 dock 附近才给停靠分
    w_settle = 1.5
    r_settle = w_settle * speed_gate * near_gate

    # ---------- 联合完成 proxy：位置近 + 对齐 + 低速 ----------
    # 位置因子：距离 < 0.5 满分，> 1.5 为 0
    pos_factor = max(0.0, min(1.0, (1.5 - dist_nxt) / 1.0))
    joint = (pos_factor * align * speed_gate) ** (1.0 / 3.0)
    w_joint = 3.0
    r_joint = w_joint * joint

    # ---------- 辅助：轻柔接触（避免硬碰撞损坏货箱） ----------
    # 仅在接触时激活；用相对速度近似撞击强度
    contact = next_obs[14]
    cart_v = next_obs[4] * 3.0
    # 相对速度近似：车速度与货箱速度之差的模
    rel_vx = cart_v * next_obs[2] - vx_n
    rel_vy = cart_v * next_obs[3] - vy_n
    rel_speed = (rel_vx * rel_vx + rel_vy * rel_vy) ** 0.5
    # 只在接触且相对速度大时惩罚（hinge），阈值 1.0 m/s
    hard_excess = max(0.0, rel_speed - 1.0)
    w_gentle = 0.8
    r_gentle = -w_gentle * hard_excess * contact

    # ---------- 辅助：边界规避（小车与货箱都不越界） ----------
    cart_x = next_obs[0] * half_w
    cart_y = next_obs[1] * half_h
    # 小车距边界的安全余量（半宽 5.0，半高 4.0），超过 4.2 / 3.2 开始惩罚
    cart_edge_x = max(0.0, abs(cart_x) - 4.2)
    cart_edge_y = max(0.0, abs(cart_y) - 3.2)
    # 货箱位置由相对偏移恢复
    crate_world_x = cart_x + next_obs[6] * 3.0 * next_obs[2] - next_obs[7] * 3.0 * next_obs[3]
    crate_world_y = cart_y + next_obs[6] * 3.0 * next_obs[3] + next_obs[7] * 3.0 * next_obs[2]
    crate_edge_x = max(0.0, abs(crate_world_x) - 4.2)
    crate_edge_y = max(0.0, abs(crate_world_y) - 3.2)
    w_bound = 1.0
    r_bound = -w_bound * (cart_edge_x + cart_edge_y + crate_edge_x + crate_edge_y)

    # ---------- 辅助：静态障碍接近（前方隔墙等） ----------
    sensor_front = next_obs[15]
    # 前方接近障碍且非接触货箱时轻微惩罚，避免撞墙
    obstacle_pen = max(0.0, sensor_front - 0.7) * (1.0 - contact)
    w_obs = 0.5
    r_obs = -w_obs * obstacle_pen

    # ---------- 总奖励 ----------
    total = r_progress + r_align + r_settle + r_joint + r_gentle + r_bound + r_obs

    components = {
        "crate_to_dock_progress": r_progress,
        "crate_dock_alignment": r_align,
        "crate_settling": r_settle,
        "joint_completion_proxy": r_joint,
        "gentle_contact": r_gentle,
        "boundary_avoidance": r_bound,
        "obstacle_proximity": r_obs,
    }
    return float(total), components