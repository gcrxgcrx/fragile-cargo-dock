def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱世界系速率 (m/s)
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向对齐度
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号: 改善量 (只有真正推进才得分) ----------
    raw_progress = dist - next_dist  # 靠近为正
    progress = raw_progress / (1.0 + abs(raw_progress) * 20.0)  # bounded

    # ---------- 联合完成度门控 (不单独收分, 只放大推进收益) ----------
    dock_factor = max(0.0, 1.0 - dist / 0.6)
    slow_factor = 1.0 / (1.0 + 3.0 * crate_speed)
    joint_gate = (dock_factor * align_factor * slow_factor) ** (1.0 / 3.0)

    # 推进收益: 基础 + 门控放大 (接近目标时推进更值钱)
    progress_reward = 8.0 * progress * (1.0 + 2.0 * joint_gate)

    # ---------- 完成度状态值 (极小权重, 仅提供接近目标的稠密梯度) ----------
    completion_reward = 1.0 * (joint_gate ** 2)

    # ---------- 边界安全 (hinge, 阈值 0.85) ----------
    boundary_penalty = 0.0
    if abs(obs[0]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[0]) - 0.85)
    if abs(obs[1]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[1]) - 0.85)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.85:
        boundary_penalty -= 0.2 * (sensor_max - 0.85)

    # ---------- 接触冲击抑制 (仅接触且高速时) ----------
    impact_penalty = 0.0
    if contact > 0.5:
        impact_excess = max(0.0, crate_speed - 1.5)
        impact_penalty = -0.2 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + completion_reward
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "joint_completion_gate": float(completion_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components