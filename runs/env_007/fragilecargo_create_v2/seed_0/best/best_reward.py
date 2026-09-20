def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱速率（世界系，m/s）
    crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号：货箱到坞距离的改善量（唯一推进信号） ----------
    progress = dist - next_dist  # 靠近为正
    progress_reward = 10.0 * progress

    # ---------- 门控：接近坞 + 对齐 + 近静止，作为乘子而非独立收分 ----------
    # 接近门：dist 越小越接近 1（阈值 0.6，终止边界约 0.15 的 4x 缓冲）
    near_gate = max(0.0, 1.0 - dist / 0.6)
    # 对齐门：朝向越对齐越接近 1（下限 0.2 防塌缩）
    align_gate = max(0.2, align_factor)
    # 静止门：货箱越慢越接近 1（下限 0.2 防塌缩）
    slow_gate = max(0.2, 1.0 - crate_speed / 0.6)
    # 几何平均，避免乘积塌缩
    gate = (near_gate * align_gate * slow_gate) ** (1.0 / 3.0)

    # 门控只放大"正在推进"的信号：推进为正时按门控加成
    if progress > 0.0:
        gated_progress = progress_reward * (1.0 + 2.0 * gate)
    else:
        gated_progress = progress_reward

    # ---------- 组件 C: 边界安全（hinge，阈值 0.85 = 终止边界 1.0 的 85%） ----------
    boundary_penalty = 0.0
    if abs(obs[0]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[0]) - 0.85)
    if abs(obs[1]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[1]) - 0.85)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.85:
        boundary_penalty -= 0.2 * (sensor_max - 0.85)

    # ---------- 组件 D: 接触冲击抑制（仅在接触且高速时，轻罚） ----------
    impact_penalty = 0.0
    if contact > 0.5:
        impact_excess = max(0.0, crate_speed - 1.5)
        impact_penalty = -0.2 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        gated_progress
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(gated_progress),
        "dock_gate": float(gate),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components