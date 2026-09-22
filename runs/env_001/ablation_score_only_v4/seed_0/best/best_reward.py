def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 计算到目标的距离 ---
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_dist = dist_old - dist_new

    progress_reward = 1.0 * delta_dist

    # --- 速度软约束 ---
    speed_new = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    speed_threshold = 0.5
    vel_excess = max(0.0, speed_new - speed_threshold)
    velocity_penalty = -0.5 * (vel_excess ** 2)

    # --- 角速度轻量惩罚 ---
    ang_vel = next_obs[5]
    angular_penalty = -0.1 * (ang_vel ** 2)

    # --- 软着陆 gate ---
    dist_thresh = 0.3
    speed_proxy_thresh = 0.3
    ang_thresh = 0.2
    f_dist  = max(0.0, 1.0 - dist_new / dist_thresh)
    f_speed = max(0.0, 1.0 - speed_new / speed_proxy_thresh)
    f_ang   = max(0.0, 1.0 - abs(ang_vel) / ang_thresh)
    landing_gate = (f_dist + f_speed + f_ang) / 3.0

    # --- 新增：最终着陆接触奖励 ---
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0          # 双脚接触程度 [0,1]
    speed_slow_factor = max(0.0, 1.0 - speed_new / 0.2)      # 速度越小越接近 1
    ang_slow_factor   = max(0.0, 1.0 - abs(ang_vel) / 0.1)  # 角速度越小越接近 1
    landing_reward = contact_avg * speed_slow_factor * ang_slow_factor * 0.5

    # --- 组合 ---
    motion_reward = progress_reward + velocity_penalty + angular_penalty
    total_reward = landing_gate * motion_reward + landing_reward

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angular_penalty": angular_penalty,
        "landing_gate": landing_gate,
        "landing_reward": landing_reward
    }
    return float(total_reward), components