def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 计算到目标的距离 ---
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_dist = dist_old - dist_new                      # 正值表示靠近目标

    progress_reward = 1.0 * delta_dist                    # 主学习信号

    # --- 速度软约束（超过阈值才惩罚）---
    speed_new = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    speed_threshold = 0.5
    vel_excess = max(0.0, speed_new - speed_threshold)
    velocity_penalty = -0.5 * (vel_excess ** 2)           # 二次惩罚，仅作用于超速部分

    # --- 角速度轻量惩罚 ---
    ang_vel = next_obs[5]
    angular_penalty = -0.1 * (ang_vel ** 2)               # 抑制剧烈旋转

    # --- 软着陆 gate：靠近 + 低速 + 姿态稳定 ---
    dist_thresh = 0.3
    speed_proxy_thresh = 0.3
    ang_thresh = 0.2
    f_dist  = max(0.0, 1.0 - dist_new / dist_thresh)
    f_speed = max(0.0, 1.0 - speed_new / speed_proxy_thresh)
    f_ang   = max(0.0, 1.0 - abs(ang_vel) / ang_thresh)
    landing_gate = (f_dist + f_speed + f_ang) / 3.0       # 平均避免乘积塌缩，gate ∈ [0,1]

    # --- 组合：gate 乘法到全部运动相关奖励 ---
    motion_reward = progress_reward + velocity_penalty + angular_penalty
    total_reward = landing_gate * motion_reward

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angular_penalty": angular_penalty,
        "landing_gate": landing_gate
    }
    return float(total_reward), components