# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 观测索引（严格按环境卡片声明） ----
    # obs[12]: crate_to_dock_x (归一化, /仓库半宽)
    # obs[13]: crate_to_dock_y (归一化, /仓库半高)
    # obs[8], obs[9]: crate_vx, crate_vy (归一化, /3.0 m/s)
    # obs[10], obs[11]: crate_cos_heading, crate_sin_heading
    # obs[0], obs[1]: cart_x, cart_y
    # obs[14]: cart_crate_contact
    # obs[15], obs[16], obs[17]: sensor_front/left/right
    # obs[18]: time_fraction

    # ---------- 基础量 ----------
    # 货箱到泊位的归一化偏移
    dx = next_obs[12]
    dy = next_obs[13]
    dx_prev = obs[12]
    dy_prev = obs[13]

    # 归一化距离（用偏移量欧氏距离，越小越近）
    dist = (dx * dx + dy * dy) ** 0.5
    dist_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5

    # 货箱速度（归一化单位）
    vx = next_obs[8]
    vy = next_obs[9]
    speed = (vx * vx + vy * vy) ** 0.5

    # 朝向误差（货箱朝向 vs 泊位朝向，泊位朝向假设为 +x 轴，cos=1）
    cos_h = next_obs[10]
    sin_h = next_obs[11]
    # 朝向误差角的正弦绝对值近似（相对 +x 轴）
    heading_err = abs(sin_h)  # sin(误差角) 的绝对值，0 表示对齐

    # ---------- 完成判据（严格取自环境事实） ----------
    # |obs[12]| <= 0.024 且 |obs[13]| <= 0.030
    # 朝向误差 < 30°  => |sin| < 0.5
    # 速度 < 0.05 m/s  => 归一化速度 < 0.05/3.0 ≈ 0.016667
    inside = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = heading_err < 0.5
    still = speed < 0.016667
    dock_condition = inside and aligned and still

    # ---------- 组件 1：货箱向泊位推进（增量形式，避免悬停陷阱） ----------
    # 只在"这一帧更接近了"时给正分；远离时给轻微负分
    progress = dist_prev - dist  # >0 表示更接近
    # 用 bounded 压缩，避免极端值
    progress_bounded = progress / (1.0 + abs(progress))
    crate_progress_reward = 1.0 * progress_bounded

    # ---------- 组件 2：接近泊位时的静止门控（仅接近时启用） ----------
    # 门控：距离泊位越近，越要求货箱慢
    # 用线性衰减门：dist < 0.15 开始激活
    near_gate = max(0.0, min(1.0, (0.15 - dist) / 0.15))
    # 静止质量：速度越小越好，仅在接近时生效
    speed_quality = max(0.0, 1.0 - speed / 0.05)  # speed<0.05 时为正
    crate_still_reward = 0.5 * near_gate * speed_quality

    # ---------- 组件 3：接近泊位时的朝向对齐门控 ----------
    align_quality = max(0.0, 1.0 - heading_err / 0.5)  # 误差<30° 时为正
    crate_align_reward = 0.5 * near_gate * align_quality

    # ---------- 组件 4：完成事件大额奖励（一次性主导信号） ----------
    # 满足完成条件时给大额奖励，压过所有过程信号
    # 量级要求：10*B > 3*(过程信号上限之和 * 400)
    # 过程信号上限估计：progress(≤1) + still(≤0.5) + align(≤0.5) ≈ 2.0
    # 3 * 2.0 * 400 = 2400  => 10*B > 2400 => B > 240
    # 取 B = 300 确保严格压过
    dock_bonus = 300.0 if dock_condition else 0.0

    # ---------- 组件 5：硬碰撞/软接触惩罚（轻量，不压制推动） ----------
    # 用接触标志 + 货箱速度突变间接推断；轻量惩罚避免噪声
    contact = next_obs[14]
    # 仅在接触且货箱速度较大时轻微惩罚（可能是硬碰撞）
    hard_contact_penalty = 0.0
    if contact > 0.5 and speed > 0.05:
        hard_contact_penalty = -0.2 * (speed - 0.05)

    # ---------- 组件 6：越界惩罚（hinge，仅在接近边界时生效） ----------
    # 小车位置 obs[0], obs[1] 归一化到 [-2, 2]，边界在 |x|>1.0 或 |y|>1.0 附近
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    # 越界 hinge：超出 0.9 开始惩罚
    oob_penalty = 0.0
    if abs(cart_x) > 0.9:
        oob_penalty -= 0.5 * (abs(cart_x) - 0.9)
    if abs(cart_y) > 0.9:
        oob_penalty -= 0.5 * (abs(cart_y) - 0.9)

    # ---------- 组件 7：动作平滑（轻量，可选） ----------
    action_smooth_penalty = -0.01 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 汇总 ----------
    components = {}
    components["crate_progress"] = crate_progress_reward
    components["crate_still"] = crate_still_reward
    components["crate_align"] = crate_align_reward
    components["dock_bonus"] = dock_bonus
    components["hard_contact"] = hard_contact_penalty
    components["out_of_bounds"] = oob_penalty
    components["action_smooth"] = action_smooth_penalty

    total_reward = (
        crate_progress_reward
        + crate_still_reward
        + crate_align_reward
        + dock_bonus
        + hard_contact_penalty
        + oob_penalty
        + action_smooth_penalty
    )

    return (float(total_reward), components)
```
