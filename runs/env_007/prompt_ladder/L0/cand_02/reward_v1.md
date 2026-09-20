```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 读取当前步信号 ----------
    dx = obs[12]          # 货箱到泊位有符号 x 偏移 / 半宽
    dy = obs[13]          # 货箱到泊位有符号 y 偏移 / 半高
    ndx = next_obs[12]
    ndy = next_obs[13]

    crate_vx = obs[8]     # 货箱世界系速度 x / 3.0
    crate_vy = obs[9]     # 货箱世界系速度 y / 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    cos_h = obs[10]
    sin_h = obs[11]
    # 朝向对齐度：|cos| 在 0 附近表示对齐（模 180° 均可入泊位）
    heading_align = 1.0 - abs(cos_h)

    contact = obs[14]

    cart_x = obs[0]
    cart_y = obs[1]

    # ---------- 1. 货箱向泊位推进（delta 形式，避免悬停陷阱） ----------
    dist_prev = (dx * dx + dy * dy) ** 0.5
    dist_next = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_prev - dist_next
    # 用有界压缩防止极端值刷分
    progress_bounded = progress / (1.0 + abs(progress))
    w_progress = 8.0
    r_progress = w_progress * progress_bounded

    # ---------- 2. 泊位内几何质量（位置收敛 + 朝向对齐） ----------
    # 位置收敛因子：距离越小越接近 1
    pos_factor = 1.0 / (1.0 + 12.0 * dist_next)
    # 朝向对齐因子
    align_factor = heading_align
    # 联合质量（几何平均，避免塌缩）
    dock_quality = (pos_factor * align_factor) ** 0.5
    w_quality = 3.0
    r_quality = w_quality * dock_quality

    # ---------- 3. 近泊位速度抑制（仅在接近泊位时启用） ----------
    near_gate = 1.0 / (1.0 + 15.0 * dist_next)   # 距离近 -> 接近 1
    speed_pen = -2.0 * near_gate * (crate_speed ** 2)
    r_speed_near = speed_pen

    # ---------- 4. 轻柔接触：接触时抑制货箱高速 ----------
    if contact > 0.5:
        soft_pen = -1.5 * (crate_speed ** 2)
    else:
        soft_pen = 0.0
    r_soft = soft_pen

    # ---------- 5. 越界 hinge 惩罚（小车与货箱） ----------
    # 小车位置在 [-2,2] 内（obs 已裁剪），真实边界接近 ±1 归一化
    cart_bound = max(0.0, abs(cart_x) - 0.9) + max(0.0, abs(cart_y) - 0.9)
    # 货箱到泊位的偏移过大 -> 可能越界
    crate_bound = max(0.0, dist_next - 1.2)
    r_bounds = -4.0 * (cart_bound + crate_bound)

    # ---------- 6. 动作平滑（轻量，不压制推动） ----------
    r_action = -0.05 * (action[0] ** 2 + action[1] ** 2)

    total = r_progress + r_quality + r_speed_near + r_soft + r_bounds + r_action

    components = {
        "crate_to_dock_progress": float(r_progress),
        "crate_docking_quality": float(r_quality),
        "crate_speed_penalty_near_dock": float(r_speed_near),
        "soft_contact_penalty": float(r_soft),
        "out_of_bounds_penalty": float(r_bounds),
        "action_smoothness": float(r_action),
    }
    return float(total), components
```