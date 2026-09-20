分析：训练反馈里组件数值与符号完全混乱（如 out_of_bounds_penalty 出现正的大额、soft_contact_penalty 出现正奖励），说明代码里存在符号/命名与实现不一致、且大量组件在无意义地互相抵消；同时 20/20 episode 全部 truncation、任务分数仅 1.53，说明没有任何信号真正把"货箱进泊位并停稳"引导出来。核心问题：进度项用 delta 太稀疏且被有界压缩后量级过小；质量项乘积塌缩导致接近 0；速度惩罚与接触惩罚在推动货箱时同时扣分，直接惩罚任务必需动作。改进方向：以"货箱到泊位距离的势能递减"为主信号并给足权重，用 hinge 只在越界/高速危险时惩罚，速度抑制只作为"接近泊位后"的门控因子乘在质量项上而非全局扣分，并新增一个连续的"入泊位联合条件"proxy（位置∩朝向∩低速）来提供完成引导。

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    crate_vx = obs[8]
    crate_vy = obs[9]
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    cos_h = obs[10]
    sin_h = obs[11]
    heading_align = 1.0 - abs(cos_h)

    contact = obs[14]
    cart_x = obs[0]
    cart_y = obs[1]

    dist_prev = (dx * dx + dy * dy) ** 0.5
    dist_next = (ndx * ndx + ndy * ndy) ** 0.5

    # 1. 主进度：势能递减，正负都保留（远离则扣分），权重给足以主导
    progress = dist_prev - dist_next
    w_progress = 60.0
    r_progress = w_progress * progress

    # 2. 泊位联合质量 proxy：位置 ∩ 朝向 ∩ 低速，几何平均避免塌缩
    pos_factor = 1.0 / (1.0 + 20.0 * dist_next)
    align_factor = heading_align
    speed_factor = 1.0 / (1.0 + 30.0 * crate_speed)
    dock_proxy = (pos_factor * align_factor * speed_factor) ** (1.0 / 3.0)
    w_quality = 12.0
    r_quality = w_quality * dock_proxy

    # 3. 接近泊位时的速度门控（只作为门控乘子，不作为全局扣分）
    #    距离近且速度快 -> 轻微抑制，但绝不惩罚"推进"本身
    near_gate = 1.0 / (1.0 + 15.0 * dist_next)
    overspeed = max(0.0, crate_speed - 0.25)
    r_speed_near = -3.0 * near_gate * (overspeed ** 2)

    # 4. 轻柔接触：仅在接触且货箱速度显著偏高时惩罚（避免误伤正常推动）
    if contact > 0.5:
        hard_proxy = max(0.0, crate_speed - 0.35)
        r_soft = -4.0 * (hard_proxy ** 2)
    else:
        r_soft = 0.0

    # 5. 越界 hinge：只在接近边界时生效
    cart_bound = max(0.0, abs(cart_x) - 0.85) + max(0.0, abs(cart_y) - 0.85)
    crate_bound = max(0.0, dist_next - 1.1)
    r_bounds = -8.0 * (cart_bound + crate_bound)

    # 6. 动作平滑：轻量，不压制推动
    r_action = -0.02 * (action[0] ** 2 + action[1] ** 2)

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