分析：上一版把 `terminal_success=300` 一次性大额奖励作为主导信号，导致回报方差极大，而 eval 只有 1/20 真正终止、19/20 超时截断，说明策略只学会把货箱推到泊位附近却完不成"完全进入+对齐+低速保持"。`settle_bonus` 是二值 done 谓词，dock 内最后一段（对齐、减速到 0.05 m/s）没有任何连续梯度，`align` 只出现在稀疏判据里、从不参与塑形。`roughness`/`hard_hit` 用接触闭合速度做持续惩罚，等于对"推箱"这一必需动作收税，且 `hard_hit` 激活率≈0、几乎无效。改法：把泊位收益拆成"包含 / 对齐 / 停稳"三段连续且被 `contain_f` 门控的非负项（dock 外恒为 0，绝不对推进产生负贡献），用连续 `speed_gate` 给出减速梯度，把一次性成功奖励降到 100，去掉逐步冲击惩罚改为累计 3 次硬接触后一次性扣分，并弱化时间与边界成本。

```python
_MEM = [1.0, 0, False, 0, False]
# _MEM = [prev_time_fraction, settled_streak, hold_paid, hard_contact_count, failure_paid]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 自检（单步总奖励）------------------------------------------------
    # ① 什么都不做（货箱静止在初始位置、小车静止、无接触）：
    #    crate_dock_progress=0, cart_crate_approach=0, dock 三项=0（contain_f=0）,
    #    action_cost≈-0.0005*(0)=0, time_cost=-0.001  ->  合计 ≈ -0.001
    # ② 正在把货箱推向泊位（接触推动，货箱因此具有朝向泊位的速度）：
    #    crate_dock_progress = 1.0*(prev_dock-next_dock) > 0（约 +0.01~+0.05）,
    #    cart_crate_approach >= 0（跟随时距离基本不变）,
    #    dock 三项仍 >= 0（货箱未进泊位时 contain_f=0，不为负）,
    #    无逐步冲击惩罚（只有累计 3 次硬接触才一次性扣分）,
    #    -> 合计 ≈ +0.01~+0.05 - 0.001 > ①  ✅ 严格更高
    #    且所有 dock/对齐/低速项均为“门控 × 非负因子”，dock 外恒为 0，
    #    不会因为“推进所必需的动作”而下降。
    # ---------------------------------------------------------------------

    t = float(next_obs[18])
    if t < _MEM[0] - 1e-6 or t < 0.01:
        _MEM[1] = 0
        _MEM[2] = False
        _MEM[3] = 0
        _MEM[4] = False
    _MEM[0] = t

    # ---------- 主职责 1：货箱 -> 泊位的势函数式进展（米） ----------
    prev_dock = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    next_dock = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    crate_dock_progress = 1.0 * (prev_dock - next_dock)

    # ---------- 辅助：小车贴近货箱（建立推挤接触，早期稠密引导） ----------
    prev_rel = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    next_rel = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5
    cart_crate_approach = 0.3 * (prev_rel - next_rel)

    # ---------- 主职责 2/3/4：泊位包含 / 朝向对齐 / 低速停稳 ----------
    # 门控：货箱必须在泊位容差内（|dx|<=0.12m, |dy|<=0.12m）才开闸，闸外恒为 0
    dx = next_obs[12] * 5.0
    dy = next_obs[13] * 4.0
    off = max(abs(dx), abs(dy)) / 0.12
    contain_f = max(0.0, 1.0 - off)          # 1 = 完全包含在泊位内

    # 朝向对齐：dock 轴与世界轴对齐，允许 90 度倍数 -> max(|cos|,|sin|)
    align = max(abs(next_obs[10]), abs(next_obs[11]))
    align_gate = max(0.0, 2.0 * align - 1.0)  # 0 @ 45°误差, 1 @ 完美对齐

    # 低速停稳：货箱速度连续衰减因子（0.30 m/s 处归零，0.05 m/s 处约 0.83）
    crate_vx_m = next_obs[8] * 3.0
    crate_vy_m = next_obs[9] * 3.0
    crate_speed = (crate_vx_m ** 2 + crate_vy_m ** 2) ** 0.5
    speed_gate = max(0.0, 1.0 - crate_speed / 0.30)

    # 三个递进的非负门控项（全部被 contain_f 门控，泊位外为 0）
    dock_containment = 0.10 * contain_f
    dock_alignment = 0.30 * contain_f * align_gate
    dock_settled = 1.00 * contain_f * align_gate * speed_gate

    # ---------- 保持判据（与成功条件同构：包含 + 对齐 + <0.05 m/s） ----------
    done_now = (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030 and
                align >= 0.8660254 and crate_speed < 0.05)
    if done_now:
        _MEM[1] += 1
    else:
        _MEM[1] = 0

    hold_bonus = 0.0
    if _MEM[1] >= 10 and not _MEM[2]:
        _MEM[2] = True
        hold_bonus = 100.0

    # ---------- 安全：硬接触计数（不做逐步惩罚，避免给推箱收税） ----------
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    closing = next_obs[4] * 3.0 - (crate_vx_m * next_obs[2] + crate_vy_m * next_obs[3])
    if contact > 0.5 and closing > 1.0:
        _MEM[3] += 1

    # ---------- 边界守卫（只在小车贴近外墙时轻罚） ----------
    pen_x = max(0.0, abs(next_obs[0]) - 0.95)
    pen_y = max(0.0, abs(next_obs[1]) - 0.95)
    boundary_guard = -10.0 * (pen_x + pen_y)

    # ---------- 一次性失败代理（越界 / 累计 3 次硬接触） ----------
    out_of_bounds_penalty = 0.0
    crate_damage_penalty = 0.0
    if not _MEM[4]:
        cart_out = (abs(next_obs[0]) > 1.05 or abs(next_obs[1]) > 1.05)

        cart_x = next_obs[0] * 5.0
        cart_y = next_obs[1] * 4.0
        cos_h = next_obs[2]
        sin_h = next_obs[3]
        rel_x = next_obs[6] * 3.0
        rel_y = next_obs[7] * 3.0
        crate_x = cart_x + rel_x * cos_h - rel_y * sin_h
        crate_y = cart_y + rel_x * sin_h + rel_y * cos_h
        crate_out = (abs(crate_x) > 5.0 or abs(crate_y) > 4.0)

        if cart_out or crate_out:
            _MEM[4] = True
            out_of_bounds_penalty = -60.0
        elif _MEM[3] >= 3:
            _MEM[4] = True
            crate_damage_penalty = -40.0

    # ---------- 轻量代价 ----------
    action_cost = -0.0005 * (action[0] ** 2 + action[1] ** 2)
    time_cost = -0.001

    total_reward = (crate_dock_progress + cart_crate_approach +
                    dock_containment + dock_alignment + dock_settled +
                    hold_bonus + boundary_guard +
                    out_of_bounds_penalty + crate_damage_penalty +
                    action_cost + time_cost)

    components = {
        "crate_dock_progress": crate_dock_progress,
        "cart_crate_approach": cart_crate_approach,
        "dock_containment": dock_containment,
        "dock_alignment": dock_alignment,
        "dock_settled": dock_settled,
        "hold_bonus": hold_bonus,
        "boundary_guard": boundary_guard,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "crate_damage_penalty": crate_damage_penalty,
        "action_cost": action_cost,
        "time_cost": time_cost,
    }
    return float(total_reward), components
```