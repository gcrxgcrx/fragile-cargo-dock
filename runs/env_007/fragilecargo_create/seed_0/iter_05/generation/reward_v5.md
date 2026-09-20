1. `evidence`：20/20 episode 全部 truncated（len=400），无 terminated，说明策略既未成功停靠也未触发失败终止；reward 组成中 `joint_dock_completion` 独占 100% 份额（episode_sum_mean=879），其余三个组件 active_rate=0，score=-1.54，历史 best 仅 3.86（iter3），远低于 target 250。
2. `behavior_diagnosis`：策略学会了"贴着 dock 附近低速对齐"这一稳态——几何平均 proxy 在货箱靠近 dock 时给出持续正奖励，agent 满足于停留刷分，从不真正完成"进入 dock 矩形 + 停稳 10 步"的联合条件，也从不推进货箱越过分隔墙。
3. `signal_completeness`：主信号缺失——`joint_dock_completion` 是"状态值"而非"完成事件"，占据好状态即持续获奖，导致 proxy 徘徊；`crate_to_dock_progress` 用 obs[12]/[13] 的差分在上一轮塌缩为 0（active_rate=0），说明差分信号被稳态行为抹平；无任何"完成度"或"阶段推进"信号。
4. `selected_level`：Level 3 重建（累积记录同骨架族已迭代 4 轮，best=3.86 未超 target×0.5=125，且 iter4 骨架变化后得分反降）。
5. `selected_intervention`：更换主信号框架——用 `improvement_delta`（货箱到 dock 距离的逐步减少量）作为主学习信号，替代"状态值"式的 `joint_dock_completion`；辅以 `joint_condition_proxy`（几何平均）作为**完成度门控**乘到 progress 上，使"接近 + 停稳 + 对齐"三者同时满足时才有强梯度，而非占据状态即得分。
6. `falsifiable_hypothesis`：若主信号从"状态值"改为"改善量 × 联合完成门控"，则 agent 无法再靠停留刷分，必须持续减小货箱到 dock 的距离才能获得正奖励，从而推动货箱真正进入 dock 区域；若下一轮 score 仍不升且 `crate_to_dock_progress` 的 episode_sum_mean 仍≈0，则说明差分信号本身不可用，需换回状态值但加时间衰减。
7. `expected_next_round`：`crate_to_dock_progress`（新主信号）active_rate 应 > 50% 且 episode_sum_mean 显著非零（>10）；`joint_dock_completion` 份额应从 100% 降至 <50%；score 应 > 3.86（刷新 best）；len 仍为 400（truncated）但货箱到 dock 距离应下降。
8. `main_risk`：差分信号在货箱被推近 dock 后自然衰减为 0，agent 可能在 dock 边缘震荡；几何平均门控若 floor 设太低会重新塌缩为 0，需保留 floor 但降低权重避免主导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    crate_cos = obs[10]
    crate_sin = obs[11]

    cart_fwd = obs[4]
    contact = obs[14]

    cart_x = obs[0]
    cart_y = obs[1]

    # ---------- A. MAIN: dock_approach_improvement (improvement_delta) ----------
    # 货箱到 dock 的距离（归一化坐标下）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new

    # 联合完成门控：接近 + 停稳 + 对齐，几何平均，带 floor 防塌缩
    f_near = max(0.0, 1.0 - dist_new / 0.6)
    f_slow = 1.0 / (1.0 + 30.0 * crate_speed)
    f_align = abs(crate_cos)
    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    gate = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 主信号 = 改善量 × 门控（门控在 0.05~1 之间，不塌缩）
    w_progress = 20.0
    r_progress = w_progress * progress * gate

    # ---------- B. AUX: dock_completion_state (bounded state, 低权重) ----------
    # 保留一个低权重的状态信号，避免货箱已到位但无梯度时完全无反馈
    w_state = 1.5
    r_state = w_state * gate

    # ---------- C. fragile_impact_penalty (hinge, only on contact) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.15)
    w_impact = 4.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. boundary_health_penalty (hinge) ----------
    bx = max(0.0, abs(cart_x) - 0.9)
    by = max(0.0, abs(cart_y) - 0.9)
    w_bounds = 6.0
    r_bounds = -w_bounds * (bx + by)

    # ---------- total ----------
    total_reward = r_progress + r_state + r_impact + r_bounds

    components = {
        "dock_approach_improvement": r_progress,
        "dock_completion_state": r_state,
        "fragile_impact_penalty": r_impact,
        "boundary_health_penalty": r_bounds,
    }

    return float(total_reward), components
```