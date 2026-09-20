1. `evidence`：score=-2.26，20/20 全 truncated（len=400 满额），无 terminated；`crate_settling_and_alignment` 独占 100% 份额（episode_sum_mean=605.97，active_rate=100%），而 `crate_to_dock_progress`、`fragile_impact_penalty`、`out_of_bounds_penalty` 全部 active_rate=0%、episode_sum_mean≈0。
2. `behavior_diagnosis`：策略发现了一个纯刷分 exploit——`settle` 项只依赖 prox/speed/align 三个状态量，与是否真正推进货箱无关；小车只要停在货箱附近、让货箱静止且朝向对齐，就能每步稳定拿 ~1.5 分，于是完全放弃推箱，全程徘徊到超时。
3. `signal_completeness`：职责覆盖名义上齐全（进度、停靠、冲击、越界），但主信号 `crate_to_dock_progress` 的 active_rate=0% 说明它从未被触发——货箱根本没被推动，进度信号不可达；真正被激活的只有可被静态刷分的 settle 项，信号结构塌缩为单一状态奖励。
4. `selected_level`：Level 2（结构变换）。触发条件：`crate_settling_and_alignment` 占据好状态即持续获奖（state→improvement 反模式），且 `crate_to_dock_progress` active_rate=0% 表明主信号被 proxy 淹没。
5. `selected_intervention`：唯一目标组件 `crate_settling_and_alignment`——把"占据静止对齐状态即得分"改为"改善量/门控"形态：用 `crate_to_dock_progress` 作为 gate 乘子，只有货箱确实在向 dock 靠近时，settle 项才发放；同时把 settle 的 prox 项改为对距离的 hinge（仅在接近 dock 时激活），避免远处静止也得分。
6. `falsifiable_hypothesis`：若 settle 项被进度门控，则"原地静止刷分"路径被切断，策略必须推动货箱才能获得正奖励，`crate_to_dock_progress` 的 active_rate 应从 0% 上升，`crate_settling_and_alignment` 的 episode_sum_mean 应大幅下降。
7. `expected_next_round`：`crate_to_dock_progress` active_rate > 30% 且 episode_sum_mean > 0；`crate_settling_and_alignment` magnitude_share 从 100% 降至 <70%；score 从 -2.26 上升（至少不再为负徘徊），len 可能因越界/冲击终止而下降。
8. `main_risk`：进度门控过严（gate 在货箱远离 dock 时≈0）会让早期探索完全无正反馈，策略可能退化为随机游走；需保证 gate 在"不理想但安全"区域 ≥ 0.3。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- unpack signals ---
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

    # --- A. crate_to_dock_progress: improvement_delta on distance ---
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0
    r_progress = w_progress * progress

    # --- B. crate_settling_and_alignment: gated by progress ---
    # proximity factor: hinge, only meaningful when crate is near dock
    # dist normalized; dock region roughly dist < 0.15, hinge at 0.6 of boundary
    prox_hinge = max(0.0, 1.0 - dist_new / 0.6)
    # speed factor: 1 when still, decays with speed
    speed_factor = 1.0 / (1.0 + 20.0 * crate_speed)
    # alignment factor: crate heading aligned with dock axis
    align = abs(crate_cos)
    # geometric mean of three continuous factors
    settle_raw = (prox_hinge * speed_factor * align) ** (1.0 / 3.0)
    # progress gate: only reward settling when crate is actually approaching dock
    # gate in [0.3, 1.0] so early exploration still gets feedback
    gate = 0.3 + 0.7 * max(0.0, min(1.0, progress * 20.0))
    w_settle = 3.0
    r_settle = w_settle * settle_raw * gate

    # --- C. fragile_impact_penalty: hinge on relative speed during contact ---
    rel_speed = abs(cart_fwd - crate_speed)
    impact_threshold = 0.15
    impact_excess = max(0.0, rel_speed - impact_threshold)
    w_impact = 5.0
    r_impact = -w_impact * impact_excess * contact

    # --- D. out_of_bounds_penalty: hinge on cart position near boundary ---
    cart_x = obs[0]
    cart_y = obs[1]
    bound_threshold = 0.85
    cart_x_excess = max(0.0, abs(cart_x) - bound_threshold)
    cart_y_excess = max(0.0, abs(cart_y) - bound_threshold)
    w_bounds = 8.0
    r_bounds = -w_bounds * (cart_x_excess + cart_y_excess)

    # --- total ---
    total_reward = r_progress + r_settle + r_impact + r_bounds

    components = {
        "crate_to_dock_progress": r_progress,
        "crate_settling_and_alignment": r_settle,
        "fragile_impact_penalty": r_impact,
        "out_of_bounds_penalty": r_bounds,
    }

    return float(total_reward), components
```