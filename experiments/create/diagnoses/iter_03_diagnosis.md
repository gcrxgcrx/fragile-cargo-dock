# Response Record

1. `evidence`：20/20 episode 全部 truncated（len=400 满额），无 terminated；`crate_settling_and_alignment` 占 magnitude 96.6%、active_rate 82.4%，而 `crate_to_dock_progress` 仅 2.7%/42.9%，`fragile_impact_penalty` 0.1%/0.2%，`out_of_bounds_penalty` 0%/0%。score=1.61 远低于 target=250。
2. `behavior_diagnosis`：策略学会了"原地贴住货箱并保持低速对齐"来持续领取 settle 奖励（proxy 徘徊），但几乎不推动货箱接近 dock（progress 信号弱且被 settle 淹没），因此永远无法触发 stable_steps>=10 的成功终止，只能超时。
3. `signal_completeness`：职责名义齐全，但主信号被 proxy 反客为主——settle 是"状态值"型奖励，占据近 dock 状态即持续获奖，与"把货箱推进 dock"这一真实进展脱钩；progress 权重(10)虽大但每步位移极小，实际贡献被压制。
4. `selected_level`：Level 2（结构变换）。触发条件：占据好状态即持续获奖（state→improvement），且 proxy 提高但外部分数不升。
5. `selected_intervention`：唯一目标组件 `crate_settling_and_alignment`。把"状态值"改为"改善量 + 完成对齐"的联合条件：仅当货箱在 dock 附近**且**本步在靠近/停稳（dist 减小或已静止）时才给奖励，并大幅降低其权重，使 progress 成为主导信号；同时把 settle 改为对"接近 dock 的进展"门控，避免原地刷分。
6. `falsifiable_hypothesis`：若 settle 奖励不再对"静止贴箱"持续发放，而只在货箱向 dock 收敛时发放，则策略必须真正推动货箱接近 dock 才能获得正反馈，`crate_to_dock_progress` 的 magnitude_share 应显著上升，settle 的 magnitude_share 应下降，且出现非零 terminated（成功或失败）而非全 truncated。
7. `expected_next_round`：`crate_settling_and_alignment` magnitude_share 从 96.6% 降至 <50%，`crate_to_dock_progress` magnitude_share 升至 >30% 且 active_rate >60%；episode 不再 20/20 全 truncated（至少部分 terminated）；score 应高于 1.61（若仍全 truncated 且 settle 仍占主导，则假设被证伪）。
8. `main_risk`：过度削弱 settle 后，策略可能只追求"靠近 dock"而高速冲入，导致硬冲击或越界失败增加；用 hinge 门控与低速因子缓解，但若 progress 权重仍不足以驱动长程推进，可能退化为原地不动。

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

    # --- A. crate_to_dock_progress: improvement_delta on distance (main signal) ---
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 20.0
    r_progress = w_progress * progress

    # --- B. crate_settling_and_alignment: improvement-gated, not state-value ---
    # Only reward when crate is near dock AND actually converging (progress>0)
    # or already nearly stopped inside the dock region.
    prox_hinge = max(0.0, 1.0 - dist_new / 0.6)
    speed_factor = 1.0 / (1.0 + 20.0 * crate_speed)
    align = abs(crate_cos)
    settle_raw = (prox_hinge * speed_factor * align) ** (1.0 / 3.0)
    # improvement gate: reward only when crate is approaching dock this step
    approach_gate = max(0.0, min(1.0, progress * 20.0))
    # allow a small floor only when already very close and slow (final settling)
    near_still = prox_hinge * speed_factor
    gate = max(approach_gate, 0.3 * near_still)
    w_settle = 1.0
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
