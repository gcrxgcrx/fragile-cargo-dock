1. `evidence`：20/20 全部 truncated（len=400），无 terminated；`dock_completion_state` 独占 100% 份额（episode_sum_mean=219.8），而 `dock_approach_improvement`、`fragile_impact_penalty`、`boundary_health_penalty` 全部 active_rate=0、episode_sum_mean=0；score=-1.69，历史 best=3.86（iter3），同骨架族已迭代 5 轮未刷新 best。
2. `behavior_diagnosis`：策略完全停滞——它发现只要停在某个"接近 dock 但未完成"的静止状态，`dock_completion_state`（纯状态值 × gate）就每步稳定给分，于是放弃移动。`dock_approach_improvement` 恒为 0 说明货箱距离根本没有净减少（agent 不动），`fragile_impact_penalty`/`boundary_health_penalty` 恒为 0 说明它既不接触也不越界——典型的"占据好状态即持续获奖"刷分 exploit。
3. `signal_completeness`：职责不完备。缺"改善量"主信号（现有 improvement 项因 agent 不动而恒 0，无法提供梯度），且状态值项无界累积（219.8/400 步 ≈ 0.55/步）成为主导，把主信号挤成僵尸。成功所需的 near+slow+align+持续 10 步四条件中，只有 near 被间接奖励，slow/align 未获独立梯度。
4. `selected_level`：Level 3 重建（触发条件：同骨架族连续 ≥3 轮未刷新 best，且 iter4/5 两次 Level 2 变换后得分反而从 3.86 跌到 -1.5/-1.7）。
5. `selected_intervention`：更换主信号框架——废弃"状态值 × gate"作为主信号，改为 **potential_based_shaping（improvement_delta）为主 + joint_condition_proxy（几何平均，连续化）为辅助**。主信号 = 货箱到 dock 距离的逐步减少量（`dist_old - dist_new`），只在货箱确实向 dock 移动时给分；辅助信号 = near/slow/align 三因子的几何平均（每因子带 floor 防塌缩），权重压低到主信号的 ~0.3x，且**不再对静止状态持续给分**（几何平均中 slow 因子在静止时高、但 near 因子在远离时低，乘积自然抑制"停在半路"）。
6. `falsifiable_hypothesis`：若 agent 停滞是因为"静止状态值可刷分"，则移除状态值主导、改为改善量主导后，`dock_approach_improvement` 的 active_rate 应从 0 升到 >30%，且 `dock_completion_state` 的 episode_sum_mean 应从 219.8 大幅下降（<50），score 应回升到 >0。
7. `expected_next_round`：`dock_approach_improvement` active_rate >30%、episode_sum_mean 由 0 变为显著正值；`dock_completion_state` episode_sum_mean 从 219.8 降至 <50；len 仍为 400（truncated）但 score 从 -1.69 升至 >0；若 score 仍 ≤0 且 improvement 仍为 0，则说明 agent 连移动都无法产生，需检查动作-观测耦合。
8. `main_risk`：改善量信号在货箱接近 dock 后自然衰减（距离趋 0 时 delta→0），可能让 agent 在最后阶段失去梯度而停在 dock 边缘；用几何平均辅助项 + 低权重状态项兜底，但若辅助项权重过大又会重现刷分，需保持辅助 ≤ 主信号 0.3x。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack (仅使用环境声明的 obs 维度) ----------
    # 货箱到 dock 的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱世界速度
    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向
    crate_cos = obs[10]
    crate_sin = obs[11]

    # 小车
    cart_fwd = obs[4]
    contact = obs[14]
    cart_x = obs[0]
    cart_y = obs[1]

    # ---------- A. MAIN: dock_approach_improvement (improvement_delta) ----------
    # 货箱到 dock 的距离（归一化坐标）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new  # >0 表示货箱确实向 dock 靠近

    # 主信号：仅对"靠近"给正分，远离给负分（对称，避免刷分）
    w_progress = 25.0
    r_progress = w_progress * progress

    # ---------- B. AUX: joint_dock_completion (joint_condition_proxy, 几何平均) ----------
    # 三因子：near / slow / align，每因子连续 bounded，带 floor 防塌缩
    f_near = max(0.0, 1.0 - dist_new / 0.6)
    f_slow = 1.0 / (1.0 + 30.0 * crate_speed)
    f_align = abs(crate_cos)

    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 辅助权重压低到主信号的 ~0.3x 量级，且不随步数无界累积
    w_joint = 0.6
    r_joint = w_joint * joint

    # ---------- C. fragile_impact_penalty (hinge, 仅接触时) ----------
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
    total_reward = r_progress + r_joint + r_impact + r_bounds

    components = {
        "dock_approach_improvement": r_progress,
        "joint_dock_completion": r_joint,
        "fragile_impact_penalty": r_impact,
        "boundary_health_penalty": r_bounds,
    }

    return float(total_reward), components
```