1. `evidence`：20/20 全 truncated（len=400），无 terminated，说明策略既没成功也没触发失败终止；`joint_completion` episode_sum_mean=604、magnitude_share=98.2%、active_rate=71.9%，而真正推进的 `crate_to_dock_progress` 仅 8.45（1.5%），即奖励被一个"近坞联合因子"完全支配，agent 学会了在坞附近低速徘徊刷 joint 分而不真正完成交付；历史 best 仅 17.69（iter4），同骨架族 5 轮未突破 target×0.5=125。

2. `behavior_diagnosis`：策略在坞附近维持"部分进坞+部分对齐+低速"的稳态，靠 `joint_completion` 的持续状态值收分（state 值而非改善量），既不完成 10 步稳定交付，也不失败——典型 proxy 徘徊 exploit。

3. `signal_completeness`：信号齐全（obs[12,13] 距离、obs[8,9] 速率、obs[10,11] 朝向、obs[14] 接触、obs[15-17] 障碍、obs[18] 时间），缺口不在信号而在**主信号形态**：把"占据好状态"当奖励，而非"完成交付的改善量"。

4. `selected_level`：Level 3 重建（同骨架族 ≥4 轮、best 17.69 远低于 target×0.5，且 iter5 的 joint 形态再次塌缩为徘徊）。

5. `selected_intervention`：更换主信号框架——废弃"状态值型 joint_completion"，改为**势能差 + 完成度改善**骨架：主信号 = 货箱到坞距离的势能改善（`Phi=-(dist)` 的 delta，即 `dist-next_dist`，但用 bounded 压缩防极端），辅以**仅在接近坞时激活的"完成度改善"**（进坞深度、朝向误差、速率的联合改善量），并把"静止/对齐"改为**门控**乘在推进信号上，绝不做全局持续奖励。

6. `falsifiable_hypothesis`：若把 joint 状态值替换为"接近坞时的完成度改善量 + 势能差"，则 agent 无法靠停在坞附近收分，必须持续减小 dist/朝向误差/速率才能得分，`crate_to_dock_progress` 的 magnitude_share 应显著上升、`joint` 类组件 episode_sum_mean 应大幅下降。

7. `expected_next_round`：`crate_to_dock_progress` magnitude_share 从 1.5% 升至 >30%；新完成度改善组件 active_rate >40% 且 episode_sum_mean 与 progress 同量级；score 应 >17.69（刷新 best），len 仍可能 400（未成功）但 score_range 上界提升。

8. `main_risk`：势能差在坞附近震荡导致正负抵消；若完成度改善权重过大，agent 可能在坞口反复进出刷改善量。自检：①什么都不做（dist 不变、crate_speed≈0、align 不变）→ progress=0，完成度改善=0，总奖励≈0；②正在推箱靠近（dist 减小 0.01，crate_speed=0.3）→ progress_reward=+0.12，完成度改善>0，总奖励>0。②严格高于①，满足。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱世界系速率 (m/s)
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    ncvx = next_obs[8] * 3.0
    ncvy = next_obs[9] * 3.0
    next_crate_speed = (ncvx * ncvx + ncvy * ncvy) ** 0.5

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]
    next_align_factor = (next_obs[10] + 1.0) * 0.5

    contact = obs[14]

    # ---------- 主信号 A: 势能差（货箱到坞距离改善量，bounded 压缩防极端） ----------
    raw_progress = dist - next_dist  # 靠近为正
    # 平滑压缩，避免单步极端值支配
    progress = raw_progress / (1.0 + abs(raw_progress) * 20.0)
    progress_reward = 15.0 * progress

    # ---------- 主信号 B: 完成度改善量（仅在接近坞时激活，改善量而非状态值） ----------
    # 接近门控：dist 越小越接近 1（终止边界约 0.15，阈值 0.6 给足缓冲）
    near_gate = max(0.0, 1.0 - dist / 0.6)
    # 完成度度量：进坞深度 + 对齐 + 近静止（作为"度量"用于求改善量）
    dock_measure = max(0.0, 1.0 - dist / 0.6)
    align_measure = align_factor
    slow_measure = 1.0 / (1.0 + 2.0 * crate_speed)
    completion = (dock_measure + align_measure + slow_measure) / 3.0

    next_dock_measure = max(0.0, 1.0 - next_dist / 0.6)
    next_align_measure = next_align_factor
    next_slow_measure = 1.0 / (1.0 + 2.0 * next_crate_speed)
    next_completion = (next_dock_measure + next_align_measure + next_slow_measure) / 3.0

    completion_delta = next_completion - completion  # 改善为正
    # 只在接近坞时激活，且用改善量（停留不再积累收益）
    completion_reward = 8.0 * completion_delta * near_gate

    # ---------- 组件 C: 边界安全（hinge，阈值 0.85 = 终止边界 1.0 的 85%） ----------
    boundary_penalty = 0.0
    if abs(obs[0]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[0]) - 0.85)
    if abs(obs[1]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[1]) - 0.85)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.85:
        boundary_penalty -= 0.2 * (sensor_max - 0.85)

    # ---------- 组件 D: 接触冲击抑制（仅在接触且高速时，轻罚） ----------
    impact_penalty = 0.0
    if contact > 0.5:
        impact_excess = max(0.0, crate_speed - 1.5)
        impact_penalty = -0.2 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + completion_reward
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "completion_improvement": float(completion_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```