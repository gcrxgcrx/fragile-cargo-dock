**evidence**：20/20 全 truncated（len=400），无终止失败；`joint_completion` 占 magnitude 99.9%、active_rate 76.1%、episode_sum_mean=3892，但外部 score 仅 1.97——这是典型的"proxy 被刷满、外部分数不动"；`crate_to_dock_progress` 仅 0.1% 份额、`boundary_avoidance` active_rate=0（僵尸组件）；历史 7 轮同骨架族 best 停在 17.69（iter4 dock_gate），连续 ≥3 轮未刷新。

**behavior_diagnosis**：策略学会了"停在某个 joint_completion 高值状态持续收分"——几何平均里 `slow_factor=1/(1+3*speed)` 在货箱静止时≈1，`dock_factor` 只要货箱靠近坞就高，于是"把货箱推到坞附近然后不动"即可每步拿 20×0.5²≈5 分，400 步累计 3892，但从未满足"完全进入+朝向<30°+速度<0.05 持续 10 步"的成功条件。这是 §3.6 明令禁止的"静止状态持续收分"失效模式。

**signal_completeness**：职责齐全（位置/朝向/速度/接触/边界/时间全部可用），但主信号形态错误——状态值型 proxy 允许停留积累，且 `slow_factor` 作为全局持续奖励与"推动货箱"对抗。

**selected_level**：Level 3 重建（同骨架族 ≥4 轮、best 未超 target×0.5、连续 ≥3 轮未刷新）。

**selected_intervention**：更换主信号框架——从"状态值 proxy"改为"**改善量 + 联合完成度门控**"：主信号 = 货箱到坞距离的 bounded 改善量（只有真正推进才得分），乘以联合完成度门控（dock×align×slow 几何平均），使"接近目标"时推进收益放大、"远离目标静止"收益≈0；删除僵尸 `boundary_avoidance`，边界改为 hinge 门控乘子。

**falsifiable_hypothesis**：若失败原因是"静止刷 proxy"，则改为改善量主信号后，`joint_completion` 类组件的 episode_sum_mean 应从 3892 量级降到与真实推进量同量级（<200），且 score 应上升或至少不再被 proxy 掩盖。

**expected_next_round**：`progress` 组件 magnitude_share 应 >60%，`joint_completion` 类份额 <30%，episode_sum_mean 从 3892 降至 <300；score 期望 >5（若仍 ≤2 则说明 agent 连推进都没学会，需检查 progress 尺度）。

**main_risk**：改善量信号在货箱被推到坞内后因震荡产生负收益，导致 agent 不敢做最后精调；用 bounded 压缩 + 门控放大缓解。

自检代入：①什么都不做、货箱静止初始位（dist≈1.0，speed=0）：progress=0，gate=dock_factor(1-1.0/0.6<0→0)×...=0 → 总奖励≈0。②正在把货箱推向坞（dist 1.0→0.9，speed=0.5）：progress=+0.1/(1+2)=0.033×8=0.27，gate=(0.33×0.9×0.4)^(1/3)≈0.49 → 总奖励≈0.27×(1+0.49)≈0.40 > 0。②严格高于①，通过。

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

    # 货箱朝向对齐度
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号: 改善量 (只有真正推进才得分) ----------
    raw_progress = dist - next_dist  # 靠近为正
    progress = raw_progress / (1.0 + abs(raw_progress) * 20.0)  # bounded

    # ---------- 联合完成度门控 (不单独收分, 只放大推进收益) ----------
    dock_factor = max(0.0, 1.0 - dist / 0.6)
    slow_factor = 1.0 / (1.0 + 3.0 * crate_speed)
    joint_gate = (dock_factor * align_factor * slow_factor) ** (1.0 / 3.0)

    # 推进收益: 基础 + 门控放大 (接近目标时推进更值钱)
    progress_reward = 8.0 * progress * (1.0 + 2.0 * joint_gate)

    # ---------- 完成度状态值 (极小权重, 仅提供接近目标的稠密梯度) ----------
    completion_reward = 1.0 * (joint_gate ** 2)

    # ---------- 边界安全 (hinge, 阈值 0.85) ----------
    boundary_penalty = 0.0
    if abs(obs[0]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[0]) - 0.85)
    if abs(obs[1]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[1]) - 0.85)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.85:
        boundary_penalty -= 0.2 * (sensor_max - 0.85)

    # ---------- 接触冲击抑制 (仅接触且高速时) ----------
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
        "joint_completion_gate": float(completion_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```