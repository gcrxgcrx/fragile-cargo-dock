1. `evidence`：9 轮全部 truncated=20/20、len≈400，无一次成功终止；best=17.69（iter4）后连续 5 轮未刷新；本轮 `joint_completion_gate` 占 magnitude 64.8% 且 active_rate 74.9%，但 score 仅 3.35，说明该门控在"未接近坞"的常态下也在大量发放，是 proxy 刷分而非任务推进；`crate_to_dock_progress` 仅 41.1% 激活、episode_sum_mean 9.68，主信号太弱。
2. `behavior_diagnosis`：策略在 400 步内既不快速失败也不成功——典型"徘徊 + 局部刷分"：靠 dock_factor 在货箱偶尔靠近坞时收 gate 分，但从未把"完全进入 + 朝向对齐 + 近静止 + 保持"这一联合条件同时满足，因此永远 truncated。
3. `signal_completeness`：职责不完备。缺"朝向对齐"的显式梯度（obs[10]/[11] 从未被用于对齐误差），缺"货箱近静止"的独立可微信号（速度只被当作门控因子），且主进展信号被门控稀释。三条件（进入/对齐/静止）从未被联合优化。
4. `selected_level`：Level 3 重建（触发条件：同骨架族 ≥4 轮、连续 ≥3 轮未刷新 best、Level 2 形态变换后得分无实质改善）。
5. `selected_intervention`：更换主信号框架——从"progress × 门控放大"改为 **potential-based 联合完成度势函数**：`Phi = 进入度 × 对齐度 × 静止度`（几何平均防塌缩），主奖励 = `gamma*Phi(next) - Phi(obs)` 的改善量 + 小权重 `Phi` 状态项；删除会自我放大的 gate 放大项，把 boundary/impact 降为轻量 hinge。
6. `falsifiable_hypothesis`：若失败源于"三条件从未被联合优化"，则把主信号换成联合完成度势函数后，`joint_completion` 的 episode_sum_mean 应显著上升且 score 上升；若下一轮 score 仍 ≤5 且 len 仍 400，则假设被证伪，说明瓶颈在探索/动作层而非奖励形态。
7. `expected_next_round`：`joint_completion` active_rate 应 >60% 且 episode_sum_mean 上升；`crate_to_dock_progress` 份额下降至 <30%；score 应 >17.69（刷新 best）；len 仍可能接近 400（任务本身难），但 terminated 中应开始出现成功迹象（score_range 上界上升）。
8. `main_risk`：势函数在货箱远离坞时 Phi≈0，梯度稀疏，可能退化为"无信号"；已用几何平均 + 小权重状态项缓解。自检：①什么都不做、货箱静止在初始位置（dist≈1.0，align≈0.5，speed≈0）→ Phi≈0.05，单步奖励≈0.007；②正在把货箱推向坞（dist 从 1.0→0.9，speed≈1.0，align≈0.7）→ Phi 从 0.05→0.09，改善量≈0.04，单步奖励≈0.05 > ①。满足 ②>①。

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

    # 货箱朝向对齐度 [0,1]：cos 误差映射
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5
    n_align_cos = next_obs[10]
    next_align_factor = (n_align_cos + 1.0) * 0.5

    contact = obs[14]

    # ---------- 联合完成度势函数 Phi ----------
    # 进入度: 距离坞越近越高, 0.5 处衰减到 0
    enter = max(0.0, 1.0 - dist / 0.5)
    next_enter = max(0.0, 1.0 - next_dist / 0.5)

    # 静止度: 速度越低越高
    still = 1.0 / (1.0 + 2.0 * crate_speed)
    next_still = 1.0 / (1.0 + 2.0 * next_crate_speed)

    # 几何平均联合 (防塌缩)
    phi = (max(enter, 1e-6) * max(align_factor, 1e-6) * max(still, 1e-6)) ** (1.0 / 3.0)
    next_phi = (max(next_enter, 1e-6) * max(next_align_factor, 1e-6) * max(next_still, 1e-6)) ** (1.0 / 3.0)

    # ---------- 主信号: 势函数改善量 ----------
    phi_delta = next_phi - phi
    progress_reward = 20.0 * phi_delta

    # 小权重状态项: 维持接近完成时的持续梯度
    completion_reward = 0.5 * next_phi

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
        "joint_completion": float(progress_reward + completion_reward),
        "crate_to_dock_progress": float(progress_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```