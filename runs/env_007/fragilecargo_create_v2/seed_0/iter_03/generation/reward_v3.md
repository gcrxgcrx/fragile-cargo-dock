**REBUILD MODE — 新骨架设计**

1. `evidence`：iter2 全部 20/20 episode 在 77 步内 terminated（early_terminal 20/20），score=-102.84；唯一活跃组件 `crate_to_dock_progress` episode_sum_mean=-16.0、signed_share=-99%、active_rate=100%，即主信号每步为负且主导全部回报；`crate_dock_alignment`/`crate_settling`/`soft_contact_penalty` active_rate=0（僵尸），`boundary_avoidance` 仅 9.3%。iter1 同骨架 len=400 全截断、score=4.50，说明 iter2 的 `-0.5*dist` 凸化项把"远离坞"变成持续负收益，agent 被推向快速失败。

2. `behavior_diagnosis`：策略在 77 步内快速触发失败终止（crate_out_of_bounds / cart_out_of_bounds / crate_damaged 之一），而非徘徊。主信号 `8.0*progress*gate + 0.5*(-dist)` 中 `-0.5*dist` 是**全局持续状态惩罚**：只要货箱离坞远就每步扣分，与"推进任务"无关，agent 学到的最优解是尽快结束 episode 以停止扣分 → 短 ep + 大负分。

3. `signal_completeness`：职责缺口在于——(a) 主信号被"距离状态惩罚"污染，不是纯推进；(b) 无任何"联合完成"引导（near + aligned + slow 同时满足）；(c) 无健康门控，失败终止前主奖励仍可累积。obs[12,13]（到坞偏移）、obs[8,9]（货箱速度）、obs[10,11]（货箱朝向）、obs[14]（接触）、obs[15-17]（障碍）均可用，信号齐全，属**校准+结构**问题，非信号缺失。

4. `selected_level`：Level 3 重建（触发条件：同骨架连续 2 轮未刷新 best 且 iter2 相对 iter1 大幅倒退 -107.35；且 iter2 主信号数学形态塌缩为"距离惩罚"）。

5. `selected_intervention`：更换主信号框架为 **joint_condition_proxy（几何平均）+ improvement_delta（纯推进，无距离状态惩罚）**：
   - 主信号 = `w_prog * (dist - next_dist)`（纯改善量，无 `-k*dist` 项）
   - 联合完成 proxy = `(f_near * f_align * f_slow) ** (1/3)`，三个连续 bounded factor，几何平均防塌缩
   - 健康门控：`boundary_avoidance` 改为 hinge（阈值 0.85，终止边界 1.0 的 85%），只在接近越界时生效
   - 删除 `crate_settling`、`soft_contact_penalty`（active_rate=0 僵尸），把"近静止"并入联合 proxy 的 `f_slow`
   - 删除 `crate_dock_alignment` 独立项（active_rate=0），把"对齐"并入联合 proxy 的 `f_align`

6. `falsifiable_hypothesis`：iter2 的负分来自 `-0.5*dist` 全局状态惩罚诱导快速终止。移除该惩罚、改用纯 improvement_delta + 联合完成 proxy 后，agent 不再因"离坞远"被持续扣分，episode 长度应回升，score 应从 -102 转正。

7. `expected_next_round`：len 从 77 → >200；`crate_to_dock_progress` episode_sum_mean 从 -16 → 正值（>0）；`joint_completion_proxy` active_rate > 0（预期 >30%）；terminated 率从 20/20 → <15/20；score 从 -102.84 → >0。

8. `main_risk`：几何平均 proxy 若三个 factor 长期有一个趋近 0（如货箱一直未对齐），proxy 会塌缩为 0，退化为稀疏信号。缓解：每个 factor 设下限（如 `f_align` 最低 0.1），保证非零梯度。

**必做自检（代入两次）**：
- ① 什么都不做，货箱静止在初始位置（假设 dist≈0.8，next_dist≈0.8，align_factor≈0.5，crate_speed≈0）：
  - progress = 0 → 主信号 = 0
  - f_near = max(0, 1-0.8/0.5) = 0 → proxy = 0
  - 总奖励 ≈ 0
- ② 正在把货箱推向坞（dist=0.8→0.75，crate_speed≈0.3，align_factor≈0.5）：
  - progress = 0.05 → 主信号 = 8.0*0.05 = 0.40
  - f_near = max(0, 1-0.75/0.5) = 0 → proxy = 0（尚未进入坞区）
  - 总奖励 ≈ 0.40 > 0 ✓
- ② > ① 成立（0.40 > 0）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱速率（世界系，m/s）
    crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号 A: 纯推进改善量（无距离状态惩罚） ----------
    progress = dist - next_dist  # 靠近为正
    progress_reward = 8.0 * progress

    # ---------- 主信号 B: 联合完成 proxy（几何平均，防塌缩） ----------
    # f_near: 货箱接近坞（阈值 0.5，终止边界约 0.15 的 3x 缓冲）
    f_near = max(0.0, 1.0 - dist / 0.5)
    # f_align: 朝向对齐（下限 0.1 防塌缩）
    f_align = max(0.1, align_factor)
    # f_slow: 货箱近静止（速度 < 0.5 m/s 时给分，下限 0.1 防塌缩）
    f_slow = max(0.1, 1.0 - crate_speed / 0.5)
    # 几何平均
    joint_proxy = (f_near * f_align * f_slow) ** (1.0 / 3.0)
    # 仅在货箱已进入坞区附近时给联合奖励，避免早期刷分
    if dist < 0.5:
        joint_reward = 3.0 * joint_proxy
    else:
        joint_reward = 0.0

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
        + joint_reward
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "joint_completion_proxy": float(joint_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```