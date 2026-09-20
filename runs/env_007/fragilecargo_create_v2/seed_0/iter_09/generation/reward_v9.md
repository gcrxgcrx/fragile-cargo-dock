**REBUILD MODE — 新骨架设计**

1. `evidence`：8 轮中 best=17.69（iter4），此后 4 轮同骨架族全部未刷新 best；本轮 `joint_completion_gate` 的 episode_sum_mean=205.3、signed_share=95.4%、active_rate=76.6%，而 `crate_to_dock_progress` 仅 7.04（3.3%）——**状态值门控彻底压过推进信号**，agent 只要"停在坞附近/低速/对齐"就持续收分，无需真正推进；terminated 仅 2/20，18/20 超时，说明策略在徘徊而非失败。

2. `behavior_diagnosis`：策略学会了"占据一个 joint_gate 高的静止状态"（低速因子 `1/(1+3*speed)` 在静止时=1，dock_factor 在近坞时≈1），于是**不推箱、不移动**即可拿 205 分/回合；这正是"辅助低速因子作为全局持续奖励"的经典失效模式——它惩罚了任务推进本身（推箱必然产生速度）。

3. `signal_completeness`：信号齐全（obs[12,13] 距离、obs[8,9] 速率、obs[10,11] 朝向、obs[14] 接触、obs[15-17] 障碍、obs[18] 时间），但**职责分配错误**：把"完成度状态"当主信号，把"推进"当辅助。需重建为"推进为主、完成度为门控放大、静止只在已接近目标时才有意义"。

4. `selected_level`：Level 3 重建（触发条件：同骨架族连续 ≥3 轮未刷新 best，且 iter4 后 4 轮全部低于 best）。

5. `selected_intervention`：更换主信号框架为 **improvement_delta 为主 + 几何平均联合门控放大 + 仅在近坞时激活的静止/对齐门**。具体：
   - 主信号：`progress = dist - next_dist`（bounded），权重 10.0；
   - 门控：`gate = (dock_factor * align_factor * slow_factor) ** (1/3)`，其中 `slow_factor` 只在 `dock_factor > 0.3`（即已接近坞）时才生效，否则=1，**杜绝"远处静止收分"**；
   - 完成度状态值权重从 1.0 降到 0.15，且乘 `dock_factor`，确保远离坞时几乎为 0；
   - 保留 hinge 边界惩罚与接触冲击惩罚（量级已很小）。

6. `falsifiable_hypothesis`：若主信号改为"推进 delta 为主、静止门控仅在近坞激活"，则 agent 无法通过"停在坞附近不动"收分，必须真正把箱子推进坞内；下一轮 `crate_to_dock_progress` 的 magnitude_share 应显著上升（>30%），`joint_completion_gate` 的 magnitude_share 应显著下降（<40%），且 score 应 > 17.69。

7. `expected_next_round`：`crate_to_dock_progress` magnitude_share 从 3.8% → >30%；`joint_completion_gate` magnitude_share 从 95.4% → <40%；episode_length 保持 350-400（不因新惩罚暴跌）；score 目标 > 17.69，理想 > 50。

8. `main_risk`：若 `slow_factor` 门控仍太强，agent 可能学会"把箱子推到坞口后停住"——但此时 dock_factor 已高，推进 delta 仍会奖励继续推入；若 progress 权重过大导致 agent 高速冲撞损坏货箱，`soft_contact_penalty` 会部分抑制，但需观察 `crate_damaged` 终止率。

**必做自检（代入两次）**：
- ① 什么都不做、箱子静止在初始位置：dist≈1.0（远），`progress=0`，`dock_factor=max(0,1-1.0/0.6)=0`，`gate=0`，`completion=0.15*0=0`，`boundary=0`，`impact=0` → **total ≈ 0.0**
- ② 正在把箱子推向坞（dist 从 1.0→0.95，箱子有速度 0.5 m/s）：`progress=(1.0-0.95)/(1+0.05*20)=0.025`，`progress_reward=10*0.025*(1+2*0)=0.25`（dock_factor 仍=0，gate=0），`completion=0`，`impact=0` → **total ≈ 0.25 > 0.0** ✅

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

    # 货箱朝向对齐度 [0,1]
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5

    contact = obs[14]

    # ---------- 主信号: 推进改善量 (bounded) ----------
    raw_progress = dist - next_dist  # 靠近为正
    progress = raw_progress / (1.0 + abs(raw_progress) * 20.0)

    # ---------- 联合门控: 仅在接近坞时激活静止/对齐因子 ----------
    dock_factor = max(0.0, 1.0 - dist / 0.6)  # 距离坞 <0.6 才 >0

    # 关键: slow_factor 只在已接近坞时生效, 否则=1 (杜绝远处静止收分)
    if dock_factor > 0.3:
        slow_factor = 1.0 / (1.0 + 3.0 * crate_speed)
    else:
        slow_factor = 1.0

    # 几何平均联合门控 (避免乘积塌缩)
    gate = (max(dock_factor, 1e-6) * max(align_factor, 1e-6) * max(slow_factor, 1e-6)) ** (1.0 / 3.0)

    # 推进收益: 基础 + 门控放大 (接近目标时推进更值钱)
    progress_reward = 10.0 * progress * (1.0 + 2.0 * gate)

    # ---------- 完成度状态值 (极小权重, 且必须乘 dock_factor) ----------
    completion_reward = 0.15 * dock_factor * (gate ** 2)

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