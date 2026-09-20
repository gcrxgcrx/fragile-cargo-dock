1. `evidence`：20/20 全部 truncated（len=400 满额），无 terminated，说明策略既未成功也未触发失败终止；`crate_settling_and_alignment` 占 magnitude 76.9% 且 active_rate 79.7%，但 score 仅 3.86（target 250 的 1.5%），`crate_to_dock_progress` 仅 14.5、active_rate 47.2%，`out_of_bounds_penalty` 完全为 0（僵尸），`fragile_impact_penalty` active_rate 0.5%（近僵尸）。三轮同骨架 best 停滞在 3.86，方向已证伪。

2. `behavior_diagnosis`：策略在 400 步内持续获得 settle 类奖励（proxy 徘徊），但从未把货箱真正推进 dock 并停稳——即"占据近似状态刷分"而非完成任务；主进展信号太弱（20.0 权重 × 每步微小 delta），被 settle 的持续小奖励淹没。

3. `signal_completeness`：职责不完备。缺失"联合完成条件"的连续代理（near + slow + aligned + 持续），且主进展信号被 proxy 主导；`out_of_bounds_penalty` 是僵尸组件（active_rate=0），`fragile_impact_penalty` 几乎不激活。

4. `selected_level`：Level 3 重建（同骨架连续 3 轮未刷新 best，且历史最佳 3.86 < target×0.5=125）。

5. `selected_intervention`：更换主信号框架——用 `joint_condition_proxy`（几何平均，非塌缩）作为主完成信号，直接对齐 success 定义（dock 内 + 低速 + 朝向对齐 + 持续），并把 `improvement_delta` 距离进展作为辅助；删除僵尸 `out_of_bounds_penalty`，把边界与冲击改为轻量 hinge 健康约束。

6. `falsifiable_hypothesis`：若主信号改为"联合完成代理"（几何平均的 near×slow×aligned），策略将被迫同时满足三个子条件而非停留在单一近似状态，score 应显著上升；若下一轮 score 仍 ≈3.86 且 joint 组件 active_rate 低，则说明该代理不可达或权重不足，假设被证伪。

7. `expected_next_round`：`joint_dock_completion` active_rate 应从 0 升至 >30%，其 episode_sum_mean 成为最大正分量；`crate_to_dock_progress` 保持正；score 应 >10（至少 2.5x 提升）；len 仍可能 400（未成功），但若成功出现则 terminated>0。

8. `main_risk`：几何平均中任一 factor（如低速）长期趋近 0 会导致主信号塌缩为 0，agent 失去梯度；用 `(f1*f2*f3)**(1/3)` 并给每个 factor 设下限缓解。

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

    # ---------- A. joint_dock_completion (MAIN, geometric mean, non-collapsing) ----------
    # factor 1: near dock (continuous, 1 at center, 0 at radius 0.6)
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    f_near = max(0.0, 1.0 - dist_new / 0.6)

    # factor 2: slow (1 at rest, decays with speed)
    f_slow = 1.0 / (1.0 + 30.0 * crate_speed)

    # factor 3: aligned (|cos| of crate heading vs dock axis; dock axis assumed x)
    f_align = abs(crate_cos)

    # geometric mean with floors to avoid total collapse
    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    w_joint = 6.0
    r_joint = w_joint * joint

    # ---------- B. crate_to_dock_progress (auxiliary improvement delta) ----------
    dist_old = (dx * dx + dy * dy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 15.0
    r_progress = w_progress * progress

    # ---------- C. fragile_impact_penalty (hinge, only on contact) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.15)
    w_impact = 4.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. boundary_health_penalty (hinge, replaces zombie) ----------
    bx = max(0.0, abs(cart_x) - 0.9)
    by = max(0.0, abs(cart_y) - 0.9)
    w_bounds = 6.0
    r_bounds = -w_bounds * (bx + by)

    # ---------- total ----------
    total_reward = r_joint + r_progress + r_impact + r_bounds

    components = {
        "joint_dock_completion": r_joint,
        "crate_to_dock_progress": r_progress,
        "fragile_impact_penalty": r_impact,
        "boundary_health_penalty": r_bounds,
    }

    return float(total_reward), components
```