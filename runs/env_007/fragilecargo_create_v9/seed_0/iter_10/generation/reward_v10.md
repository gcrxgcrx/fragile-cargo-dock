1. `evidence`：score=309.77（123.9% target），20/20 terminated、len=167.85、无 truncation/early_terminal，terminal_success=300 一次性（active 0.6%）；settle=3.62、active 21.9%；上一轮唯一改动是 settle 系数 3.0→4.0，只换回 +0.22（预测 +0.9）。

2. `behavior_diagnosis`：策略已完成任务链（入坞→对齐→静止 10 步→终止），无 fast-fail、无徘徊、无刷分；失败在信号层——`settle` 是 γ=1 势函数差分，沿整条轨迹望远镜式求和＝`4.0·(Φ_T−Φ_0)` 常数，对策略梯度零贡献。

3. `signal_completeness`：作者 9 项职责齐备且可达（成功链完整、20/20 终止）；缺的是近坞逐帧梯度而非新职责。

4. `selected_level`：Level 2（结构变换）。触发：Level-1 尺度修复无效，框架禁止重复调同一系数。

5. `selected_intervention`：只改 `settle` —— 保留 4.0 势函数差分，叠加有界稠密项 `dense = 1.0·align_next·sfactor_next`，仅在成功谓词区（|obs[12]|≤0.024 且 |obs[13]|≤0.030）发放；其余 9 项权重与所有门带/斜坡不动。本轮为格式修复，方向不变。

6. `falsifiable_hypothesis`：因环境在连续 10 步成立后立即终止，红利最多累积约 10 步、无法徘徊刷取，应给出与行为同向的小幅正增量且不改变成功率/回合长度。证伪标志：settle active_rate 降至 0；或 len 显著变长（坞内滞留）；或成功数跌破 20/20。

7. `expected_next_round`：settle episode_sum_mean 3.62→~7–9、active_rate→~6–10%；score→~312–316；len 维持 165–175；terminated 20/20；hard_hit / terminal_failure 仍 0；progress / approach_cargo / dock_enter 基本不变。

8. `main_risk`：容差内部分满足（未对齐/偏快）滞留可抬高 settle 而不终止，拉长 episode 并增加 time_cost；若下轮 len 上升，应把稠密项系数 1.0 降到 0.3。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Faithful reconstruction of the author reward structure.
    # 9 author terms (exact weights) + potential-based settle term
    # + NEW bounded dense in-dock settle bonus.
    #
    # THIS ROUND (iter 10): LEVEL-2 STRUCTURAL CHANGE on the ONLY
    #   self-built component, `settle`.
    #   iter 9 raised the settle coefficient 3.0 -> 4.0 and got only
    #   +0.22 (predicted +0.9): a gamma=1 potential difference
    #   telescopes to a CONSTANT, so it carries zero per-step
    #   learning gradient. Instead of re-tuning that coefficient
    #   (forbidden direction) we ADD a bounded DENSE term that pays
    #   only while the crate is FULLY INSIDE the dock tolerance and
    #   aligned and slow -- i.e. the success predicate region.
    #   Safe from farming: the environment terminates the episode
    #   after 10 consecutive such steps, so the dense term is capped
    #   at ~10 * align * sfactor (~a few points).
    #   All author weights, the gate band (0.35 / 0.08), the align
    #   ramp (0.70) and the speed ramp (0.30) are UNCHANGED.
    #
    # NOTE: this revision is a FORMAT FIX only -- components dict is
    #   fully assigned and (float(total_reward), components) is
    #   returned. No change of direction.
    # ============================================================

    # ---------- module-level episode state (no self / helper / info) ----------
    st = getattr(compute_reward, "_r10_state", None)
    if st is None:
        st = {
            "prev_cc": None,      # previous cart->crate distance (m)
            "prev_cd": None,      # previous crate->dock distance (m)
            "prev_t": None,       # previous time_fraction (episode boundary detector)
            "entered": False,     # crate ever fully inside dock tolerance
            "stable": 0,          # consecutive settled steps inside dock
            "hard": 0,            # accumulated hard-impact proxy count
            "succ_paid": False,
            "fail_paid": False,
        }
        compute_reward._r10_state = st

    # ---------- episode boundary detection ----------
    # time_fraction is monotone inside one episode; a drop means a new episode.
    t_now = float(obs[18])
    if st["prev_t"] is not None and t_now < st["prev_t"] - 0.05:
        st["prev_cc"] = None
        st["prev_cd"] = None
        st["entered"] = False
        st["stable"] = 0
        st["hard"] = 0
        st["succ_paid"] = False
        st["fail_paid"] = False
    st["prev_t"] = t_now

    # ---------- geometry (all restored to metres) ----------
    hx = float(obs[2])
    hy = float(obs[3])

    rel_x = float(obs[6]) * 3.0          # cart->crate, body frame x (m)
    rel_y = float(obs[7]) * 3.0          # cart->crate, body frame y (m)
    nrel_x = float(next_obs[6]) * 3.0
    nrel_y = float(next_obs[7]) * 3.0
    cc_next = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5      # cart->crate dist (m)

    dnx = float(next_obs[12]) * 5.0      # crate->dock signed x (m)
    dny = float(next_obs[13]) * 4.0      # crate->dock signed y (m)
    cd_next = (dnx * dnx + dny * dny) ** 0.5                  # crate->dock dist (m)

    # ---------- approach_cargo : +1.0 / m (signed potential difference) ----------
    if st["prev_cc"] is None:
        st["prev_cc"] = cc_next
    approach_cargo = st["prev_cc"] - cc_next
    st["prev_cc"] = cc_next

    # ---------- progress : +1.0 / m (signed potential difference) ----------
    if st["prev_cd"] is None:
        st["prev_cd"] = cd_next
    progress = st["prev_cd"] - cd_next
    st["prev_cd"] = cd_next

    # ---------- dock_enter : +5.0 one-shot on first full containment ----------
    inside = (abs(float(next_obs[12])) <= 0.024) and (abs(float(next_obs[13])) <= 0.030)
    dock_enter = 0.0
    if inside and not st["entered"]:
        dock_enter = 5.0
        st["entered"] = True

    # ---------- roughness (weak) + hard_hit (sparse) ----------
    # No impulse channel in obs -> closing-speed x contact proxy.
    # Deliberately sub-dominant so the policy is never afraid to push.
    cart_speed = float(obs[4]) * 3.0
    crate_along = float(obs[8]) * 3.0 * hx + float(obs[9]) * 3.0 * hy
    closing = cart_speed - crate_along
    contact = float(obs[14]) > 0.5

    roughness = 0.0
    hard_hit = 0.0
    if contact and closing > 0.0:
        roughness = -0.02 * closing           # scale kept << progress
        if closing > 1.5:                     # high-severity proxy
            hard_hit = -0.5
            st["hard"] += 1

    # ---------- action_cost / time_cost ----------
    action_cost = -0.0005 * (float(action[0]) ** 2 + float(action[1]) ** 2)
    time_cost = -0.002

    # ---------- terminal_success chain: inside + aligned + slow, 10 consecutive ----------
    crate_speed = ((float(next_obs[8]) * 3.0) ** 2 + (float(next_obs[9]) * 3.0) ** 2) ** 0.5
    aligned = float(next_obs[10]) > 0.866     # |crate heading error| < 30 deg
    if inside and aligned and crate_speed < 0.05:
        st["stable"] += 1
    else:
        st["stable"] = 0

    terminal_success = 0.0
    if st["stable"] >= 10 and not st["succ_paid"]:
        terminal_success = 300.0
        st["succ_paid"] = True

    # ---------- terminal_failure chain: OOB or >=3 hard hits ----------
    crate_wx = float(obs[0]) * 5.0 + rel_x * hx - rel_y * hy
    crate_wy = float(obs[1]) * 4.0 + rel_x * hy + rel_y * hx
    out_of_bounds = (
        abs(float(obs[0])) > 1.05
        or abs(float(obs[1])) > 1.05
        or abs(crate_wx) > 5.25
        or abs(crate_wy) > 4.2
    )

    terminal_failure = 0.0
    if (out_of_bounds or st["hard"] >= 3) and not st["fail_paid"]:
        terminal_failure = -100.0
        st["fail_paid"] = True

    # ---------- settle : potential-based shaping + NEW dense bonus ----------
    # Phi = gate(crate->dock dist) * align * ground-speed factor, all in [0,1].
    # potential part = 4.0 * (Phi(next) - Phi(prev)) -> telescopes, net-zero on excursions.
    dpx = float(obs[12]) * 5.0
    dpy = float(obs[13]) * 4.0
    d_prev = (dpx * dpx + dpy * dpy) ** 0.5

    gate_prev = (0.35 - d_prev) / 0.08
    if gate_prev < 0.0:
        gate_prev = 0.0
    if gate_prev > 1.0:
        gate_prev = 1.0

    align_prev = (float(obs[10]) - 0.70) / 0.30
    if align_prev < 0.0:
        align_prev = 0.0
    if align_prev > 1.0:
        align_prev = 1.0

    spd_prev = ((float(obs[8]) * 3.0) ** 2 + (float(obs[9]) * 3.0) ** 2) ** 0.5
    sfactor_prev = (0.30 - spd_prev) / 0.30
    if sfactor_prev < 0.0:
        sfactor_prev = 0.0
    if sfactor_prev > 1.0:
        sfactor_prev = 1.0

    phi_prev = gate_prev * align_prev * sfactor_prev

    gate_next = (0.35 - cd_next) / 0.08
    if gate_next < 0.0:
        gate_next = 0.0
    if gate_next > 1.0:
        gate_next = 1.0

    align_next = (float(next_obs[10]) - 0.70) / 0.30
    if align_next < 0.0:
        align_next = 0.0
    if align_next > 1.0:
        align_next = 1.0

    sfactor_next = (0.30 - crate_speed) / 0.30
    if sfactor_next < 0.0:
        sfactor_next = 0.0
    if sfactor_next > 1.0:
        sfactor_next = 1.0

    phi_next = gate_next * align_next * sfactor_next

    settle_potential = 4.0 * (phi_next - phi_prev)

    # NEW dense term: paid ONLY while the crate is FULLY INSIDE dock tolerance.
    # Bounded in [0, 1]; the environment terminates after 10 consecutive
    # success steps, so this can never be farmed by loitering in the gate band.
    settle_dense = 0.0
    if inside:
        settle_dense = 1.0 * align_next * sfactor_next

    settle = settle_potential + settle_dense

    # ---------- assemble ----------
    components = {
        "approach_cargo": float(approach_cargo),
        "progress": float(progress),
        "dock_enter": float(dock_enter),
        "settle": float(settle),
        "settle_potential": float(settle_potential),
        "settle_dense": float(settle_dense),
        "roughness": float(roughness),
        "action_cost": float(action_cost),
        "time_cost": float(time_cost),
        "hard_hit": float(hard_hit),
        "terminal_success": float(terminal_success),
        "terminal_failure": float(terminal_failure),
    }

    total_reward = (
        approach_cargo
        + progress
        + dock_enter
        + settle
        + roughness
        + action_cost
        + time_cost
        + hard_hit
        + terminal_success
        + terminal_failure
    )

    return float(total_reward), components
```