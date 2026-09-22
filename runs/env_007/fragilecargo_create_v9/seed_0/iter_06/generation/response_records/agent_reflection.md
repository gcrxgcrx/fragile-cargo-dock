# Response Record

## 诊断

1. `evidence`：score=309.55 已超 target 250（达成率 123.8%），terminated 20/20、truncated 0/20，len=180.9；组件表中 terminal_success 占 95.5% 份额（active 0.6%），progress/approach_cargo/settle 激活但量级很小（合计 ~7.8%），hard_hit 与 terminal_failure 完全休眠（0%）。历史 best 由 iter5 刷新至 309.55。
2. `behavior_diagnosis`：策略已能稳定完成全流程——20/20 回合都以 docked_success 终止，没有快速失败、没有徘徊、没有刷分；由于成功几乎必发，得分由 +300 主导，其余塑形项只贡献个位数。
3. `signal_completeness`：9 个作者职责（approach_cargo / progress / dock_enter / roughness / action_cost / time_cost / hard_hit / terminal_success / terminal_failure）在 obs 上全部可达且已被实现；hard_hit、terminal_failure 休眠是"策略主动避开了它们"而非信号缺失。无需新增职责。
4. `selected_level`：Level 3 重建被触发，但证据（20/20 成功、score 远超 target）表明失败并不存在——因此"重建"的正确含义是**忠实重写作者结构**，而不是换一套新的数学骨架；换骨架只会破坏已收敛的激励。
5. `selected_intervention`：以 `improvement_delta`（approach_cargo、progress）、一次性 bonus（dock_enter）、弱代理惩罚（roughness）、二次动作代价、稀疏事件（hard_hit / terminal_*）重写 9 项；另保留一个**势函数形式**的 settle 项（Phi(next)-Phi(prev)，策略不变），并把 roughness 量级压到推进项之下。
6. `falsifiable_hypothesis`：若重建确实忠实，策略会重新取得 20/20 docked_success 且 score≈300-313；若我误把 roughness/terminal 权重放大（如 iter2 的 204）或把 settle 写成非势函数形式，成功率会下降、terminated 中会出现非成功终止。
7. `expected_next_round`：score ∈ [300, 313]（不低于 280），terminated=20/20、truncated=0/20，len≈170-200，terminal_success active_rate≈0.5-0.7%，hard_hit≈0%、terminal_failure≈0%。
8. `main_risk`：episode 边界处若 next_obs 已是新回合初态，prev-势会注入虚假的 approach/progress；以及窄的 settle 门（0.35 m）可能给最终精细对准带来偏差。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Faithful reconstruction of the author reward structure.
    # 9 author terms + one policy-invariant potential-based settle term.
    # ============================================================

    # ---------- module-level episode state (no self / helper / info) ----------
    st = getattr(compute_reward, "_r6_state", None)
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
        compute_reward._r6_state = st

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

    # ---------- settle : potential-based shaping (policy-invariant) ----------
    # Phi = gate(crate->dock dist) * align * ground-speed factor, all in [0,1].
    # reward = 3.0 * (Phi(next) - Phi(prev)) -> telescopes, net-zero on excursions,
    # provides a dense gradient for the 0.05-0.35 m capture band.
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

    settle = 3.0 * (phi_next - phi_prev)

    # ---------- assemble ----------
    components = {
        "approach_cargo": float(approach_cargo),
        "progress": float(progress),
        "dock_enter": float(dock_enter),
        "settle": float(settle),
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
