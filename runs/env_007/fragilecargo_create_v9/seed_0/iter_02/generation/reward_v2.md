1. `evidence`：score=279.63 已高于 target 250（达成率 111.85%），len=194.9，terminated=18/20、truncated=2/20；component 证据显示 terminal_success 的 episode_sum_mean=270=300×0.9，即 18/20 回合拿到成功奖励，但其 active_rate 仅 0.5%（每回合一次、极稀疏）；dock_enter=5.0、progress=4.07、approach_cargo=1.20，roughness=−0.03（16.5%），hard_hit=0、terminal_failure=0；累积记录仅 iter1，new best，无“改A无效”的历史。

2. `behavior_diagnosis`：策略已能可靠把货箱送入坞并达成成功条件（90%），但在 2/20 个种子上超时（truncated），个别回合分数极低（min 8.6），表现为部分 seeds 无法在预算内完成“入坞后对齐+减速并保持 10 步”的收尾阶段；没有失败终止，说明不是撞墙/越界问题，而是收尾子技能学习不足。

3. `signal_completeness`：距离类职责（approach_cargo、progress、dock_enter、terminal_success）齐全且尺度正常（无 >0.5 的惩罚、无极端值支配）。真正的缺口是：成功条件中两个**非距离前提**——货箱朝向对齐（obs[10]）与货箱低速（obs[8]/obs[9]）——目前没有任何 dense 梯度，只能靠极稀疏的 terminal_success（active_rate 0.5%）隐式学到。这是可复现的 sparse→dense 缺口，而非“任务描述关键词联想”。

4. `selected_level`：Level 2（结构变换：sparse→dense）。触发条件：terminal_success active_rate 0.5% < 5%，且 2/20 truncated 表明收尾阶段缺少局部反馈。

5. `selected_intervention`：只新增一个组件 `settle`——在**货箱完全进入坞**（|obs[12]|≤0.024 且 |obs[13]|≤0.030）的门控下，用 obs[10] 构造连续对齐因子、用 obs[8]/obs[9] 构造货箱低速因子，二者相乘乘系数 0.05。其余所有组件（含 terminal_success=+300、dock_enter=+5、progress、approach_cargo、roughness、hard_hit、terminal_failure、action_cost、time_cost）保持与上一轮逐字一致。

6. `falsifiable_hypothesis`：若 2/20 超时确实源于“缺乏收尾梯度”而非种子固有难度，则加入这个有界 dense 项后，策略能更快学会“入坞后对齐并减速”，把超时回合转化为成功（terminal_success 触发率上升）；反之若超时与收尾子技能无关，则成功率和 score 不会改善。

7. `expected_next_round`：terminal_success 的 episode_sum_mean 由 270 升向 ≥285（成功 ≥19/20）；truncated 由 2/20 降至 ≤1/20；score 由 279.63 升到 ~285–300；`settle` 出现 active_rate>0 且 magnitude_share≤0.03%（量级极小、不改变分解结构）；best 不下降。

8. `main_risk`：该项可能与成功终止条件耦合而奖励“在坞内滞留”，但由于环境在满足后 10 步即终止，单回合最多累计 ~0.5，可利用空间被硬性封顶；次风险是轻微扰动策略导致个别种子成功率下降，故把系数压到 0.05 并做 bounded、门控处理。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- module-level episode state (no self, no helper, no info) ----
    state = getattr(compute_reward, "_v2_state", None)
    if state is None:
        state = {
            "prev_cc": None,      # previous cart->crate distance (m)
            "prev_cd": None,      # previous crate->dock distance (m)
            "prev_t": None,       # previous time_fraction, used to detect a fresh episode
            "entered": False,     # has crate ever fully entered dock tolerance
            "stable": 0,          # consecutive settled steps inside dock
            "hard": 0,            # accumulated hard-hit proxy count
            "success_paid": False,
            "fail_paid": False,
        }
        compute_reward._v2_state = state

    # time_fraction is monotone inside an episode; a drop means a new episode started
    t_now = float(obs[18])
    if state["prev_t"] is not None and t_now < state["prev_t"] - 0.05:
        state["prev_cc"] = None
        state["prev_cd"] = None
        state["entered"] = False
        state["stable"] = 0
        state["hard"] = 0
        state["success_paid"] = False
        state["fail_paid"] = False
    state["prev_t"] = t_now

    heading_x = float(obs[2])
    heading_y = float(obs[3])

    # ---- cart -> crate body-frame relative vector (metres) ----
    rel_x = float(obs[6]) * 3.0
    rel_y = float(obs[7]) * 3.0
    cc_now = (rel_x * rel_x + rel_y * rel_y) ** 0.5

    nrel_x = float(next_obs[6]) * 3.0
    nrel_y = float(next_obs[7]) * 3.0
    cc_next = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5

    # ---- crate -> dock offset (metres) ----
    dock_x = float(next_obs[12]) * 5.0
    dock_y = float(next_obs[13]) * 4.0
    cd_next = (dock_x * dock_x + dock_y * dock_y) ** 0.5

    # ---- signed potential-difference signals ----
    if state["prev_cc"] is None:
        state["prev_cc"] = cc_next
    approach_cargo = 1.0 * (state["prev_cc"] - cc_next)
    state["prev_cc"] = cc_next

    if state["prev_cd"] is None:
        state["prev_cd"] = cd_next
    progress = 1.0 * (state["prev_cd"] - cd_next)
    state["prev_cd"] = cd_next

    # ---- one-shot bonus on first full dock containment, inferred from obs only ----
    inside = (abs(float(next_obs[12])) <= 0.024) and (abs(float(next_obs[13])) <= 0.030)
    dock_enter = 0.0
    if inside and not state["entered"]:
        dock_enter = 5.0
        state["entered"] = True

    # ---- NEW dense settling shaping (sparse->dense for the success prerequisites) ----
    # Gated strictly on full dock containment; bounded, cannot dominate.
    # Supplies a local gradient toward the two non-distance success conditions:
    #   crate alignment (obs[10]) and low crate speed (obs[8], obs[9]).
    settle = 0.0
    if inside:
        cos_err = float(next_obs[10])
        align_factor = (cos_err - 0.5) / 0.5      # 0 at 60deg, 0.73 at 30deg, 1 at 0deg
        if align_factor < 0.0:
            align_factor = 0.0
        elif align_factor > 1.0:
            align_factor = 1.0

        crate_spd = ((float(next_obs[8]) * 3.0) ** 2 + (float(next_obs[9]) * 3.0) ** 2) ** 0.5
        speed_factor = 1.0 - crate_spd / 0.5      # 1 at rest, 0.9 at 0.05 m/s, 0 at 0.5 m/s
        if speed_factor < 0.0:
            speed_factor = 0.0
        elif speed_factor > 1.0:
            speed_factor = 1.0

        settle = 0.05 * align_factor * speed_factor

    # ---- roughness proxy from contact x closing speed (no impulse channel in obs) ----
    cart_speed = float(obs[4]) * 3.0
    crate_along = float(obs[8]) * 3.0 * heading_x + float(obs[9]) * 3.0 * heading_y
    closing = cart_speed - crate_along
    contact = float(obs[14]) > 0.5

    roughness = 0.0
    hard_hit = 0.0
    if contact and closing > 0.0:
        roughness = -0.02 * closing            # deliberately much weaker than progress
        if closing > 1.5:                      # high-severity contact proxy
            hard_hit = -0.5
            state["hard"] += 1

    # ---- effort / time bookkeeping ----
    action_cost = -0.0005 * (float(action[0]) ** 2 + float(action[1]) ** 2)
    time_cost = -0.002

    # ---- derived settling chain feeding terminal_success ----
    crate_speed = ((float(next_obs[8]) * 3.0) ** 2 + (float(next_obs[9]) * 3.0) ** 2) ** 0.5
    aligned = float(next_obs[10]) > 0.866       # |heading error| < 30 deg via cos proxy
    if inside and aligned and crate_speed < 0.05:
        state["stable"] += 1
    else:
        state["stable"] = 0

    terminal_success = 0.0
    if state["stable"] >= 10 and not state["success_paid"]:
        terminal_success = 300.0
        state["success_paid"] = True

    # ---- derived failure chain feeding terminal_failure ----
    crate_wx = float(obs[0]) * 5.0 + rel_x * heading_x - rel_y * heading_y
    crate_wy = float(obs[1]) * 4.0 + rel_x * heading_y + rel_y * heading_x
    out_of_bounds = (
        abs(float(obs[0])) > 1.05
        or abs(float(obs[1])) > 1.05
        or abs(crate_wx) > 5.25
        or abs(crate_wy) > 4.2
    )

    terminal_failure = 0.0
    if (out_of_bounds or state["hard"] >= 3) and not state["fail_paid"]:
        terminal_failure = -100.0
        state["fail_paid"] = True

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