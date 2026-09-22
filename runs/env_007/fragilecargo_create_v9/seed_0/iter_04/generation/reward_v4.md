1. `evidence`: score=279.63 (>target 250); 18/20 terminated and `terminal_success` mean=270.0=0.90×300 → all 18 terminations are successes; `dock_enter` mean=5.0 with 0.5% active → all 20 episodes (including the 2 truncations) got the crate fully inside the dock; `progress`/`approach_cargo` carry only 1.4%/0.4% of the magnitude, `hard_hit`=0.0 and `terminal_failure`=0.0 inactive.

2. `behavior_diagnosis`: the policy reliably pushes the crate into the dock (20/20 `dock_enter`), but 2/20 episodes truncate **while inside the dock** — they never hold aligned + crate_speed<0.05 for 10 consecutive steps before the time budget ends. The failure is in the *settle* phase, not the transport phase.

3. `signal_completeness`: all 9 author terms are present and correctly scaled; the one missing responsibility is a dense gradient for the success preconditions (crate alignment and crate speed) once the crate is inside the dock — the only signal there is the 0.5%-active binary `terminal_success`.

4. `selected_level`: Level 2 — sparse→dense on the success precondition (active_rate ≈ 0.5% for the only settle-phase signal; the `dock_enter`=20/20 vs `terminal_success`=18/20 gap pins the failure to settling).

5. `selected_intervention`: add **one** dense potential-based `settle` term, `Φ(next)−Φ(prev)` with `Φ = gate(crate→dock distance) × align_factor(crate_cos_heading) × speed_factor(crate speed)`, coefficient 3.0; the gate saturates at 1 inside dock tolerance and ramps to 0 by 0.20 m, so it cannot pull the crate short of the dock. Every existing component (including `terminal_success`=+300) is left byte-identical.

6. `falsifiable_hypothesis`: the two truncating episodes stall because nothing rewards reducing alignment error / crate speed after the crate is in the dock. Because the new term is a bounded potential difference (Φ(next)−Φ(prev), net-zero on any excursion and non-farmable by hovering) whose gradient points exactly along "align + slow down inside the dock", the two stalling episodes should reach the 10-step stability chain within budget.

7. `expected_next_round`: `settle` active_rate > 3% and `terminal_success` episode_sum_mean ≥ 270 with `truncated` ≤ 1/20; `score` ≥ 279.63 (should rise toward 300–310 via the two previously-truncated episodes); `episode_length` mean should fall slightly (more episodes end by success before timeout); `dock_enter` stays 5.0 (20/20).

8. `main_risk`: the pre-dock ramp of the gate (0.12–0.20 m) could slow the approach before full containment and turn some 20/20 entries into near-misses if the crate is left stopped just outside tolerance; the ramp width is deliberately narrow so Φ is ~0 beyond 0.20 m, and any such regression would show up immediately as `dock_enter` < 5.0 next round.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- module-level episode state (no self, no helper, no info) ----
    state = getattr(compute_reward, "_v4_state", None)
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
        compute_reward._v4_state = state

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

    # ---- NEW: dense potential shaping for the settle phase (align + slow inside dock) ----
    # Potential Phi(s) = gate(crate->dock dist) * align_factor * speed_factor, in [0, 1].
    # Reward = 3.0 * (Phi(next) - Phi(prev)): bounded, net-zero on any excursion,
    # non-farmable by hovering, gradient points along "align + decelerate in the dock".
    # Gate saturates at 1 inside the dock tolerance and reaches 0 by 0.20 m, so it
    # gives no pull to stop short of full containment.
    d_prev_x = float(obs[12]) * 5.0
    d_prev_y = float(obs[13]) * 4.0
    d_prev = (d_prev_x * d_prev_x + d_prev_y * d_prev_y) ** 0.5
    gate_prev = (0.20 - d_prev) / 0.08
    if gate_prev < 0.0:
        gate_prev = 0.0
    if gate_prev > 1.0:
        gate_prev = 1.0
    align_prev = (float(obs[10]) - 0.866) / 0.134
    if align_prev < 0.0:
        align_prev = 0.0
    if align_prev > 1.0:
        align_prev = 1.0
    spd_prev = ((float(obs[8]) * 3.0) ** 2 + (float(obs[9]) * 3.0) ** 2) ** 0.5
    sfactor_prev = (0.6 - spd_prev) / 0.6
    if sfactor_prev < 0.0:
        sfactor_prev = 0.0
    if sfactor_prev > 1.0:
        sfactor_prev = 1.0
    phi_prev = gate_prev * align_prev * sfactor_prev

    d_next = cd_next
    gate_next = (0.20 - d_next) / 0.08
    if gate_next < 0.0:
        gate_next = 0.0
    if gate_next > 1.0:
        gate_next = 1.0
    align_next = (float(next_obs[10]) - 0.866) / 0.134
    if align_next < 0.0:
        align_next = 0.0
    if align_next > 1.0:
        align_next = 1.0
    sfactor_next = (0.6 - crate_speed) / 0.6
    if sfactor_next < 0.0:
        sfactor_next = 0.0
    if sfactor_next > 1.0:
        sfactor_next = 1.0
    phi_next = gate_next * align_next * sfactor_next

    settle = 3.0 * (phi_next - phi_prev)

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