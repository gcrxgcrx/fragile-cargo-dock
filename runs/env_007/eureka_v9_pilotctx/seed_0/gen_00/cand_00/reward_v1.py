# Module-level state for episode boundary detection and one-time events.
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_FAIL_PAID = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------------
    # Self-check records (approximate per-step values):
    #   R_idle    ~ -0.002  (only time cost)
    #   R_push    ~  0.05   (progress + approach - roughness - small costs)
    #   R_settled ~  0.498  (settled bonus +0.5 - time cost - action cost)
    # => R_push > R_idle, R_settled > R_push.
    #
    # Self-check ③: closing=1.0 m/s -> roughness ~ -0.50
    #                closing=0.05 m/s -> roughness ~ -0.00125
    #                difference ~ 0.50, >= normal push progress ~0.05.
    #
    # Self-check ④: same settled state 12x -> each step adds ~0.498,
    #                so cumulative reward grows linearly.
    #
    # Self-check ⑤: boundary at center -> boundary_guard = 0.
    #                |obs[0]|=1.05 -> boundary_guard ~ -0.4 (plus other terms).
    # ------------------------------------------------------------------

    # Detect new episode via monotonic time fraction obs[18].
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAIL_PAID[0] = False
    _PREV_T[0] = t

    components = {}

    # --------------------------------------------------------------
    # 1. approach_cargo: cart -> crate distance decrease this frame.
    #    obs[6],obs[7] are body-frame offsets normalised by 3.0 m.
    # --------------------------------------------------------------
    dx1 = obs[6] * 3.0
    dy1 = obs[7] * 3.0
    dist_cc_prev = (dx1 * dx1 + dy1 * dy1) ** 0.5

    dx2 = next_obs[6] * 3.0
    dy2 = next_obs[7] * 3.0
    dist_cc_next = (dx2 * dx2 + dy2 * dy2) ** 0.5

    approach_cargo = dist_cc_prev - dist_cc_next
    components["approach_cargo"] = 1.0 * approach_cargo

    # --------------------------------------------------------------
    # 2. progress: crate -> dock distance decrease this frame.
    #    obs[12] is x offset / warehouse half-width  (5.0 m)
    #    obs[13] is y offset / warehouse half-height (4.0 m)
    # --------------------------------------------------------------
    ddx1 = obs[12] * 5.0
    ddy1 = obs[13] * 4.0
    dist_cd_prev = (ddx1 * ddx1 + ddy1 * ddy1) ** 0.5

    ddx2 = next_obs[12] * 5.0
    ddy2 = next_obs[13] * 4.0
    dist_cd_next = (ddx2 * ddx2 + ddy2 * ddy2) ** 0.5

    progress = dist_cd_prev - dist_cd_next
    components["progress"] = 1.0 * progress

    # --------------------------------------------------------------
    # 3. dock_enter: +5 once when crate first fully enters tolerance.
    #    Tolerance from environment facts:
    #    |obs[12]| <= 0.024 and |obs[13]| <= 0.030
    # --------------------------------------------------------------
    in_dock = (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030)
    dock_enter = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0
    components["dock_enter"] = dock_enter

    # --------------------------------------------------------------
    # 4. roughness: observable proxy for contact impulse.
    #    closing = cart forward speed minus crate velocity along heading.
    #    Quadratic form keeps normal pushing cheap but strongly punishes
    #    high-speed contact (self-check ③).
    # --------------------------------------------------------------
    closing = obs[4] * 3.0 - (next_obs[8] * 3.0 * obs[2] + next_obs[9] * 3.0 * obs[3])
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.5 * contact * (closing ** 2)
    components["roughness"] = roughness

    # --------------------------------------------------------------
    # 5. hard_hit: fixed penalty for one hard impact in this step.
    #    Proxy: contact and closing > 0.8 m/s.
    #    Also count hard hits for terminal_failure.
    # --------------------------------------------------------------
    hard_hit = 0.0
    if contact > 0.5 and closing > 0.8:
        hard_hit = -0.5
        _HARD_HITS[0] += 1
    components["hard_hit"] = hard_hit

    # --------------------------------------------------------------
    # 6. action_cost: -0.0005 * sum(action^2)
    # --------------------------------------------------------------
    action_cost = -0.0005 * (action[0] ** 2 + action[1] ** 2)
    components["action_cost"] = action_cost

    # --------------------------------------------------------------
    # 7. time_cost: -0.002 per step
    # --------------------------------------------------------------
    time_cost = -0.002
    components["time_cost"] = time_cost

    # --------------------------------------------------------------
    # 8. settled_bonus: required per-step positive reward when the
    #    crate is inside dock, aligned, and slow.
    #    This is the "停稳期每步收益" required by the prompt.
    #    It is paid every step the predicate holds; no gate on _PAID.
    # --------------------------------------------------------------
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    aligned = next_obs[10] > 0.8660254  # cos(30 deg)
    slow = crate_speed < 0.05
    settled = 1.0 if (in_dock and aligned and slow) else 0.0
    settled_bonus = 0.5 * settled
    components["settled_bonus"] = settled_bonus

    # --------------------------------------------------------------
    # 9. terminal_success: +300 once when settled predicate holds
    #    for 10 consecutive steps.
    # --------------------------------------------------------------
    if settled > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0
    components["terminal_success"] = terminal_success

    # --------------------------------------------------------------
    # 10. boundary_guard: monotonic penalty as cart or crate nears
    #     the warehouse boundary.  Cart position: obs[0], obs[1].
    #     Crate world position reconstructed from cart pose + body offset.
    # --------------------------------------------------------------
    ax = abs(next_obs[0])
    ay = abs(next_obs[1])

    cx = next_obs[0] * 5.0
    cy = next_obs[1] * 4.0
    cos_h = next_obs[2]
    sin_h = next_obs[3]
    rx = next_obs[6] * 3.0
    ry = next_obs[7] * 3.0
    wx = rx * cos_h - ry * sin_h
    wy = rx * sin_h + ry * cos_h
    px = cx + wx
    py = cy + wy
    px_n = px / 5.0
    py_n = py / 4.0

    boundary_penalty = 0.0
    if ax > 0.95:
        boundary_penalty -= 4.0 * (ax - 0.95)
    if ay > 0.95:
        boundary_penalty -= 4.0 * (ay - 0.95)
    if abs(px_n) > 0.95:
        boundary_penalty -= 4.0 * (abs(px_n) - 0.95)
    if abs(py_n) > 0.95:
        boundary_penalty -= 4.0 * (abs(py_n) - 0.95)
    components["boundary_guard"] = boundary_penalty

    # --------------------------------------------------------------
    # 11. terminal_failure: -100 once if cart/crate out of bounds
    #     or cumulative hard hits >= 3.
    # --------------------------------------------------------------
    out_of_bounds = (ax > 1.05 or ay > 1.05 or
                     abs(px_n) > 1.05 or abs(py_n) > 1.05)

    terminal_failure = 0.0
    if (out_of_bounds or _HARD_HITS[0] >= 3) and not _FAIL_PAID[0]:
        _FAIL_PAID[0] = True
        terminal_failure = -100.0
    components["terminal_failure"] = terminal_failure

    # --------------------------------------------------------------
    # Total reward
    # --------------------------------------------------------------
    total_reward = (
        components["approach_cargo"] +
        components["progress"] +
        components["dock_enter"] +
        components["roughness"] +
        components["hard_hit"] +
        components["action_cost"] +
        components["time_cost"] +
        components["settled_bonus"] +
        components["terminal_success"] +
        components["boundary_guard"] +
        components["terminal_failure"]
    )

    return float(total_reward), components