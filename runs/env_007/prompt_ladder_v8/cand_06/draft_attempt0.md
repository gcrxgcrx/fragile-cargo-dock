```python
# ---- module level state (episode boundary detection + one-shot flags) ----
_PREV_T = [-1.0]
_SETTLE_STREAK = [0]
_ENTERED = [False]
_PAID = [False]

# =========================================================================
# Self-check log (estimates, per-step totals)
#   R_idle    : crate static, cart static, no contact, mid-field
#               -> progress 0.0, settle 0.0, gentleness 0.0,
#                  bounds ~0.0, time 0.0  =>  ~0.0
#   R_push    : normal push, contact=1, closing~0.2 m/s, crate closing on
#               dock by ~0.02 m/frame
#               -> progress +5.0*0.02 = +0.10 ... plus push_gate term
#                  (see below, ~+0.30), gentleness -0.05*0.2 = -0.01
#                  =>  ~ +0.39
#   R_settled : crate inside dock, aligned, slow
#               -> settle +20.0, progress ~0, gentleness 0
#                  =>  ~ +20.0
#   monotonic: R_settled (20) >> R_push (0.4) > R_idle (0.0)
#   self-check (3) hover 400 steps: ~0.0/step -> ~0
#   self-check (4) settled 400 steps: ~20/step -> ~8000 (linear growth)
#   self-check (5) closing 1.0 m/s contact vs 0.05 m/s contact:
#       gentleness diff = -0.05*1.0 - (-0.05*0.05) = -0.0475
#       -> too weak, so k raised to 2.0 below: diff = -1.9, same order
#          as push progress magnitude (~0.4).  See _K_GENTLE.
# =========================================================================

_K_GENTLE = 2.0          # gentleness penalty coefficient (raised per env card)
_GENTLE_CAP = 3.0        # cap on closing speed used in gentleness

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- episode boundary ----------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _SETTLE_STREAK[0] = 0
        _ENTERED[0] = False
        _PAID[0] = False
    _PREV_T[0] = t

    # ---------------- geometry recovery ----------------
    # dock offsets are normalized by half-width / half-height
    # env card: |obs[12]| <= 0.024 and |obs[13]| <= 0.030 means fully inside
    dx = float(next_obs[12])     # signed x offset / half-width
    dy = float(next_obs[13])     # signed y offset / half-height
    dx_now = float(obs[12])
    dy_now = float(obs[13])

    # metric distance to dock center, using the env-declared half extents
    # (half-width ~ 0.30/0.024 ... we only need relative scale, use ratios)
    # Use normalized offsets directly: tolerance region is tiny, so we
    # compute a normalized distance in "tolerance units".
    TOL_X = 0.024
    TOL_Y = 0.030
    dist_now = ((dx_now / TOL_X) ** 2 + (dy_now / TOL_Y) ** 2) ** 0.5
    dist_nxt = ((dx / TOL_X) ** 2 + (dy / TOL_Y) ** 2) ** 0.5

    # ---------------- crate / cart velocities ----------------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    cart_cos = float(obs[2])
    cart_sin = float(obs[3])
    cart_v = float(obs[4]) * 3.0

    crate_along_heading = crate_vx * cart_cos + crate_vy * cart_sin
    closing = cart_v - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0

    # ---------------- crate orientation error ----------------
    # crate heading vs dock "into" direction; env requires < 30 deg.
    # We align crate heading to the +x axis of the dock frame (0 rad),
    # using atan2 of the crate heading and folding to [-pi, pi].
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # angle of crate heading
    # (no math import allowed) -> use a small polynomial approx for atan2
    # We only need a bounded error proxy; use (1 - cos) form.
    # Aligned to the world +x axis (dock "into" direction assumed = +x).
    align_cos = crate_cos            # cos(angle error) if target is 0 rad
    # 30 deg -> cos = 0.866
    align_factor = align_cos
    if align_factor < 0.0:
        align_factor = 0.0
    # normalized alignment error in [0, 1]; 0 = perfectly aligned to +x
    align_err = 1.0 - align_cos
    if align_err < 0.0:
        align_err = 0.0

    # ---------------- components ----------------
    components = {}

    # (1) crate_to_dock_progress : SIGNED improvement delta, in tolerance units.
    #     Positive only when the crate actually got closer this frame;
    #     negative (symmetric) when it moved away.  Prevents "hover and farm".
    delta = dist_now - dist_nxt          # >0 means closer
    progress = 5.0 * delta
    # clamp to avoid single-frame spikes from teleports / resets
    if progress > 2.0:
        progress = 2.0
    if progress < -2.0:
        progress = -2.0
    components["crate_to_dock_progress"] = progress

    # (2) push_gate : reward *pushing the crate toward the dock*, gated on
    #     contact so idle cart motion gets nothing.  This is an incremental
    #     signal (only when the crate actually closes distance), so it does
    #     NOT violate the "no persistent state bonus" rule.
    push_gate = 0.0
    if contact > 0.5 and delta > 0.0:
        push_gate = 0.30 * delta / max(1e-6, TOL_X)
        if push_gate > 1.5:
            push_gate = 1.5
    components["push_gate"] = push_gate

    # (3) gentleness : contact + closing speed penalty.
    #     k raised to 2.0 so that at closing ~1.0 m/s the penalty (~2.0)
    #     is the same order as / larger than the push progress reward.
    #     Zero when closing ~ 0 (normal steady pushing is not punished).
    gentleness = 0.0
    if contact > 0.5 and closing > 0.0:
        c = closing
        if c > _GENTLE_CAP:
            c = _GENTLE_CAP
        gentleness = -_K_GENTLE * c
    components["gentleness"] = gentleness

    # (4) settle_per_step : THE required completion-side signal.
    #     Predicate: crate fully inside dock + aligned + slow.
    #     Must fire every step the predicate holds, unconditionally.
    inside = 1.0 if (abs(dx) <= TOL_X and abs(dy) <= TOL_Y) else 0.0
    aligned = 1.0 if align_factor >= 0.866 else 0.0     # within 30 deg
    slow = 1.0 if crate_speed < 0.05 else 0.0

    settle_pred = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0
    settle_reward = 20.0 * settle_pred
    components["settle_per_step"] = settle_reward

    # maintain streak (for the optional one-shot event)
    if settle_pred > 0.5:
        _SETTLE_STREAK[0] += 1
    else:
        _SETTLE_STREAK[0] = 0

    # (5) first_entry_bonus : one-shot, when the crate first enters the dock.
    first_entry = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        first_entry = 5.0
    components["first_entry_bonus"] = first_entry

    # (6) success_event : optional one-shot, kept small (per env card it does
    #     not carry weight due to per-step clipping).  Emitted once.
    success_event = 0.0
    if _SETTLE_STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 20.0
    components["success_event"] = success_event

    # (7) out_of_bounds_penalty : monotone hinge on cart position.
    #     +-1.0 is the wall; start biting at 0.90, saturate near 1.05.
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    ax = cart_x if cart_x >= 0.0 else -cart_x
    ay = cart_y if cart_y >= 0.0 else -cart_y
    m = ax if ax > ay else ay
    oob = 0.0
    if m > 0.90:
        over = m - 0.90
        if over > 0.20:
            over = 0.20
        oob = -60.0 * over
    components["out_of_bounds_penalty"] = oob

    # (8) crate_out_of_bounds_penalty : crate must stay on the warehouse floor.
    #     Reconstruct crate world position from cart pose + body-frame offset.
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    crate_wx = cart_x * 5.0 + rel_x * cart_cos - rel_y * cart_sin
    crate_wy = cart_y * 4.0 + rel_x * cart_sin + rel_y * cart_cos
    # normalized by the same half extents (5.0, 4.0)
    cnx = crate_wx / 5.0
    cny = crate_wy / 4.0
    cax = cnx if cnx >= 0.0 else -cnx
    cay = cny if cny >= 0.0 else -cny
    cm = cax if cax > cay else cay
    crate_oob = 0.0
    if cm > 0.90:
        over = cm - 0.90
        if over > 0.20:
            over = 0.20
        crate_oob = -60.0 * over
    components["crate_out_of_bounds_penalty"] = crate_oob

    # (9) time_pressure : mild penalty for burning budget without progress.
    #     Kept small so it never dominates.
    time_pen = -0.02 * float(next_obs[18])
    components["time_pressure"] = time_pen

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components
```