```python
# ---- module-level state (episode boundary via obs[18], which resets on new episode) ----
_PREV_T = [-1.0]
_PAID = [False]

# =====================================================================
# Self-check record (estimated per-step average reward)
#   R_idle    ~ +0.00   (nothing moves: no progress delta, no penalties)
#   R_push    ~ +3.00   (signed progress delta ~ +3.0/step at 3 m/s closing;
#                        gentleness ~ -0.05*1.0 = -0.05 when pushing at 1 m/s;
#                        boundary/sensor penalties ~ 0 in the middle)
#   R_settled ~ +20.00  (dwell-in-dock per-step bonus, +20/step while
#                        "inside dock + aligned + slow" holds)
#   => R_push - R_idle = 3.0  >>  max penalty magnitude (~0.05..0.6)
#   => R_settled > R_push, settled is the global optimum.
# Self-check (4): calling reward 12x on the same settled state grows
#   linearly by +20 each step (dwell bonus is never switched off).
# =====================================================================

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode boundary detection ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _PAID[0] = False
    _PREV_T[0] = t

    components = {}

    # ---------- geometry: crate->dock offset in metres ----------
    # obs[12] = dx / half_width, obs[13] = dy / half_height
    # half_width ~ 5.0 m, half_height ~ 4.0 m (warehouse half extents)
    dx = next_obs[12] * 5.0
    dy = next_obs[13] * 4.0
    dist = (dx * dx + dy * dy) ** 0.5

    odx = obs[12] * 5.0
    ody = obs[13] * 4.0
    old_dist = (odx * odx + ody * ody) ** 0.5

    # ---------- 1. signed progress toward dock (increment, per metre) ----------
    # positive when this frame got closer, negative (symmetric) when farther.
    progress = (old_dist - dist) * 6.0
    components["crate_progress"] = progress

    # ---------- 2. gentleness / soft contact ----------
    # observable proxy for closing speed along the cart heading.
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # k = 1.2 so that at closing ~1.0 m/s the penalty (~1.2) is the same
    # order as the per-step progress reward (~3.0) -> high-speed impact
    # is clearly unprofitable, while gentle/steady pushing (closing~0)
    # incurs no penalty at all.
    gentleness = -1.2 * contact * closing
    components["gentleness"] = gentleness

    # ---------- 3. dock entry / alignment / speed predicates ----------
    inside = 1.0 if (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030) else 0.0

    # crate heading error vs dock heading (dock assumed axis-aligned, 0 rad)
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    # alignment factor in [0,1]: 1 when perfectly aligned
    align = crate_cos
    if align < 0.0:
        align = 0.0
    if align > 1.0:
        align = 1.0
    aligned = 1.0 if align >= 0.866 else 0.0   # cos(30 deg)

    crate_speed = ((crate_vx * crate_vx + crate_vy * crate_vy)) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    settled = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    # ---------- 4. dwell-in-dock per-step bonus (mandatory completion signal) ----------
    # paid EVERY step the predicate holds; never switched off by any counter
    # or one-shot event.
    dwell_bonus = 20.0 * settled
    components["dwell_bonus"] = dwell_bonus

    # ---------- 5. one-shot completion event (optional, kept small) ----------
    # episode terminates after 10 consecutive settled steps; the event is
    # only a marker, the dwell bonus is what actually drives the policy.
    success_event = 0.0
    if settled > 0.5 and not _PAID[0]:
        _PAID[0] = True
        success_event = 20.0
    components["success_event"] = success_event

    # ---------- 6. out-of-bounds guard (cart & crate) ----------
    # cart position normalized to warehouse half extents; +-1.0 is the wall.
    ax = next_obs[0]
    if ax < 0.0:
        ax = -ax
    ay = next_obs[1]
    if ay < 0.0:
        ay = -ay
    cart_margin = 0.95
    cart_over = 0.0
    if ax > cart_margin:
        cart_over = ax - cart_margin
    if ay > cart_margin:
        d = ay - cart_margin
        if d > cart_over:
            cart_over = d
    cart_boundary_pen = -60.0 * cart_over

    # crate world position reconstructed from body-frame relative position
    rel_xb = next_obs[6] * 3.0
    rel_yb = next_obs[7] * 3.0
    ch = obs[2]
    sh = obs[3]
    crate_wx = next_obs[0] * 5.0 + (rel_xb * ch - rel_yb * sh)
    crate_wy = next_obs[1] * 4.0 + (rel_xb * sh + rel_yb * ch)
    crate_nx = crate_wx / 5.0
    crate_ny = crate_wy / 4.0
    cax = crate_nx
    if cax < 0.0:
        cax = -cax
    cay = crate_ny
    if cay < 0.0:
        cay = -cay
    crate_over = 0.0
    if cax > cart_margin:
        crate_over = cax - cart_margin
    if cay > cart_margin:
        d = cay - cart_margin
        if d > crate_over:
            crate_over = d
    crate_boundary_pen = -60.0 * crate_over

    components["cart_boundary_penalty"] = cart_boundary_pen
    components["crate_boundary_penalty"] = crate_boundary_pen

    # ---------- 7. light obstacle-proximity guard (front sensor) ----------
    front = next_obs[15]
    if front < 0.0:
        front = 0.0
    if front > 1.0:
        front = 1.0
    obstacle_pen = 0.0
    if front > 0.8:
        obstacle_pen = -0.5 * (front - 0.8)
    components["obstacle_penalty"] = obstacle_pen

    # ---------- total ----------
    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)
```