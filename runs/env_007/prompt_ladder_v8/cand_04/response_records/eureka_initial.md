# Response Record

```python
# ---- module-level state (must be declared before compute_reward) ----
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# ---- constants from environment facts ----
# dock tolerance (normalized units): |obs[12]| <= 0.024, |obs[13]| <= 0.030
_TOLX = 0.024
_TOLY = 0.030
# cart half-width/half-height in meters (from obs[0]/obs[1] scaling)
_HALF_W = 5.0
_HALF_H = 4.0


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode boundary detection via obs[18] ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- recover crate-to-dock offset in meters ----------
    # obs[12], obs[13] are signed offsets normalized by half-width / half-height
    dx_m = next_obs[12] * _HALF_W
    dy_m = next_obs[13] * _HALF_H
    dist_now = (dx_m * dx_m + dy_m * dy_m) ** 0.5

    dx_m_prev = obs[12] * _HALF_W
    dy_m_prev = obs[13] * _HALF_H
    dist_prev = (dx_m_prev * dx_m_prev + dy_m_prev * dy_m_prev) ** 0.5

    # ---------- 1) signed progress toward dock (per-meter, symmetric) ----------
    # positive if this frame moved closer, negative if farther
    progress_delta = dist_prev - dist_now  # meters, signed

    # ---------- 2) contact & closing speed (gentleness) ----------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # strong enough to dominate push reward at ~1.0 m/s closing
    gentleness = -3.0 * contact * closing

    # ---------- 3) orientation alignment (used as gate, not standalone bonus) ----------
    crate_h = next_obs[10]
    crate_s = next_obs[11]
    # desired crate heading points toward dock: use cart heading as reference proxy
    cart_h = next_obs[2]
    cart_s = next_obs[3]
    dot = crate_h * cart_h + crate_s * cart_s
    if dot > 1.0:
        dot = 1.0
    if dot < -1.0:
        dot = -1.0
    align = 0.5 * (1.0 + dot)  # 1 = perfectly aligned, 0 = opposite

    # ---------- 4) settled predicate: inside dock + aligned + slow ----------
    inside = 1.0 if (abs(next_obs[12]) <= _TOLX and abs(next_obs[13]) <= _TOLY) else 0.0
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0
    aligned = 1.0 if dot > 0.866 else 0.0  # cos(30 deg)
    settled = 1.0 if (inside > 0.5 and slow > 0.5 and aligned > 0.5) else 0.0

    # ---------- 5) out-of-bounds guard (cart + crate) ----------
    cx = abs(next_obs[0])
    cy = abs(next_obs[1])
    bound_cart = 0.0
    if cx > 0.95:
        bound_cart += (cx - 0.95) * 60.0
    if cy > 0.95:
        bound_cart += (cy - 0.95) * 60.0

    # crate world position reconstructed from cart pose + relative offset
    # cart world pos approx (obs[0]*5, obs[1]*4); crate_rel_body scaled by 3
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    crate_wx = next_obs[0] * _HALF_W + rel_x * cart_h - rel_y * cart_s
    crate_wy = next_obs[1] * _HALF_H + rel_x * cart_s + rel_y * cart_h
    crate_nx = abs(crate_wx) / _HALF_W
    crate_ny = abs(crate_wy) / _HALF_H
    bound_crate = 0.0
    if crate_nx > 0.95:
        bound_crate += (crate_nx - 0.95) * 60.0
    if crate_ny > 0.95:
        bound_crate += (crate_ny - 0.95) * 60.0
    bounds_penalty = -(bound_cart + bound_crate)

    # ---------- 6) obstacle proximity penalty (front sensor) ----------
    obs_pen = -1.0 * max(0.0, next_obs[15] - 0.7)

    # ---------- 7) gentleness penalty for high-speed contact ----------
    # (already computed as gentleness above)

    # ---------- 8) settled per-step reward (must always fire when settled) ----------
    settled_reward = 20.0 * settled

    # ---------- 9) one-shot success event (optional, kept for structure) ----------
    if settled > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 0.0  # single-step clipping makes large event useless; keep 0

    # ---------- 10) one-shot first-entry bonus ----------
    first_entry = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        first_entry = 0.0  # keep small; settled per-step is the real signal

    # ---------- assemble ----------
    # progress reward: signed per-meter, scaled
    prog_reward = 30.0 * progress_delta

    # gate progress by alignment only mildly (avoid zeroing)
    # (kept simple: alignment already enters via settled predicate)

    components = {
        "progress_to_dock": prog_reward,
        "settled_step": settled_reward,
        "gentleness_penalty": gentleness,
        "bounds_penalty": bounds_penalty,
        "obstacle_penalty": obs_pen,
        "success_event": success_event,
        "first_entry": first_entry,
    }

    total = (
        prog_reward
        + settled_reward
        + gentleness
        + bounds_penalty
        + obs_pen
        + success_event
        + first_entry
    )

    # ---- self-check record ----
    # R_idle  : progress_delta ~ 0, no contact, no bounds -> ~0
    # R_push  : progress_delta ~ +0.01 m/frame -> +0.3 ; no penalty if gentle
    # R_settled: +20 per step
    # Check: R_push > R_idle (0.3 > 0), R_settled (20) > R_push (0.3)  OK
    # Check gentleness: closing=1.0 -> -3.0 vs push +0.3 -> clearly worse OK
    # Check settled 12x: settled_reward fires every step, linear growth OK

    return float(total), components
```
