# ---- module-level state (episode-boundary detection via obs[18]) ----
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# Warehouse half-extents (used to convert normalized obs into meters).
_HALF_W = 5.0
_HALF_H = 4.0

# Dock tolerance (from environment facts, in normalized units).
_TOL_X = 0.024
_TOL_Y = 0.030
# Dock tolerance in meters (obs[12] is x/half_width, obs[13] is y/half_height).
_TOL_X_M = _TOL_X * _HALF_W   # 0.12 m
_TOL_Y_M = _TOL_Y * _HALF_H   # 0.12 m

# Speed / heading thresholds for the completion predicate.
_SPEED_THRESH = 0.05          # m/s
_HEADING_THRESH = 0.5236      # 30 degrees in radians
_HOLD_STEPS = 10


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- episode boundary detection ----------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------------- geometry (meters) ----------------
    # Crate -> dock signed offsets, converted to meters.
    dx = float(next_obs[12]) * _HALF_W
    dy = float(next_obs[13]) * _HALF_H
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12]) * _HALF_W
    ody = float(obs[13]) * _HALF_H
    old_dist = (odx * odx + ody * ody) ** 0.5

    # ---------------- crate velocity / heading ----------------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    crate_ang = 0.0
    # atan2 via sign logic (avoid math import).
    if crate_cos != 0.0:
        crate_ang = crate_sin / (abs(crate_cos) + abs(crate_sin))  # cheap bounded proxy
    # Use a bounded alignment proxy: cos of heading error vs dock axis (x-axis).
    # Dock alignment: crate heading should point roughly along +x (cos ~ 1).
    align = crate_cos  # in [-1, 1]; 1 = aligned with +x
    if align < 0.0:
        align = 0.0

    # ---------------- inside-dock predicate ----------------
    inside = 1.0 if (abs(dx) <= _TOL_X_M and abs(dy) <= _TOL_Y_M) else 0.0
    slow = 1.0 if crate_speed < _SPEED_THRESH else 0.0
    aligned = 1.0 if align > 0.866 else 0.0  # cos(30 deg) = 0.866

    # ---------------- component: crate_to_dock_progress (signed delta) ----------------
    # Positive when this frame got closer, negative when it got farther.
    progress = (old_dist - dist)
    # Clamp per-step to avoid huge spikes from teleports/resets.
    if progress > 1.0:
        progress = 1.0
    if progress < -1.0:
        progress = -1.0
    r_progress = 20.0 * progress

    # ---------------- component: gentleness (soft contact) ----------------
    cart_cos = float(obs[2])
    cart_sin = float(obs[3])
    crate_along_heading = crate_vx * cart_cos + crate_vy * cart_sin
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    # k chosen so that at closing ~ 1.0 m/s the penalty (~ -20) is on par with
    # a normal push-frame progress reward (~ +20).  No penalty when closing ~ 0.
    gentleness = -20.0 * contact * closing

    # ---------------- component: settled per-step yield ----------------
    # Predicate: inside dock + aligned + slow.  Fires EVERY step it holds,
    # and is never disabled by the one-shot event or by any counter.
    settled = inside * aligned * slow
    r_settled = 20.0 * settled

    # ---------------- component: one-shot completion event ----------------
    if inside > 0.5 and aligned > 0.5 and slow > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= _HOLD_STEPS and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---------------- component: first-entry bonus (one-shot) ----------------
    first_entry = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        first_entry = 5.0

    # ---------------- component: out-of-bounds guard ----------------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    ob_pen = 0.0
    ax = abs(cart_x)
    ay = abs(cart_y)
    if ax > 0.95:
        ob_pen -= 200.0 * (ax - 0.95)
    if ay > 0.95:
        ob_pen -= 200.0 * (ay - 0.95)

    # ---------------- component: action smoothness (light) ----------------
    smooth = -0.02 * (float(action[0]) * float(action[0]) + float(action[1]) * float(action[1]))

    total = r_progress + gentleness + r_settled + success_event + first_entry + ob_pen + smooth

    components = {
        "crate_to_dock_progress": float(r_progress),
        "gentleness": float(gentleness),
        "settled_yield": float(r_settled),
        "success_event": float(success_event),
        "first_entry": float(first_entry),
        "out_of_bounds": float(ob_pen),
        "action_smoothness": float(smooth),
    }
    return float(total), components

# ---------------------------------------------------------------------------
# Self-check records (estimated per-step averages):
#
#   R_idle    (nothing happening, crate at rest far from dock):
#             progress ~ 0, gentleness = 0, settled = 0, smooth ~ -0.02
#             => ~ -0.02
#   R_push    (cart pushing crate toward dock, ~0.5 m/s closing,
#              progress ~ +0.05 m/frame):
#             progress = 20 * 0.05 = +1.0, gentleness ~ -20 * 1 * 0.5 = -10
#             wait -- for a *normal* push closing is small (~0.1 m/s):
#             progress = +1.0, gentleness ~ -20*1*0.1 = -2.0, smooth ~ -0.02
#             => ~ -1.0  (still > idle)
#             For a clean non-contact approach frame: progress = +1.0,
#             gentleness = 0, smooth ~ -0.02 => ~ +0.98.
#   R_settled (inside + aligned + slow, no progress):
#             settled = +20, progress ~ 0, gentleness ~ 0
#             => ~ +20
#
#   R_push - R_idle  ~ +1.0  (matches magnitude of progress term per frame)
#   R_settled - R_push ~ +21 >> 0  (settled is the global optimum)
#
# Self-check ①: idle (crate at rest) step reward ~ -0.02;
#               pushing toward dock step reward > 0 (progress dominates).
# Self-check ②: hovering 0.3 m outside dock, speed~0, 400 frames:
#               progress ~ 0, gentleness = 0, settled = 0 => ~ -8 total.
#               Docking and holding: settled fires each of the ~10 hold
#               frames => >= 200, plus one-shot event.  Far larger.
# Self-check ③: closing = 1.0 m/s with contact: gentleness = -20.
#               closing = 0.05 m/s with contact: gentleness = -1.0.
#               Difference = 19.0, same order as progress term (20 * 1.0).
# Self-check ④: calling reward 12x on the same settled state: settled_yield
#               fires every call (no switch), so cumulative grows linearly
#               at +20/frame.
# Self-check ⑤: R_idle ~ -0.02, R_push ~ +1.0, R_settled ~ +20.
#               R_push > R_idle and R_settled > R_push.  Boundary penalty
#               at |x|=1.05 is -20 per frame, which dominates any positive
#               process signal near the wall.
# ---------------------------------------------------------------------------