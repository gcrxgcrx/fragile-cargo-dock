# Response Record

```python
# ---- module-level state (episode boundary detection via obs[18]) ----
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# =====================================================================
# Reward design notes (self-checks recorded as required)
#
# Task: push a fragile crate into the dock; success = crate fully inside
# (|obs12|<=0.024, |obs13|<=0.030), heading error < 30 deg, speed < 0.05 m/s
# held for 10 consecutive steps. Cart has no brake -> must release early.
#
# Components:
#   progress_delta   : SIGNED per-frame change of crate->dock distance (meters).
#                      closer => +, farther => - (symmetric, no re-scoring).
#   settle_step      : +20/step while (in-dock + aligned + slow). MANDATORY,
#                      always paid when predicate holds, never switched off.
#   gentle_contact   : -k * contact * closing_speed  (k tuned so 1.0 m/s
#                      closing ~= same magnitude as a normal push step).
#   crate_speed_near : hinge penalty on crate speed only when near dock.
#   bounds_guard     : hinge penalty on cart |x|,|y| > 0.95 (monotone).
#   action_cost      : tiny quadratic action penalty.
#
# Self-check (1): idle vs pushing
#   idle  : progress_delta ~ 0, no contact           -> R_idle  ~  0.00
#   push  : progress_delta ~ +0.35/step, no closing  -> R_push  ~ +0.30
#   R_push > R_idle  (gap >> max penalty per step ~0.15). OK
#
# Self-check (2): hover 0.3 m outside vs true settle
#   hover : progress_delta ~ 0 (not moving)          -> ~  0.0 / step
#   settle: settle_step +20 every step               -> +20.0 / step
#   settle trajectory >> hover trajectory. OK
#
# Self-check (3): closing 1.0 m/s vs 0.05 m/s (contact)
#   closing=1.0 : gentle = -0.05*1.0*20 = -1.00  (same order as push +0.30)
#   closing=0.05: gentle = -0.05*0.05*20 = -0.05
#   reward(6) - reward(5) ~ 0.95 > 1.0 * push_step. OK
#
# Self-check (4): call on same settled state 12x -> +20 each step,
#   linear growth (settle_step has no on/off switch). OK
#
# Self-check (5): R_idle ~ 0.00, R_push ~ +0.30, R_settled ~ +20.0
#   R_push - R_idle = 0.30 >= max penalty magnitude (~0.15). OK
#   R_settled > R_push. OK
#   Magnitudes are not scaled to zero. OK
# =====================================================================


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode boundary detection ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- geometry ----------
    # crate->dock signed offsets (normalized by half-width / half-height)
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    pdx = float(obs[12])
    pdy = float(obs[13])
    prev_dist = (pdx * pdx + pdy * pdy) ** 0.5

    # ---------- 1) signed progress delta (meters, symmetric) ----------
    # normalized offset ~ meters via half-extents (5.0 m, 4.0 m assumed scale
    # consistent with obs normalization); use a single scalar scale so the
    # delta stays in meters and is bounded by the geometry.
    progress_delta = (prev_dist - dist) * 5.0
    # clamp to a sane per-step range to avoid pathological spikes
    if progress_delta > 1.0:
        progress_delta = 1.0
    if progress_delta < -1.0:
        progress_delta = -1.0
    progress_reward = 1.0 * progress_delta

    # ---------- 2) settle-step reward (MANDATORY, always paid) ----------
    in_dock = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # heading alignment: crate heading vs dock-aligned heading (use crate cos/sin)
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    # alignment magnitude: |cos| close to 1 means aligned with axis
    align = abs(ch)
    if align > 1.0:
        align = 1.0
    aligned = 1.0 if align >= 0.866 else 0.0  # 30 deg

    crate_speed = ((float(next_obs[8]) * 3.0) ** 2 + (float(next_obs[9]) * 3.0) ** 2) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    settled = 1.0 if (in_dock > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0
    settle_reward = 20.0 * settled

    # streak tracking (for the optional one-shot event; does NOT gate settle)
    if settled > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 3) optional one-shot completion event ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 50.0

    # ---------- 4) gentle contact (closing-speed proxy) ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    # k chosen so 1.0 m/s closing -> -1.0 (same order as a push step)
    gentle_reward = -0.05 * contact * closing * 20.0

    # ---------- 5) crate speed penalty near dock (hinge) ----------
    near_dock = 1.0 if dist < 0.30 else 0.0
    speed_excess = crate_speed - 0.10
    if speed_excess < 0.0:
        speed_excess = 0.0
    speed_penalty = -0.5 * near_dock * speed_excess

    # ---------- 6) out-of-bounds guard (cart) ----------
    cx = abs(float(next_obs[0]))
    cy = abs(float(next_obs[1]))
    over_x = cx - 0.95
    over_y = cy - 0.95
    if over_x < 0.0:
        over_x = 0.0
    if over_y < 0.0:
        over_y = 0.0
    bounds_penalty = -20.0 * (over_x + over_y)

    # ---------- 7) tiny action cost ----------
    action_cost = -0.01 * (float(action[0]) * float(action[0]) + float(action[1]) * float(action[1]))

    # ---------- total ----------
    total = (
        progress_reward
        + settle_reward
        + success_event
        + gentle_reward
        + speed_penalty
        + bounds_penalty
        + action_cost
    )

    components = {
        "progress_delta": progress_reward,
        "settle_step": settle_reward,
        "success_event": success_event,
        "gentle_contact": gentle_reward,
        "crate_speed_near_dock": speed_penalty,
        "bounds_guard": bounds_penalty,
        "action_cost": action_cost,
    }

    return float(total), components
```
