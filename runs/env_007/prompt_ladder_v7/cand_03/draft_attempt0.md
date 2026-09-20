```python
# Module-level state for one-shot completion event and episode-boundary detection.
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]

# ---------------------------------------------------------------------------
# Design notes / self-check records (numbers are per-step estimates)
#
# Completion predicate (from env facts, tolerance taken from card):
#   |obs[12]| <= 0.024  and  |obs[13]| <= 0.030  (crate fully inside dock)
#   heading error < 30 deg
#   crate speed < 0.05 m/s
#
# Self-check (1): idle (crate at initial pos, still) vs pushing toward dock.
#   R_idle  ~= -0.05 (small boundary/action terms only, no progress)
#   R_push  ~= +6.0 (progress increment) + small penalties
#   => R_push > R_idle  OK
#
# Self-check (2): hover 0.3 m outside dock for 400 steps vs real docking.
#   hover:  ~ +2.0/step * 400 = 800 (but progress increments shrink to 0 as it
#           stops moving; realistically ~ 0/step once stalled) -> ~ 0
#   docked: +300 one-shot + ~20/step settled (<=10 steps) -> ~ 500
#   => docking trajectory beats hovering  OK
#
# Self-check (3): contact closing 1.0 m/s vs 0.05 m/s.
#   gentleness = -0.05 * contact * closing
#   at 1.0 m/s: -0.05*1.0 = -0.05 ... too weak alone; we use k=6.0 for impact
#   so at 1.0 m/s: -6.0 ; at 0.05 m/s: -0.3  => diff 5.7 ~ push term (6.0) OK
#
# Self-check (4): settled state, 12 consecutive calls -> linear growth from
#   settled per-step term (+20) until one-shot fires.
#
# Self-check (5): R_idle ~ 0, R_push ~ +6, R_settled ~ +20  (settled is max)
#
# In the fully-settled state, all persistent components are gated to zero
# except the settled per-step term, and the one-shot completion event.
# ---------------------------------------------------------------------------


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode boundary detection via time_fraction (monotone in episode) --
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---- crate-to-dock geometry -------------------------------------------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # ---- crate heading error ----------------------------------------------
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # dock alignment: crate heading should be ~ aligned with +x (dock axis)
    head_err = abs(crate_sin)  # sin of heading angle; 0 when aligned
    align = 1.0 - head_err
    if align < 0.0:
        align = 0.0

    # ---- crate speed ------------------------------------------------------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- completion predicate (strict, from env facts) --------------------
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    aligned = 1.0 if head_err < 0.5 else 0.0   # sin(30deg) ~ 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0
    complete = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    # ---- streak counter for one-shot completion event ---------------------
    if complete > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---- progress increment (delta form, no hover farming) ----------------
    progress = 0.0
    if _PREV_DIST[0] >= 0.0:
        d_prev = _PREV_DIST[0]
        d_now = dist
        delta = d_prev - d_now          # positive when moving closer
        if delta > 0.0:
            # gate by alignment so pushing straight helps most
            progress = 6.0 * delta * (0.5 + 0.5 * align)
        else:
            progress = 4.0 * delta      # mild penalty for moving away
    _PREV_DIST[0] = dist

    # ---- first-entry one-shot (does NOT replace settled per-step) ---------
    entry_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entry_event = 40.0

    # ---- settled per-step income (only when fully settled) ----------------
    settled_step = 0.0
    if complete > 0.5:
        settled_step = 20.0

    # ---- gentleness / soft contact ----------------------------------------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -6.0 * contact * closing

    # ---- boundary guard ---------------------------------------------------
    ax = abs(float(next_obs[0]))
    ay = abs(float(next_obs[1]))
    bound = 0.0
    if ax > 0.90:
        bound -= 20.0 * (ax - 0.90)
    if ay > 0.90:
        bound -= 20.0 * (ay - 0.90)

    # ---- gentle action cost (small) ---------------------------------------
    act_cost = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---- in settled state, zero out all persistent components -------------
    if complete > 0.5:
        progress = 0.0
        gentleness = 0.0
        act_cost = 0.0
        # keep bound guard (it should be ~0 inside dock anyway)

    components = {
        "progress": progress,
        "settled_step": settled_step,
        "entry_event": entry_event,
        "success_event": success_event,
        "gentleness": gentleness,
        "boundary": bound,
        "action_cost": act_cost,
    }

    total = (
        progress
        + settled_step
        + entry_event
        + success_event
        + gentleness
        + bound
        + act_cost
    )
    return (float(total), components)
```