```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode boundary detection via time_fraction (obs[18]) ----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---- geometry: crate to dock offset (normalized) ----
    ex = float(next_obs[12])
    ey = float(next_obs[13])
    dist = (ex * ex + ey * ey) ** 0.5

    # ---- completion condition (from obs only) ----
    # crate fully inside dock: |ex| <= 0.024, |ey| <= 0.030
    inside = 1.0 if (abs(ex) <= 0.024 and abs(ey) <= 0.030) else 0.0
    # heading alignment error < 30 deg
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    align_dot = ch  # dot with dock heading (1,0)
    aligned = 1.0 if align_dot >= 0.8660254 else 0.0  # cos(30 deg)
    # crate speed < 0.05 m/s
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    cspeed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if cspeed < 0.05 else 0.0

    done_state = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if done_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---- if in completion state, all other components must be exactly 0 ----
    if done_state > 0.5:
        components = {
            "crate_progress": 0.0,
            "contact_gentleness": 0.0,
            "speed_near_dock": 0.0,
            "align_gate": 0.0,
            "first_enter_bonus": 0.0,
            "success_event": success_event,
            "action_smoothness": 0.0,
        }
        return (float(success_event), components)

    # ---- main progress: incremental approach of crate to dock ----
    pex = float(obs[12])
    pey = float(obs[13])
    prev_dist = (pex * pex + pey * pey) ** 0.5
    delta = prev_dist - dist  # positive when closer
    crate_progress = 10.0 * delta
    if crate_progress < -1.0:
        crate_progress = -1.0
    if crate_progress > 1.0:
        crate_progress = 1.0

    # ---- alignment gate: only multiplies incremental progress ----
    # continuous alignment factor in [0,1]
    align_raw = ch
    if align_raw < 0.0:
        align_raw = 0.0
    if align_raw > 1.0:
        align_raw = 1.0
    align_gate = 0.0
    if delta > 0.0:
        align_gate = 0.5 * delta * align_raw

    # ---- gentleness: penalty only when contact AND closing ----
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- speed penalty near dock (only when close and moving fast) ----
    speed_near_dock = 0.0
    if dist < 0.15:
        over = cspeed - 0.3
        if over > 0.0:
            speed_near_dock = -0.5 * over * over

    # ---- first-entry one-time bonus ----
    first_enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        first_enter_bonus = 5.0

    # ---- action smoothness (light) ----
    a0 = float(action[0])
    a1 = float(action[1])
    action_smoothness = -0.01 * (a0 * a0 + a1 * a1)

    components = {
        "crate_progress": crate_progress,
        "contact_gentleness": gentleness,
        "speed_near_dock": speed_near_dock,
        "align_gate": align_gate,
        "first_enter_bonus": first_enter_bonus,
        "success_event": success_event,
        "action_smoothness": action_smoothness,
    }

    total = (
        crate_progress
        + gentleness
        + speed_near_dock
        + align_gate
        + first_enter_bonus
        + success_event
        + action_smoothness
    )
    return (float(total), components)
```