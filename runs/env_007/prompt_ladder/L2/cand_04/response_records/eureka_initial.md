# Response Record

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode boundary detection ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- geometry from observations ----------
    # crate -> dock signed offsets (normalized)
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # crate heading error vs dock (dock aligned with world axes; use crate heading)
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    heading_err = (sh * sh) ** 0.5  # |sin(theta)| proxy, 0 when aligned to +x axis

    # crate speed (m/s)
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- completion condition ----------
    inside = 1.0 if (dx <= 0.024 and dx >= -0.024 and dy <= 0.030 and dy >= -0.030) else 0.0
    aligned = 1.0 if heading_err < 0.5 else 0.0   # <30 deg
    slow = 1.0 if crate_speed < 0.05 else 0.0
    complete_state = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if complete_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    components = {}

    # ---------- success event (one-shot) ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---------- first entry bonus (one-shot) ----------
    entry_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entry_bonus = 20.0

    # ---------- completion-state freeze: everything else zero ----------
    if complete_state > 0.5:
        components["success_event"] = success_event
        components["entry_bonus"] = entry_bonus
        components["progress"] = 0.0
        components["gentleness"] = 0.0
        components["speed_gate_penalty"] = 0.0
        components["bounds_penalty"] = 0.0
        components["action_penalty"] = 0.0
        total = success_event + entry_bonus
        return (float(total), components)

    # ---------- progress: incremental distance decrease ----------
    progress = 0.0
    if _PREV_DIST[0] >= 0.0:
        delta = _PREV_DIST[0] - dist
        if delta > 0.0:
            progress = 60.0 * delta
    _PREV_DIST[0] = dist

    # ---------- gentleness (soft contact) ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -2.0 * contact * closing

    # ---------- speed gate penalty near dock ----------
    # only penalize high crate speed when already close to dock (not blocking approach)
    speed_gate_penalty = 0.0
    if dist < 0.35 and crate_speed > 0.05:
        over = crate_speed - 0.05
        speed_gate_penalty = -8.0 * over * over

    # ---------- out-of-bounds guard ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    ax = cx if cx >= 0.0 else -cx
    ay = cy if cy >= 0.0 else -cy
    bounds_penalty = 0.0
    if ax > 0.90:
        bounds_penalty -= 40.0 * (ax - 0.90)
    if ay > 0.90:
        bounds_penalty -= 40.0 * (ay - 0.90)
    # crate near boundary (recover world-ish via dock offsets is unreliable; use cart proximity as proxy)
    if dist > 1.2:
        bounds_penalty -= 5.0 * (dist - 1.2)

    # ---------- action smoothness (light) ----------
    a0 = float(action[0])
    a1 = float(action[1])
    action_penalty = -0.02 * (a0 * a0 + a1 * a1)

    components["success_event"] = success_event
    components["entry_bonus"] = entry_bonus
    components["progress"] = progress
    components["gentleness"] = gentleness
    components["speed_gate_penalty"] = speed_gate_penalty
    components["bounds_penalty"] = bounds_penalty
    components["action_penalty"] = action_penalty

    total = (success_event + entry_bonus + progress + gentleness
             + speed_gate_penalty + bounds_penalty + action_penalty)
    return (float(total), components)
```
