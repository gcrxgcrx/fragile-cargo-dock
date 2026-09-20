# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- module-level state ----
    _PREV_T = [-1.0]
    _STREAK = [0]
    _PAID = [False]
    _ENTERED = [False]

    # ---- episode boundary detection ----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    components = {}

    # ================= geometry =================
    # crate-to-dock signed offsets (normalized by half-width / half-height)
    dock_x = float(next_obs[12]) * 5.0
    dock_y = float(next_obs[13]) * 4.0
    dist = (dock_x * dock_x + dock_y * dock_y) ** 0.5

    old_dock_x = float(obs[12]) * 5.0
    old_dock_y = float(obs[13]) * 4.0
    old_dist = (old_dock_x * old_dock_x + old_dock_y * old_dock_y) ** 0.5

    # ================= 1. progress (incremental) =================
    progress = old_dist - dist
    if progress > 0.0:
        progress = progress * 2.0
    else:
        progress = progress * 0.5
    if progress > 1.0:
        progress = 1.0
    if progress < -1.0:
        progress = -1.0
    components["crate_to_dock_progress"] = progress * 1.0

    # ================= 2. docking quality (gated, near dock only) =================
    # only active when crate is reasonably close to dock
    near_gate = 1.0
    if dist > 1.0:
        near_gate = 0.0
    else:
        near_gate = 1.0 - dist

    # orientation error: crate heading vs desired (aligned with dock axis, i.e. +x)
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # heading angle error relative to +x axis
    align = crate_cos  # cos(angle) ; 1.0 = perfectly aligned with +x
    align_err = 1.0 - align
    if align_err < 0.0:
        align_err = 0.0
    align_factor = 1.0 - align_err
    if align_factor < 0.0:
        align_factor = 0.0

    # crate speed
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    speed_factor = 1.0 / (1.0 + 10.0 * crate_speed)

    # position factor: how deep inside the dock
    pos_factor = 1.0 / (1.0 + 5.0 * dist)

    quality = near_gate * (0.4 * pos_factor + 0.3 * align_factor + 0.3 * speed_factor)
    components["crate_docking_quality"] = quality * 0.5

    # ================= 3. soft contact / gentleness =================
    cart_cos_h = float(obs[2])
    cart_sin_h = float(obs[3])
    crate_along_heading = crate_vx * cart_cos_h + crate_vy * cart_sin_h
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact_penalty"] = gentleness

    # ================= 4. speed penalty near dock (gated) =================
    if dist < 1.0:
        speed_pen = -0.3 * near_gate * crate_speed
    else:
        speed_pen = 0.0
    components["crate_speed_penalty_near_dock"] = speed_pen

    # ================= 5. out of bounds penalty (hinge) =================
    oob = 0.0
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    if abs(cart_x) > 0.95:
        oob -= (abs(cart_x) - 0.95) * 2.0
    if abs(cart_y) > 0.95:
        oob -= (abs(cart_y) - 0.95) * 2.0
    components["out_of_bounds_penalty"] = oob

    # ================= 6. action smoothness (light) =================
    a0 = float(action[0])
    a1 = float(action[1])
    smooth = -0.01 * (a0 * a0 + a1 * a1)
    components["action_smoothness"] = smooth

    # ================= 7. first-entry bonus (one-shot) =================
    inside_dock = (abs(float(next_obs[12])) <= 0.024 and abs(float(next_obs[13])) <= 0.030)
    entry_bonus = 0.0
    if inside_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        entry_bonus = 30.0
    components["dock_entry_bonus"] = entry_bonus

    # ================= 8. success event (one-shot) =================
    # completion condition: crate fully inside dock + aligned < 30deg + speed < 0.05
    aligned = align >= 0.866  # cos(30 deg)
    slow = crate_speed < 0.05
    if inside_dock and aligned and slow:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components
```
