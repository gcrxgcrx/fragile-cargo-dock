# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- module-level state (one-time events + episode boundary detection) ----
    _PREV_T = [-1.0]
    _STREAK = [0]
    _PAID = [False]
    _ENTERED = [False]

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---- geometry / signal recovery ----
    # warehouse half-widths (from env card: cart_x scale 5.0, cart_y scale 4.0)
    HALF_W = 5.0
    HALF_H = 4.0

    dock_dx = next_obs[12] * HALF_W          # signed x offset of crate to dock (m)
    dock_dy = next_obs[13] * HALF_H          # signed y offset of crate to dock (m)
    prev_dock_dx = obs[12] * HALF_W
    prev_dock_dy = obs[13] * HALF_H

    dist_new = (dock_dx * dock_dx + dock_dy * dock_dy) ** 0.5
    dist_old = (prev_dock_dx * prev_dock_dx + prev_dock_dy * prev_dock_dy) ** 0.5

    # ---- component 1: crate -> dock progress (incremental, avoids hover farming) ----
    progress = dist_old - dist_new                 # >0 when crate moves toward dock
    crate_progress = 1.5 * progress

    # ---- component 2: docking quality (only meaningful when near dock) ----
    # inside-dock tolerances from env card: |dx|<=0.024, |dy|<=0.030 (normalized)
    inside_x = 1.0 - min(1.0, abs(next_obs[12]) / 0.024)
    inside_y = 1.0 - min(1.0, abs(next_obs[13]) / 0.030)
    inside_x = max(0.0, inside_x)
    inside_y = max(0.0, inside_y)
    inside_factor = inside_x * inside_y            # 1.0 when fully inside

    # heading alignment: crate heading error < 30 deg
    crate_ang = (next_obs[11] * next_obs[11]) ** 0.5   # |sin| proxy for heading magnitude
    # use cos component to gauge alignment with dock axis (assume dock axis ~ world x)
    align = max(0.0, next_obs[10])                 # cos(heading), 1.0 aligned, 0 orthogonal
    align_factor = min(1.0, align / 0.866)         # 0.866 = cos(30 deg)

    # crate speed (world frame), success needs < 0.05 m/s
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    speed_factor = max(0.0, 1.0 - crate_speed / 0.30)

    # joint condition proxy (geometric mean of continuous factors)
    joint = (inside_factor * align_factor * speed_factor) ** (1.0 / 3.0)
    docking_quality = 0.5 * joint

    # ---- component 3: soft contact gentleness (required) ----
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- component 4: action smoothness (light, optional) ----
    smoothness = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---- component 5: out-of-bounds hinge (cart + crate near walls) ----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    oob = 0.0
    if abs(cart_x) > 0.90:
        oob -= 0.5 * (abs(cart_x) - 0.90)
    if abs(cart_y) > 0.90:
        oob -= 0.5 * (abs(cart_y) - 0.90)
    # crate world position recovery (approx): cart world + body-frame crate offset rotated
    cw_x = obs[0] * HALF_W + (obs[6] * 3.0) * obs[2] - (obs[7] * 3.0) * obs[3]
    cw_y = obs[1] * HALF_H + (obs[6] * 3.0) * obs[3] + (obs[7] * 3.0) * obs[2]
    if abs(cw_x) > HALF_W * 0.95:
        oob -= 0.5 * (abs(cw_x) / HALF_W - 0.95)
    if abs(cw_y) > HALF_H * 0.95:
        oob -= 0.5 * (abs(cw_y) / HALF_H - 0.95)
    oob = max(oob, -1.0)

    # ---- component 6: static obstacle proximity (front/left/right) ----
    obstacle = -0.1 * (next_obs[15] + next_obs[16] + next_obs[17])

    # ---- component 7: one-time events ----
    # success conditions (explicit from obs)
    inside_now = (abs(next_obs[12]) <= 0.024) and (abs(next_obs[13]) <= 0.030)
    aligned_now = next_obs[10] >= 0.866
    slow_now = crate_speed < 0.05

    if inside_now and aligned_now and slow_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    enter_event = 0.0
    if inside_now and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0

    total_reward = (
        crate_progress
        + docking_quality
        + gentleness
        + smoothness
        + oob
        + obstacle
        + success_event
        + enter_event
    )

    components = {
        "crate_progress": crate_progress,
        "docking_quality": docking_quality,
        "gentleness": gentleness,
        "smoothness": smoothness,
        "out_of_bounds": oob,
        "obstacle_proximity": obstacle,
        "success_event": success_event,
        "enter_event": enter_event,
    }

    return (float(total_reward), components)
```
