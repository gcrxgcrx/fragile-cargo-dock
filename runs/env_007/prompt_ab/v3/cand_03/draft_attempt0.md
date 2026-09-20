```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- module-level state (persist across calls) ----
    if not hasattr(compute_reward, "_state"):
        compute_reward._state = {
            "prev_t": -1.0,
            "streak": 0,
            "paid": False,
            "entered": False,
            "prev_dock_dist": None,
        }
    st = compute_reward._state

    # ---- episode boundary detection via obs[18] (monotone within episode) ----
    t = float(next_obs[18])
    if t < st["prev_t"] or t <= 1.0 / 400.0:
        st["streak"] = 0
        st["paid"] = False
        st["entered"] = False
        st["prev_dock_dist"] = None
    st["prev_t"] = t

    components = {}

    # ---- geometry: crate-to-dock offsets (normalized) ----
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    # convert to meters using warehouse half-width / half-height
    # half-width = 5.0, half-height = 4.0 (from obs scaling of cart pos)
    mx = dx * 5.0
    my = dy * 4.0
    dock_dist = (mx * mx + my * my) ** 0.5

    # ---- crate speed (m/s) ----
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- crate heading error (rad) ----
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    heading_err = abs(2.718281828 ** 0.0 * 0.0)  # placeholder-free: compute via atan2 approx
    # use atan2-free approximation: heading_err ~ sqrt(2*(1-cos)) for small angles
    # cos of crate heading relative to dock axis (assume dock axis = +x, cos=ch)
    # angle error = arccos(clamp(ch)) ; approximate with bounded signal
    if ch > 1.0:
        ch = 1.0
    if ch < -1.0:
        ch = -1.0
    heading_err = (2.0 * (1.0 - ch)) ** 0.5  # radians approx, valid for |err|<pi

    # ---- 1. crate_to_dock_progress : incremental delta (avoid hovering) ----
    if st["prev_dock_dist"] is None:
        st["prev_dock_dist"] = dock_dist
    progress = st["prev_dock_dist"] - dock_dist
    st["prev_dock_dist"] = dock_dist
    if progress > 0.0:
        crate_progress = 2.0 * progress
    else:
        crate_progress = 1.0 * progress
    components["crate_progress"] = float(crate_progress)

    # ---- 2. docking quality : gated by proximity, only near dock ----
    # proximity gate: 1 when dock_dist ~ 0, decays to 0 at ~1.5 m
    prox_gate = max(0.0, 1.0 - dock_dist / 1.5)
    # speed factor: 1 when slow, 0 when fast (0.5 m/s)
    speed_factor = max(0.0, 1.0 - crate_speed / 0.5)
    # alignment factor: 1 when aligned, 0 at 30 deg (0.5236 rad)
    align_factor = max(0.0, 1.0 - heading_err / 0.5236)
    # joint continuous quality (geometric mean to avoid collapse)
    q = (prox_gate * speed_factor * align_factor)
    if q > 0.0:
        q = q ** (1.0 / 3.0)
    components["dock_quality"] = float(0.6 * q)

    # ---- 3. soft contact gentleness (only when contacting AND closing) ----
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["gentleness"] = float(gentleness)

    # ---- 4. out-of-bounds hinge penalty (cart & crate near boundary) ----
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    # crate world position recovery (approx via cart frame + cart world)
    cx_world = cart_x * 5.0 + (float(next_obs[6]) * 3.0) * float(obs[2]) - (float(next_obs[7]) * 3.0) * float(obs[3])
    cy_world = cart_y * 4.0 + (float(next_obs[6]) * 3.0) * float(obs[3]) + (float(next_obs[7]) * 3.0) * float(obs[2])
    oob = 0.0
    if abs(cart_x) > 0.85:
        oob += (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        oob += (abs(cart_y) - 0.85)
    if abs(cx_world) > 4.3:
        oob += (abs(cx_world) - 4.3) * 0.5
    if abs(cy_world) > 3.4:
        oob += (abs(cy_world) - 3.4) * 0.5
    components["out_of_bounds"] = float(-2.0 * oob)

    # ---- 5. action smoothness (light, optional) ----
    a0 = float(action[0])
    a1 = float(action[1])
    components["action_smoothness"] = float(-0.01 * (a0 * a0 + a1 * a1))

    # ---- completion condition (explicit from obs, per env facts) ----
    inside = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = heading_err < 0.5236
    slow = crate_speed < 0.05
    complete = inside and aligned and slow

    if complete:
        st["streak"] += 1
    else:
        st["streak"] = 0

    # one-time "first entered dock" bonus (helps cross the "get in but can't stop" gap)
    if inside and not st["entered"]:
        st["entered"] = True
        components["first_entry"] = 20.0
    else:
        components["first_entry"] = 0.0

    # one-time success event
    success_event = 0.0
    if st["streak"] >= 10 and not st["paid"]:
        st["paid"] = True
        success_event = 300.0
    components["success_event"] = float(success_event)

    total = 0.0
    for k in components:
        total += components[k]
    return float(total), components
```