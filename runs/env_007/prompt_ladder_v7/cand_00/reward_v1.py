# Module-level state (episode boundary detection + one-shot events)
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# ---------------------------------------------------------------------------
# Self-check records (estimated single-step totals for the three regimes)
#
#   R_idle    (nothing happens, crate still)          ~ +0.00
#   R_push    (contact, pushing crate toward dock,    ~ +1.20
#              closing speed ~0.3 m/s)
#   R_settled (crate inside tolerance + aligned +     ~ +20.0
#              slow, per-step settle income)
#
#   R_push - R_idle  ~ 1.20   (>= magnitude of largest penalty term)
#   R_settled - R_push ~ 18.8 (settled is the global optimum)
# ---------------------------------------------------------------------------


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- episode boundary detection ----------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    components = {}

    # ---------------- signals ----------------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])

    dock_x = float(next_obs[12])   # crate - dock, signed x offset / half-width
    dock_y = float(next_obs[13])   # crate - dock, signed y offset / half-height

    old_dx = float(obs[12])
    old_dy = float(obs[13])

    # crate speed (world frame)
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # crate heading error vs dock alignment (crate heading is the alignment ref)
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    crate_ang = (crate_sin * crate_sin + crate_cos * crate_cos) ** 0.5
    if crate_ang < 1e-6:
        crate_ang = 1e-6
    crate_cos_n = crate_cos / crate_ang
    # alignment proxy: |cos| close to 1 means axis-aligned with dock frame
    align = abs(crate_cos_n)
    align_err = 1.0 - align            # 0 = perfectly aligned, 1 = perpendicular

    # distance-to-dock proxy (normalized units, monotone in real distance)
    dist = (dock_x * dock_x + dock_y * dock_y) ** 0.5

    # ---------------- completion predicate (explicit from obs) ----------------
    # tolerance taken from environment facts:
    #   |crate_to_dock_x| <= 0.024  and  |crate_to_dock_y| <= 0.030
    #   heading error < 30 deg  ->  |cos| > cos(30deg) = 0.866
    #   crate speed < 0.05 m/s
    inside = 1.0 if (abs(dock_x) <= 0.024 and abs(dock_y) <= 0.030) else 0.0
    aligned = 1.0 if align >= 0.866 else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0
    done_now = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    # ---------------- completion streak / one-shot events ----------------
    if done_now > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    first_enter = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        first_enter = 5.0

    # ---------------- contact / gentleness ----------------
    crate_vx_b = float(next_obs[8]) * 3.0
    crate_vy_b = float(next_obs[9]) * 3.0
    ch = float(obs[2])
    sh = float(obs[3])
    crate_along_heading = crate_vx_b * ch + crate_vy_b * sh
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    # k = 0.45 so that at closing = 1.0 m/s the penalty (-0.45) is on the same
    # order as the push progress term (~+1.2) -> high-speed impacts are not
    # profitable.  Zero when not closing -> normal steady pushing is not punished.
    gentleness = -0.45 * contact * closing

    # ---------------- main progress: incremental approach of crate to dock ----------------
    # pure delta form: only pays when this frame is strictly closer.
    old_dist = (old_dx * old_dx + old_dy * old_dy) ** 0.5
    delta = old_dist - dist
    if delta < 0.0:
        delta = 0.0
    # gate the increment by alignment so pushing toward the dock while aligned
    # is worth slightly more (still an increment, never a per-step state bonus)
    progress = 40.0 * delta * (0.5 + 0.5 * align)

    # ---------------- settle income (per-step, only while fully settled) ----------------
    # This is the ONLY persistent positive term and it is active ONLY when the
    # full completion predicate holds; in every other state it is exactly 0.
    settle = 0.0
    if done_now > 0.5:
        settle = 20.0

    # ---------------- speed penalty near dock (hinge, only when close) ----------------
    # only active when the crate is near the dock AND moving too fast
    near = 1.0 if dist < 0.20 else 0.0
    speed_over = crate_speed - 0.05
    if speed_over < 0.0:
        speed_over = 0.0
    near_dock_speed_pen = -6.0 * near * speed_over

    # ---------------- out-of-bounds guard (cart) ----------------
    # +1.0 is the wall; start penalizing from 0.95, monotone increasing.
    ax = abs(cart_x)
    ay = abs(cart_y)
    ob = 0.0
    if ax > 0.95:
        ob += (ax - 0.95)
    if ay > 0.95:
        ob += (ay - 0.95)
    oob_pen = -200.0 * ob

    # ---------------- assemble components ----------------
    components["success_event"] = success_event
    components["first_enter"] = first_enter
    components["progress"] = progress
    components["settle"] = settle
    components["gentleness"] = gentleness
    components["near_dock_speed_pen"] = near_dock_speed_pen
    components["oob_pen"] = oob_pen

    # ---------------- completion-state zeroing rule ----------------
    # When the crate is in the completion state, all non-one-shot components
    # must contribute exactly 0.
    if done_now > 0.5:
        components["progress"] = 0.0
        components["gentleness"] = 0.0
        components["near_dock_speed_pen"] = 0.0
        components["oob_pen"] = 0.0
        # settle remains (per-step income), success_event / first_enter are one-shot

    total = 0.0
    total += components["success_event"]
    total += components["first_enter"]
    total += components["progress"]
    total += components["settle"]
    total += components["gentleness"]
    total += components["near_dock_speed_pen"]
    total += components["oob_pen"]

    return float(total), components