"""DIAGNOSTIC PROBE C - roughness AND the event terminal together.

Probe A (control v1 + the native roughness term) reached 40.0 % on fresh seeds.
Probe B (control v1 with the per-step settled stream replaced by the native
one-off terminal event) reached 45.0 %. Control v1 itself reaches 0 % at the
same budget. Each differs from the control in exactly one place.

This probe applies both changes at once, on the same shaping and the same
boundary guard, to test whether the two defects are independent and additive.
If they are, this should approach the native reward's 95 %.

Diagnostic only - it reads `info["official_reward_terms"]`. Never enters any
population.
"""


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- control v1 shaping -------------------------------------------------
    dx0, dy0 = obs[12] * 5.0, obs[13] * 4.0
    dx1, dy1 = next_obs[12] * 5.0, next_obs[13] * 4.0
    dist_prev = (dx0 * dx0 + dy0 * dy0) ** 0.5
    dist_now = (dx1 * dx1 + dy1 * dy1) ** 0.5
    crate_progress = dist_prev - dist_now

    rx0, ry0 = obs[6] * 3.0, obs[7] * 3.0
    rx1, ry1 = next_obs[6] * 3.0, next_obs[7] * 3.0
    gap_prev = (rx0 * rx0 + ry0 * ry0) ** 0.5
    gap_now = (rx1 * rx1 + ry1 * ry1) ** 0.5
    cart_approach = gap_prev - gap_now

    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    edge = abs(next_obs[0])
    if abs(next_obs[1]) > edge:
        edge = abs(next_obs[1])
    boundary = -5.0 * (edge - 0.95) if edge > 0.95 else 0.0

    shove = 0.0
    if next_obs[14] > 0.5 and crate_speed > 1.5:
        shove = -0.2 * (crate_speed - 1.5)

    # ---- change 1: native roughness (contact gentleness) --------------------
    terms = (info or {}).get("official_reward_terms", {}) or {}
    roughness = float(terms.get("roughness", 0.0))

    # ---- change 2: native one-off terminal event, no per-step settled stream
    dock_enter = float(terms.get("dock_enter", 0.0))
    terminal_success = float(terms.get("terminal_success", 0.0))
    terminal_failure = float(terms.get("terminal_failure", 0.0))

    components = {
        "crate_progress": float(crate_progress),
        "cart_approach": float(cart_approach),
        "dock_settled_hold": 0.0,
        "boundary": float(boundary),
        "shove": float(shove),
        "roughness": roughness,
        "dock_enter": dock_enter,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }
    total = (crate_progress + cart_approach + boundary + shove
             + roughness + dock_enter + terminal_success + terminal_failure)
    return float(total), components
