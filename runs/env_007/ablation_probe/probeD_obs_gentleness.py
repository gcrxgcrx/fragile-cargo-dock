"""DIAGNOSTIC PROBE D - can an observation-only proxy replace `roughness`?

Probe A showed that adding the native roughness term (a penalty proportional to
the peak normal contact impulse) lifts the hand-written control from 0 % to
40 % on fresh seeds. That term is *not* observable: the observation vector only
carries a boolean contact flag (`obs[14]`), the cart's forward speed (`obs[4]`),
the cart heading (`obs[2]`, `obs[3]`) and the crate's world velocity
(`obs[8]`, `obs[9]`). There is no impulse channel.

This probe keeps control v1 exactly as it is and adds, instead of `roughness`,
the best impact proxy the observation can actually support:

    closing speed along the cart heading = cart_forward_speed
                                           - (crate velocity . cart heading)
    gentleness = -0.05 * contact * max(0, closing speed)

i.e. penalise the *rate at which the cart is closing on the crate while in
contact* - the observable part of "how hard am I hitting it".

If this recovers probe A's performance, no environment change is needed and the
fix is a prompt rule. If it does not, the observation contract is genuinely
missing the signal and `FragileCargoDock-v0` has to expose an impulse-like
channel.

Not a search candidate: it is an ablation arm. Never enters any population.
"""


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- control v1, unchanged ---------------------------------------------
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

    inside_x = abs(next_obs[12]) <= 0.024
    inside_y = abs(next_obs[13]) <= 0.030
    aligned = next_obs[10] >= 0.8660254037844387
    slow = crate_speed < 0.05
    settled_hold = 20.0 if (inside_x and inside_y and aligned and slow) else 0.0

    edge = abs(next_obs[0])
    if abs(next_obs[1]) > edge:
        edge = abs(next_obs[1])
    boundary = -5.0 * (edge - 0.95) if edge > 0.95 else 0.0

    shove = 0.0
    if next_obs[14] > 0.5 and crate_speed > 1.5:
        shove = -0.2 * (crate_speed - 1.5)

    # ---- the added term: an OBSERVABLE gentleness proxy ---------------------
    cos_h = obs[2]
    sin_h = obs[3]
    cart_forward_speed = obs[4] * 3.0
    crate_speed_along_heading = cvx * cos_h + cvy * sin_h
    closing_speed = cart_forward_speed - crate_speed_along_heading
    if closing_speed < 0.0:
        closing_speed = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -0.05 * contact * closing_speed

    components = {
        "crate_progress": float(crate_progress),
        "cart_approach": float(cart_approach),
        "dock_settled_hold": float(settled_hold),
        "boundary": float(boundary),
        "shove": float(shove),
        "gentleness_obs": float(gentleness),
    }
    total = (crate_progress + cart_approach + settled_hold + boundary + shove + gentleness)
    return float(total), components
