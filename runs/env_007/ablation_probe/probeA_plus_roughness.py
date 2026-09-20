"""DIAGNOSTIC PROBE A - control v1 plus the native contact-roughness term.

The hand-written observation-only control in `../control_obs_only/reward.py`
plateaus at 6.7 % on fresh seeds while the native reward - clipped to the same
+/-20 - reaches 95 %. Both enter the dock on 90 % of episodes; the control then
spends 43.5 steps inside the dock and gets the crate below 0.05 m/s on only 4.8
of them, while the native policy settles within ~22 steps.

The two rewards share their shaping (`progress` + `approach` increments) and
differ only in what the native reward has on top. This probe adds exactly one of
those terms:

    roughness = -0.02 * peak normal contact impulse, every step in contact

It is the only native term that measures *how gently* the crate is being
handled, which is the skill both the control and every LLM candidate fail to
learn. If adding it alone recovers the native performance, that is the missing
signal.

`info["contact_impulse"]` is used here only because this is a diagnostic probe;
the search contract forbids info fields. Must never enter any population.
Self-contained on purpose: no cross-module import, so cloudpickle cannot mangle
it when the env factories are shipped to the SubprocVecEnv workers.
"""


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- identical to control v1 ------------------------------------------
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

    # ---- the single added term --------------------------------------------
    # exactly the environment's own roughness term:
    #   -0.02 * peak normal impulse, only while in contact
    roughness = float((info or {}).get("official_reward_terms", {}).get("roughness", 0.0))

    components = {
        "crate_progress": float(crate_progress),
        "cart_approach": float(cart_approach),
        "dock_settled_hold": float(settled_hold),
        "boundary": float(boundary),
        "shove": float(shove),
        "roughness": roughness,
    }
    total = (crate_progress + cart_approach + settled_hold + boundary + shove + roughness)
    return float(total), components
