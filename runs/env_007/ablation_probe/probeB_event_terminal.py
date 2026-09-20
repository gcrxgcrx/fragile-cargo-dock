"""DIAGNOSTIC PROBE B - the native terminal *event*, instead of a per-step stream.

Control v1 pays `+20` for every step the crate is inside the dock, aligned and
slower than 0.05 m/s. The native reward instead pays `+300` **once**, at the
step the episode terminates in success, plus `+5` once on first entering the
dock, plus `-100` on failure.

Those forms are not equivalent: a per-step stream on the success *state* can be
kept alive by never completing the ten consecutive settling steps, whereas a
one-off terminal event can only be collected by actually finishing.

This probe keeps control v1's shaping and boundary guard, drops the per-step
settled stream, and substitutes the native event terms. It therefore isolates
"per-step state stream vs one-off terminal event" from every other difference.

The event terms are read from `info["official_reward_terms"]`, which a real
search candidate may not do. Diagnostic only; never enters any population.
Self-contained on purpose (no cross-module import) so cloudpickle cannot mangle
it when the env factories are shipped to the SubprocVecEnv workers.
"""


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- identical to control v1, except the settled stream is dropped -----
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

    # ---- the native event terms, substituted for the per-step stream ------
    terms = (info or {}).get("official_reward_terms", {}) or {}
    dock_enter = float(terms.get("dock_enter", 0.0))          # +5, once per episode
    terminal_success = float(terms.get("terminal_success", 0.0))   # +300 at the terminal step
    terminal_failure = float(terms.get("terminal_failure", 0.0))   # -100 at the terminal step

    components = {
        "crate_progress": float(crate_progress),
        "cart_approach": float(cart_approach),
        "dock_settled_hold": 0.0,
        "boundary": float(boundary),
        "shove": float(shove),
        "dock_enter": dock_enter,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }
    total = (crate_progress + cart_approach + boundary + shove
             + dock_enter + terminal_success + terminal_failure)
    return float(total), components
