"""DIAGNOSTIC PROBE E - the fully observation-only target.

Probe C (98.3 % fresh success) proves the recipe, but it reads
`info["official_reward_terms"]`, which no search candidate may do. Probe D
(73.3 %) proved the gentleness half is expressible from observations alone.

This probe assembles the whole recipe from observations only, so it measures the
ceiling that a *generateable* reward can reach:

  1. shaping, as in every arm: `crate_progress` + `cart_approach` increments;
  2. gentleness: `-0.05 * contact * max(0, cart_forward_speed - crate velocity
     along the cart heading)` - computable from obs[14], obs[4], obs[2..3],
     obs[8..9];
  3. a **one-off** terminal event instead of a per-step stream. The environment
     terminates the episode on the tenth consecutive settled step, so a reward
     that pays per settled step competes with finishing; a one-off does not.
     The episode boundary is detected from `obs[18]` (fraction of the time
     budget consumed): it increases monotonically inside an episode and drops at
     reset. Module-level state is per worker process, which is where each
     SubprocVecEnv environment lives;
  4. a latched `+5` on first entering the dock, again as an event;
  5. the boundary guard, unchanged.

No `info` field is read. If this lands near probe C, the entire fix is
expressible in a prompt rule.
"""

# per-process state; each SubprocVecEnv worker owns one copy of this module
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_T = [-1.0]

STABLE_STEPS_REQUIRED = 10
SUCCESS_EVENT = 300.0
DOCK_ENTER_EVENT = 5.0


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode boundary ---------------------------------------------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---- shaping (unchanged across every arm) ------------------------------
    dx0, dy0 = obs[12] * 5.0, obs[13] * 4.0
    dx1, dy1 = next_obs[12] * 5.0, next_obs[13] * 4.0
    crate_progress = ((dx0 * dx0 + dy0 * dy0) ** 0.5) - ((dx1 * dx1 + dy1 * dy1) ** 0.5)

    rx0, ry0 = obs[6] * 3.0, obs[7] * 3.0
    rx1, ry1 = next_obs[6] * 3.0, next_obs[7] * 3.0
    cart_approach = ((rx0 * rx0 + ry0 * ry0) ** 0.5) - ((rx1 * rx1 + ry1 * ry1) ** 0.5)

    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- gentleness, from observations only --------------------------------
    cart_forward_speed = obs[4] * 3.0
    crate_along_heading = cvx * obs[2] + cvy * obs[3]
    closing = cart_forward_speed - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- boundary guard ----------------------------------------------------
    edge = abs(next_obs[0])
    if abs(next_obs[1]) > edge:
        edge = abs(next_obs[1])
    boundary = -5.0 * (edge - 0.95) if edge > 0.95 else 0.0

    # ---- events ------------------------------------------------------------
    inside = abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030
    aligned = next_obs[10] >= 0.8660254037844387
    settled = inside and aligned and crate_speed < 0.05

    dock_enter = 0.0
    if inside and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = DOCK_ENTER_EVENT

    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= STABLE_STEPS_REQUIRED and not _PAID[0]:
        _PAID[0] = True
        success_event = SUCCESS_EVENT

    components = {
        "crate_progress": float(crate_progress),
        "cart_approach": float(cart_approach),
        "gentleness": float(gentleness),
        "boundary": float(boundary),
        "dock_enter": float(dock_enter),
        "success_event": float(success_event),
    }
    total = crate_progress + cart_approach + gentleness + boundary + dock_enter + success_event
    return float(total), components
