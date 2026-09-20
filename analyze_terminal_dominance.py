"""Mechanical structure check for a generated reward function.

Why this exists
---------------
Every candidate reward produced so far fails for what looks like the same
reason: the dense (process) signal keeps paying while the crate is *near* the
dock, and nothing pays enough for the one-off completion event.  This script
measures that directly, without training anything:

1. Four scripted observation states are pushed through the reward function:

       docked        crate exactly at the dock centre, at rest, aligned
       hover_0.30    crate 0.30 m from the dock centre, at rest, aligned
       transit_2.0   crate 2.0 m from the dock centre, moving 0.8 m/s
       initial       crate at its spawn offset, at rest

   A reward that lets the completion event dominate must satisfy
   ``docked >> hover_0.30``.

2. Three scripted controllers are rolled out in the real environment:

       idle          zero action
       push_forever  the calibrated heuristic, but never releases
       expert        the calibrated heuristic (RELEASE_DIST = 0.25, ~50% success)

   A reward whose structure matches the task must satisfy
   ``cum(expert) > cum(push_forever) > cum(idle)``.

Both raw and clipped per-step values are reported, because the training
harness clips the generated reward per step (``reward_clip``, default 20.0).

Usage
-----
    python analyze_terminal_dominance.py REWARD.py [REWARD.py ...]
    python analyze_terminal_dominance.py --clip 20 runs/.../cand_*/reward_v1.py
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_envs.registration  # noqa: F401  (registers FragileCargoDock-v0)
import gymnasium as gym

from run_fragilecargo_baseline import heuristic_action
import run_fragilecargo_baseline as baseline

ENV_ID = "FragileCargoDock-v0"
HALF_W, HALF_H = 5.0, 4.0
SPEED_SCALE = 3.0


def load_reward(path: str):
    spec = importlib.util.spec_from_file_location(f"cand_{abs(hash(path))}", str(Path(path)))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.compute_reward


def _obs(**kw) -> np.ndarray:
    """Build a synthetic 19-D observation; every entry defaults to a sane value."""
    o = np.zeros(19, dtype=np.float32)
    o[0], o[1] = -1.5 / HALF_W, 0.0          # cart behind the crate, near side
    o[2], o[3] = 1.0, 0.0                    # cart heading +x
    o[4], o[5] = 0.0, 0.0                    # cart at rest
    o[6], o[7] = 0.6 / SPEED_SCALE, 0.0      # crate 0.6 m ahead of the cart
    o[8], o[9] = 0.0, 0.0                    # crate at rest
    o[10], o[11] = 1.0, 0.0                  # crate aligned
    o[12], o[13] = 0.0, 0.0                  # crate at the dock centre
    o[14] = 0.0                              # no contact
    o[15], o[16], o[17] = 0.0, 0.0, 0.0
    o[18] = 0.9
    for k, v in kw.items():
        o[int(k[1:])] = v
    return o


STATES = {
    # name                  crate→dock offset (m)        crate velocity (m/s)
    "docked":              dict(o12=0.00 / HALF_W, o13=0.00 / HALF_H, o8=0.0, o9=0.0),
    "hover_0.30":          dict(o12=0.30 / HALF_W, o13=0.00 / HALF_H, o8=0.0, o9=0.0),
    "hover_edge":          dict(o12=0.13 / HALF_W, o13=0.00 / HALF_H, o8=0.0, o9=0.0),
    "in_dock_too_fast":    dict(o12=0.00 / HALF_W, o13=0.00 / HALF_H, o8=0.50 / SPEED_SCALE, o9=0.0),
    "hover_0.30_mv":       dict(o12=0.30 / HALF_W, o13=0.00 / HALF_H, o8=0.30 / SPEED_SCALE, o9=0.0),
    "transit_2.0":         dict(o12=2.00 / HALF_W, o13=0.00 / HALF_H, o8=0.80 / SPEED_SCALE, o9=0.0),
    "initial":             dict(o12=4.10 / HALF_W, o13=0.00 / HALF_H, o8=0.0, o9=0.0),
    # contact states, for the gentleness check: same contact, different closing speed
    "contact_fast":        dict(o6=0.6 / SPEED_SCALE, o12=1.00 / HALF_W, o4=1.00 / SPEED_SCALE,
                                o8=0.0, o9=0.0, o14=1.0),
    "contact_slow":        dict(o6=0.6 / SPEED_SCALE, o12=1.00 / HALF_W, o4=0.05 / SPEED_SCALE,
                                o8=0.0, o9=0.0, o14=1.0),
}

# The environment ends the episode on success, so a "docked" step can be
# collected at most STABLE_STEPS_REQUIRED times; a hover step can be collected
# for the whole episode.
STABLE_STEPS_REQUIRED = 10
EPISODE_STEPS = 400

# States that are *not* completions but that a policy can sit in forever.
HACK_STATES = ("hover_0.30", "hover_edge", "in_dock_too_fast")


def terminal_form_test(reward_fn, clip: float | None, repeats: int = 12):
    """Is the completion paid as a one-off event, or as a per-step stream?

    Call the reward `repeats` times on the *identical* settled state and look at
    the SHAPE of the raw increments, not at their concentration:

        per-step stream : [20, 20, 20, 20, ...]        flat
        one-off event   : [5, 0, 0, 0, 0, 0, 0, 0, 0, 300, 0, 0]   a spike

    Concentration of the *clipped* total cannot tell these apart, because the
    per-step clip flattens a +300 event to +20 - the same size as a +20
    per-step term - so an event looks like a stream.

    Two separate questions are answered:

    * `has_event` - is there a spike at all (max >= 3x the next largest)?
    * `sufficient` - does the clipped event beat the clipped residual stream
      accumulated over a whole episode, `event > 3 * residual * 400`? A +0.3
      per-step residual looks negligible but is worth 120 over 400 steps, i.e.
      more than a +300 event flattened to +20 by the clip.

    Returns a dict.
    """
    obs = _obs(**STATES["docked"])
    raw = []
    errors = 0
    for _ in range(repeats):
        try:
            val, _comps = reward_fn(obs, np.zeros(2, dtype=np.float32), obs, 0.0, {}, 0.0)
            raw.append(float(val))
        except Exception:  # noqa: BLE001
            raw.append(0.0)
            errors += 1

    order = sorted(raw, reverse=True)
    spike = order[0] if order else 0.0
    # A one-off event fires on one call out of twelve, so it cannot move the
    # median; a per-step stream fires on every call and IS the median. Using the
    # second-largest value instead would mistake a second one-off event (e.g. a
    # `+5` on first entering the dock) for a stream.
    residual = sorted(raw)[len(raw) // 2] if raw else 0.0
    # negative medians mean "no stream", not a negative stream
    if residual < 0.0:
        residual = 0.0

    def cl(v: float) -> float:
        return v if clip is None else max(-clip, min(clip, v))

    event_credit = cl(spike)
    stream_credit = EPISODE_STEPS * cl(residual)
    ratio = float("inf") if stream_credit <= 0 else event_credit / stream_credit
    has_event = residual <= 0 or spike >= 3.0 * residual

    return {
        "spike_raw": spike,
        "residual_raw": residual,
        "event_credit": event_credit,
        "stream_credit": stream_credit,
        "ratio": ratio,
        "has_event": has_event,
        "sufficient": has_event and ratio > 3.0,
        "errors": errors,
        "increments": [round(v, 3) for v in raw],
    }


def gentleness_test(reward_fn, clip: float | None):
    """Does hitting the crate faster cost more?

    Compares two states that are identical except for the closing speed while in
    contact. A reward that teaches gentle handling must score the fast approach
    strictly lower.
    """
    def score(name: str):
        obs = _obs(**STATES[name])
        try:
            val, _comps = reward_fn(obs, np.zeros(2, dtype=np.float32), obs, 0.0, {}, 0.0)
        except Exception:  # noqa: BLE001
            score.errors += 1
            return 0.0
        val = float(val)
        return val if clip is None else max(-clip, min(clip, val))

    score.errors = 0
    fast = score("contact_fast")
    slow = score("contact_slow")
    return fast, slow, slow - fast, score.errors


def episode_dominance(probed, clip: float | None) -> tuple[float, float, float, str]:
    """Can the completion event out-pay the best legal hover, over an episode?

    Returns (terminal_credit, best_hover_credit, ratio, worst_hack_state).
    Both credits use the same per-step clipping the trainer applies.
    """
    def clipped(name: str) -> float:
        val = probed.get(name)
        if not val or val[0] == "ERROR":
            return 0.0
        raw = float(val[0])
        return raw if clip is None else max(-clip, min(clip, raw))

    terminal = STABLE_STEPS_REQUIRED * clipped("docked")
    worst_state, worst = "", -1e18
    for name in HACK_STATES:
        credit = EPISODE_STEPS * clipped(name)
        if credit > worst:
            worst_state, worst = name, credit
    ratio = float("inf") if worst <= 0 else terminal / worst
    return terminal, worst, ratio, worst_state


def probe_states(reward_fn, clip: float | None):
    out = {}
    for name, kw in STATES.items():
        obs = _obs(**kw)
        nxt = _obs(**kw)
        try:
            total, comps = reward_fn(obs, np.zeros(2, dtype=np.float32), nxt, 0.0, {}, 0.0)
        except Exception as exc:  # noqa: BLE001
            out[name] = ("ERROR", f"{type(exc).__name__}: {exc}")
            continue
        clipped = total if clip is None else max(-clip, min(clip, total))
        out[name] = (float(total), float(clipped), comps)
    return out


def rollout(reward_fn, policy, seed: int, clip: float | None):
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=seed)
    cum = 0.0
    steps = 0
    success = False
    while True:
        action = policy(obs, info, env) if callable(policy) else np.zeros(2, dtype=np.float32)
        nxt, native_r, terminated, truncated, info = env.step(action)
        try:
            total, comps = reward_fn(obs, action, nxt, native_r, dict(info), 0.0)
        except Exception:  # noqa: BLE001
            total, comps = 0.0, {}
        if clip is not None:
            total = max(-clip, min(clip, float(total)))
        cum += float(total)
        steps += 1
        obs = nxt
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break
    env.close()
    return cum, steps, success


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rewards", nargs="+")
    ap.add_argument("--clip", type=float, default=20.0,
                    help="per-step clip used by the training harness (default 20.0; "
                         "pass a negative number to disable)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=40000)
    ap.add_argument("--summary", action="store_true",
                    help="print one CSV row per reward file instead of the full report")
    args = ap.parse_args()

    clip = None if args.clip < 0 else args.clip

    if args.summary:
        print("file,docked_raw,best_hover_step,ratio,event,event_ok,term_ratio,gentle_gap,raised,"
              "idle,push,expert,expert_success")
        for path in args.rewards:
            try:
                # each load_reward() re-executes the module, giving fresh
                # module-level state, so the stateful checks cannot contaminate
                # the states probe
                form = terminal_form_test(load_reward(path), clip)
                fast, slow, gap, err2 = gentleness_test(load_reward(path), clip)
                reward_fn = load_reward(path)
                probed = probe_states(reward_fn, clip)
                terminal, hover, ratio, worst = episode_dominance(probed, clip)
            except Exception as exc:  # noqa: BLE001
                print(f"{path},ERROR,{type(exc).__name__}")
                continue
            docked_val = probed.get("docked", (0.0,))
            if docked_val and docked_val[0] == "ERROR":
                print(f"{path},STATE_ERROR,{docked_val[1]}")
                continue
            docked_raw = docked_val[0]
            cols = []
            for release in (None, 0.0, 0.25):
                cums, succ = [], 0
                for ep in range(max(1, args.episodes)):
                    if release is None:
                        pol = None
                    else:
                        baseline.RELEASE_DIST = release
                        pol = heuristic_action
                    c, _st, ok = rollout(reward_fn, pol, args.seed_offset + ep, clip)
                    cums.append(c)
                    succ += int(ok)
                cols += [float(np.mean(cums)), succ]
            print(f"{path},{float(docked_raw):.2f},{hover / EPISODE_STEPS:.3f},"
                  f"{ratio:.3f},{'yes' if form['has_event'] else 'no'},"
                  f"{'Y' if form['sufficient'] else 'n'},{form['ratio']:.2f},"
                  f"{gap:.3f},{form['errors'] + err2},"
                  f"{cols[0]:.2f},{cols[2]:.2f},{cols[4]:.2f},{int(cols[5])}")
        return

    print(f"per-step clip: {clip}")

    for path in args.rewards:
        print("\n" + "=" * 78)
        print(path)
        print("=" * 78)
        # stateful checks on separate module instances, so they cannot
        # contaminate the states probe below
        form = terminal_form_test(load_reward(path), clip)
        fast, slow, gap, err2 = gentleness_test(load_reward(path), clip)
        reward_fn = load_reward(path)

        print("\n-- scripted states (raw / clipped per step) --")
        probed = probe_states(reward_fn, clip)
        for name, val in probed.items():
            if val[0] == "ERROR":
                print(f"  {name:<18} {val[1]}")
                continue
            raw, clipped, comps = val
            detail = ", ".join(f"{k}={v:+.4f}" for k, v in (comps or {}).items())
            print(f"  {name:<18} raw={raw:+10.4f}  clipped={clipped:+8.4f}   [{detail}]")

        terminal, hover, ratio, worst = episode_dominance(probed, clip)
        verdict = "PASS" if ratio > 3.0 else ("marginal" if ratio > 1.0 else "FAIL")
        print(f"\n-- episode-level dominance (clip={clip}, "
              f"{STABLE_STEPS_REQUIRED} settling steps vs {EPISODE_STEPS} episode steps) --")
        print(f"  completion credit : {STABLE_STEPS_REQUIRED} x clipped(docked) = {terminal:+.2f}")
        print(f"  best hover credit : {EPISODE_STEPS} x clipped({worst}) = {hover:+.2f}")
        print(f"  ratio             : {ratio:.3f}   -> {verdict}"
              f"   (hacking is optimal once the policy converges if ratio <= 1)")

        if form["errors"]:
            form_txt, form_ok = f"RAISED {form['errors']}/12 CALLS", "FAIL"
        elif form["sufficient"]:
            form_txt, form_ok = "ONE-OFF EVENT, and it dominates the residual", "PASS"
        elif form["has_event"]:
            form_txt, form_ok = "one-off event present, but the per-step residual still wins", "FAIL"
        else:
            form_txt, form_ok = "PER-STEP STREAM (no event at all)", "FAIL"
        print(f"\n-- terminal form (reward called 12 times on the identical settled state) --")
        print(f"  increments: {form['increments']}")
        print(f"  spike {form['spike_raw']:+.3f} (clipped {form['event_credit']:+.3f})  "
              f"residual {form['residual_raw']:+.3f} (clipped {form['stream_credit']:+.1f} over 400 steps)")
        print(f"  event/residual ratio {form['ratio']:.2f} -> {form_txt} ({form_ok})")
        if form["errors"]:
            print("     the reward raises on the settled state; it would crash at the first step")
        elif form["has_event"] and not form["sufficient"]:
            print("     a small residual per-step term looks negligible but collects 400 times;")
            print("     once clipped, the one-off event can no longer out-pay it")
        elif not form["has_event"]:
            print("     a per-step stream on the success state pays the policy for NOT finishing")

        g_ok = "PASS" if gap > 0 else "FAIL"
        if err2:
            g_ok = "FAIL"
        print(f"\n-- gentleness (in contact, identical except closing speed) --")
        print(f"  closing 1.00 m/s -> {fast:+.4f}    closing 0.05 m/s -> {slow:+.4f}   "
              f"gap {gap:+.4f} -> {g_ok}")
        if gap <= 0:
            print("     nothing discourages hitting the crate hard; this is the skill the")
            print("     task is actually about, and no candidate has ever encoded it")

        if args.episodes <= 0:
            continue

        print("\n-- scripted controllers --")
        for pname, release in (("idle", None), ("push_forever", 0.0), ("expert", 0.25)):
            cums, succ, steps = [], 0, []
            for ep in range(args.episodes):
                seed = args.seed_offset + ep
                if release is None:
                    pol = None
                else:
                    baseline.RELEASE_DIST = release
                    pol = heuristic_action
                c, st, ok = rollout(reward_fn, pol, seed, clip)
                cums.append(c)
                steps.append(st)
                succ += int(ok)
            print(f"  {pname:<14} cum={np.mean(cums):+10.2f}  "
                  f"(min {min(cums):+.1f} / max {max(cums):+.1f})  "
                  f"native success {succ}/{args.episodes}  "
                  f"mean steps {np.mean(steps):.0f}")


if __name__ == "__main__":
    main()
