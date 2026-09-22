"""Render the 'was it just short of time?' diagnosis.

Three panels on the same seed:

  1. CREATE best reward, 400-step limit   -> looks like it was inches away
  2. CREATE best reward, time limit lifted -> the crate blows straight through
  3. native-reward PPO                     -> releases early and settles

Panel 1 is truncated inside the TimeLimit wrapper exactly as evaluation does it.
Panel 2 steps the unwrapped environment with truncation suppressed but keeps
max_episode_steps at 400, so observation index 18 (elapsed / budget) stays in
the distribution the policy was trained on.

    PY=D:\\Code\\python\\research\\llm_env_310\\Scripts\\python.exe
    $PY record_overshoot_diagnosis.py --seed 20003 --steps 600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from custom_envs import registration  # noqa: E402,F401
from run_fragilecargo_baseline import load_trained_policy  # noqa: E402
from record_fragilecargo_policies import annotate, hstack  # noqa: E402

RUNS = Path("runs/env_007")
CREATE_MODEL = RUNS / "fragilecargo_create/seed_0/iter_10/training"
NATIVE_MODEL = RUNS / "ac3_s3"

HUD_FG = (238, 238, 238)
HUD_DIM = (165, 165, 172)
HUD_BAD = (250, 130, 130)
HUD_WARN = (250, 210, 120)
HUD_OK = (120, 235, 140)


def hud(t, dist, speed, inside, stable, done_line=None):
    if done_line:
        return done_line
    return [
        (f"t {t}", HUD_FG),
        (f"crate->dock {dist:.3f} m", HUD_FG),
        (f"crate speed {speed:.3f} m/s", HUD_BAD if speed > 0.05 else HUD_OK),
        ("INSIDE DOCK" if inside else "", HUD_OK),
        (f"hold {stable}/10", HUD_OK if stable else HUD_DIM),
    ]


def run_truncated(policy, obs_rms, seed, max_steps):
    """Normal evaluation path: TimeLimit wrapper cuts the episode at 400."""
    env = gym.make("FragileCargoDock-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    frames, meta, done_line = [], [], None
    for t in range(1, max_steps + 1):
        frames.append(np.asarray(env.render(), dtype=np.uint8))
        meta.append((t, info["cargo_goal_distance"], info["cargo_speed"],
                     info["cargo_inside_dock"], info["stable_steps"]))
        o = obs if obs_rms is None else np.clip(
            (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10).astype("float32")
        obs, r, term, trunc, info = env.step(policy(o, info, env))
        if term or trunc:
            done_line = [("TIME LIMIT at t=400" if trunc else "TERMINATED", HUD_WARN)]
            break
    frames.append(np.asarray(env.render(), dtype=np.uint8))
    meta.append((len(meta) + 1, info["cargo_goal_distance"], info["cargo_speed"],
                 info["cargo_inside_dock"], info["stable_steps"]))
    env.close()
    return frames, meta, done_line


def run_extended(policy, obs_rms, seed, max_steps):
    """Same budget for the observation, but truncation suppressed."""
    env = gym.make("FragileCargoDock-v0", render_mode="rgb_array")
    base = env.unwrapped
    orig_step = base.step

    def step_no_truncate(action):
        o, r, term, _trunc, info = orig_step(action)
        return o, r, term, False, info

    base.step = step_no_truncate
    obs, info = base.reset(seed=seed)
    frames, meta, done_line = [], [], None
    best = 1e9
    for t in range(1, max_steps + 1):
        frames.append(np.asarray(base.render(), dtype=np.uint8))
        best = min(best, info["cargo_goal_distance"])
        meta.append((t, info["cargo_goal_distance"], info["cargo_speed"],
                     info["cargo_inside_dock"], info["stable_steps"]))
        o = obs if obs_rms is None else np.clip(
            (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10).astype("float32")
        obs, r, term, trunc, info = base.step(policy(o, info, env))
        if term:
            done_line = [("TERMINATED", HUD_WARN)]
            break
    frames.append(np.asarray(base.render(), dtype=np.uint8))
    meta.append((max_steps, info["cargo_goal_distance"], info["cargo_speed"],
                 info["cargo_inside_dock"], info["stable_steps"]))
    env.close()
    return frames, meta, done_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20003)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out-dir", default="runs/env_007/overshoot_videos")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-fps", type=int, default=15)
    ap.add_argument("--gif-seconds", type=float, default=10.0)
    ap.add_argument("--gif-scale", type=float, default=0.42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    print("loading policies")
    create_pol, create_rms = load_trained_policy(
        str(CREATE_MODEL / "model.zip"), str(CREATE_MODEL / "vecnormalize.pkl"))
    native_pol, native_rms = load_trained_policy(
        str(NATIVE_MODEL / "model.zip"), str(NATIVE_MODEL / "vecnormalize.pkl"))

    print(f"seed {args.seed}")
    a_frames, a_meta, a_done = run_truncated(create_pol, create_rms, args.seed, args.steps)
    b_frames, b_meta, b_done = run_extended(create_pol, create_rms, args.seed, args.steps)
    c_frames, c_meta, c_done = run_truncated(native_pol, native_rms, args.seed, args.steps)
    print(f"  1) CREATE, 400-step limit : {len(a_frames)} frames")
    print(f"  2) CREATE, limit lifted   : {len(b_frames)} frames")
    print(f"  3) native-reward PPO      : {len(c_frames)} frames")

    runs = [
        ("CREATE best  |  400-step limit", (215, 140, 80), a_frames, a_meta, a_done),
        ("CREATE best  |  time limit lifted", (245, 185, 90), b_frames, b_meta, b_done),
        ("native-reward PPO  (reference)", (140, 215, 140), c_frames, c_meta, c_done),
    ]

    max_len = max(len(r[2]) for r in runs)
    out_frames = []
    for i in range(max_len):
        row = []
        for label, colour, frames, meta, done_line in runs:
            idx = min(i, len(frames) - 1)
            frozen = i >= len(frames) - 1
            t, dist, speed, inside, stable = meta[idx]
            row.append(annotate(
                frames[idx], f"{label}   (seed {args.seed})", colour,
                hud(t, dist, speed, inside, stable, done_line if frozen else None),
            ))
        out_frames.append(hstack(row))

    mp4 = out_dir / f"overshoot_seed{args.seed}.mp4"
    imageio.mimsave(str(mp4), out_frames, fps=args.fps, quality=8, macro_block_size=None)
    print(f"  wrote {mp4.name}  ({len(out_frames)} frames, {mp4.stat().st_size/1e6:.2f} MB)")

    stride = max(1, int(round(args.fps / args.gif_fps)))
    sel = out_frames[::stride][: int(args.gif_seconds * args.gif_fps)]
    imgs = []
    for f in sel:
        im = Image.fromarray(f)
        im = im.resize((max(2, int(im.width * args.gif_scale)),
                        max(2, int(im.height * args.gif_scale))), Image.LANCZOS)
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    gif = out_dir / f"overshoot_seed{args.seed}.gif"
    imgs[0].save(str(gif), save_all=True, append_images=imgs[1:],
                 duration=int(1000 / args.gif_fps), loop=0, optimize=True)
    print(f"  wrote {gif.name}  ({len(imgs)} frames, {gif.stat().st_size/1e6:.2f} MB)")

    # A focused clip over the critical window: the crate enters the dock,
    # crosses it and leaves the far side without ever satisfying the hold.
    focus_start, focus_end = 355, 480
    focus = out_frames[focus_start:focus_end:max(1, int(round(args.fps / args.gif_fps)))]
    if focus:
        fimgs = []
        for f in focus:
            im = Image.fromarray(f)
            im = im.resize((max(2, int(im.width * args.gif_scale)),
                            max(2, int(im.height * args.gif_scale))), Image.LANCZOS)
            fimgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
        fgif = out_dir / f"overshoot_seed{args.seed}_focus.gif"
        fimgs[0].save(str(fgif), save_all=True, append_images=fimgs[1:],
                      duration=int(1000 / args.gif_fps), loop=0, optimize=True)
        print(f"  wrote {fgif.name}  ({len(fimgs)} frames, {fgif.stat().st_size/1e6:.2f} MB)")

    Image.fromarray(out_frames[min(len(out_frames) - 1, 430)]).save(
        out_dir / f"overshoot_seed{args.seed}_t430.png")
    Image.fromarray(out_frames[-1]).save(out_dir / f"overshoot_seed{args.seed}_final.png")
    print(f"files in {out_dir}")


if __name__ == "__main__":
    main()
