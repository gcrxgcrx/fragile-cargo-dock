"""Render FragileCargoDock-v0 rollouts to MP4 / GIF.

Produces a side-by-side comparison of the random, heuristic and trained-PPO
policies on the same episode seed, plus individual clips and still frames.

Usage (from the repository root):

    PY=D:\\Code\\python\\research\\llm_env_310\\Scripts\\python.exe
    $PY record_fragilecargo_policies.py --seed 20000
    $PY record_fragilecargo_policies.py --seeds 20000,20003 --fps 30
    $PY record_fragilecargo_policies.py --no-compare --keep-frames
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless-safe rendering

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym  # noqa: E402
import pygame  # noqa: E402

from custom_envs import registration  # noqa: E402,F401
from run_fragilecargo_baseline import (  # noqa: E402
    ENV_ID,
    heuristic_action,
    make_random_policy,
    load_trained_policy,
)

PANEL_HEADER_H = 52
HUD_BG = (18, 18, 22)
HUD_FG = (238, 238, 238)
HUD_DIM = (165, 165, 172)
HUD_OK = (120, 235, 140)
HUD_BAD = (250, 138, 138)
HUD_WARN = (250, 210, 120)
COLORS = {"random": (205, 95, 95), "heuristic": (215, 165, 75), "ppo": (95, 165, 240)}
PANEL_ORDER = ("random", "heuristic", "ppo")


# --------------------------------------------------------------------------- #
# HUD helpers
# --------------------------------------------------------------------------- #

_font_cache = {}


def _font(size):
    if size not in _font_cache:
        if not pygame.font.get_init():
            pygame.font.init()
        _font_cache[size] = pygame.font.SysFont(
            "consolas,couriernew,dejavusansmono,monospace", size, bold=True)
    return _font_cache[size]


def to_surface(frame: np.ndarray) -> pygame.Surface:
    return pygame.surfarray.make_surface(np.transpose(np.ascontiguousarray(frame), (1, 0, 2)))


def to_array(surface: pygame.Surface) -> np.ndarray:
    return np.transpose(pygame.surfarray.pixels3d(surface), (1, 0, 2)).copy()


def annotate(frame: np.ndarray, title: str, title_color, lines) -> np.ndarray:
    """Header strip with the policy name plus a row of status fields."""
    h, w, _ = frame.shape
    out = pygame.Surface((w, h + PANEL_HEADER_H))
    out.fill(HUD_BG)
    out.blit(to_surface(frame), (0, PANEL_HEADER_H))

    img = _font(25).render(title, True, title_color)
    out.blit(img, (12, 6))

    x = 12
    for text, color in lines:
        f = _font(17)
        out.blit(f.render(text, True, color), (x, 33))
        x += f.size(text)[0] + 20
    return to_array(out)


def hstack(frames) -> np.ndarray:
    return np.concatenate(frames, axis=1)


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #

def rollout(env, policy, seed, obs_rms=None):
    """Run one episode; return (frames, frame_meta, record)."""
    obs, info = env.reset(seed=seed)
    frames, meta = [], []
    total, steps = 0.0, 0

    while True:
        frames.append(np.asarray(env.render(), dtype=np.uint8))
        meta.append({
            "t": steps,
            "dist": float(info["cargo_goal_distance"]),
            "speed": float(info["cargo_speed"]),
            "contact": bool(obs[14] > 0.5),
            "inside": bool(info["cargo_inside_dock"]),
        })

        policy_obs = obs
        if obs_rms is not None:
            policy_obs = np.clip(
                (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0
            ).astype(np.float32)

        action = policy(policy_obs, info, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1

        if terminated or truncated:
            frames.append(np.asarray(env.render(), dtype=np.uint8))
            meta.append({
                "t": steps,
                "dist": float(info["cargo_goal_distance"]),
                "speed": float(info["cargo_speed"]),
                "contact": bool(obs[14] > 0.5),
                "inside": bool(info["cargo_inside_dock"]),
            })
            break

    return frames, meta, {
        "seed": seed, "return": total, "steps": steps,
        "success": bool(info["is_success"]),
        "reason": info["termination_reason"],
        "final_distance": float(info["cargo_goal_distance"]),
        "hard_collisions": int(info["hard_collision_count"]),
    }


def live_lines(meta):
    return [
        (f"t {meta['t']:3d}", HUD_FG),
        (f"dist {meta['dist']:.2f} m", HUD_FG),
        (f"crate_v {meta['speed']:.2f}", HUD_DIM),
        (("CONTACT" if meta["contact"] else "free"),
         HUD_WARN if meta["contact"] else HUD_DIM),
        (("IN DOCK" if meta["inside"] else ""), HUD_OK),
    ]


def final_lines(rec):
    ok = rec["success"]
    return [
        ("SUCCESS" if ok else rec["reason"].upper(), HUD_OK if ok else HUD_BAD),
        (f"return {rec['return']:.1f}", HUD_FG),
        (f"steps {rec['steps']}", HUD_DIM),
        (f"final dist {rec['final_distance']:.2f} m", HUD_DIM),
    ]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Render FragileCargoDock rollouts")
    ap.add_argument("--env-id", default=ENV_ID)
    ap.add_argument("--seeds", default="20000", help="comma separated episode seeds")
    ap.add_argument("--model", default="runs/env_007/ac3_s3/model.zip")
    ap.add_argument("--vecnormalize", default="runs/env_007/ac3_s3/vecnormalize.pkl")
    ap.add_argument("--out-dir", default="runs/env_007/videos")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-fps", type=int, default=15)
    ap.add_argument("--gif-seconds", type=float, default=6.0)
    ap.add_argument("--gif-scale", type=float, default=0.45)
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    from PIL import Image

    print(f"env={args.env_id}  seeds={seeds}  out={out_dir}")
    ppo_policy, obs_rms = load_trained_policy(args.model, args.vecnormalize)
    policies = {
        "random": (make_random_policy(0), None),
        "heuristic": (heuristic_action, None),
        "ppo": (ppo_policy, obs_rms),
    }

    def write_video(path, frames, fps):
        imageio.mimsave(str(path), frames, fps=fps, quality=8, macro_block_size=None)
        print(f"  wrote {path.name}  ({len(frames)} frames, {path.stat().st_size/1e6:.2f} MB)")

    def write_gif(path, frames, fps, seconds, scale):
        stride = max(1, int(round(fps / args.gif_fps)))
        sel = frames[::stride][: int(seconds * args.gif_fps)]
        imgs = []
        for f in sel:
            im = Image.fromarray(f)
            if scale != 1.0:
                im = im.resize((max(2, int(im.width * scale)), max(2, int(im.height * scale))),
                               Image.LANCZOS)
            imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
        imgs[0].save(str(path), save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.gif_fps), loop=0, optimize=True)
        print(f"  wrote {path.name}  ({len(imgs)} frames, {path.stat().st_size/1e6:.2f} MB)")

    all_results = []

    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        env = gym.make(args.env_id, render_mode="rgb_array")
        runs = {}
        for name in PANEL_ORDER:
            policy, rms = policies[name]
            frames, meta, rec = rollout(env, policy, seed, obs_rms=rms)
            runs[name] = {"frames": frames, "meta": meta, "rec": rec}
            flag = "SUCCESS" if rec["success"] else rec["reason"]
            print(f"  {name:10s} return={rec['return']:8.2f} steps={rec['steps']:3d} "
                  f"final_dist={rec['final_distance']:.3f}  {flag}")
            all_results.append({"policy": name, **rec})

        # ---- individual clips ---------------------------------------------
        for name in PANEL_ORDER:
            r = runs[name]
            frames, meta, rec = r["frames"], r["meta"], r["rec"]
            ann = []
            for i, f in enumerate(frames):
                last = (i == len(frames) - 1)
                ann.append(annotate(
                    f, f"FragileCargoDock-v0  |  {name}  |  seed {seed}",
                    COLORS[name],
                    final_lines(rec) if last else live_lines(meta[i]),
                ))
            write_video(out_dir / f"{name}_seed{seed}.mp4", ann, args.fps)

            if args.keep_frames:
                for tag, idx in (("start", 0), ("mid", len(frames) // 2), ("end", len(frames) - 1)):
                    Image.fromarray(frames[idx]).save(out_dir / f"{name}_seed{seed}_{tag}.png")

        # ---- side-by-side comparison --------------------------------------
        if not args.no_compare:
            max_len = max(len(runs[n]["frames"]) for n in PANEL_ORDER)
            panels = []
            for i in range(max_len):
                row = []
                for n in PANEL_ORDER:
                    r = runs[n]
                    frames, meta, rec = r["frames"], r["meta"], r["rec"]
                    idx = min(i, len(frames) - 1)
                    done = i >= len(frames) - 1
                    row.append(annotate(
                        frames[idx],
                        f"{n}   (seed {seed})",
                        COLORS[n],
                        final_lines(rec) if done else live_lines(meta[i]),
                    ))
                panels.append(hstack(row))
            write_video(out_dir / f"comparison_seed{seed}.mp4", panels, args.fps)
            write_gif(out_dir / f"comparison_seed{seed}.gif", panels,
                      args.fps, args.gif_seconds, args.gif_scale)

            # still frame at the moment the trained policy docks
            ppo_frames = runs["ppo"]["frames"]
            if runs["ppo"]["rec"]["success"]:
                idx = len(ppo_frames) - 1
                row = []
                for n in PANEL_ORDER:
                    r = runs[n]
                    j = min(idx, len(r["frames"]) - 1)
                    done = idx >= len(r["frames"]) - 1
                    row.append(annotate(
                        r["frames"][j], f"{n}   (seed {seed})", COLORS[n],
                        final_lines(r["rec"]) if done else live_lines(r["meta"][j]),
                    ))
                Image.fromarray(hstack(row)).save(out_dir / f"comparison_seed{seed}_final.png")

        env.close()

    (out_dir / "render_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n--- summary ---")
    for name in PANEL_ORDER:
        rows = [r for r in all_results if r["policy"] == name]
        if not rows:
            continue
        succ = sum(1 for r in rows if r["success"])
        print(f"  {name:10s} success {succ}/{len(rows)}   "
              f"mean return {sum(r['return'] for r in rows)/len(rows):8.2f}   "
              f"mean steps {sum(r['steps'] for r in rows)/len(rows):6.1f}")
    print(f"\nfiles in {out_dir}")


if __name__ == "__main__":
    main()
