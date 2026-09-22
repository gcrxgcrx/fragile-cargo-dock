"""Render the best reward-search policies side by side.

Panels: EUREKA-style best, CREATE best, and the native-reward PPO policy as a
reference for what "task solved" actually looks like.

    PY=D:\\Code\\python\\research\\llm_env_310\\Scripts\\python.exe
    $PY record_reward_search_policies.py --seeds 20000,20003,20018
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
from PIL import Image  # noqa: E402

from custom_envs import registration  # noqa: E402,F401
from run_fragilecargo_baseline import load_trained_policy  # noqa: E402
from record_fragilecargo_policies import (  # noqa: E402
    annotate,
    final_lines,
    hstack,
    live_lines,
    rollout,
)

RUNS = Path("runs/env_007")

#: (label, model.zip, vecnormalize.pkl, colour)
DEFAULT_PANELS = [
    ("EUREKA best  (native +5.13)",
     RUNS / "fragilecargo_eureka/seed_0/gen_02/cand_03/training/model.zip",
     RUNS / "fragilecargo_eureka/seed_0/gen_02/cand_03/training/vecnormalize.pkl",
     (215, 140, 80)),
    ("CREATE best  (native +4.86)",
     RUNS / "fragilecargo_create/seed_0/iter_10/training/model.zip",
     RUNS / "fragilecargo_create/seed_0/iter_10/training/vecnormalize.pkl",
     (120, 175, 235)),
    ("native-reward PPO  (reference)",
     RUNS / "ac3_s3/model.zip",
     RUNS / "ac3_s3/vecnormalize.pkl",
     (140, 215, 140)),
]


def main():
    ap = argparse.ArgumentParser(description="Render reward-search policies")
    ap.add_argument("--seeds", default="20000,20003,20018")
    ap.add_argument("--out-dir", default="runs/env_007/reward_search_videos")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-fps", type=int, default=15)
    ap.add_argument("--gif-seconds", type=float, default=8.0)
    ap.add_argument("--gif-scale", type=float, default=0.42)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    print("loading policies")
    panels = []
    for label, model_path, vec_path, colour in DEFAULT_PANELS:
        if not Path(model_path).exists():
            print(f"  MISSING model for {label}: {model_path}")
            continue
        policy, obs_rms = load_trained_policy(str(model_path), str(vec_path))
        panels.append((label, policy, obs_rms, colour))
        print(f"  ok  {label}")

    if not panels:
        raise SystemExit("no models found")

    results = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        env = gym.make("FragileCargoDock-v0", render_mode="rgb_array")
        runs = []
        for label, policy, obs_rms, colour in panels:
            frames, meta, rec = rollout(env, policy, seed, obs_rms=obs_rms)
            flag = "SUCCESS" if rec["success"] else rec["reason"]
            print(f"  {label:34s} return={rec['return']:8.2f} steps={rec['steps']:3d} "
                  f"final_dist={rec['final_distance']:.3f}  {flag}")
            results.append({"seed": seed, "label": label, **rec})
            runs.append((label, colour, frames, meta, rec))

        max_len = max(len(r[2]) for r in runs)
        out_frames = []
        for i in range(max_len):
            row = []
            for label, colour, frames, meta, rec in runs:
                idx = min(i, len(frames) - 1)
                done = i >= len(frames) - 1
                row.append(annotate(
                    frames[idx],
                    f"{label}   (seed {seed})",
                    colour,
                    final_lines(rec) if done else live_lines(meta[i]),
                ))
            out_frames.append(hstack(row))

        mp4 = out_dir / f"reward_search_seed{seed}.mp4"
        imageio.mimsave(str(mp4), out_frames, fps=args.fps, quality=8,
                        macro_block_size=None)
        print(f"  wrote {mp4.name}  ({len(out_frames)} frames, "
              f"{mp4.stat().st_size/1e6:.2f} MB)")

        stride = max(1, int(round(args.fps / args.gif_fps)))
        sel = out_frames[::stride][: int(args.gif_seconds * args.gif_fps)]
        imgs = []
        for f in sel:
            im = Image.fromarray(f)
            im = im.resize((max(2, int(im.width * args.gif_scale)),
                            max(2, int(im.height * args.gif_scale))), Image.LANCZOS)
            imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
        gif = out_dir / f"reward_search_seed{seed}.gif"
        imgs[0].save(str(gif), save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.gif_fps), loop=0, optimize=True)
        print(f"  wrote {gif.name}  ({len(imgs)} frames, {gif.stat().st_size/1e6:.2f} MB)")

        Image.fromarray(out_frames[-1]).save(out_dir / f"reward_search_seed{seed}_final.png")
        env.close()

    (out_dir / "render_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nfiles in {out_dir}")


if __name__ == "__main__":
    main()
