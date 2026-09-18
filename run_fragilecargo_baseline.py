"""Baseline runner and calibration tool for FragileCargoDock-v0.

Policies
--------
  1. ``random``    - uniform random actions (floor)
  2. ``heuristic`` - hand-written push controller (solvability proof)
  3. ``ppo``       - PPO trained on the environment's own native reward
                     (the "official reward" reference for CREATE)

For ``ppo`` the script also runs a *fixed-seed periodic evaluation during
training* and stores the resulting learning curve, which is what the
convergence / ``target_score`` calibration is based on.

Usage (from the repository root):

    PY=D:\\Code\\python\\research\\llm_env_310\\Scripts\\python.exe
    $PY run_fragilecargo_baseline.py --quick
    $PY run_fragilecargo_baseline.py --policies random,heuristic
    $PY run_fragilecargo_baseline.py --policies ppo --gamma 0.999 \
        --total-timesteps 3000000 --tag g999
    $PY run_fragilecargo_baseline.py --policies ppo --load-model runs/env_007/baseline_g999/model.zip \
        --load-vecnormalize runs/env_007/baseline_g999/vecnormalize.pkl
    $PY run_fragilecargo_baseline.py --trace

Outputs (``--out-dir``) : ``model.zip``, ``vecnormalize.pkl``,
``learning_curve.json``, ``baseline_results.json``, ``baseline_results.csv``,
``baseline_summary.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from custom_envs import registration  # noqa: E402,F401  (registers the env)

ENV_ID = "FragileCargoDock-v0"

# Must stay in sync with custom_envs/fragile_cargo_dock_env.py. Only the
# heuristic controller uses these to recover world coordinates from the
# normalised observation.
FIELD_HALF_W, FIELD_HALF_H = 5.0, 4.0
DOCK_X, DOCK_Y = 2.6, 0.0
SPEED_SCALE, REL_SCALE, OMEGA_SCALE = 3.0, 3.0, 8.0
CARGO_LIN_DAMP = 2.0
RELEASE_DIST = 0.25   # swept on eval seeds 10000..10059: 40% success across the
                      # 0.18-0.33 plateau, 0% outside [0.18, 0.36].
OBS_CLIP = 10.0
NORM_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #

def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _world_from_obs(obs):
    heading = math.atan2(float(obs[3]), float(obs[2]))
    robot_x = float(obs[0]) * FIELD_HALF_W
    robot_y = float(obs[1]) * FIELD_HALF_H
    rel_x = float(obs[6]) * REL_SCALE
    rel_y = float(obs[7]) * REL_SCALE
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    cargo_x = robot_x + rel_x * cos_h - rel_y * sin_h
    cargo_y = robot_y + rel_x * sin_h + rel_y * cos_h
    return robot_x, robot_y, heading, cargo_x, cargo_y


def make_random_policy(seed):
    rng = np.random.default_rng(seed)

    def policy(obs, info, env):
        return rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

    return policy


def heuristic_action(obs, info, env):
    """Line up behind the crate, push toward the dock, release and let it settle.

    The cart is torque-steered, so it can rotate in place at zero throttle; the
    controller uses that to line up instead of circling the crate. The cart
    cannot brake the crate from behind, so it releases before the dock and lets
    floor drag carry the crate into the bay.
    """
    robot_x, robot_y, heading, cargo_x, cargo_y = _world_from_obs(obs)
    omega = float(obs[5]) * OMEGA_SCALE
    forward_speed = float(obs[4]) * SPEED_SCALE

    dx, dy = DOCK_X - cargo_x, DOCK_Y - cargo_y
    dist = math.hypot(dx, dy)
    ux, uy = (dx / dist, dy / dist) if dist > 1e-6 else (1.0, 0.0)

    inside = bool(info.get("cargo_inside_dock", False))
    cargo_speed = float(info.get("cargo_speed", 0.0))

    if inside or dist < RELEASE_DIST:
        tx, ty = cargo_x - ux * 1.8, cargo_y - uy * 1.8
        target_speed = 0.9
    else:
        standoff = 0.62
        sx, sy = cargo_x - ux * standoff, cargo_y - uy * standoff
        behind_dist = math.hypot(sx - robot_x, sy - robot_y)
        if behind_dist > 0.30:
            tx, ty = sx, sy
            target_speed = min(1.0, 0.9 * behind_dist)
        else:
            tx, ty = cargo_x + ux * 0.55, cargo_y + uy * 0.55
            if dist > 1.4:
                target_speed = 0.70
            elif dist > 0.8:
                target_speed = 0.22
            else:
                target_speed = 0.14
            if cargo_speed > target_speed + 0.30:
                target_speed = -0.25

    want = math.atan2(ty - robot_y, tx - robot_x)
    err = wrap_pi(want - heading)
    steer = float(np.clip(2.2 * err - 0.35 * omega, -1.0, 1.0))

    if abs(err) > 1.25:
        throttle = 0.0
    else:
        throttle = float(np.clip(math.cos(err) * target_speed, -1.0, 1.0))
        if forward_speed > target_speed + 0.15:
            throttle = -0.35

    return np.array([throttle, steer], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _episode(env, policy, seed, obs_rms=None, clip_obs=OBS_CLIP):
    obs, info = env.reset(seed=seed)
    total, steps, impulses = 0.0, 0, []
    while True:
        policy_obs = obs
        if obs_rms is not None:
            policy_obs = np.clip(
                (obs - obs_rms.mean) / np.sqrt(obs_rms.var + NORM_EPS),
                -clip_obs, clip_obs,
            ).astype(np.float32)
        action = policy(policy_obs, info, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        impulses.append(float(info["contact_impulse"]))
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "return": total,
        "steps": steps,
        "success": bool(info["is_success"]),
        "reason": info["termination_reason"],
        "hard_collisions": int(info["hard_collision_count"]),
        "final_goal_distance": float(info["cargo_goal_distance"]),
        "final_angle_error": float(info["cargo_angle_error"]),
        "final_cargo_speed": float(info["cargo_speed"]),
        "dock_entered": bool(info["dock_entered"]),
        "peak_impulse": max(impulses) if impulses else 0.0,
        "contact_steps": sum(1 for v in impulses if v > 0.0),
    }


def evaluate(env_id, policy, episodes, seed_offset, obs_rms=None, verbose=False):
    """Run fixed-seed episodes with the raw (native) reward."""
    env = gym.make(env_id)
    records = []
    for i in range(episodes):
        rec = _episode(env, policy, seed_offset + i, obs_rms=obs_rms)
        records.append(rec)
        if verbose:
            print(f"    ep{rec['seed']}: return={rec['return']:9.3f} steps={rec['steps']:3d} "
                  f"{'SUCCESS' if rec['success'] else rec['reason']}")
    env.close()
    return records


def summarize(name, records):
    returns = [r["return"] for r in records]
    n = len(records)
    successes = sum(1 for r in records if r["success"])
    return {
        "policy": name,
        "episodes": n,
        "mean_return": statistics.mean(returns),
        "stdev_return": statistics.pstdev(returns) if n > 1 else 0.0,
        "min_return": min(returns),
        "max_return": max(returns),
        "success_rate": successes / n if n else 0.0,
        "successes": successes,
        "dock_rate": sum(1 for r in records if r["dock_entered"]) / n if n else 0.0,
        "mean_steps": statistics.mean(r["steps"] for r in records),
        "mean_final_goal_distance": statistics.mean(r["final_goal_distance"] for r in records),
        "mean_final_angle_error": statistics.mean(r["final_angle_error"] for r in records),
        "mean_final_cargo_speed": statistics.mean(r["final_cargo_speed"] for r in records),
        "mean_hard_collisions": statistics.mean(r["hard_collisions"] for r in records),
        "terminations": dict(Counter(r["reason"] for r in records)),
    }


def print_summary(s):
    print(f"  return          mean={s['mean_return']:9.3f}  stdev={s['stdev_return']:8.3f}  "
          f"min={s['min_return']:9.3f}  max={s['max_return']:9.3f}")
    print(f"  success         {s['successes']}/{s['episodes']} "
          f"({s['success_rate']:.1%})   dock entered {s['dock_rate']:.1%}")
    print(f"  episode length  mean={s['mean_steps']:.1f}")
    print(f"  end state       goal_dist={s['mean_final_goal_distance']:.3f} m  "
          f"angle_err={s['mean_final_angle_error']:.3f} rad  "
          f"crate_speed={s['mean_final_cargo_speed']:.3f} m/s")
    print(f"  hard collisions mean={s['mean_hard_collisions']:.2f}")
    print(f"  terminations    {s['terminations']}")


# --------------------------------------------------------------------------- #
# Environment contract
# --------------------------------------------------------------------------- #

def check_env_contract(env_id):
    print("--- environment contract ---")
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    print(f"  obs    shape={obs.shape} dtype={obs.dtype} "
          f"range=[{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  action {env.action_space}")
    print(f"  max_episode_steps={env.spec.max_episode_steps}")
    assert obs.shape == (19,), f"unexpected observation shape {obs.shape}"
    assert env.action_space.shape == (2,), env.action_space
    assert np.all(np.abs(obs) <= 2.0 + 1e-6), "observation outside documented bounds"

    actions = np.linspace(-1.0, 1.0, 40).reshape(20, 2).astype(np.float32)

    def fixed(_obs, _info, _env):
        a = actions[fixed.i % len(actions)]
        fixed.i += 1
        return a

    fixed.i = 0
    first = evaluate(env_id, fixed, 1, seed_offset=123)[0]
    fixed.i = 0
    second = evaluate(env_id, fixed, 1, seed_offset=123)[0]
    same = abs(first["return"] - second["return"]) < 1e-9
    print(f"  determinism (same seed, same actions): "
          f"{first['return']:.6f} vs {second['return']:.6f} -> {'OK' if same else 'MISMATCH'}")
    assert same, "environment is not deterministic for a fixed seed and action sequence"
    env.close()
    print()


# --------------------------------------------------------------------------- #
# PPO training
# --------------------------------------------------------------------------- #

def _make_env(env_id, rank, seed):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init


class FixedSeedEvalCallback(BaseCallback):
    """Periodic deterministic evaluation on a fixed seed set, in native-reward units.

    Records mean native return *and* task success rate, which SB3's EvalCallback
    does not provide. Used to locate the convergence point.
    """

    def __init__(self, env_id, every_steps, eval_episodes, seed_offset,
                 clip_obs=OBS_CLIP, verbose=1, curve_path=None):
        super().__init__(verbose)
        self.env_id = env_id
        self.every_steps = every_steps
        self.eval_episodes = eval_episodes
        self.seed_offset = seed_offset
        self.clip_obs = clip_obs
        self.history = []
        self._eval_env = None
        self._next = every_steps
        self.curve_path = Path(curve_path) if curve_path else None

    def _obs_rms(self):
        return getattr(self.training_env, "obs_rms", None)

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every_steps
        self.history.append(self._evaluate())
        self._flush_curve()
        point = self.history[-1]
        if self.verbose:
            print(f"    [eval @ {point['timesteps']:>9,}] "
                  f"return={point['mean_return']:9.2f}  "
                  f"success={point['success_rate']:5.1%}  "
                  f"dock={point['dock_rate']:5.1%}  "
                  f"goal_dist={point['mean_final_goal_distance']:.3f}", flush=True)
        return True

    def _flush_curve(self):
        """Write the curve after every checkpoint so a long run can be monitored."""
        if self.curve_path is None:
            return
        try:
            self.curve_path.parent.mkdir(parents=True, exist_ok=True)
            self.curve_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _evaluate(self):
        if self._eval_env is None:
            self._eval_env = gym.make(self.env_id)
        obs_rms = self._obs_rms()

        def policy(o, _info, _env):
            action, _ = self.model.predict(o, deterministic=True)
            return action

        recs = [_episode(self._eval_env, policy, self.seed_offset + i, obs_rms=obs_rms,
                         clip_obs=self.clip_obs)
                for i in range(self.eval_episodes)]
        n = len(recs)
        return {
            "timesteps": int(self.num_timesteps),
            "mean_return": statistics.mean(r["return"] for r in recs),
            "success_rate": sum(1 for r in recs if r["success"]) / n,
            "dock_rate": sum(1 for r in recs if r["dock_entered"]) / n,
            "mean_steps": statistics.mean(r["steps"] for r in recs),
            "mean_final_goal_distance": statistics.mean(r["final_goal_distance"] for r in recs),
        }

    def _on_training_end(self):
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None


def build_ppo_kwargs(args):
    import torch

    arch = [int(x) for x in str(args.net_arch).split(",") if x.strip()]
    activation = {
        "tanh": torch.nn.Tanh, "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU, "gelu": torch.nn.GELU,
    }[args.activation.lower()]
    return dict(
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=dict(ortho_init=not args.no_ortho_init,
                           activation_fn=activation,
                           net_arch=dict(pi=arch, vf=arch)),
    )


def train_ppo(env_id, total_timesteps, seed, n_envs, out_dir, device, ppo_kwargs, args):
    print("--- PPO on the native reward ---")
    print(f"  total_timesteps={total_timesteps:,} n_envs={n_envs} seed={seed} "
          f"device={device} gamma={ppo_kwargs['gamma']} "
          f"ent_coef={ppo_kwargs['ent_coef']} arch={args.net_arch}")

    if n_envs > 1:
        try:
            venv = SubprocVecEnv([_make_env(env_id, r, seed) for r in range(n_envs)])
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"  SubprocVecEnv unavailable ({exc}); falling back to DummyVecEnv")
            venv = DummyVecEnv([_make_env(env_id, r, seed) for r in range(n_envs)])
    else:
        venv = DummyVecEnv([_make_env(env_id, 0, seed)])

    venv = VecNormalize(venv, training=True, norm_obs=True,
                        norm_reward=args.normalize_reward, clip_obs=OBS_CLIP)

    set_random_seed(seed)
    model = PPO("MlpPolicy", venv, seed=seed, device=device, verbose=0,
                tensorboard_log=None, **ppo_kwargs)

    callback = None
    if args.eval_every > 0:
        callback = FixedSeedEvalCallback(
            env_id=env_id,
            every_steps=args.eval_every,
            eval_episodes=args.train_eval_episodes,
            seed_offset=args.eval_seed_offset,
            verbose=1,
            curve_path=out_dir / "learning_curve.json",
        )

    model.learn(total_timesteps=total_timesteps, tb_log_name=None,
                progress_bar=False, callback=callback)

    model.save(str(out_dir / "model.zip"))
    venv.training = False
    venv.norm_reward = False
    venv.save(str(out_dir / "vecnormalize.pkl"))
    obs_rms = venv.obs_rms
    venv.close()

    if callback is not None and callback.history:
        (out_dir / "learning_curve.json").write_text(
            json.dumps(callback.history, indent=2), encoding="utf-8")
        print(f"  saved: {out_dir / 'learning_curve.json'}")

    def policy(obs, info, env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    print(f"  saved: {out_dir / 'model.zip'}")
    return policy, obs_rms


def load_trained_policy(model_path, vecnormalize_path=None):
    print("--- loading trained model ---")
    print(f"  model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    obs_rms = None
    if vecnormalize_path:
        venv = DummyVecEnv([_make_env(ENV_ID, 0, 0)])
        venv = VecNormalize.load(vecnormalize_path, venv)
        venv.training = False
        venv.norm_reward = False
        obs_rms = venv.obs_rms
        venv.close()
        print(f"  vecnormalize: {vecnormalize_path}")

    def policy(obs, info, env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return policy, obs_rms


# --------------------------------------------------------------------------- #
# Convergence analysis
# --------------------------------------------------------------------------- #

def convergence_report(history, success_threshold=0.5):
    """Locate the first checkpoint that reaches a stable fraction of the final level."""
    if not history:
        return {}
    final_success = max(p["success_rate"] for p in history)
    final_return = max(p["mean_return"] for p in history)
    target = success_threshold * final_success

    first_hit = None
    for p in history:
        if p["success_rate"] >= target and target > 0:
            first_hit = p
            break

    # Last checkpoint that is still below half of the best success rate.
    last_below = None
    for p in history:
        if p["success_rate"] < 0.5 * final_success:
            last_below = p

    return {
        "checkpoints": len(history),
        "best_success_rate": final_success,
        "best_mean_return": final_return,
        "first_timesteps_at_or_above_half_of_best": (
            first_hit["timesteps"] if first_hit else None),
        "last_timesteps_below_half_of_best": (
            last_below["timesteps"] if last_below else None),
        "final_checkpoint": history[-1],
    }


# --------------------------------------------------------------------------- #
# Trace mode
# --------------------------------------------------------------------------- #

def trace_episode(env_id, seed):
    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)
    print(f"--- heuristic trace (seed={seed}) ---")
    print(f"{'t':>4} {'cart_x':>7} {'cart_y':>7} {'hdg':>7} {'crate_x':>8} {'crate_y':>8} "
          f"{'d_dock':>7} {'cspeed':>7} {'contact':>7} {'inside':>6} {'thr':>6} {'str':>6}")
    for t in range(env.spec.max_episode_steps):
        action = heuristic_action(obs, info, env)
        obs, _r, term, trunc, info = env.step(action)
        if t % 20 == 0 or t < 3 or term:
            rx, ry, hdg, cx, cy = _world_from_obs(obs)
            print(f"{t:>4} {rx:>7.2f} {ry:>7.2f} {math.degrees(hdg):>7.1f} "
                  f"{cx:>8.2f} {cy:>8.2f} {info['cargo_goal_distance']:>7.2f} "
                  f"{info['cargo_speed']:>7.2f} {int(obs[14]):>7} "
                  f"{int(info['cargo_inside_dock']):>6} {action[0]:>6.2f} {action[1]:>6.2f}")
        if term or trunc:
            print(f"--- end at t={t} reason={info['termination_reason']} "
                  f"success={info['is_success']}")
            break
    env.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="FragileCargoDock-v0 baselines")
    ap.add_argument("--env-id", default=ENV_ID)
    ap.add_argument("--policies", default="random,heuristic,ppo",
                    help="comma separated subset of random,heuristic,ppo")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--eval-seed-offset", type=int, default=10000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None, help="label appended to the default out-dir")

    # PPO hyper-parameters
    ap.add_argument("--total-timesteps", type=int, default=1500000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--net-arch", default="128,128")
    ap.add_argument("--activation", default="tanh")
    ap.add_argument("--no-ortho-init", action="store_true")
    ap.add_argument("--normalize-reward", action="store_true",
                    help="let VecNormalize scale rewards during training "
                         "(evaluation always uses the raw native reward)")

    # In-training fixed-seed evaluation
    ap.add_argument("--eval-every", type=int, default=100000,
                    help="timesteps between periodic fixed-seed evaluations (0 disables)")
    ap.add_argument("--train-eval-episodes", type=int, default=10)

    ap.add_argument("--load-model", default=None,
                    help="evaluate an already trained model.zip instead of training")
    ap.add_argument("--load-vecnormalize", default=None,
                    help="VecNormalize .pkl matching --load-model")
    ap.add_argument("--quick", action="store_true",
                    help="short run for smoke testing (200k steps, 5 episodes)")
    ap.add_argument("--trace", action="store_true",
                    help="print one heuristic episode and exit")
    ap.add_argument("--skip-contract-check", action="store_true")
    args = ap.parse_args()

    if args.trace:
        trace_episode(args.env_id, args.seed)
        return

    if args.quick:
        args.total_timesteps = 200_000
        args.episodes = 5
        args.n_envs = min(args.n_envs, 4)
        args.eval_every = max(args.eval_every, 50_000)

    if args.out_dir is None:
        args.out_dir = f"runs/env_007/baseline_{args.tag}" if args.tag else "runs/env_007/baseline"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"FragileCargoDock baseline   env={args.env_id}")
    print(f"  interpreter={sys.executable}")
    print(f"  out_dir={out_dir}")
    print()

    if not args.skip_contract_check:
        check_env_contract(args.env_id)

    wanted = [p.strip() for p in args.policies.split(",") if p.strip()]
    summaries = []
    curve = []

    if "random" in wanted:
        print("--- random policy ---")
        recs = evaluate(args.env_id, make_random_policy(args.seed), args.episodes,
                        args.eval_seed_offset)
        s = summarize("random", recs)
        print_summary(s)
        summaries.append(s)
        print()

    if "heuristic" in wanted:
        print("--- heuristic push controller ---")
        recs = evaluate(args.env_id, heuristic_action, args.episodes, args.eval_seed_offset)
        s = summarize("heuristic", recs)
        print_summary(s)
        summaries.append(s)
        print()

    if "ppo" in wanted:
        if args.load_model:
            policy, obs_rms = load_trained_policy(args.load_model, args.load_vecnormalize)
        else:
            policy, obs_rms = train_ppo(args.env_id, args.total_timesteps, args.seed,
                                        args.n_envs, out_dir, args.device,
                                        build_ppo_kwargs(args), args)
        curve_path = out_dir / "learning_curve.json"
        if curve_path.exists():
            curve = json.loads(curve_path.read_text(encoding="utf-8"))
        recs = evaluate(args.env_id, policy, args.episodes, args.eval_seed_offset,
                        obs_rms=obs_rms, verbose=args.episodes <= 10)
        s = summarize("ppo_native_reward", recs)
        print_summary(s)
        summaries.append(s)
        print()

    if not summaries:
        print("nothing to do: --policies selected no known policy")
        return

    conv = convergence_report(curve) if curve else {}

    # ---- persist results ---------------------------------------------------
    (out_dir / "baseline_results.json").write_text(
        json.dumps({"config": vars(args), "summaries": summaries,
                    "convergence": conv}, indent=2, default=str),
        encoding="utf-8",
    )

    fields = ["policy", "episodes", "mean_return", "stdev_return", "min_return",
              "max_return", "successes", "success_rate", "dock_rate", "mean_steps",
              "mean_final_goal_distance", "mean_final_angle_error",
              "mean_final_cargo_speed", "mean_hard_collisions"]
    with (out_dir / "baseline_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    lines = [
        "# FragileCargoDock-v0 baseline",
        "",
        f"- env id: `{args.env_id}`",
        f"- evaluation: {args.episodes} episodes, seeds "
        f"{args.eval_seed_offset}..{args.eval_seed_offset + args.episodes - 1}",
        f"- PPO: {args.total_timesteps:,} timesteps, n_envs={args.n_envs}, seed={args.seed}, "
        f"gamma={args.gamma}, ent_coef={args.ent_coef}, arch={args.net_arch}",
        "",
        "| policy | mean return | stdev | success | dock entered | mean steps | "
        "final goal dist (m) | final angle err (rad) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['policy']} | {s['mean_return']:.2f} | {s['stdev_return']:.2f} | "
            f"{s['success_rate']:.1%} | {s['dock_rate']:.1%} | {s['mean_steps']:.1f} | "
            f"{s['mean_final_goal_distance']:.3f} | {s['mean_final_angle_error']:.3f} |"
        )

    if curve:
        lines += ["", "## Learning curve (fixed-seed evaluation during training)", "",
                  "| timesteps | mean return | success | dock entered | goal dist (m) |",
                  "|---:|---:|---:|---:|---:|"]
        for p in curve:
            lines.append(f"| {p['timesteps']:,} | {p['mean_return']:.2f} | "
                         f"{p['success_rate']:.1%} | {p['dock_rate']:.1%} | "
                         f"{p['mean_final_goal_distance']:.3f} |")
        if conv:
            lines += ["", "## Convergence", "",
                      f"- best success rate observed: {conv['best_success_rate']:.1%}",
                      f"- best mean return observed: {conv['best_mean_return']:.2f}",
                      f"- first checkpoint at/above half of best success: "
                      f"{conv['first_timesteps_at_or_above_half_of_best']}",
                      f"- last checkpoint below half of best success: "
                      f"{conv['last_timesteps_below_half_of_best']}"]

    lines += ["", "## Notes", "",
              "- `ppo_native_reward` trains on the environment's own native reward, i.e. the",
              "  `--use-original-reward` reference used to bound what a generated reward can",
              "  realistically reach.",
              "- One successful episode contributes about +311 to that episode's native",
              "  return, so each additional success shifts `mean_return` by roughly",
              "  311/episodes.",
              ""]
    (out_dir / "baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")

    if conv:
        print("--- convergence ---")
        print(f"  best success rate: {conv['best_success_rate']:.1%}   "
              f"best mean return: {conv['best_mean_return']:.2f}")
        print(f"  first checkpoint at/above half of best success: "
              f"{conv['first_timesteps_at_or_above_half_of_best']}")
        print(f"  last checkpoint below half of best success: "
              f"{conv['last_timesteps_below_half_of_best']}")

    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
