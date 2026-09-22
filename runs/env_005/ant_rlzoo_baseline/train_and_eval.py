import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == '__main__':
    ENV_ID = "Ant-v4"
    TOTAL_TIMESTEPS = 1_000_000
    N_ENVS = 4          # 虚拟并行数，加快采样
    SEED = 0

    # ---------- 训练（使用向量化环境 + 观测归一化） ----------
    def make_train_env():
        def _init():
            return gym.make(ENV_ID)
        return _init

    train_env = DummyVecEnv([make_train_env() for _ in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(
            ortho_init=True,
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        verbose=1,
        seed=SEED,
        device="cpu",
    )

    print("Training... (this may take ~20 minutes)")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 保存模型和归一化统计量
    model.save("ant_rlzoo_tuned")
    train_env.save("vecnormalize_rlzoo.pkl")
    train_env.close()
    print("Training finished.\n")

    # ---------- 评估（单环境 + 手动观测归一化） ----------
    # 加载统计量
    norm = VecNormalize.load("vecnormalize_rlzoo.pkl")
    obs_rms = norm.obs_rms   # 包含 mean 和 var

    def normalize_obs(obs):
        """使用保存的均值和标准差归一化观测"""
        return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)

    # 创建普通环境（无向量化，关闭健康终止）
    eval_env = gym.make(ENV_ID, terminate_when_unhealthy=False)
    model = PPO.load("ant_rlzoo_tuned")

    episode_rewards = []
    for ep in range(100):
        obs, _ = eval_env.reset(seed=SEED * 100 + ep)
        done = False
        truncated = False
        total_reward = 0.0
        while not (done or truncated):
            # 归一化观测后再预测
            obs_normalized = normalize_obs(obs)
            action, _ = model.predict(obs_normalized, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
        print(f"Ep {ep+1:3d}: {total_reward:.2f}")

    rewards = np.array(episode_rewards)
    print(f"\n--- Ant-v4 RL Zoo Tuned Baseline (seed {SEED}) ---")
    print(f"Mean : {np.mean(rewards):.2f}")
    print(f"Std  : {np.std(rewards):.2f}")
    print(f"Min  : {np.min(rewards):.2f}")
    print(f"Max  : {np.max(rewards):.2f}")

    np.savetxt("ant_rlzoo_tuned_baseline.csv", rewards, header="reward", comments="")
    eval_env.close()