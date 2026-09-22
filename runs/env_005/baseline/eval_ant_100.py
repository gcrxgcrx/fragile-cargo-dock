import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path

MODEL_PATH = Path("seed_0/model")
VECNORM_PATH = Path("seed_0/vecnormalize")
N_EPISODES = 100
ENV_ID = "Ant-v4"

env = gym.make(ENV_ID)
if VECNORM_PATH.exists():
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(str(VECNORM_PATH), env)
    env.training = False
    env.norm_reward = False
    is_vecenv = True
else:
    is_vecenv = False

model = PPO.load(MODEL_PATH)

episode_rewards = []

for ep in range(N_EPISODES):
    seed = 1000 + ep
    if is_vecenv:
        env.seed(seed)
        obs = env.reset()
    else:
        obs, _ = env.reset(seed=seed)

    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        if is_vecenv:
            obs, reward, done, truncated = env.step(action)
            total_reward += reward[0]
            done = done[0]
            truncated = truncated[0]
        else:
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    print(f"Episode {ep+1:3d}: reward = {total_reward:.2f}, steps = {steps}")

rewards = np.array(episode_rewards)
mean = np.mean(rewards)
std = np.std(rewards)

print("\n========== Statistics (100 episodes) ==========")
print(f"Mean   : {mean:.2f}")
print(f"Std    : {std:.2f}")
print(f"Min    : {np.min(rewards):.2f}")
print(f"Max    : {np.max(rewards):.2f}")
print(f"Median : {np.median(rewards):.2f}")

output_file = Path("ant_baseline_100_episodes.csv")
np.savetxt(output_file, rewards, header="reward", comments="")
print(f"Rewards saved to {output_file.resolve()}")