import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

ENV_ID = "Ant-v4"
MODEL_PATH = "ant_rlzoo_baseline_seed0"
VECNORM_PATH = "vecnormalize_rlzoo_seed0.pkl"
N_EPISODES = 100

def make_env():
    env = gym.make(ENV_ID, terminate_when_unhealthy=False)
    return env

env = DummyVecEnv([make_env])
env = VecNormalize.load(VECNORM_PATH, env)
env.training = False
env.norm_reward = False

model = PPO.load(MODEL_PATH)

rewards = []
for ep in range(N_EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        done = done[0]
    rewards.append(total_reward)
    print(f"Ep {ep+1:3d}: {total_reward:.2f}")

arr = np.array(rewards)
print(f"\nMean: {np.mean(arr):.2f}, Std: {np.std(arr):.2f}")
np.savetxt("ant_rlzoo_100ep_stable.csv", arr, header="reward", comments="")