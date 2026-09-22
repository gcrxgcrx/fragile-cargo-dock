import gymnasium as gym
from stable_baselines3 import PPO
from pathlib import Path

MODEL_PATH = Path("seed_0/model")
ENV_ID = "Ant-v4"

env = gym.make(ENV_ID)
model = PPO.load(MODEL_PATH)

obs, _ = env.reset(seed=42)
action, _ = model.predict(obs, deterministic=True)
obs_next, reward, done, truncated, info = env.step(action)

print("reward:", reward)
print("done:", done, "truncated:", truncated)
print("info:", info)