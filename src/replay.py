import os
import gymnasium as gym
import gym_panda
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

models_dir = "models/vanilla/ppo/0.25_co1_1300_0908_random-target"

env = make_vec_env("panda-v0")

dones = False
model = PPO.load(models_dir+'/1000000.zip')

obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #print(obs[0])

    #if terminated or truncated:
    #    dones = True

    if dones:
        obs = env.reset()
    #env.render(mode='human')