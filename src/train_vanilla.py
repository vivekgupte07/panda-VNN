import os
import gym_panda
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy



models_dir = "models/t_reach/ppo/0.25_cu2_2325_0408_random-target"
logdir = "logs/t_reach"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = make_vec_env("panda-v0")


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
i = 1
while i <= 10:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="t_reach_ppo_0.25_cu2_2325_0408_random-target")  
    # Name convention: type_algo_target-radius_t-reach-interval_reward-type
    #c:curriculum,sp:sparse, co:cont)_time(hhmm)_date(dddmm)_state#noof-elements
    print("Saving...")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1







#Reward Curriculums
#curriculum-1(cu1): Award 0.5 for reaching the goal in any number of steps, 1 for reaching within the T-reach interval - did not work in 100k steps
#cu2 - cu1 plus T-reach-interval starts wide and shrinks as robot succeeds.
