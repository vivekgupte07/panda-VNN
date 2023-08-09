import os
import gym_panda
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


mode = 'vanilla' # or 't_reach'

models_dir = "models/"+mode+"/ppo/0.25_co1_1900_0808_random-target"
logdir = "logs/"+mode

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = make_vec_env("panda-v0")


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100000
i = 1
while i <= 10:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=mode+"_ppo_0.25_co1_1900_0808_random-target")  
    # Name convention: mode_algo_target-radius_t-reach-interval_reward-type
    #c:curriculum,sp:sparse, co:cont)_time(hhmm)_date(dddmm)_state#noof-elements
    print("Saving...")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1







#Reward Curriculums
#curriculum-1(cu1): Award 0.5 for reaching the goal in any number of steps, 1 for reaching within the T-reach interval - did not work in 100k steps
#cu2 - cu1 plus T-reach-interval starts wide and shrinks (0.1 per step) as robot succeeds.


#co 1 : continuous rewards - 0.001 cost of taking a step 

#1900_0808 : Changed n_timesteps in sb3.ppo from 2048 to 20480 >> learned a suboptimal policy >> might be worth changing batch_size and epochs

#