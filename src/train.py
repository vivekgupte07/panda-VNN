import os
import gym_panda
import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


mode = 't_reach' # or 't_reach'
time = '1530_1208' #hhmm_ddmm

models_dir = "models/"+mode+"/sac/0.25_cu1_"+time+"_random-target"
logdir = "logs/"+mode

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = make_vec_env("panda-v0")


model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100000
i = 1
while True  :
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=mode+"_sac_0.25_cu1_"+time+"_random-target")  
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
#            Included t-reach related features in state, but did not reward it - oops! >> Still it confirms the scale of training episodes required

#1300_0908 : Removing t-reach features in state  : Most optimal - 250 steps

#2000_0908 : t-reach with cu1, t-reach=1000 +- 30%, also, constant cost per step of -0.00005
#2200_0908 : same as previous, fixed some bugs ----learns a suboptimal policy -- can try three things: 1)less cost for in_range 2) keep cu1 but change 
#              interval to (t-reach,t-reach+delta) -- reward going close to t-reach 3) less curiculum reward?
#2230_0908 : tried 1,2,3 from previous - didnt learn

#1150_1008 : tried prev again, same settings

#1200_1008 : retuned reward and tried 2200_0908 again

#1200_1208 : re trained 2230_0908 using SAC

#1500_1208 : re training 2230_0908 forever...

#1530_1208 : retraining 1200_1208 with SAC using buffer_size=1e5