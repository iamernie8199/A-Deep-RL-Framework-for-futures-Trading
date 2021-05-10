import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import gym
from stable_baselines3 import PPO, DDPG, A2C, TD3
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Disable the warnings
import warnings
warnings.filterwarnings('ignore')

from env.env_long import TradingEnvLong
from utils import Futures

def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
        all_episode_rewards.append(sum(episode_rewards))
    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    return mean_episode_reward


data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
train = data_df[(data_df.Date >= '2000-01-01') & (data_df.Date < '2019-01-01')]
# the index needs to start from 0
train = train.reset_index(drop=True)

txf = Futures()
env_kwargs = {}
e_train_gym = TradingEnvLong(df=train, futures=txf, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
#model_ppo = PPO('MlpPolicy', env_train, tensorboard_log="./trading_2_tensorboard/")
model_ppo = PPO.load("ppo_trading", env=env_train, tensorboard_log="./trading_2_tensorboard/")
#%%
evaluate(model_ppo, num_episodes=5)
#%%
model_ppo.learn(total_timesteps=100000, tb_log_name="run_ppo")
model_ppo.save("ppo_trading")