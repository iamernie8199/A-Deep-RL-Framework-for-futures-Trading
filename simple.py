import pandas as pd
import matplotlib.pyplot as plt

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


data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
train = data_df[(data_df.Date >= '2000-01-01') & (data_df.Date < '2019-01-01')]
# the index needs to start from 0
train = train.reset_index(drop=True)

txf = Futures()
env_kwargs = {}
e_train_gym = TradingEnvLong(df=train, futures=txf, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
model_ppo = PPO('MlpPolicy', env_train, tensorboard_log="./trading_2_tensorboard/")
model_ppo.learn(total_timesteps=100000, tb_log_name="run_ppo")
