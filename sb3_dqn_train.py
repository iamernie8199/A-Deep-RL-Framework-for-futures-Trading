import warnings

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from env.env_long2 import TradingEnvLong

warnings.filterwarnings('ignore')

data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
train = data_df[(data_df.Date >= '2010-01-01') & (data_df.Date < '2020-01-01')]
# the index needs to start from 0
train = train.reset_index(drop=True)

env_kwargs = {}
e_train_gym = TradingEnvLong(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
check_env(e_train_gym)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(env_train, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
# %%
"""
DQN
"""
#model_dqn = DQN('MlpPolicy', env_train, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.9, batch_size=4096,optimize_memory_usage=True)
model_dqn = DQN.load("./logs/dqn_best_model", env=env_train, tensorboard_log="./trading_2_tensorboard/", device='cuda',
                    gamma=0.9, batch_size=4096,optimize_memory_usage=True, target_update_interval=5000)
# %%
model_dqn.learn(total_timesteps=10000, tb_log_name="run_ppo", callback=eval_callback)
# model_ppo.save("ppo")
