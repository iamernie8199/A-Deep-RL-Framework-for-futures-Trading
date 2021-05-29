import warnings

import numpy as np
import pandas as pd
from sb3_contrib import QRDQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from env.env_long2 import TradingEnvLong
from utils import Futures

warnings.filterwarnings('ignore')

data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
train = data_df[(data_df.Date >= '2010-01-01') & (data_df.Date < '2020-01-01')]
# the index needs to start from 0
train = train.reset_index(drop=True)

txf = Futures()
env_kwargs = {}
e_train_gym = TradingEnvLong(df=train, futures=txf, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
check_env(e_train_gym)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(env_train, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
# %%
"""
QRDQN
"""
policy_kwargs = dict(n_quantiles=50)
"""model_qrdqn = QRDQN("MlpPolicy", env_train, policy_kwargs=policy_kwargs, gamma=0.9, batch_size=4096,
                    learning_starts=10000, buffer_size=2000000, exploration_initial_eps=0.1,
                    optimize_memory_usage=True, device='cuda', tensorboard_log="./trading_2_tensorboard/")"""
model_qrdqn = QRDQN.load("./logs/qrdqn_best_model", env_train, tensorboard_log="./trading_2_tensorboard/",
                         device='cuda', gamma=0.9)
# %%
model_qrdqn.learn(total_timesteps=10000, tb_log_name="run_DQN", callback=eval_callback, n_eval_episodes=5)
# %%
"""
PPO
"""
# model_ppo = PPO('MlpPolicy', env_train, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.7)
model_ppo = PPO.load("./logs/ppo_best_model", env=env_train, tensorboard_log="./trading_2_tensorboard/", device='cuda',
                     gamma=0.8, batch_size=1024)
# %%
model_ppo.learn(total_timesteps=10000, tb_log_name="run_ppo", callback=eval_callback)
# model_ppo.save("ppo")

# %%
model = PPO.load("./logs/ppo_best_model", env=env_train, tensorboard_log="./trading_2_tensorboard/", device='cuda',
                 gamma=0.8)
# model = QRDQN.load("./logs/qrdqn_best_model", env_train, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.9)

e_test_gym = TradingEnvLong(df=data_df[data_df.Date >= '2000-01-01'], futures=txf, log=True, **env_kwargs)
env_test, _ = e_test_gym.get_sb_env()
# %%
obs_test = env_test.reset()
done = False
while not done:
    action, _states = model.predict(obs_test)
    obs_test, rewards, done, _ = env_test.step(action)
    env_test.render()

"""
%load_ext tensorboard
%tensorboard --logdir ./trading_2_tensorboard/run_DQN_19/
"""
