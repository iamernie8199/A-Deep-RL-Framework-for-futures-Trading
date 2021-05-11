import pandas as pd
import numpy as np
import os

import gym
from stable_baselines3 import PPO, DDPG, A2C, TD3
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback

# Disable the warnings
import warnings
warnings.filterwarnings('ignore')

from env.env_long import TradingEnvLong
from utils import Futures

#%%
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
        return True

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
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and done are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
        all_episode_rewards.append(reward)
    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    return mean_episode_reward


data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
train = data_df[(data_df.Date >= '2010-01-01') & (data_df.Date < '2020-01-01')]
# the index needs to start from 0
train = train.reset_index(drop=True)

txf = Futures()
env_kwargs = {}
e_train_gym = TradingEnvLong(df=train, futures=txf, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(e_train_gym, log_dir)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(env_train, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
#%%
model_ppo = PPO('MlpPolicy', env_train, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.7)
#model_ppo = PPO.load("./logs/best_model", env=env_train, tensorboard_log="./trading_2_tensorboard/", device='cuda')
#%%
#evaluate(model_ppo, num_episodes=5)
#%%
model_ppo.learn(total_timesteps=1000000, tb_log_name="run_ppo", callback=eval_callback)
"""
%load_ext tensorboard
%tensorboard --logdir ./trading_2_tensorboard/run_ppo_1
"""
