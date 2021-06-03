import warnings
from glob import glob

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env.env_long2 import TradingEnvLong
from utils import Futures

warnings.filterwarnings('ignore')


def random_rollout(env):
    state = env.reset()
    dones = False
    # Keep looping as long as the simulation has not finished.
    while not dones:
        # Choose a random action (0, 1, 2).
        act = np.random.choice(3, 1)[0]
        # Take the action in the environment.
        state, reward, dones, _ = env.step(act)
    # Return the cumulative reward.
    return reward


def result(title=''):
    equitylist = glob('results_pic/equity_*.csv')
    random_df = pd.read_csv(equitylist[0])[['date', 'equity_tmp']]
    for path in equitylist[1:]:
        random_df2 = pd.read_csv(path)[['date', 'equity_tmp']]
        random_df = random_df.merge(random_df2, how='left', on='date')
    random_df = random_df.set_index(pd.to_datetime(random_df['date'])).drop(columns='date')
    random_df.columns = range(len(random_df.columns))
    # random_df.plot(legend=False, title=title)
    sns.lineplot(data=random_df, legend=False, linewidth=1.25).set_title(title)
    plt.show()


data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
txf = Futures()
env_kwargs = {}

e_test_gym = TradingEnvLong(df=data_df[data_df.Date >= '2000-01-01'], futures=txf, log=True, **env_kwargs)
env_test, _ = e_test_gym.get_sb_env()
# %%
model = PPO.load("./logs/ppo_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda',
                 gamma=0.8)
# model = QRDQN.load("./logs/qrdqn_best_model", env_train, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.9)
"""
# %%
for _ in range(20):
    random_rollout(e_test_gym)

# %%
obs_test = e_test_gym.reset()
done = False
while not done:
    action, _states = model.predict(obs_test)
    obs_test, rewards, done, _ = e_test_gym.step(action)
    e_test_gym.render()"""
