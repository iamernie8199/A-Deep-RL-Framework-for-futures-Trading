import warnings
from glob import glob
import shutil
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from sb3_contrib import QRDQN

from env.env_long2 import TradingEnvLong
from utils import Futures

warnings.filterwarnings('ignore')


def random_rollout(env, bnh=False):
    state = env.reset()
    dones = False
    # Keep looping as long as the simulation has not finished.
    while not dones:
        # Choose a random action (0, 1, 2).
        if bnh:
            act = 1
        else:
            act = np.random.choice(3, 1)[0]
        # Take the action in the environment.
        state, reward, dones, info = env.step(act)
    return info


def result(title=''):
    equitylist = glob('results_pic/equity_*.csv')
    random_df = pd.read_csv(equitylist[0])[['date', 'equity_tmp']]
    for path in equitylist[1:]:
        random_df2 = pd.read_csv(path)[['date', 'equity_tmp']]
        random_df = random_df.merge(random_df2, how='left', on='date')
    random_df = random_df.set_index(pd.to_datetime(random_df['date'])).drop(columns='date')
    random_df.columns = range(len(random_df.columns))
    # random_df.plot(legend=False, title=title)
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=random_df, legend=False, linewidth=1.25)
    ax.set_title(title)
    ax.axhline(100000, ls='-.', c='grey')
    plt.savefig('results_pic/{}.png'.format(title), bbox_inches='tight')


def price():
    data = pd.read_csv('data/clean/WTX&.csv')
    data = data.set_index(pd.to_datetime(data['Date']))
    data = data.drop(columns=['Date', 'OI'])
    mc = mpf.make_marketcolors(up='#fe3032', down='#00b060', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-.', gridaxis='horizontal')
    mpf.plot(data.loc['2000-01-01':], type='candle', volume=True, panel_ratios=(5, 1), style=s, figratio=(12, 7),
             ylabel='', ylabel_lower='', tight_layout=True, datetime_format="'%y-%b-%d", xrotation=0,
             returnfig=True)
    plt.savefig('TAIEX Futures.png', bbox_inches='tight')
    plt.show()


data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
txf = Futures()
env_kwargs = {}

e_test_gym = TradingEnvLong(df=data_df[data_df.Date >= '2000-01-01'], futures=txf, log=True, **env_kwargs)
env_test, _ = e_test_gym.get_sb_env()
# %%
model = PPO.load("./logs/ppo_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.8)
#model = QRDQN.load("./logs/qrdqn_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.9)
"""
# %%
for _ in range(20):
    random_rollout(e_test_gym)
result(title='random')
shutil.rmtree('results')
shutil.rmtree('results_pic')
"""
out = []
# %%
for _ in range(1):
    obs_test = e_test_gym.reset()
    done = False
    while not done:
        action, _states = model.predict(obs_test)
        obs_test, rewards, done, tmp = e_test_gym.step(action)
        env_test.render()
    out.append(tmp)
out_df = pd.DataFrame(out, columns=['Net Pnl','rtn_on_MDD','PF','CAGR','num','winning_rate'])