import shutil
import warnings

import pandas as pd
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN

from env.env_long2 import TradingEnvLong
from utils import random_rollout, result_plt, year_frac

warnings.filterwarnings('ignore')

data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
env_kwargs = {}

e_test_gym = TradingEnvLong(df=data_df[data_df.Date >= '2000-01-01'], log=True, **env_kwargs)
"""
# %%
year_num = year_frac(e_test_gym.equity_memory['date'].iloc[0],
                     e_test_gym.equity_memory[e_test_gym.equity_memory.equity_tmp > 0]['date'].iloc[-1])
cagr = ((out_df['Net Pnl'].mean() + e_test_gym.init_equity) / e_test_gym.init_equity) ** (1 / year_num) - 1
print(round(cagr * 100, 3))
"""
# %%
model = DQN.load("./logs/dqn_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda',
                    gamma=0.9, batch_size=4096, optimize_memory_usage=True,target_update_interval=5000)
# model = PPO.load("./logs/ppo_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.8)
# model = QRDQN.load("./logs/qrdqn_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.9)

out = []
# %%
for _ in range(1):
    obs_test = e_test_gym.reset()
    done = False
    while not done:
        action, _states = model.predict(obs_test)
        obs_test, rewards, done, tmp = e_test_gym.step(action)
        # env_test.render()
    out.append(tmp)
out_df = pd.DataFrame(out, columns=['Net Pnl', 'rtn_on_MDD', 'PF', 'CAGR', 'num', 'winning_rate'])
