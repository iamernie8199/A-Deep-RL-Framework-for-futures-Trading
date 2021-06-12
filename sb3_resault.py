import warnings

import pandas as pd
from stable_baselines3 import PPO

from env.env_long2 import TradingEnvLong
from utils import Futures

warnings.filterwarnings('ignore')

data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
txf = Futures()
env_kwargs = {}

e_test_gym = TradingEnvLong(df=data_df[data_df.Date >= '2000-01-01'], log=True, **env_kwargs)
env_test, _ = e_test_gym.get_sb_env()
# %%
model = PPO.load("./logs/ppo_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda',
                 gamma=0.8)
# model = QRDQN.load("./logs/qrdqn_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda', gamma=0.9)
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
for _ in range(10):
    obs_test = e_test_gym.reset()
    done = False
    while not done:
        action, _states = model.predict(obs_test)
        obs_test, rewards, done, tmp = e_test_gym.step(action)
        # env_test.render()
    # out.append(tmp)
# out_df = pd.DataFrame(out, columns=['Net Pnl','rtn_on_MDD','PF','CAGR','num','winning_rate'])
