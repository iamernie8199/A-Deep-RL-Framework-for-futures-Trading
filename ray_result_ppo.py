import warnings

import pandas as pd
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

from env.env_long2 import TradingEnvLong
from utils import result_plt, year_frac

warnings.filterwarnings('ignore')


def create_env(env_kwargs={}):
    data_df = pd.read_csv("data_simple2.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date >= '2000-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    env = TradingEnvLong(df=train, log=True, **env_kwargs)
    return env


register_env("TestEnv", create_env)
ray.init()
checkpoint_path = 'model/PPO/checkpoint_000700/checkpoint-700'
# 'PPO/checkpoint_000500/checkpoint-500'

# Restore agent
agent = ppo.PPOTrainer(
    env="TestEnv",
    config={
        "env": "TradingEnv",
        "log_level": "WARN",
        "framework": "tf",
        "ignore_worker_failures": True,
        "num_gpus": 1,
        "gamma": 0.9,
        "lambda": 0.95,
        "clip_param": 0.2,
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": .0001,
        "sgd_minibatch_size": 32768,
        "horizon": 5000,
        "train_batch_size": 320000,
        "vf_clip_param": 5000.0,
        "model": {
            "vf_share_layers": False,
        },
        "num_workers": 1,
    }
)
agent.restore(checkpoint_path)

test_gym = create_env()
out = []
#%%
for _ in range(1):
    done = False
    obs = test_gym.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, tmp = test_gym.step(action)
        #test_gym.render()
    out.append(tmp)
out_df = pd.DataFrame(out, columns=['Net Pnl', 'rtn_on_MDD', 'PF', 'CAGR', 'num', 'winning_rate'])
result_plt(title='PPO')
year_num = year_frac(test_gym.equity_memory['date'].iloc[0],
                     test_gym.equity_memory[test_gym.equity_memory.equity_tmp > 0]['date'].iloc[-1])
cagr = ((out_df['Net Pnl'].mean() + test_gym.init_equity) / test_gym.init_equity) ** (1 / year_num) - 1
print(round(cagr * 100, 3))
