import warnings

import pandas as pd
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.pg as pg

from env.env_long2 import TradingEnvLong

warnings.filterwarnings('ignore')


def create_env(env_kwargs={}):
    data_df = pd.read_csv("D:\Documents\GitHub\A-Deep-RL-Framework-for-Index-futures-Trading\data_simple2.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date >= '2000-01-01') & (data_df.Date < '2020-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    env = TradingEnvLong(df=train, **env_kwargs)
    return env


register_env("TestEnv", create_env)
ray.init()
checkpoint_path = "C:\\Users\\iamer\\ray_results\\PG\\PG_TradingEnv_a689c_00000_0_2021-06-11_17-08-56\\checkpoint_000003\\checkpoint-3"
# Restore agent
agent = pg.PGTrainer(
    env="TestEnv",
    config={
        "env": "TradingEnv",
        "log_level": "WARN",
        "framework": "tf",
        "ignore_worker_failures": True,
        "num_workers": 0,
        "num_cpus_per_worker": 2,
        "num_gpus": 1,
        "lr": 8e-6,
        "gamma": 0,
        "observation_filter": "NoFilter",
    }
)
agent.restore(checkpoint_path)

test_gym = create_env()
done = False
obs = test_gym.reset()
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, _ = test_gym.step(action)
    test_gym.render()

