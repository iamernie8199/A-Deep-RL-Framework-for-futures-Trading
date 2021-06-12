import warnings

import pandas as pd
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.pg as pg

from env.env_long2 import TradingEnvLong

warnings.filterwarnings('ignore')


def create_env(env_kwargs={}):
    data_df = pd.read_csv("D:\Documents\GitHub\A-Deep-RL-Framework-for-Index-futures-Trading\data_simple2.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date >= '2010-01-01') & (data_df.Date < '2020-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    environment = TradingEnvLong(df=train, **env_kwargs)
    return environment


register_env("TradingEnv", create_env)
analysis = tune.run(
    "PG",
    stop={
        "episode_reward_mean": 500
    },
    mode='max',
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
    },
    # restore from the last checkpoint
    #restore=checkpoint_path,
    reuse_actors=True,
    checkpoint_freq=3,
    checkpoint_at_end=True
)

# Get checkpoint
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean"
)
checkpoint_path = checkpoints[0][0]
print(checkpoint_path)
# C:\\Users\\iamer\\ray_results\\PPO\\PPO_TradingEnv_d524f_00000_0_2021-06-10_22-47-52\\checkpoint_000006\\checkpoint-6
# C:\Users\iamer\ray_results\PG\PG_TradingEnv_a689c_00000_0_2021-06-11_17-08-56\checkpoint_000003\checkpoint-3
# Restore agent
agent = pg.PGTrainer(
    env="TradingEnv",
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
