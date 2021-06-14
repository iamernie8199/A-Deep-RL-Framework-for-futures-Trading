import warnings

import pandas as pd
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.pg as pg

from env.env_long2 import TradingEnvLong

warnings.filterwarnings('ignore')


def create_env(env_kwargs={}):
    data_df = pd.read_csv("/home/sean/Docs/GitHub/A-Deep-RL-Framework-for-Index-futures-Trading/data_simple2.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date >= '2007-01-01') & (data_df.Date < '2020-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    environment = TradingEnvLong(df=train, **env_kwargs)
    return environment

#%%
register_env("TradingEnv", create_env)
checkpoint_path = '/home/sean/ray_results/PG/PG_TradingEnv_b2a5d_00000_0_2021-06-12_16-10-49/checkpoint_000006/checkpoint-6'
# '/home/sean/ray_results/PG/PG_TradingEnv_b2a5d_00000_0_2021-06-12_16-10-49/checkpoint_000006/checkpoint-6'
# '/home/sean/ray_results/PG/PG_TradingEnv_78ea6_00000_0_2021-06-12_21-02-42/checkpoint_000009/checkpoint-9'
# '/home/sean/ray_results/PG/PG_TradingEnv_314de_00000_0_2021-06-13_01-54-11/checkpoint_000012/checkpoint-12'
# '/home/sean/ray_results/PG/PG_TradingEnv_c5e0e_00000_0_2021-06-13_12-56-54/checkpoint_000013/checkpoint-13'
#%%
analysis = tune.run(
    "PG",
    stop={
        "episode_reward_mean": 1000
    },
    mode='max',
    config={
        "env": "TradingEnv",
        "log_level": "WARN",
        "framework": "tf",
        "ignore_worker_failures": True,
        "num_workers": 3,
        "num_cpus_per_worker": 2,
        "num_gpus": 1,
        "lr": 8e-6,
        "gamma": 0,
        "observation_filter": "NoFilter",
    },
    # restore from the last checkpoint
    restore=checkpoint_path,
    reuse_actors=True,
    checkpoint_freq=3,
    checkpoint_at_end=True
)
#%%
# Get checkpoint
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean"
)
checkpoint_path = checkpoints[0][0]
print(checkpoint_path)
# C:\\Users\\iamer\\ray_results\\PPO\\PPO_TradingEnv_d524f_00000_0_2021-06-10_22-47-52\\checkpoint_000006\\checkpoint-6
# C:\Users\iamer\ray_results\PG\PG_TradingEnv_a689c_00000_0_2021-06-11_17-08-56\checkpoint_000003\checkpoint-3
