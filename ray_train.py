import warnings

import pandas as pd
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

from env.env_long2 import TradingEnvLong

warnings.filterwarnings('ignore')


def create_env(env_kwargs={}):
    data_df = pd.read_csv("data_simple.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date >= '2010-01-01') & (data_df.Date < '2020-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    environment = TradingEnvLong(df=train, **env_kwargs)
    return environment


register_env("TradingEnv", create_env)
analysis = tune.run(
    "PPO",
    stop={
        "episode_reward_mean": 500
    },
    mode='max',
    config={
        "env": "TradingEnv",
        "log_level": "WARN",
        "framework": "tf",
        "ignore_worker_failures": True,
        "num_workers": 2,
        "num_cpus_per_worker": 2,
        "num_gpus": 1,
        "clip_rewards": True,
        "lr": 8e-6,
        "lr_schedule": [
            [0, 1e-1],
            [int(1e2), 1e-2],
            [int(1e3), 1e-3],
            [int(1e4), 1e-4],
            [int(1e5), 1e-5],
            [int(1e6), 1e-6],
            [int(1e7), 1e-7]
        ],
        "gamma": 0,
        "observation_filter": "MeanStdFilter",
        "lambda": 0.72,
        "vf_loss_coeff": 10,
        "entropy_coeff": 0.01,
    },
    # restore from the last checkpoint
    restore=checkpoint_path,
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
# C:\Users\iamer\ray_results\PPO\PPO_TradingEnv_c4d50_00000_0_2021-06-10_22-33-05\checkpoint_000005\checkpoint-5
# Restore agent
agent = ppo.PPOTrainer(
    env="TradingEnv",
    config={
        "framework": "tf",
        "log_level": "WARN",
        "ignore_worker_failures": True,
        "num_workers": 2,
        "num_gpus": 1,
        "clip_rewards": True,
        "lr": 8e-6,
        "lr_schedule": [
            [0, 1e-1],
            [int(1e2), 1e-2],
            [int(1e3), 1e-3],
            [int(1e4), 1e-4],
            [int(1e5), 1e-5],
            [int(1e6), 1e-6],
            [int(1e7), 1e-7]
        ],
        "gamma": 0,
        "observation_filter": "MeanStdFilter",
        "lambda": 0.72,
        "vf_loss_coeff": 10,
        "entropy_coeff": 0.01
    }
)
agent.restore(checkpoint_path)
