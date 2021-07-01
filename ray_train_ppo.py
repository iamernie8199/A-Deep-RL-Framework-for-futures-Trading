import warnings

import pandas as pd
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

from env.env_long2 import TradingEnvLong

warnings.filterwarnings('ignore')


def create_env(env_kwargs={}):
    data_df = pd.read_csv("/home/sean/Docs/GitHub/A-Deep-RL-Framework-for-Index-futures-Trading/data_simple2.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date >= '2010-01-01') & (data_df.Date < '2020-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    env = TradingEnvLong(df=train, **env_kwargs)
    return env


register_env("TestEnv", create_env)
ray.init()
checkpoint_path = '/home/sean/ray_results/PPO_TestEnv_2021-06-28_21-24-141rfn176u/checkpoint_000500/checkpoint-500'
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
        "num_workers": 10,
    }
)

for i in range(500):
    # Perform one iteration of training the policy with PPO
    result = agent.train()

    if i % 100 == 0:
        checkpoint = agent.save()
        print("checkpoint saved at", checkpoint)
