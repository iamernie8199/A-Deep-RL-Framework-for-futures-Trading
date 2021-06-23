import warnings

import pandas as pd
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.dqn as dqn

from env.env_long2 import TradingEnvLong

warnings.filterwarnings('ignore')


def create_env(env_kwargs={}):
    data_df = pd.read_csv("data_simple2.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date >= '2010-01-01') & (data_df.Date < '2020-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    env = TradingEnvLong(df=train, **env_kwargs)
    return env


register_env("TestEnv", create_env)
ray.init()
checkpoint_path = '/home/sean/ray_results/DQN_TestEnv_2021-06-18_15-33-25_c944tj8/checkpoint_000500/checkpoint-500'
# Restore agent
agent = dqn.DQNTrainer(
    env="TestEnv",
    config={
        "env": "TradingEnv",
        "log_level": "WARN",
        "framework": "tf",
        "ignore_worker_failures": True,
        "num_gpus": 1,
        "num_atoms": 1,
        "noisy": True,
        "v_min": -10000.0,
        "v_max": 10000.0,
        "gamma": 0.9,
        "lr": .0001,
        "dueling": False,
        "hiddens": [512],
        "learning_starts": 10000,
        "buffer_size": 50000,
        "rollout_fragment_length": 4,
        "train_batch_size": 32,
        "double_q": False,
        "exploration_config": {
            "epsilon_timesteps": 2,
            "final_epsilon": 0.0,
        },
        "target_network_update_freq": 500,
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.5,
        "final_prioritized_replay_beta": 1.0,
        "prioritized_replay_beta_annealing_timesteps": 400000,
        "n_step": 1
    }
)
agent.restore(checkpoint_path)

for i in range(500):
    # Perform one iteration of training the policy with PPO
    result = agent.train()

    if i % 100 == 0:
        checkpoint = agent.save()
        print("checkpoint saved at", checkpoint)