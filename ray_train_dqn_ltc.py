import warnings

import pandas as pd
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.dqn as dqn

from env.env_long2 import TradingEnvLong

warnings.filterwarnings('ignore')


def create_env(env_kwargs={}):
    data_df = pd.read_csv("/home/sean/Docs/GitHub/A-Deep-RL-Framework-for-Index-futures-Trading/LTCday.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date < '2020-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    env = TradingEnvLong(df=train, big_point_value=1, cost=0, **env_kwargs)
    return env


register_env("trainEnv", create_env)
ray.init()
# Restore agent
checkpoint_path = '/home/sean/ray_results/DQN_trainEnv_2021-06-16_20-34-07zfn10y3o/checkpoint_004601/checkpoint-4601'
# '/home/sean/ray_results/DQN_trainEnv_2021-06-16_03-06-12lbhm2aw3/checkpoint_004500/checkpoint-4500'

agent = dqn.DQNTrainer(
    env="trainEnv",
    config={
        "env": "trainEnv",
        "log_level": "WARN",
        "framework": "tf",
        "ignore_worker_failures": True,
        "num_gpus": 1,
        "num_atoms": 51,
        "noisy": True,
        "v_min": -10000.0,
        "v_max": 10000.0,
        "gamma": 0.9,
        "lr": .0001,
        "hiddens": [512],
        "learning_starts": 10000,
        "buffer_size": 50000,
        "rollout_fragment_length": 4,
        "train_batch_size": 32,
        "exploration_config": {
            "epsilon_timesteps": 2,
            "final_epsilon": 0.0,
        },
        "target_network_update_freq": 500,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.5,
        "final_prioritized_replay_beta": 1.0,
        "prioritized_replay_beta_annealing_timesteps": 400000,
        "n_step": 3
    }
)
agent.restore(checkpoint_path)

for i in range(100):
    result = agent.train()

    if i % 100 == 0:
        checkpoint = agent.save()
        print("checkpoint saved at", checkpoint)