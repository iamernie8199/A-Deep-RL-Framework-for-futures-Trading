import warnings

import pandas as pd
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.dqn as dqn

from env.env_long2 import TradingEnvLong
from utils import result, year_frac

warnings.filterwarnings('ignore')


def create_env(env_kwargs={}):
    data_df = pd.read_csv("/home/sean/Docs/GitHub/A-Deep-RL-Framework-for-Index-futures-Trading/BTCday.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date >= '2019-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    env = TradingEnvLong(df=train, big_point_value=1, cost=0, log=True, **env_kwargs)
    return env


register_env("TestEnv", create_env)
ray.init()
checkpoint_path = '/home/sean/ray_results/DQN_trainEnv_2021-06-15_21-23-183tf_lf8i/checkpoint_001500/checkpoint-1500'
#'/home/sean/ray_results/DQN_trainEnv_2021-06-15_18-55-32yxp9433d/checkpoint_001000/checkpoint-1000'

# Restore agent
agent = dqn.DQNTrainer(
    env="TestEnv",
    config={
        "env": "TestEnv",
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

test_gym = create_env()
out = []
#%%
for _ in range(20):
    done = False
    obs = test_gym.reset()
    while not done:
        action = agent.compute_action(obs)
        #action = 1
        obs, reward, done, tmp = test_gym.step(action)
        test_gym.render()
    out.append(tmp)
out_df = pd.DataFrame(out, columns=['Net Pnl', 'rtn_on_MDD', 'PF', 'CAGR', 'num', 'winning_rate'])
#%%
result(title='dqn_Rainbow_btc')
year_num = year_frac(test_gym.equity_memory['date'].iloc[0],
                     test_gym.equity_memory[test_gym.equity_memory.equity_tmp > 0]['date'].iloc[-1])
cagr = ((out_df['Net Pnl'].mean() + test_gym.init_equity) / test_gym.init_equity) ** (1 / year_num) - 1
print(round(cagr * 100, 3))