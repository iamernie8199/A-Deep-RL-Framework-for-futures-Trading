import shutil
from datetime import datetime

import pandas as pd
from stable_baselines3 import DQN, PPO
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.dqn as dqn

from env.env_long2 import TradingEnvLong
from utils import random_rollout, result_plt, split_result, year_frac


def create_env(env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}
    data_df = pd.read_csv("data_simple2.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    train = data_df[(data_df.Date >= '2000-01-01')]
    # the index needs to start from 0
    train = train.reset_index(drop=True)
    env = TradingEnvLong(df=train, log=True, **env_kwargs)
    return env


def latexprint(row):
    print(row.name, end=' & ')
    print(' & '.join(map(str, row.values[:3])), end=' & ')
    print(f'{round(row.values[3] * 100, 2)}\%', end=' & ')
    print(f'{row.values[4]} & {round(row.values[5] * 100, 2)}\% ', end=r'\\')
    print()


def avg_print(tmp):
    print('\hline')
    print('Avg. & ', end='')
    print(f"{tmp[0]} & "
          f"{round(tmp[1], 4)} & {round(tmp[2], 4)}"
          f" & {round(tmp[3], 3) * 100}\% & {tmp[4]} & {round(tmp[5] * 100, 3)}\% "
          r"\\")


def split_print():
    result = split_result().round(4)
    result_list = result.values.tolist()
    print("\multicolumn{7}{c}{'00 - '10} ", end=r"\\")
    print()
    print("\hline")
    t = result[col1].apply(latexprint, axis=1)
    t = result[col1].mean().values.tolist()
    avg_print(t)
    print("\hline")
    print("\multicolumn{7}{c}{'10 - '20} ", end=r"\\")
    print()
    print("\hline")
    t = result[col2].apply(latexprint, axis=1)
    t = result[col2].mean().values.tolist()
    avg_print(t)
    print("\hline")
    print("\multicolumn{7}{c}{'20 - '21} ", end=r"\\")
    print()
    print("\hline")
    t = result[col3].apply(latexprint, axis=1)
    t = result[col3].mean().values.tolist()
    avg_print(t)
    print("\hline")


def latexsummary(o):
    out_df = pd.DataFrame(o, columns=['Net Pnl', 'rtn_on_MDD', 'PF', 'CAGR', 'num', 'winning_rate'])
    for i in range(len(o)):
        print(f"{i} & {int(out[i][0])} & {out[i][1]} & {out[i][2]} & {out[i][3]}\% & {out[i][4]} & {out[i][5]}\% "
              r"\\")
    year_num = year_frac(datetime.strptime("2000-01-01", "%Y-%m-%d"), datetime.strptime("2021-03-11", "%Y-%m-%d"))
    cagr = ((out_df['Net Pnl'].mean() + test_gym.init_equity) / test_gym.init_equity) ** (1 / year_num) - 1
    print('\hline')
    print('Avg. & ', end='')
    print(f"{int(out_df['Net Pnl'].mean())} & "
          f"{round(out_df['rtn_on_MDD'].mean(), 4)} & {round(out_df['PF'].mean(), 4)}"
          f" & {round(cagr * 100, 2)}\% & {round(out_df['num'].mean(), 2)} & {round(out_df['winning_rate'].mean(), 2)}\% "
          r"\\")


test_gym = create_env()
col1 = ['t1_net', 't1_rtn_mdd', 't1_cagr', 't1_pf', 't1_num', 't1_rate']
col2 = ['t2_net', 't2_rtn_mdd', 't2_cagr', 't2_pf', 't2_num', 't2_rate']
col3 = ['t3_net', 't3_rtn_mdd', 't3_cagr', 't3_pf', 't3_num', 't3_rate']

# random
out = []
for _ in range(10):
    info = random_rollout(test_gym)
    out.append(info[0])
latexsummary(out)
result_plt(title='random')
split_print()
shutil.move("results_pic", "results/random")
# bnh
out = []
info = random_rollout(test_gym, bnh=True)
out.append(info[0])
latexsummary(out)
result_plt(title='BnH')
split_print()
shutil.move("results_pic", "results/BnH")

# sb3 env
data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
env_kwargs = {}
e_test_gym = TradingEnvLong(df=data_df[data_df.Date >= '2000-01-01'], log=True, **env_kwargs)

# DQN
model = DQN.load("./logs/dqn_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda',
                 gamma=0.9, batch_size=4096, optimize_memory_usage=True, target_update_interval=5000)
out = []
for _ in range(10):
    obs_test = e_test_gym.reset()
    done = False
    while not done:
        action, _states = model.predict(obs_test)
        obs_test, rewards, done, tmp = e_test_gym.step(action)
        # env_test.render()
    out.append(tmp)
result_plt(title='DQN')
latexsummary(out)
split_print()
shutil.move("results_pic", "results/DQN")
# PPO
model = PPO.load("./logs/ppo_best_model", env=e_test_gym, tensorboard_log="./trading_2_tensorboard/", device='cuda',
                 gamma=0.8)
out = []
for _ in range(10):
    obs_test = e_test_gym.reset()
    done = False
    while not done:
        action, _states = model.predict(obs_test)
        obs_test, rewards, done, tmp = e_test_gym.step(action)
        # env_test.render()
    out.append(tmp)
result_plt(title='PPO')
latexsummary(out)
split_print()
shutil.move("results_pic", "results/PPO")
# ray
register_env("TestEnv", create_env)
ray.init()

# Double-DQN
checkpoint_path = 'DQN_TestEnv_2021-06-18_03-07-18dlvhiswk/checkpoint_001000/checkpoint-1000'
agent = dqn.DQNTrainer(
    env="TestEnv",
    config={
        "env": "TestEnv",
        "log_level": "WARN",
        "framework": "tf",
        "ignore_worker_failures": True,
        "num_gpus": 1,
        "num_atoms": 1,
        "v_min": -10000.0,
        "v_max": 10000.0,
        "noisy": False,
        "dueling": False,
        "hiddens": [512],
        "n_step": 1,
        "double_q": True,
        "gamma": 0.9,
        "lr": .0001,
        "learning_starts": 10000,
        "buffer_size": 50000,
        "rollout_fragment_length": 4,
        "train_batch_size": 32,
        "exploration_config": {
            "epsilon_timesteps": 2,
            "final_epsilon": 0.0,
        },
        "target_network_update_freq": 500,
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.5,
        "final_prioritized_replay_beta": 1.0,
        "prioritized_replay_beta_annealing_timesteps": 400000,
    }
)
agent.restore(checkpoint_path)
out = []
for _ in range(10):
    done = False
    obs = test_gym.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, tmp = test_gym.step(action)
        # test_gym.render()
    out.append(tmp)
result_plt(title='dqn_double')
latexsummary(out)
split_print()
shutil.move("results_pic", "results/Double-DQN")

# Dueling-DQN
checkpoint_path = 'DQN_TestEnv_2021-06-18_04-44-503upfoi0h/checkpoint_002000/checkpoint-2000'
agent = dqn.DQNTrainer(
    env="TestEnv",
    config={
        "env": "TradingEnv",
        "log_level": "WARN",
        "framework": "tf",
        "ignore_worker_failures": True,
        "num_gpus": 1,
        "num_atoms": 1,
        "v_min": -10000.0,
        "v_max": 10000.0,
        "noisy": False,
        "dueling": True,
        "hiddens": [512],
        "n_step": 1,
        "double_q": False,
        "gamma": 0.9,
        "lr": .0001,
        "learning_starts": 10000,
        "buffer_size": 50000,
        "rollout_fragment_length": 4,
        "train_batch_size": 32,
        "exploration_config": {
            "epsilon_timesteps": 2,
            "final_epsilon": 0.0,
        },
        "target_network_update_freq": 500,
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.5,
        "final_prioritized_replay_beta": 1.0,
        "prioritized_replay_beta_annealing_timesteps": 400000,
    }
)
agent.restore(checkpoint_path)
out = []
for _ in range(10):
    done = False
    obs = test_gym.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, tmp = test_gym.step(action)
        # test_gym.render()
    out.append(tmp)
result_plt(title='dqn_duel')
latexsummary(out)
split_print()
shutil.move("results_pic", "results/Dueling-DQN")

# Noisy-DQN
checkpoint_path = 'DQN_TestEnv_2021-06-18_15-33-25_c944tj8/checkpoint_000500/checkpoint-500'
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