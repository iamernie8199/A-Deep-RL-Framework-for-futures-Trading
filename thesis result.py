import shutil
from datetime import datetime

import pandas as pd
from stable_baselines3 import DQN

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
    for i in range(len(out)):
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
"""for i in range(len(out)):
    print(f"{i} & {int(out[i][0])} & {out[i][1]} & {out[i][2]} & {out[i][3]}\% & {out[i][4]} & {out[i][5]}\% "
          r"\\")
year_num = year_frac(datetime.strptime("2000-01-01", "%Y-%m-%d"), datetime.strptime("2021-03-11", "%Y-%m-%d"))
cagr = ((out_df['Net Pnl'].mean() + test_gym.init_equity) / test_gym.init_equity) ** (1 / year_num) - 1
print('\hline')
print('Avg. & ', end='')
print(f"{int(out_df['Net Pnl'].mean())} & "
      f"{round(out_df['rtn_on_MDD'].mean(), 4)} & {round(out_df['PF'].mean(), 4)}"
      f" & {round(cagr, 4) * 100}\% & {out_df['num'].mean()} & {out_df['winning_rate'].mean()}\% "
      r"\\")"""
result_plt(title='random')
split_print()
shutil.move("results_pic", "results/random")
# bnh
out = []
info = random_rollout(test_gym, bnh=True)
out.append(info[0])
latexsummary(out)
"""out_df = pd.DataFrame(out, columns=['Net Pnl', 'rtn_on_MDD', 'PF', 'CAGR', 'num', 'winning_rate'])
for i in range(len(out)):
    print(f"{i} & {int(out[i][0])} & {out[i][1]} & {out[i][2]} & {out[i][3]}\% & {out[i][4]} & {out[i][5]}\% "
          r"\\")"""
result_plt(title='BnH')
split_print()
shutil.move("results_pic", "results/BnH")
# DQN
data_df = pd.read_csv("data_simple.csv")
data_df['Date'] = pd.to_datetime(data_df['Date'])
env_kwargs = {}
e_test_gym = TradingEnvLong(df=data_df[data_df.Date >= '2000-01-01'], log=True, **env_kwargs)
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