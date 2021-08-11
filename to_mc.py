import pandas as pd
import shutil

from utils import action_result

model = 'ppo_ray'
mc_path = 'mc/ppo_ray.csv'

action_result(title=model)
mc_df = pd.read_csv(mc_path)
mc_df['date'] = pd.to_datetime(mc_df['date'])
mc_df = mc_df.set_index('date')
new = pd.read_csv(f"{model}.csv")
new['date'] = pd.to_datetime(new['date'])
new = new.set_index('date')
new = new.loc[mc_df.index[-1]:].drop(mc_df.index[-1])
mc_df = pd.concat([mc_df, new])
mc_df.to_csv(mc_path)

shutil.rmtree('results_pic')
