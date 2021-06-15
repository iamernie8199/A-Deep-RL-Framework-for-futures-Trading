import numpy as np
import pandas as pd

from utils import vol_feature, candle, norm_ohlc, kalman

df = pd.read_csv("data/raw/BTCUSDT 1 Day.txt")
df.columns = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
df['Date'] = pd.to_datetime(df['Date'])
df = vol_feature(df)  # volume feature
df = candle(df)  # candlestick feature
df = norm_ohlc(df)  # normalized ohlc
df['log_rtn'] = np.log(df['Close']).diff(1).round(5)
# kalman filter
df['kalman_log_rtn1'] = np.log(kalman(df.Close)).diff(1).round(5)
df = df[1:]  # df[0] has nan
df = df.drop(columns=["Time"])
df.to_csv("BTCday.csv", index=False)
