import numpy as np
import pandas as pd
import psycopg2
from utils import basis, vol_feature, settlement_cal, oi_feature, candle, norm_ohlc, kalman, hurst
from config import pg_config

conn = psycopg2.connect(user=pg_config['user'],
                        password=pg_config['password'],
                        host=pg_config['host'],
                        port=pg_config['port'],
                        database=pg_config['dbname'])
sql = """
select DISTINCT ON (date) 
date, open as o, high, low, last as close, vt as volume, oi
from api_taifex_futures_price 
where contract='TX' and session='regular' and date>'2021-03-15'
order by date desc, contract_month asc
"""
df = pd.read_sql(sql, conn)
sql = """
select DISTINCT ON (date) 
date, open, high, low
from api_taifex_futures_price 
where contract='TX' and session='ah' and date>'2021-03-15'
order by date desc, contract_month asc
"""
df = df.merge(pd.read_sql(sql, conn), how='left', on='date')
df['High'] = df[['high_x', 'high_y']].max(axis=1)
df['Low'] = df[['low_x', 'low_y']].min(axis=1)
df['Open'] = df[['open', 'o']].apply(lambda x: x.open if x.open == x.open else x.o, axis=1)
df = df[['date', 'Open', 'High', 'Low', 'close', 'volume', 'oi']]
df = df.rename(columns={"date": "Date",
                        "close": "Close",
                        "volume": "Volume",
                        "oi": "OI"})
df = df.sort_index(ascending=False).reset_index(drop=True)
# concat with old data(dq2)
df = pd.concat([pd.read_csv("data/clean/WTX&.csv"), df], ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv(f"data/clean/TX.csv", index=False)

settlement = pd.read_csv("data/txf_settlement.csv")
settlement['txf_settlement'] = pd.to_datetime(settlement['txf_settlement'])
settlement = settlement.set_index(settlement['txf_settlement'])

df['basis'] = basis(df)  # basis feature
df = vol_feature(df)  # volume feature
df = oi_feature(df)  # oi feature
df = candle(df)  # candlestick feature
df = norm_ohlc(df)  # normalized ohlc

# hurst
# 2Q/3Q/4Q
# df['hurst_120'] = df['Close'].rolling(120).apply(lambda x: hurst(x)).round(1)

df['log_rtn'] = np.log(df['Close']).diff(1).round(5)
# kalman filter
df['kalman_log_rtn1'] = np.log(kalman(df.Close)).diff(1).round(5)
df = df[1:]  # df[0] has nan
# wiener filter
# df['wiener_log_rtn'] = wiener(df['log_rtn'].values).round(5)

df = df[df[df.Volume == 0].index.values[-1]:].reset_index(drop=True)
df = df.set_index(df['Date'])['1998/09':]
# settlement
df = settlement_cal(df)
df['until_expiration'] = df['until_expiration'].apply(lambda x: x / 45).round(2)  # minmax scale, max=1.5 month
# df['kalman_log_rtn2'] = kalman(df.log_rtn).round(5)
# df.rename(columns={"Date": "Timestamp"})
df.to_csv("data_simple_v2.csv", index=False)
