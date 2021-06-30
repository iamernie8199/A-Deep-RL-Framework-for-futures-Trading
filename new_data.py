import pandas as pd
import psycopg2
import numpy as np

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
where contract='TX' and session='regular' and date>='2021-03-01'
order by date desc, contract_month asc
"""
df1 = pd.read_sql(sql, conn)
sql = """
select DISTINCT ON (date) 
date, open, high, low
from api_taifex_futures_price 
where contract='TX' and session='ah' and date>='2021-03-01'
order by date desc, contract_month asc
"""
df2 = pd.read_sql(sql, conn)
df = df1.merge(df2, how='left', on='date')
df['High'] = df[['high_x', 'high_y']].max(axis=1)
df['Low'] = df[['low_x', 'low_y']].min(axis=1)
df['Open'] = df[['open', 'o']].apply(lambda x: x.open if x.open == x.open else x.o, axis=1)
df = df[['date', 'Open', 'High', 'Low', 'close', 'volume', 'oi']]
df = df.rename(columns={"date": "Date",
                        "close": "Close",
                        "volume": "Volume",
                        "oi": "OI"})
df = df.sort_index(ascending=False)
