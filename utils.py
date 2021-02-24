import pandas as pd
import requests
import json
import datetime


df = pd.read_csv('data/txf_settlement.csv', delimiter=',')
df['txf_settlement'] = pd.to_datetime(df['txf_settlement'])
today = datetime.date.today()
if (df['txf_settlement'].tail(1) < today).values[0]:
    df['txf_settlement'] = df['txf_settlement'].apply(lambda x: datetime.datetime.strftime(x, "%Y/%m/%d"))
    r = requests.get(
        f"https://www.yuantafutures.com.tw/api/TradeCal01?format=json&select01=%E5%8F%B0%E7%81%A3%E6%9C%9F%E4%BA%A4%E6%89%80TAIFEX&select02=%E6%9C%9F%E8%B2%A8&y={today.year}&o=TE"
    )
    tmp_json = json.loads(r.text)
    for tmp in tmp_json['result01']:
        # utc+0
        d = tmp['d'].split('T')[0]
        d = datetime.datetime.strptime(d, "%Y-%m-%d")
        # utc+8
        d += datetime.timedelta(days=1)
        d = d.strftime("%Y/%m/%d")
        if d not in df['txf_settlement'].tolist():
            df = df.append({'txf_settlement': d}, ignore_index=True)

    df.to_csv('data/txf_settlement.csv', index=False)