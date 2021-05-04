import datetime
import json
import os
from fractions import Fraction
from glob import glob

import numpy as np
import pandas as pd
import requests
from pykalman import KalmanFilter
from scipy.signal.signaltools import wiener


class Futures():
    def __init__(self,
                 min_movement_point=1,
                 big_point_value=200):
        self.min_movement_point = min_movement_point
        self.big_point_value = big_point_value


def settlement_day():
    """
    更新結算日列表
    :return:
    """
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
    return df


def hurst(ts=None, lags=None):
    """
    USAGE:
        Returns the Hurst Exponent of the time series
        hurst < 0.5: mean revert
        hurst = 0.5: random
        hurst > 0.5: trend
    PARAMETERS:
        ts[,]   a time-series, with 100+ elements
    RETURNS:
        float - a Hurst Exponent approximation
    """
    if ts is None:
        ts = [None, ]
    if lags is None:
        lags = [2, 80]

    if isinstance(ts, pd.Series):
        ts = ts.dropna().to_list()

    too_short_list = lags[1] + 1 - len(ts)
    if 0 < too_short_list:  # IF NOT:
        # 序列長度不足則以第一筆補滿
        ts = too_short_list * ts[:1] + ts  # PRE-PEND SUFFICIENT NUMBER of [ts[0],]-as-list REPLICAS TO THE LIST-HEAD
    # Create the range of lag values
    lags = range(lags[0], lags[1])
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Return the Hurst exponent from the polyfit output ( a linear fit to estimate the Hurst Exponent
    return 2.0 * np.polyfit(np.log(lags), np.log(tau), 1)[0]


def frac2float(x):
    """
    covert something like '387 2/4' to float
    :param x: fractions(str)
    :return: float
    """
    if len(x.split(' ')) == 2:
        return float(x.split(' ')[0]) + float(Fraction(x.split(' ')[-1]))
    elif len(x.split(' ')) == 3:
        return float(x.split(' ')[0]) + float(Fraction(x.split(' ')[-2])) * float(Fraction(x.split(' ')[-1]))
    else:
        return float(x.split(' ')[0])


def dq2():
    file = glob('data/raw/*.csv')
    for f in file:
        # UTF-8會亂碼
        tmp = pd.read_csv(f, skiprows=[0], encoding='big5')
        tmp = tmp.dropna(axis='columns')
        # 調整日期格式
        tmp['日期'] = tmp['日期'].apply(lambda x: str(x // 10000) + '/' + str(x % 10000 // 100) + '/' + str(x % 100))
        # 與Yahoo統一欄位名稱
        tmp = tmp.rename(columns={"日期": "Date",
                                  "開盤價": "Open",
                                  "最高價": "High",
                                  "最低價": "Low",
                                  "收盤價": "Close",
                                  "成交量": "Volume",
                                  "未平倉量": "OI",
                                  "上漲家數": "advancing",
                                  "下跌家數": "declining",
                                  "成交金額": "trade_value"})
        filename = f.split('\\')[-1].split('(')[-2].replace(')', '.csv')
        # 捨棄全0欄位
        tmp = tmp.loc[:, (tmp != 0).any(axis=0)]

        try:
            tmp = tmp.astype({'Open': 'float',
                              'High': 'float',
                              'Low': 'float',
                              'Close': 'float'})
        except:
            tmp['Open'] = tmp['Open'].apply(lambda x: frac2float(x))
            tmp['High'] = tmp['High'].apply(lambda x: frac2float(x))
            tmp['Low'] = tmp['Low'].apply(lambda x: frac2float(x))
            tmp['Close'] = tmp['Close'].apply(lambda x: frac2float(x))
        if not os.path.exists('data/clean/'):
            os.makedirs('data/clean/')
        tmp.to_csv(f"data/clean/{filename}", index=False)


def kalman(ts=None):
    if ts is None:
        ts = [None, ]
    if ts[0] is None:
        return
    kf = KalmanFilter(initial_state_mean=0,
                      initial_state_covariance=1,
                      transition_matrices=[1],
                      observation_matrices=[1],
                      observation_covariance=1,
                      transition_covariance=.01)
    state_means, _ = kf.filter(ts)
    state_means = pd.Series(state_means.flatten(), index=ts.index)
    return state_means


def expiration_cal(x):
    remain = (settlement[x.strftime('%Y-%m')].index - x)[0]
    if remain >= pd.Timedelta("0 days"):
        return remain
    else:
        remain = (settlement[(x + pd.Timedelta(30, unit="d")).strftime('%Y-%m')].index - x)[0]
        return remain


def settlement_cal(d):
    d['until_expiration'] = d.Date.apply(lambda x: expiration_cal(x))
    d['until_expiration'] = d['until_expiration'].apply(lambda x: x.days)
    return d


if __name__ == "__main__":
    settlement = pd.read_csv("data/txf_settlement.csv")
    settlement['txf_settlement'] = pd.to_datetime(settlement['txf_settlement'])
    settlement = settlement.set_index(settlement['txf_settlement'])
    # dq2()
    df = pd.read_csv("data/clean/WTX&.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    # volume degree change(minmax scale)
    # df['vol_deg_change'] = df['Volume'].diff(1).apply(lambda x: np.arctan2(x, 100) * 180 / np.pi)
    df['Volume'].diff(1).apply(lambda x: (np.arctan2(x, 100) / np.pi) + 0.5)

    # candlestick feature
    df['range'] = df['High'] - df['Low']
    df['body'] = np.abs(df['Open'] - df['Close']) / df['range']
    df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['range']
    df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['range']

    df['log_rtn'] = np.log(df['Close']) - np.log(df['Close'].shift(1))

    # normalized ohlc
    df['norm_o'] = np.log(df['Open']) - np.log(df['Close'].shift(1))
    df['norm_h'] = np.log(df['High']) - np.log(df['Open'])
    df['norm_l'] = np.log(df['Low']) - np.log(df['Open'])
    df['norm_c'] = np.log(df['Close']) - np.log(df['Open'])
    # kf
    df['kalman'] = kalman(df.Close)
    df['kalman_log_rtn'] = np.log(df['kalman']) - np.log(df['kalman'].shift(1))
    # hurst
    # print(hurst(df.Close))
    # 2Q/3Q/4Q
    df['hurst_120'] = df['Close'].rolling(120).apply(lambda x: hurst(x))
    # df['hurst_180'] = df['Close'].rolling(180).apply(lambda x: hurst(x))
    # df['hurst_240'] = df['Close'].rolling(240).apply(lambda x: hurst(x))
    # wf
    df = df[1:]
    df['wiener_log_rtn'] = wiener(df['log_rtn'].values)
    # filter compare
    df.plot(x='Date', y=['log_rtn', 'wiener_log_rtn'], kind='kde')
    df.plot(x='Date', y='Volume')
    df = df[df[df.Volume == 0].index.values[-1]:-1].reset_index(drop=True)
    df = df.drop(columns=['range']).set_index(df['Date'])['1998/09':]
    df = settlement_cal(df)
    # df['until_expiration'] = df.Date.apply(lambda x: expiration_cal(x))
