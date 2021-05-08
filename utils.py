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


class Futures:
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
    settle = pd.read_csv('data/txf_settlement.csv', delimiter=',')
    settle['txf_settlement'] = pd.to_datetime(settle['txf_settlement'])
    today = datetime.date.today()
    if (settle['txf_settlement'].tail(1) < today).values[0]:
        settle['txf_settlement'] = settle['txf_settlement'].apply(lambda x: datetime.datetime.strftime(x, "%Y/%m/%d"))
        url = f"https://www.yuantafutures.com.tw/api/TradeCal01?format=json&select01=台灣期交所TAIFEX&select02=期貨&y={today.year}&o=TE"
        r = requests.get(url)
        tmp_json = json.loads(r.text)
        for tmp in tmp_json['result01']:
            # utc+0
            d = tmp['d'].split('T')[0]
            d = datetime.datetime.strptime(d, "%Y-%m-%d")
            # utc+8
            d += datetime.timedelta(days=1)
            d = d.strftime("%Y/%m/%d")
            if d not in settle['txf_settlement'].tolist():
                settle = settle.append({'txf_settlement': d}, ignore_index=True)

        settle.to_csv('data/txf_settlement.csv', index=False)
    return settle


def hurst(ts=None, lags=None):
    """
    Returns the Hurst Exponent of the time series
    hurst < 0.5: mean revert
    hurst = 0.5: random
    hurst > 0.5: trend
    :param:
        ts[,]   a time-series, with 100+ elements
    :return:
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
        return (settlement[(x + pd.Timedelta(15, unit="d")).strftime('%Y-%m')].index - x)[0]


def settlement_cal(d):
    d['until_expiration'] = d.Date.apply(lambda x: expiration_cal(x))
    d['until_expiration'] = d['until_expiration'].apply(lambda x: x.days)
    return d


def vol_feature(d):
    """
    add 2 volume feature to df
    vol_deg_change: degree change in minmax scale, origin value belong to [-90, 90]
    vol_percentile: volume percentile
    :param d: df
    """
    d['vol_deg_change'] = d['Volume'].diff(1).apply(lambda x: (np.arctan2(x, 100) / np.pi) + 0.5)
    d['vol_percentile'] = d['Volume'].rolling(len(df), min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).values[-1], raw=False)
    return d


def oi_feature(d):
    """
    same as vol_feature, for futures
    :param d: df
    """
    d['oi_deg_change'] = d['OI'].diff(1).apply(lambda x: (np.arctan2(x, 100) / np.pi) + 0.5)
    d['oi_percentile'] = d['OI'].rolling(len(df), min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).values[-1], raw=False)
    return d


def candle(d):
    """
    add candlestick feature to df
    body: abs(o - c)
    upper_shadow: h - max(o, c)
    lower_shadow: min(o, c) - l
    divide by range in order to make all candlestick in same scale
    :param d: df
    """
    d['range'] = d['High'] - d['Low']
    d['body'] = np.abs(d['Open'] - d['Close']) / d['range']
    d['upper_shadow'] = (d['High'] - d[['Open', 'Close']].max(axis=1)) / d['range']
    d['lower_shadow'] = (d[['Open', 'Close']].min(axis=1) - d['Low']) / d['range']
    return d


def norm_ohlc(d):
    """
    normalize ohlc
    :param d: df
    """
    tmp_o = np.log(d['Open'])
    d['norm_o'] = tmp_o - np.log(d['Close'].shift(1))
    d['norm_h'] = np.log(d['High']) - tmp_o
    d['norm_l'] = np.log(d['Low']) - tmp_o
    d['norm_c'] = np.log(d['Close']) - tmp_o
    return d


def basis(d):
    """
    add futures basis feature to df
    :param d: df
    :return: basis feature
    """
    d = d.set_index(d['Date'])
    taiex = pd.read_csv('data/clean/#001.csv')
    taiex['Date'] = pd.to_datetime(taiex['Date'])
    taiex = taiex.set_index(taiex['Date'])
    d['basis'] = (taiex['Close'] - d['Close']).fillna(method='ffill')
    return d['basis'].reset_index(drop=True)


if __name__ == "__main__":
    settlement = pd.read_csv("data/txf_settlement.csv")
    settlement['txf_settlement'] = pd.to_datetime(settlement['txf_settlement'])
    settlement = settlement.set_index(settlement['txf_settlement'])
    # dq2()
    df = pd.read_csv("data/clean/WTX&.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['basis'] = basis(df)
    df = vol_feature(df)  # volume feature
    df = oi_feature(df)  # oi feature
    df = candle(df)  # candlestick feature
    df = norm_ohlc(df)  # normalized ohlc

    # hurst
    # print(hurst(df.Close))
    # 2Q/3Q/4Q
    df['hurst_120'] = df['Close'].rolling(120).apply(lambda x: hurst(x))
    # df['hurst_180'] = df['Close'].rolling(180).apply(lambda x: hurst(x))
    # df['hurst_240'] = df['Close'].rolling(240).apply(lambda x: hurst(x))

    df['log_rtn'] = np.log(df['Close']).diff(1)
    # kalman filter
    df['kalman_log_rtn1'] = np.log(kalman(df.Close)).diff(1)
    df = df[1:]  # df[0] has nan
    df['kalman_log_rtn2'] = kalman(df.log_rtn.reset_index(drop=True))
    # wiener filter
    df['wiener_log_rtn'] = wiener(df['log_rtn'].values)
    """
    # filter compare
    df.plot(x='Date', y=['log_rtn', 'wiener_log_rtn'], kind='kde')
    df.plot(x='Date', y='Volume')
    """
    df = df[df[df.Volume == 0].index.values[-1]:].reset_index(drop=True)
    df = df.drop(columns=['range']).set_index(df['Date'])['1998/09':]
    # settlement
    df = settlement_cal(df)
    df['until_expiration'] = df['until_expiration'].apply(lambda x: x / 45)  # minmax scale, max=1.5 month
