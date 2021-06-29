import datetime
import json
import os
# Disable the warnings
import warnings
from datetime import datetime
from fractions import Fraction
from glob import glob

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from pykalman import KalmanFilter

warnings.filterwarnings('ignore')


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
    d['vol_deg_change'] = d['Volume'].diff(1).apply(lambda x: (np.arctan2(x, 100) / np.pi) + 0.5).round(4)
    d['vol_percentile'] = d['Volume'].rolling(len(d), min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).values[-1], raw=False).round(3)
    return d


def oi_feature(d):
    """
    same as vol_feature, for futures
    :param d: df
    """
    d['oi_deg_change'] = d['OI'].diff(1).apply(lambda x: (np.arctan2(x, 100) / np.pi) + 0.5).round(4)
    d['oi_percentile'] = d['OI'].rolling(len(d), min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).values[-1], raw=False).round(3)
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
    d['body'] = (np.abs(d['Open'] - d['Close']) / d['range']).fillna(1).round(2)
    d['upper_shadow'] = ((d['High'] - d[['Open', 'Close']].max(axis=1)) / d['range']).fillna(0).round(2)
    d['lower_shadow'] = ((d[['Open', 'Close']].min(axis=1) - d['Low']) / d['range']).fillna(0).round(2)
    return d.drop(columns=['range'])


def norm_ohlc(d):
    """
    normalize ohlc
    :param d: df
    """
    tmp_o = np.log(d['Open'])
    d['norm_o'] = (tmp_o - np.log(d['Close'].shift(1))).round(4)
    d['norm_h'] = (np.log(d['High']) - tmp_o).round(4)
    d['norm_l'] = (np.log(d['Low']) - tmp_o).round(4)
    d['norm_c'] = (np.log(d['Close']) - tmp_o).round(4)
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
    tmp = taiex['Close'].reindex(d['Date']).fillna(method='ffill')
    d['basis'] = (tmp - d['Close'])
    # d['basis'] = d['basis'] / tmp  # scale to [-1,1]
    d['basis'] = d['basis'] / 1000
    return d['basis'].reset_index(drop=True).round(5)


def year_frac(start, end):
    """
    a year fraction between two dates (i.e. 1.53 years).
    Approximation using the average number of seconds in a year.
    Parameters
    ----------
    start(datetime): start date
    end(datetime): end date

    Returns: year fraction approximation between two dates
    -------
    """
    if start > end:
        raise ValueError("start cannot be larger than end")

    # obviously not perfect but good enough
    return (end - start).total_seconds() / (31557600)


def random_rollout(env, bnh=False):
    state = env.reset()
    dones = False
    info = []
    # Keep looping as long as the simulation has not finished.
    while not dones:
        # Choose a random action (0, 1, 2).
        if bnh:
            act = 1
        else:
            act = np.random.choice(3, 1)[0]
        # Take the action in the environment.
        state, reward, dones, i = env.step(act)
    info.append(i)
    return info


def result_plt(title='', init_equity=1000000, path='results_pic', time1='2010-01-01', time2="2020-01-01"):
    equitylist = glob(f'{path}/equity_*.csv')
    tmp_df = pd.read_csv(equitylist[0])[['date', 'equity_tmp']]
    for p in equitylist[1:]:
        tmp_df2 = pd.read_csv(p)[['date', 'equity_tmp']]
        tmp_df = tmp_df.merge(tmp_df2, how='left', on='date')
    tmp_df = tmp_df.set_index(pd.to_datetime(tmp_df['date'])).drop(columns='date')
    tmp_df.columns = range(len(tmp_df.columns))
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=tmp_df, legend=False, linewidth=1.25)
    ax.set_title(title)
    ax.set_ylabel('Equity')
    ax.axhline(init_equity, ls='-.', c='grey')
    if time1 is not None:
        ax.axvline(x=datetime.strptime(time1, "%Y-%m-%d"), ls=':', c='black')
    if time2 is not None:
        ax.axvline(x=datetime.strptime(time2, "%Y-%m-%d"), ls=':', c='black')
    plt.savefig(f'{path}/{title}.png', bbox_inches='tight')


def split_result(time1="2010-01-01", time2="2020-01-01", path='results_pic'):
    equitylist = glob(f"{path}/equity_*.csv")
    tradelist = glob(f"{path}/trades_list_*.csv")

    def cagr_cal(df):
        if np.sign(df.equity_tmp.values[-1]) == np.sign(df.equity_tmp.values[0]) and df.equity_tmp.values[0] > 0:
            cagr = df.equity_tmp.values[-1] / df.equity_tmp.values[0]
        elif df.equity_tmp.values[-1] < 0 and df.equity_tmp.values[0] > 0:
            cagr = df[df.equity_tmp > 0].equity_tmp.values[-1] / df.equity_tmp.values[0]
            cagr = cagr ** (1 / year_frac(df.index[0], df[df.equity_tmp > 0].index[-1])) - 1
            return cagr
        else:
            cagr = df.equity_tmp.values[-1] - df.equity_tmp.values[0] + np.absolute(df.equity_tmp.values[0])
            cagr /= np.absolute(df.equity_tmp.values[0])
        cagr = cagr ** (1 / year_frac(df.index[0], df.index[-1])) - 1
        return cagr

    def net_cal(df):
        return df.equity_tmp.values[-1] - df.equity_tmp.values[0]

    def mdd_cal(df):
        df['high'] = df['equity_tmp'].rolling(len(df), min_periods=1).max()
        df['dd'] = df['high'] - df['equity_tmp']
        return df['dd'].max()

    result = []
    for e in equitylist:
        e_tmp_df = pd.read_csv(e)[['date', 'equity_tmp']]
        e_tmp_df = e_tmp_df.set_index(pd.to_datetime(e_tmp_df['date'])).drop(columns='date')
        tmp = []
        e_tmp_df1 = e_tmp_df.loc[:time1]
        e_tmp_df2 = e_tmp_df.loc[time1:time2]
        e_tmp_df3 = e_tmp_df.loc[time2:]

        net1 = net_cal(e_tmp_df1)
        cagr1 = round(cagr_cal(e_tmp_df1), 4)
        rtn_mdd1 = net1 / mdd_cal(e_tmp_df1)
        tmp.extend([net1, rtn_mdd1, cagr1])

        net2 = net_cal(e_tmp_df2)
        cagr2 = round(cagr_cal(e_tmp_df2), 4)
        rtn_mdd2 = net2 / mdd_cal(e_tmp_df2)
        tmp.extend([net2, rtn_mdd2, cagr2])

        net3 = net_cal(e_tmp_df3)
        cagr3 = round(cagr_cal(e_tmp_df3), 4)
        rtn_mdd3 = net3 / mdd_cal(e_tmp_df3)
        tmp.extend([net3, rtn_mdd3, cagr3])
        result.append(tmp)

    result_df1 = pd.DataFrame(data=result,
                              columns=['t1_net', 't1_rtn_mdd', 't1_cagr',
                                       't2_net', 't2_rtn_mdd', 't2_cagr',
                                       't3_net', 't3_rtn_mdd', 't3_cagr'])

    def pf_cal(df):
        win = df.profit > 0
        loss = df.profit < 0
        pf = float(df[win].sum() / -df[loss].sum())
        winrate = len(df[win]) / len(df)
        return pf, winrate

    result = []
    for t in tradelist:
        t_tmp_df = pd.read_csv(t)[['date', 'profit']]
        t_tmp_df = t_tmp_df[t_tmp_df['profit'].notna()]
        t_tmp_df = t_tmp_df.set_index(pd.to_datetime(t_tmp_df['date'])).drop(columns='date')
        t_tmp_df1 = t_tmp_df[:time1]
        t_tmp_df2 = t_tmp_df[time1:time2]
        t_tmp_df3 = t_tmp_df[time2:]
        tmp = []
        num1 = len(t_tmp_df1)
        num2 = len(t_tmp_df2)
        num3 = len(t_tmp_df3)
        pf1, winrate1 = pf_cal(t_tmp_df1)
        pf2, winrate2 = pf_cal(t_tmp_df2)
        pf3, winrate3 = pf_cal(t_tmp_df3)
        tmp.extend([pf1, num1, winrate1])
        tmp.extend([pf2, num2, winrate2])
        tmp.extend([pf3, num3, winrate3])
        result.append(tmp)

    result_df2 = pd.DataFrame(data=result,
                              columns=['t1_pf', 't1_num', 't1_rate',
                                       't2_pf', 't2_num', 't2_rate',
                                       't3_pf', 't3_num', 't3_rate'])
    result_df = pd.concat([result_df1, result_df2], axis=1)
    columns = ['t1_net', 't1_rtn_mdd', 't1_pf', 't1_cagr', 't1_num', 't1_rate',
               't2_net', 't2_rtn_mdd', 't2_pf', 't2_cagr', 't2_num', 't2_rate',
               't3_net', 't3_rtn_mdd', 't3_pf', 't3_cagr', 't3_num', 't3_rate']
    result_df = result_df[columns]
    return result_df


def price():
    data = pd.read_csv('data/clean/WTX&.csv')
    data = data.set_index(pd.to_datetime(data['Date']))
    data = data.drop(columns=['Date', 'OI'])
    mc = mpf.make_marketcolors(up='#fe3032', down='#00b060', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-.', gridaxis='horizontal')
    mpf.plot(data.loc['2000-01-01':], type='candle', volume=True, panel_ratios=(5, 1), style=s, figratio=(12, 7),
             ylabel='', ylabel_lower='', tight_layout=True, datetime_format="'%y-%b-%d", xrotation=0,
             returnfig=True)
    plt.savefig('TAIEX Futures.png', bbox_inches='tight')
    plt.show()


def action_result():
    action_dict = {
        "hold": 0,
        "buy_next": 1,
        "sell_next": -1
    }
    action_list = glob('results_pic/actions_*.csv')
    tmp = pd.read_csv(action_list[0])
    tmp['action'] = tmp['action'].apply(lambda x: action_dict[x])
    for path in action_list[1:]:
        df2 = pd.read_csv(path)
        df2['action'] = df2['action'].apply(lambda x: action_dict[x])
        tmp = tmp.merge(df2, how='left', on='date')
    tmp = tmp.set_index(pd.to_datetime(tmp['date']))
    tmp.columns = range(len(tmp.columns))
    tmp['mean'] = tmp.mean(axis=1).round(2)
    # to multicharts
    tmp['open'] = 0
    tmp['high'] = tmp['mean'].apply(lambda x: x if x >= 0 else 0)
    tmp['low'] = tmp['mean'].apply(lambda x: x if x < 0 else 0)
    tmp['close'] = tmp['mean']
    tmp[['open', 'high', 'low', 'close']].to_csv('rainbow_btc2.csv')
    return tmp


if __name__ == "__main__":
    settlement = pd.read_csv("data/txf_settlement.csv")
    settlement['txf_settlement'] = pd.to_datetime(settlement['txf_settlement'])
    settlement = settlement.set_index(settlement['txf_settlement'])
    # dq2()
    df = pd.read_csv("data/clean/WTX&.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['basis'] = basis(df)  # basis feature
    df = vol_feature(df)  # volume feature
    df = oi_feature(df)  # oi feature
    df = candle(df)  # candlestick feature
    df = norm_ohlc(df)  # normalized ohlc

    # hurst
    # print(hurst(df.Close))
    # 2Q/3Q/4Q
    # df['hurst_120'] = df['Close'].rolling(120).apply(lambda x: hurst(x)).round(1)
    # df['hurst_180'] = df['Close'].rolling(180).apply(lambda x: hurst(x))
    # df['hurst_240'] = df['Close'].rolling(240).apply(lambda x: hurst(x))

    df['log_rtn'] = np.log(df['Close']).diff(1).round(5)
    # kalman filter
    df['kalman_log_rtn1'] = np.log(kalman(df.Close)).diff(1).round(5)
    df = df[1:]  # df[0] has nan
    # wiener filter
    # df['wiener_log_rtn'] = wiener(df['log_rtn'].values).round(5)
    """
    # filter compare
    df.plot(x='Date', y=['log_rtn', 'wiener_log_rtn'], kind='kde')
    df.plot(x='Date', y='Volume')
    """
    df = df[df[df.Volume == 0].index.values[-1]:].reset_index(drop=True)
    df = df.set_index(df['Date'])['1998/09':]
    # settlement
    df = settlement_cal(df)
    df['until_expiration'] = df['until_expiration'].apply(lambda x: x / 45).round(2)  # minmax scale, max=1.5 month
    # df['kalman_log_rtn2'] = kalman(df.log_rtn).round(5)
    # df.rename(columns={"Date": "Timestamp"})
    df.to_csv("data_simple2.csv", index=False)
