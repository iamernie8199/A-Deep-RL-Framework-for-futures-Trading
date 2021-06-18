import os

import gym
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Box

from utils import year_frac


class TradingEnvLong(gym.Env):
    """
    A futures trading environment for OpenAI gym
    only long

    Attributes
    ----------
        df: DataFrame
            input data
        init_equity: float
            init equity for backtest
        max_position: int
            maximum number of shares to trade
        log: bool
            whether output the log
        min_movement_point: float
            the min movement point of the futures contract
            set 1 if not futures contract
        big_point_value: float
            the big point value of the futures contract
            set 1 if not futures contract
        cost: float
            ticks number as the slippage/cost per trade
            set 0 if not futures contract
        cost_pct: float
            cost for stock/forex
    """

    def __init__(self, df, cost=6, init_equity=1000000, max_position=1, log=False, r='rtn_on_mdd',
                 min_movement_point=1,
                 big_point_value=200,
                 cost_pct=0.001):
        self.df = df
        self.min_movement_point = min_movement_point
        self.big_point_value = big_point_value
        self.current_idx = 0
        self.episode = 0
        self.max_position = max_position
        self.init_equity = init_equity
        self.done = False
        self.log = log
        # buy & hold return
        self.bnh = self.init_equity
        # cost = (cost//2) tick/per trade considering slippage
        self.cost = self.min_movement_point * self.big_point_value * (cost // 2)
        # cost for stock or forex
        self.cost_pct = cost_pct
        try:
            self.prices = self.df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']]
        except:
            self.prices = self.df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        self.prices['signal_E'] = np.nan
        self.prices['signal_X'] = np.nan

        # action spaces: 0: 'hold'/1: 'long'/2: 'sell'
        self.action_space = Discrete(3)
        # observation space: [-1, 1]
        try:
            self.observation = self.df.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        except:
            self.observation = self.df.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        # feature + position
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.observation.shape[1] + 1,))

        # initialize reward
        self.r = r
        self.reward = 0
        self.position = 0
        self.points = 0
        self.equity = self.init_equity
        self.equity_tmp = self.init_equity
        self.equity_l = self.init_equity
        self.equity_h = self.init_equity
        self._entryprice = None
        self.drawdown = 0
        self.winrate = 0
        self.avg_win = 0
        self.avg_loss = -0
        self.rtn_on_mdd = 0
        self.profit_factor = 0
        self.ratio_winloss = 0
        self.avg_trade = 0
        self.out_perform = 0
        self.sharpe = 0
        self.trade_num = 0
        self.gross_profit = 0
        self.gross_loss = 0
        self.cagr = 0

        self.actions_memory = pd.DataFrame(columns=['date', 'action'])
        """
        equity_tmp: end of the day equity
        equity_l: lowest equity of the day
        equity_h: highest  equity of the day
        """
        self.equity_col = ['date', 'equity_tmp', 'equity_l', 'equity_h', 'equity_dd', 'BnH']
        self.tradeslist_col = ['date', 'action', 'price', 'contracts', 'profit', 'drawdown']
        self.equity_memory = pd.DataFrame(columns=self.equity_col)
        self.trades_list = pd.DataFrame(columns=self.tradeslist_col)

    def reset(self):
        self.episode += 1
        self.current_idx = 0
        self.done = False
        self.bnh = self.init_equity
        self.equity = self.init_equity
        self.equity_tmp = self.init_equity
        self.equity_l = self.init_equity
        self.equity_h = self.init_equity
        self.reward = 0
        self.position = 0
        self.points = 0
        self.out_perform = 0
        self.drawdown = 0
        self.winrate = 0
        self.profit_factor = 0
        self.avg_win = 0
        self.avg_loss = -0
        self.ratio_winloss = 0
        self.rtn_on_mdd = 0
        self.avg_trade = 0
        self.sharpe = 0
        self.trade_num = 0
        self.gross_profit = 0
        self.gross_loss = 0
        self.cagr = 0
        self._entryprice = None

        self.actions_memory = pd.DataFrame(columns=['date', 'action'])
        self.equity_memory = pd.DataFrame(columns=self.equity_col)
        self.trades_list = pd.DataFrame(columns=self.tradeslist_col)

        self.prices['signal_E'] = np.nan
        self.prices['signal_X'] = np.nan

        return np.append(self.observation.iloc[0].values, 0)

    def commission_cost(self, contracts, price):
        if self.cost == 0:
            self.equity -= price * self.cost_pct * contracts
        else:
            self.equity -= self.cost * contracts

    def _long(self, price, contracts):
        if self.position < self.max_position:
            self.position += contracts
            self.commission_cost(contracts, price)
            self.points += price * contracts
            self._entryprice = price

    def _sell(self, price, contracts):
        if self.position > 0:
            self.position -= contracts
            self.points -= price * contracts
            self.commission_cost(contracts, price)
            # Close out of long position
            if self.position == 0:
                profit = -(self.points * self.big_point_value)
                self.equity += profit
                self.points = 0
                self._entryprice = None
                return profit

    def render(self, mode='human'):
        if self.current_idx >= (len(self.df) - 10):
            plt.close()
            data = self.prices[max(0, self.current_idx - 200):self.current_idx]
            data = data.set_index(pd.to_datetime(data['Date']))
            data = data.drop(columns=['Date'])
            mc = mpf.make_marketcolors(up='#fe3032', down='#00b060', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-.', gridaxis='horizontal', y_on_right=True)
            apds = [mpf.make_addplot(data['signal_E'], type='scatter', markersize=10, marker='^', color='r'),
                    mpf.make_addplot(data['signal_X'], type='scatter', markersize=10, marker='v', color='g')]
            try:
                mpf.plot(data, type='candle', volume=True, panel_ratios=(5, 1), style=s, figratio=(12, 7),
                         ylabel='', ylabel_lower='', tight_layout=True, datetime_format="'%y-%b-%d", xrotation=0,
                         addplot=apds, returnfig=True)
            except:
                mpf.plot(data, type='candle', volume=True, panel_ratios=(5, 1), style=s, figratio=(12, 7),
                         ylabel='', ylabel_lower='', tight_layout=True, datetime_format="'%y-%b-%d", xrotation=0,
                         returnfig=True)
            finally:
                plt.savefig('render.png', bbox_inches='tight')
                plt.show(block=False)
                plt.pause(0.0001)

    def step(self, actions):
        self.done = (self.current_idx >= len(self.df.index.unique()) - 2)  # or (self.equity <= 0)

        # calculate buy and hold return
        if self.current_idx == 0:
            self.bnh += self.big_point_value * (
                    self.prices['Close'].iloc[self.current_idx] - self.prices['Open'].iloc[self.current_idx])
        else:
            self.bnh += self.big_point_value * (
                    self.prices['Close'].iloc[self.current_idx] - self.prices['Close'].iloc[self.current_idx - 1])

        if self.position > 0:
            self.equity_tmp = self.equity + (
                    self.prices['Close'].iloc[self.current_idx] - self._entryprice) * self.big_point_value
            self.equity_l = self.equity + (
                    self.prices['Low'].iloc[self.current_idx] - self._entryprice) * self.big_point_value
            self.equity_h = self.equity + (
                    self.prices['High'].iloc[self.current_idx] - self._entryprice) * self.big_point_value
        else:
            self.equity_tmp = self.equity
            self.equity_l = self.equity
            self.equity_h = self.equity

        self.equity_memory = self.equity_memory.append({
            'date': self.prices['Date'].iloc[self.current_idx],
            'equity_tmp': self.equity_tmp,
            'equity_l': self.equity_l,
            'equity_h': self.equity_h,
            'equity_dd': max(self.equity_memory['equity_h'].max(), self.equity_h) - self.equity_l,
            'BnH': self.bnh
        }, ignore_index=True)

        if self.done:
            print(f"Net Profit:\t{self.equity_tmp - self.init_equity}")
            print(f"Gross Profit:\t{self.gross_profit}")
            print(f"Gross Loss:\t{self.gross_loss}")
            print(f"Return on Initial Capital:\t{(self.equity_tmp - self.init_equity) / self.init_equity}")
            print(f"MDD:\t{self.drawdown}")
            print(f"Return on MDD:\t{self.rtn_on_mdd}")
            print(f"Profit Factor:\t{self.profit_factor}")
            print(f"Annual Rate of Return:\t{round(self.cagr * 100, 2)}%")
            print(f"Total # of Trades:\t{self.trade_num}")
            print(f"% Profitable:\t{round(self.winrate * 100, 2)}%")
            print(f"Sharpe:\t{self.sharpe}")
            print(f"Avg Trade:\t{self.avg_trade}")
            print(f"Ratio Avg Win / Loss:\t{round(self.ratio_winloss, 4)}")
            print("=============================================")
            self._make_plot()
            if self.log:
                self._make_log()
                # for thesis
                out = [self.equity_tmp - self.init_equity, self.rtn_on_mdd, self.profit_factor,
                       round(self.cagr * 100, 2),
                       self.trade_num, round(self.winrate * 100, 2)]
                return np.append(self.observation.iloc[self.current_idx].values,
                                 self.position), self.reward, self.done, out
            else:
                return np.append(self.observation.iloc[self.current_idx].values,
                                 self.position), self.reward, self.done, {}
        else:
            try:
                # settlement
                if self.df['until_expiration'].iloc[self.current_idx] == 0 and self.position > 0:
                    c = self.prices['Close'].iloc[self.current_idx]
                    p = self._sell(c, self.position)
                    self.prices['signal_X'].iloc[self.current_idx] = self.prices['High'].iloc[self.current_idx] * 1.01
                    self.trades_list = self.trades_list.append(
                        {'date': self.prices['Date'].iloc[self.current_idx],
                         'action': 'settlement',
                         'price': c,
                         'contracts': 1,
                         'profit': p - 2 * self.cost,
                         'drawdown': self.equity_memory['equity_h'].max() - (
                                 self.init_equity + self.trades_list['profit'].sum() + p)
                         }, ignore_index=True)
            except:
                pass

            if actions == 1:
                if self.position < self.max_position:
                    action_str = 'buy_next'
                    o = self.prices['Open'].iloc[self.current_idx + 1]
                    self._long(o, 1)
                    self.trades_list = self.trades_list.append({
                        'date': self.prices['Date'].iloc[self.current_idx + 1],
                        'action': 'buy',
                        'price': o,
                        'contracts': 1,
                        'profit': np.nan,
                        'drawdown': np.nan
                    }, ignore_index=True)
                else:
                    action_str = 'hold'
            elif actions == 2 and self.position > 0:
                action_str = 'sell_next'
                o = self.prices['Open'].iloc[self.current_idx + 1]
                p = self._sell(o, 1)
                self.trades_list = self.trades_list.append({
                    'date': self.prices['Date'].iloc[self.current_idx + 1],
                    'action': 'sell',
                    'price': o,
                    'profit': p - 2 * self.cost,
                    'contracts': 1,
                    'drawdown': self.equity_memory['equity_h'].max() - (
                            self.init_equity + self.trades_list['profit'].sum() + p)
                }, ignore_index=True)
            else:
                action_str = 'hold'

            # for render
            if action_str == 'buy_next':
                self.prices['signal_E'].iloc[self.current_idx + 1] = self.prices['Low'].iloc[
                                                                         self.current_idx + 1] * 0.99
            elif action_str == 'sell_next':
                self.prices['signal_X'].iloc[self.current_idx + 1] = self.prices['High'].iloc[
                                                                         self.current_idx + 1] * 1.01

            # equity new high
            dd = self.equity_memory['equity_h'].max() - self.equity_l
            # MDD new high
            if dd > self.drawdown:
                self.drawdown = dd

            self.actions_memory = self.actions_memory.append(
                {'date': self.prices['Date'].iloc[self.current_idx],
                 'action': action_str}, ignore_index=True)

            # reward
            tradelist = self.trades_list[self.trades_list['profit'].notna()]
            win = tradelist.profit >= 0
            loss = tradelist.profit < 0
            self.rtn_on_mdd = np.round((self.equity - self.init_equity) / max(self.drawdown, 1), 2)
            self.winrate = round(len(tradelist[win]) / len(tradelist), 4) if len(tradelist) else 0
            self.gross_profit = tradelist[win]['profit'].sum()
            self.gross_loss = tradelist[loss]['profit'].sum()
            self.profit_factor = round(self.gross_profit / max(-self.gross_loss, 1), 2)
            if self.gross_profit < -self.gross_loss:
                self.profit_factor = -self.profit_factor
            self.avg_win = round(self.gross_profit / max(len(tradelist[win]), 1), 0)
            self.avg_loss = round(self.gross_loss / max(len(tradelist[loss]), 1), 0)
            self.ratio_winloss = self.avg_win / -self.avg_loss if self.avg_loss else self.avg_win
            self.avg_trade = round(tradelist['profit'].mean(), 0)
            self.trade_num = len(tradelist)
            if self.current_idx > 0:
                # year_num = year_frac(self.equity_memory['date'].iloc[0], self.prices['Date'].iloc[self.current_idx])
                year_num = year_frac(self.equity_memory['date'].iloc[0],
                                     self.equity_memory[self.equity_memory.equity_tmp > 0]['date'].iloc[-1])
                if self.equity > 0:
                    self.cagr = (self.equity_tmp / self.init_equity) ** (1 / year_num) - 1
                else:
                    self.cagr = (1 / self.init_equity) ** (1 / year_num) - 1
            lattest_profit = self.equity_memory['equity_tmp'].diff(1).values[-1]
            daily_rtn = self.equity_memory['equity_tmp'].pct_change(1)
            # daily_rtn = daily_rtn[(daily_rtn != -np.inf) & (daily_rtn != np.inf)]
            daily_rtn = daily_rtn.replace([np.inf, -np.inf], [1, -1])
            if daily_rtn.std():
                self.sharpe = round((252 ** 0.5) * daily_rtn.mean() / daily_rtn.std(), 2)
            else:
                self.sharpe = round((252 ** 0.5) * daily_rtn.mean(), 2)
            self.out_perform = self.equity_tmp - self.bnh

            if self.drawdown:
                self.reward = self.rtn_on_mdd
                # self.reward = self.sharpe - self.reward
            elif self.equity == self.init_equity:
                self.reward = -100

            self.current_idx += 1

            return np.append(self.observation.iloc[self.current_idx].values,
                             self.position), self.reward, self.done, {}

    def _make_plot(self):
        self.equity_memory.set_index(['date'])['equity_tmp'].plot.line(legend=True)
        self.equity_memory.set_index(['date'])['BnH'].plot.line(legend=True, color='r')
        if not os.path.exists("./results"):
            os.makedirs("./results")
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def _make_log(self):
        if not os.path.exists("./results_pic"):
            os.makedirs("./results_pic")
        self.equity_memory.to_csv(f'results_pic/equity_{self.episode}.csv', index=False)
        self.trades_list.to_csv(f'results_pic/trades_list_{self.episode}.csv', index=False)
        self.actions_memory.to_csv(f'results_pic/actions_{self.episode}.csv', index=False)
