import os

import gym
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Box
from stable_baselines3.common.vec_env import DummyVecEnv


class TradingEnvLong(gym.Env):
    """
    A futures trading environment for OpenAI gym
    only long

    Attributes
    ----------
        df: DataFrame
            input data
        futures: futures class
            min_movement_point & big_point_value
        max_position: int
            maximum number of shares to trade
    """

    def __init__(self, df, futures, cost=6, init_equity=100000, max_position=1, log=False):
        self.df = df
        self.futures = futures
        self.current_idx = 0
        self.episode = 0
        self.max_position = max_position
        self.init_equity = init_equity
        self.done = False
        self.log = log
        # buy & hold return
        self.bnh = self.init_equity
        # cost = 3 tick/per trade considering slippage
        self.cost = self.futures.min_movement_point * self.futures.big_point_value * (cost // 2)

        self.prices = self.df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        self.prices['signal_E'] = np.nan
        self.prices['signal_X'] = np.nan

        # action spaces: 0: 'hold'/1: 'long'/2: 'sell'
        self.action_space = Discrete(3)
        # observation space: [-1, 1]
        self.observation = self.df.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        # feature + position
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.observation.shape[1] + 1,))

        # initialize reward
        self.reward = 0
        self.position = 0
        self.points = 0
        self.equity = self.init_equity
        self.equity_l = self.init_equity
        self.equity_h = self.init_equity
        self._entryprice = None
        self.drawdown = 0

        self.actions_memory = pd.DataFrame(columns=['date', 'action'])
        self.equity_memory = pd.DataFrame(columns=['date', 'equity_l', 'equity_h', 'BnH'])
        self.tradeslist_col = ['date', 'action', 'price', 'profit', 'drawdown']
        self.trades_list = pd.DataFrame(columns=self.tradeslist_col)

    def reset(self):
        self.episode += 1
        self.current_idx = 0
        self.done = False

        self.equity = self.init_equity
        self.bnh = self.init_equity

        self.reward = 0
        self.position = 0
        self.points = 0
        self.equity_l = self.init_equity
        self.equity_h = self.init_equity
        self.drawdown = 0
        self._entryprice = None

        self.actions_memory = pd.DataFrame(columns=['date', 'action'])
        self.equity_memory = pd.DataFrame(columns=['date', 'equity_l', 'equity_h', 'BnH'])
        self.trades_list = pd.DataFrame(columns=self.tradeslist_col)

        self.prices['signal_E'] = np.nan
        self.prices['signal_X'] = np.nan

        return np.append(self.observation.iloc[0].values, 0)

    def commission_cost(self, contracts):
        self.equity -= self.cost * contracts

    def _long(self, price, contracts):
        """
        :param price:
        :param contracts:
        :return:
        """
        if self.position < self.max_position:
            self.position += contracts
            self.commission_cost(contracts)
            self.points += price * contracts
            self._entryprice = price

    def _sell(self, price, contracts):
        if self.position > 0:
            self.position -= contracts
            self.points -= price * contracts
            self.commission_cost(contracts)
            # Close out of long position
            if self.position == 0:
                self.equity -= self.points * self.futures.big_point_value
                self.points = 0
                self._entryprice = None

    def render(self, mode='human'):
        if self.current_idx >= (len(self.df) // 2):
            plt.close()
            data = self.prices[:self.current_idx]
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
                plt.show(block=False)
                plt.pause(0.0001)

    def step(self, actions):
        self.done = (self.current_idx >= len(self.df.index.unique()) - 2)  # or (self.equity <= 0)
        # if self.current_idx % 2:
        # print(self.prices['Date'].iloc[self.current_idx])
        if self.current_idx == 0:
            self.bnh += self.futures.big_point_value * (
                    self.prices['Close'].iloc[self.current_idx] - self.prices['Open'].iloc[self.current_idx])
        else:
            self.bnh += self.futures.big_point_value * (
                    self.prices['Close'].iloc[self.current_idx] - self.prices['Close'].iloc[self.current_idx - 1])

        if self.done:
            print(self.reward)
            self._make_plot()
            if self.log:
                self._make_log()
            return np.append(self.observation.iloc[self.current_idx].values,
                             self.position), self.reward, self.done, {}
        else:
            # print(self.df['Date'].iloc[self.current_idx], actions)
            if self.df['until_expiration'].iloc[self.current_idx] == 0 and self.position > 0:
                c = self.prices['Close'].iloc[self.current_idx]
                self._sell(self.prices['Close'].iloc[self.current_idx], self.position)
                self.prices['signal_X'].iloc[self.current_idx] = self.prices['High'].iloc[self.current_idx] * 1.01
                self.trades_list = self.trades_list.append(
                    {'date': self.prices['Date'].iloc[self.current_idx + 1],
                     'action': 'settlement',
                     'price': c,
                     'profit': c - self._entryprice - 2 * self.cost,
                     'drawdown': self.equity_memory['equity_h'].max() - (
                                 self.init_equity + self.trades_list['profit'].sum())
                     }, ignore_index=True)

            if actions == 1:
                action_str = 'buy_next' if self.position < self.max_position else 'hold'
                self._long(self.prices['Open'].iloc[self.current_idx + 1], 1)
            elif actions == 2:
                action_str = 'sell_next'
                self._sell(self.prices['Open'].iloc[self.current_idx + 1], 1)
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
            if self.equity > self.equity_h:
                self.equity_h = self.equity

            if self.position > 0:
                self.equity_l += (self.prices['Low'].iloc[
                                      self.current_idx] - self._entryprice) * self.futures.big_point_value
            else:
                self.equity_l = self.equity
            dd = self.equity_h - self.equity_l
            # MDD new high
            if dd > self.drawdown:
                self.drawdown = dd

            self.actions_memory = self.actions_memory.append(
                {'date': self.prices['Date'].iloc[self.current_idx],
                 'action': action_str}, ignore_index=True)
            self.equity_memory = self.equity_memory.append(
                {'date': self.prices['Date'].iloc[self.current_idx],
                 'equity': self.equity_l,
                 'BnH': self.bnh}, ignore_index=True)
            self.current_idx += 1
            if self.drawdown:
                self.reward = np.round(self.equity / self.drawdown, 2)  # reward = return / MDD
            elif self.equity == self.init_equity:
                self.reward = -999
            else:
                self.reward = self.equity
            # self.reward = self.equity - self.bnh
            return np.append(self.observation.iloc[self.current_idx].values,
                             self.position), self.reward, self.done, {}

    def _make_plot(self):
        self.equity_memory.set_index(['date'])['equity'].plot(legend=True)
        self.equity_memory.set_index(['date'])['BnH'].plot(legend=True, colors='r')
        if not os.path.exists("./results"):
            os.makedirs("./results")
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def _make_log(self):
        self.equity_memory.to_csv(f'results_pic/{self.episode}.csv')

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
