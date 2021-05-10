import os

import gym
import matplotlib.pyplot as plt
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

    def __init__(self, df, futures, init_equity=100000, max_position=1):
        self.df = df
        self.current_idx = 0
        self.init_equity = init_equity
        # cost = 3 tick/per trade considering slippage
        self.futures = futures
        self.cost = self.futures.min_movement_point * self.futures.big_point_value * 3
        self.prices = self.df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        self.max_position = max_position
        self.trades_list_col = ['action', 'date', 'price', 'position', 'profit', 'equity', 'drawdown']

        # spaces
        self.action_describe = {
            0: 'hold',
            1: 'long',
            2: 'sell'
        }
        self.action_space = Discrete(len(self.action_describe))
        # observation_space 值域為[0,1]
        self.observation = self.df.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.observation.shape[1] + 1,))
        self.done = False

        # initialize reward
        self.reward = 0
        self.position = 0
        self.points = 0
        self.equity = self.init_equity
        self.equity_tmp = self.init_equity
        self.equity_h = self.init_equity
        self._entryprice = None
        self.episode = 0
        self.drawdown = self.equity_h - self.equity_tmp
        self.trades_list = pd.DataFrame(columns=self.trades_list_col)

        self.actions_memory = pd.DataFrame(columns=['date', 'action'])
        self.equity_memory = pd.DataFrame(columns=['date', 'equity'])
        self.rewards_memory = pd.DataFrame(columns=['date', 'rewards'])

    def reset(self):
        self.equity = self.init_equity
        self.reward = 0
        self.position = 0
        self.current_idx = 0
        self.points = 0
        self.equity_tmp = self.init_equity
        self.equity_h = self.init_equity
        self._entryprice = None
        self.drawdown = self.equity_h - self.equity_tmp
        self.trades_list = pd.DataFrame(columns=self.trades_list_col)
        self.done = False
        self._entryprice = None
        self.episode += 1
        self.actions_memory = pd.DataFrame(columns=['date', 'action'])
        self.equity_memory = pd.DataFrame(columns=['date', 'equity'])
        self.rewards_memory = pd.DataFrame(columns=['date', 'rewards'])
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

    def render(self, mode='human'):
        pass

    def step(self, actions):
        self.done = self.current_idx >= len(self.df.index.unique()) - 2
        if self.done:
            print(self.reward)
            self._make_plot()
            return np.append(self.observation.iloc[self.current_idx].values,
                             self.position), self.reward, self.done, {}
        else:
            # print(self.df['Date'].iloc[self.current_idx], actions)
            if self.df['until_expiration'].iloc[self.current_idx] == 0:
                action_str = 'hold' if not self.position else 'settlement'
                if self.position > 0:
                    self._sell(self.prices['Close'].iloc[self.current_idx], self.position)
            else:
                if actions == 1:
                    action_str = 'buy_next' if self.position < self.max_position else 'hold'
                    self._long(self.prices['Open'].iloc[self.current_idx + 1], 1)
                elif actions == 2:
                    action_str = 'sell_next'
                    self._sell(self.prices['Open'].iloc[self.current_idx + 1], 1)
                else:
                    action_str = 'hold'
            # equity new high
            if self.equity > self.equity_h:
                self.equity_h = self.equity

            if self.position > 0:
                self.equity_tmp += (self.prices['Low'].iloc[
                                        self.current_idx] - self._entryprice) * self.futures.big_point_value
            else:
                self.equity_tmp = self.equity
            dd = self.equity_h - self.equity_tmp
            # MDD new high
            if dd > self.drawdown:
                self.drawdown = dd

            self.actions_memory.append(
                {'date': self.prices['Date'].iloc[self.current_idx],
                 'action': action_str}, ignore_index=True)
            self.equity_memory.append(
                {'date': self.prices['Date'].iloc[self.current_idx],
                 'equity': self.equity_tmp}, ignore_index=True)
            self.rewards_memory.append(
                {'date': self.prices['Date'].iloc[self.current_idx],
                 'rewards': self.scale_reward()}, ignore_index=True)
            self.current_idx += 1
            self.data = self.df.loc[self.current_idx, :]
            self.reward = self.equity / self.drawdown if self.drawdown else self.equity  # reward = return / MDD

            return np.append(self.observation.iloc[self.current_idx].values,
                             self.position), self.reward, self.done, {}

    def scale_reward(self):
        scaled = self.reward / 50
        if scaled < 0:
            return 0
        else:
            return scaled

    def _make_plot(self):
        plt.plot(self.equity_memory, 'r')
        if not os.path.exists("./results"):
            os.makedirs("./results")
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
