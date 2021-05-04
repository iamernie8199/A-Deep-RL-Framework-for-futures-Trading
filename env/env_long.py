import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Box

from stable_baselines3.common import logger


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

    def __init__(self, df, futures, window_size, init_equity=100000, max_position=1):
        self.df = df
        self.current_idx = 0
        self.data = self.df.loc[self.current_idx, :]
        self.init_equity = init_equity
        # cost = 3 tick/per trade considering slippage
        self.futures = futures
        self.cost = self.futures.min_movement_point * self.futures.big_point_value * 3
        self.window_size = window_size
        self.prices = self.data[['Date', 'Open', 'High', 'Low', 'Close']]
        self.max_position = max_position
        self.trades_list_col = ['action', 'date', 'price', 'position', 'profit', 'equity', 'drawdown']

        # spaces
        self.action = {
            -1: 'sell',
            0: 'hold',
            1: 'long',
        }
        self.action_space = Discrete(len(self.action))
        # observation_space 值郁為[0,1]
        self.observation = self.data.drop(columns=[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
        self.observation_space = Box(low=0, high=1, shape=(self.observation.shape[1],))
        self.done = False

        # initialize reward
        self.reward = 0
        self.position = 0
        self.points = 0
        self.equity = self.init_equity
        self.trades_list = pd.DataFrame(columns=self.trades_list_col)

        self.actions_memory = pd.DataFrame(columns=['date', 'action'])

    def reset(self):
        self.equity = self.init_equity
        self.reward = 0
        self.position = 0
        self.current_idx = 0
        self.points = 0
        self.trades_list = pd.DataFrame(columns=self.trades_list_col)
        self.done = False
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
        self.done = self.current_idx >= len(self.df.index.unique())-1

        if self.df['until_expiration'].iloc[self.current_idx] == 0:
            action_str = 'hold' if not self.position else 'settlement'
            if self.position > 0:
                self._sell(self.prices['Close'].iloc[self.current_idx], self.position)
        else:
            if actions == 1:
                action_str = 'buy_next' if self.position < self.max_position else 'hold'
                self._long(self.prices['Open'].iloc[self.current_idx + 1], 1)
            elif actions == -1:
                action_str = 'sell_next'
                self._sell(self.prices['Open'].iloc[self.current_idx + 1], 1)
            else:
                action_str = 'hold'

        self.actions_memory.append({'date': self.prices['Date'].iloc[self.current_idx],
                                    'action': action_str})
        self.current_idx += 1
        self.data = self.df.loc[self.current_idx, :]

        return np.append(self.observation.iloc[self.current_idx].values, self.position), self.reward, done, {}
