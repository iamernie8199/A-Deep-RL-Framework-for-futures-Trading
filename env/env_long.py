import gym
import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box


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
        self.init_equity = init_equity
        # cost = 3 tick/per trade considering slippage
        self.futures = futures
        self.cost = self.futures.min_movement_point * self.futures.big_point_value * 3
        self.window_size = window_size
        self.prices = self.df['Close']
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
        self.observation = self.df.drop(columns=[['Open', 'High', 'Low', 'Close']])
        self.observation_space = Box(low=0, high=1, shape=(self.df.shape[1],))

        self.current_idx = 0
        # initialize reward
        self.reward = 0
        self.position = 0
        self.points = 0
        self.equity = self.init_equity
        self.trades_list = pd.DataFrame(columns=self.trades_list_col)

    def reset(self):
        self.equity = self.init_equity
        self.reward = 0
        self.position = 0
        self.current_idx = 0
        self.points = 0
        self.trades_list = pd.DataFrame(columns=self.trades_list_col)
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
