import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf
import pendulum

from data_preprocessing import *

# Risk/Investment Management
balance_init = 1000 # initial balance in hand
k = 5 # max number of shares
min_balance = -100 # tolerance balance

# action
A = [a for a in range(-k, k+1, 1)]

# transition
def get_features(s, new_record):
    return list(new_record[5:])

def update_state(s, a, new_record):
    # s: (price, balance, shares)
    price, balance, shares = s[0], s[1], s[2]
    # a: (0 is hold, -k is sell, +k is buying)
    
    # Constraints
    # if is sell, check if there are enough number of shares
    if a < 0:
        if shares <= abs(a):
            a = -shares
    elif a > 0: # if buying, check if there are enough balance
        if balance - (a * price) < min_balance:
            possible_balance = np.array([balance - (a_ * price) for a_ in range(a)]) >= min_balance
            a = np.argmax(possible_balance)
    new_shares = shares + a
    new_balance = balance - (a * price)
    
    # apply fee (approx 0.1%)
    new_balance -= (a * price) * 1e-3
    
    # update state
    features = get_features(s, new_record)
    return [new_record['Close'], new_balance, new_shares] + features

# reward
def reward(s, s_next):
    return (s[1] + s[0]*s[2]) - (s_next[1] + s_next[0]*s_next[2])

# interact
def simulate(series, state_init, pi, greedy, eps=0.2):
    Rs = list()
    actions = list()
    states = [state_init]
    for index, row in series.iterrows():
        a = pi(states[-1], greedy=greedy, eps=eps)
        actions.append(a)
        states.append(update_state(states[-1], a, row))
        Rs.append(reward(states[-2], states[-1]))
    return states, Rs, actions

def interact_test(pi, series_name='test', verbose=True):
    if series_name == 'test':
        series = test_series
        prev_series = train_series
        prev_ind = -1
    elif series_name == 'train':
        series = train_series[1:]
        prev_series = train_series
        prev_ind = 0

    state_init = [prev_series['Close'][prev_ind], balance_init, 0] + list(prev_series.iloc[prev_ind][5:]) # price, balance, shares, Index

    # start a trajectory
    states, rewards, actions = simulate(series, state_init, pi, True)

    # verbose
    portforlio = np.array([s[1] + s[0]*s[2] for s in states])
    if verbose:
        print("Profit at The End of Trajactory:", portforlio[-1] - balance_init)

        plt.style.use('dark_background')
        plt.plot(series['Close'])
        plt.title("Price")
        plt.xlabel("Time (1 day inter val)")
        plt.ylabel("Price ($)")
        plt.show()

        plt.style.use('dark_background')
        plt.plot([s[2] for s in states])
        plt.title("Number of Shares")
        plt.xlabel("Time (1 day interval)")
        plt.ylabel("Num shares")
        plt.show()

        plt.style.use('dark_background')
        plt.plot(portforlio)
        plt.title("Portfolio ($)")
        plt.xlabel("Time (1 day inter val)")
        plt.ylabel("Portfolio ($)")
        plt.show()

        plt.style.use('dark_background')
        plt.plot(portforlio - balance_init)
        plt.title("Trading Profit")
        plt.xlabel("Time (1 day interval)")
        plt.ylabel("Profit ($)")
        plt.show()
    
    return portforlio[-1] - balance_init