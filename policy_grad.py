import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from mdp import *
from deep_q_learning import *

def pi_random(s, greedy=False, eps=0.2):
    return np.random.choice(A)


def simulate_pg(series, state_init, pi, greedy, eps=0.2):
    Rs = list()
    actions = list()
    state_init_norm = np.array(state_init)
    state_init_norm = (state_init_norm-min(state_init_norm)) / (max(state_init_norm) - min(state_init_norm))
    states = [state_init]

    states_normed = [list(state_init_norm)]

    for index, row in series.iterrows():
        a = pi(states_normed[-1], greedy=greedy, eps=eps)
        actions.append(a)
        s = update_state(states[-1], a, row)
        states.append(s)
        state = np.array(update_state(states[-1], a, row))
        state_norm = (state-min(state)) / (max(state)-min(state))
        states_normed.append(list(state_norm))
        Rs.append(reward(states[-2], states[-1]))
    return states_normed, Rs, actions


class Rpg(nn.Module):
    def __init__(self, input_size=7, num_classes=len(A)):
        super().__init__()
        self.fc_liner = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Softmax(dim=-1)
        )


    def forward(self, x):
        return self.fc_liner(x)


def policy_gradient_learning(rpg,
                    series,
                    state_init,
                    pi,
                    optimizer,
                    epochs=10,
                    episode=100,
                    gamma=0.9,
                    eps=0.5,
                    min_eps=0.05,
                    decay=0.9,
                    greedy=False,
                    verbose=True,
                    ):
    losses = list()

    profits_lc = list()
    # loop for each episode
    for epi in tqdm(range(episode)):
        # generate a trajectory
        eps *= decay
        if epi == 0:
            states, rewards, actions = simulate_pg(series, state_init, pi_random, greedy, eps=max(min_eps, eps))


        else:
            states, rewards, actions = simulate_pg(series, state_init, pi, greedy, eps=max(min_eps, eps))

        # form dataset and data loader
        dataset = StatesDataset(states, rewards, actions)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # fit the deep NN
        for epo in range(epochs):
            for data_pack in dataloader:
                input_tensor = data_pack['states']

                action_tensor = data_pack['actions']+k
                out = rpg(input_tensor)

                logprob = torch.log(out)

                action_tensor = torch.unsqueeze(action_tensor, 1)

                # Calculate the G_t at each t
                batch_reward = data_pack['rewards']
                discount_rewards = np.array([gamma** i * batch_reward[i] for i in range(len(batch_reward))])
                discount_rewards = discount_rewards[::-1].cumsum()[::-1]
                discount_rewards = torch.Tensor(discount_rewards.copy())



                # torch.gather() is used to calculate the log of pi(s_t, a_t)
                selected_logprobs = discount_rewards * torch.gather(logprob, 1, action_tensor).squeeze()
                loss = -selected_logprobs.mean()


                # update weights
                rpg.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())

        profits_lc.append(interact_test(pi, series_name='train', verbose=False))



    # verbose
    if verbose:
        print("Last loss", losses[-1])
        plt.plot(losses)
        plt.xlabel('Iterations (Not Epochs)')
        plt.ylabel('Loss')
        plt.show()

        plt.clf()
        plt.plot(profits_lc)
        plt.xlabel('Iterations (Not Epochs)')
        plt.ylabel('Profit')
        plt.show()

def train_policy_gradient(verbose=False):
    rpg = Rpg(input_size=7, num_classes=len(A))
    state_init = [train_series['Close'][0], balance_init, 0] + list(train_series.iloc[0][5:])
    series = train_series[1:]

    optimizer = optim.SGD(rpg.parameters(), lr=1e-4)

    def pi_gradient(s, eps=0.2, greedy=False):
        with torch.no_grad():
            out_rpg = rpg(torch.Tensor(s).float()).squeeze()
            action = out_rpg.argmax().item() - k

            if not greedy:
                r = np.random.rand()
                # if it is on the less side, the explore other actions
                if r > 1 - eps + (eps / len(A)):
                    a_ = np.random.choice(A)
                    while a_ == action:
                        a_ = np.random.choice(A)
                    action = a_

        return action

    # greedy is set to True in policy gradient.
    policy_gradient_learning(rpg,
                    series,
                    state_init,
                    pi_gradient,
                    optimizer,
                    epochs=10, # number of epochs for training NN in each episode 10
                    episode=30, # 30
                    gamma=0.6, # discount coefficient 0.618
                    eps=0.8, # eps greedy policy
                    min_eps=0.2, # 0.2
                    decay=0.9, # 0.9
                    greedy=False,
                    verbose=verbose
                   )
    return pi_gradient, rpg

