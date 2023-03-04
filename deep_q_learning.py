import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from mdp import *

class Qsa(nn.Module):
    def __init__(self, input_size=7, num_classes=len(A)):
        super().__init__()
        self.fc_liner = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.fc_liner(x)

class StatesDataset(Dataset):
    def __init__(self, states, rewards, actions):
        self.states = torch.Tensor(states[:-1]).float()
        self.states_next = torch.Tensor(states[1:]).float()
        self.rewards = torch.Tensor(rewards).float()
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],
            'states_next': self.states_next[idx],
            'rewards': self.rewards[idx],
            'actions': self.actions[idx]
        }

def deep_q_learning(qsa, 
                    series, 
                    state_init, 
                    pi, 
                    optimizer,
                    loss_func,
                    epochs=10,
                    episode=100, 
                    gamma=0.9,
                    lr=0.7,
                    eps=0.5,
                    min_eps=0.05,
                    decay=0.9,
                    greedy=False,
                    verbose=True,
                    sarsa=False
                   ):
    losses = list()
    learning_curve = list()
    # loop for each episode
    for epi in tqdm(range(episode)):
        # generate a trajectory
        eps *= decay
        states, rewards, actions = simulate(series, state_init, pi, greedy, eps=max(min_eps, eps))
        
        # form dataset and data loader
        dataset = StatesDataset(states, rewards, actions)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        # fit the deep NN
        for epo in range(epochs):
            for data_pack in dataloader:
                input_tensor = data_pack['states']
                out = qsa(input_tensor)
                output_tensor = out[[i for i in range(len(data_pack['rewards']))], [a+k for a in data_pack['actions']]]

                with torch.no_grad():
                    max_qsa_out = qsa(data_pack['states_next'])
                    if not sarsa:
                        max_qsa = max_qsa_out[[i for i in range(len(data_pack['rewards']))], max_qsa_out.argmax(dim=1)]
                        max_qsa = torch.Tensor(data_pack['rewards']).float() + (gamma * max_qsa)
                    else:
                        max_qsa = max_qsa_out[[i for i in range(len(data_pack['rewards']))], [a+k for a in data_pack['actions'][1:]] + [0]]
                    target_tensor = (1 - lr) * output_tensor + lr * max_qsa

                # update weights
                loss = loss_func(output_tensor, target_tensor)
                qsa.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())
                
        learning_curve.append(interact_test(pi, series_name='test', verbose=False))
        
    # verbose
    if verbose:
        print("Last loss", losses[-1])
        plt.plot(losses)
        plt.xlabel('Iterations (Not Epochs)')
        plt.ylabel('Loss')
        plt.savefig('loss.pdf')
        plt.show()
    return learning_curve

def train_deep_q(verbose=False):
    qsa = Qsa(input_size=7, num_classes=len(A))
    state_init = [train_series['Close'][0], balance_init, 0] + list(train_series.iloc[0][5:])
    series = train_series[1:]

    optimizer = optim.Adam(
        qsa.parameters(),
        lr=1e-5, # 1e-4
    #     weight_decay=1e-6
    )
#     loss_func = nn.MSELoss()
    loss_func = nn.HuberLoss()

    def pi_deep(s, eps=0.2, greedy=False):
        with torch.no_grad():
            out_qsa = qsa(torch.Tensor(s).float()).squeeze()
            action = out_qsa.argmax().item() - k

            if not greedy:
                r = np.random.rand()
                # if it is on the less side, the explore other actions
                if r > 1 - eps + (eps / len(A)):
                    a_ = np.random.choice(A)
                    while a_ == action:
                        a_ = np.random.choice(A)
                    action = a_
        return action

    learning_curve = deep_q_learning(qsa, 
                                     series, 
                                     state_init, 
                                     pi_deep, 
                                     optimizer,
                                     loss_func,
                                     epochs=10, # number of epochs for training NN in each episode 10
                                     episode=30, # 30
                                     gamma=0.6, # discount coefficient 0.618
                                     lr=0.7, # learning rate for update q function
                                     eps=0.8, # eps greedy policy
                                     min_eps=0.2, # 0.2
                                     decay=0.9, # 0.9
                                     greedy=False,
                                     verbose=verbose
                                    )
    return pi_deep, qsa, learning_curve

def train_deep_sarsa(verbose=False, sarsa=False):
    qsa = Qsa(input_size=7, num_classes=len(A))
    state_init = [train_series['Close'][0], balance_init, 0] + list(train_series.iloc[0][5:])
    series = train_series[1:]

    optimizer = optim.Adam(
        qsa.parameters(),
        lr=1e-5, # 1e-4
    #     weight_decay=1e-6
    )
#     loss_func = nn.MSELoss()
    loss_func = nn.HuberLoss()

    def pi_deep(s, eps=0.2, greedy=False):
        with torch.no_grad():
            out_qsa = qsa(torch.Tensor(s).float()).squeeze()
            action = out_qsa.argmax().item() - k

            if not greedy:
                r = np.random.rand()
                # if it is on the less side, the explore other actions
                if r > 1 - eps + (eps / len(A)):
                    a_ = np.random.choice(A)
                    while a_ == action:
                        a_ = np.random.choice(A)
                    action = a_
        return action

    deep_q_learning(qsa,
                    series,
                    state_init,
                    pi_deep,
                    optimizer,
                    loss_func,
                    epochs=10, # number of epochs for training NN in each episode 10
                    episode=30, # 30
                    gamma=0.6, # discount coefficient 0.618
                    lr=0.7, # learning rate for update q function
                    eps=0.8, # eps greedy policy
                    min_eps=0.2, # 0.2
                    decay=0.9, # 0.9
                    greedy=False,
                    verbose=verbose,
                    sarsa=sarsa
                   )
    return pi_deep, qsa