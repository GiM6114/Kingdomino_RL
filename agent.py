import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch
torch.autograd.set_detect_anomaly(True)
from collections import namedtuple
from torchviz import make_dot
from IPython.display import display

from board import Board
from setup import N_TILE_TYPES
from networks import Shared, NeuralNetwork


class Pipeline:
    def __init__(self, player, env):
        self.player = player
        self.env = env


class HumanPlayer:      
    def action(self, state):
        tile_id = input("Which tile do you choose ?")
        x1 = int(input("x1 ? "))
        y1 = int(input("y1 ? "))      
        x2 = int(input("x2 ? "))
        y2 = int(input("y2 ? "))
        return tile_id, (x1,y1,x2,y2)


class RandomPlayer:
    def action(self, state, kingdomino):
        return (kingdomino.selectTileRandom(),
                kingdomino.selectTilePositionRandom())

NetworksInfo = namedtuple('NetworksInfo', 
                          ['in_channels', 
                           'output_size',
                           'conv_channels',
                           'conv_l',
                           'conv_kernel_size', 
                           'conv_stride',
                           'pool_place',
                           'pool_kernel_size', 
                           'pool_stride',
                           'board_rep_size',
                           'board_fc_n',
                           'board_fc_l',
                           'player_rep_size',
                           'board_prev_tile_fc_l',
                           'shared_rep_size',
                           'shared_l',
                           'shared_n',
                           'actor_l',
                           'actor_n',
                           'critic_l',
                           'critic_n'])

class PlayerAC:
    
    def __init__(self, 
                 batch_size,
                 n_players, 
                 gamma, lr_a, lr_c, coordinate_std,
                 network_info):
        
        self.gamma = gamma
        self.shared = Shared(
            n_players, 
            network_info)

        self.critic = NeuralNetwork(
            input_size=network_info.shared_rep_size, 
            output_size=1, 
            l=network_info.critic_l, 
            n=network_info.critic_n)
        self.actor = NeuralNetwork(
            input_size=network_info.shared_rep_size, 
            output_size=2+n_players, 
            # coordinates to place previous tile + which tile to take
            l=network_info.actor_l, 
            n=network_info.actor_n)
   
        self.coordinates_cov = torch.eye(2).repeat(batch_size,1,1) * coordinate_std**2
        
        self.optimizer_c = torch.optim.AdamW(
            list(self.shared.parameters()) + list(self.critic.parameters()),
            amsgrad=True,
            lr=lr_c)
        self.criterion_c = nn.SmoothL1Loss()
        self.optimizer_a = torch.optim.AdamW(
            list(self.shared.parameters()) + list(self.actor.parameters()),
            amsgrad=True,
            lr=lr_a)
        
        self.reset()
        
    def reset(self):
        self.rewards = None
        self.tile_choice_distributions = None
        self.tiles_chosen = None
        self.coordinates_distributions = None
        self.coordinates_chosen = None
        self.values = None
    
    def action(self, states):
        shared_rep = self.shared(states)

        print('Version :', shared_rep._version)
        
        # Here we have the previous state value, action, reward and state is the next state
        # -> Update
        self.prev_values = self.values
        self.values = self.critic(shared_rep).squeeze()
        graph = make_dot(self.critic(shared_rep), params=dict(self.shared.named_parameters()) | dict(self.critic.named_parameters()))
        display(graph)

        if self.prev_values is not None:
            targets = self.rewards + self.gamma*self.values
            # Critic update
            loss_c = self.criterion_c(
                targets,
                self.prev_values)
            self.optimizer_c.zero_grad()
            loss_c.backward(retain_graph=True)
            self.optimizer_c.step()
            print('Version :', shared_rep._version)
            print('Version :', loss_c._version)

            # Actor update
            # product of proba as tile choice and coordinates chosen for previous tiles
            # assumeed independant (they are not)
            loss_a = torch.mean((-self.coordinates_distributions.log_prob(self.coordinates_chosen) \
                     -self.tile_choice_distributions.log_prob(self.tile_chosen))) * loss_c
            print('Version :', shared_rep._version)
            print('Version :', loss_a._version)
            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()
        
        action_outputs = self.actor(shared_rep)
        graph = make_dot(self.actor(shared_rep), params=dict(self.shared.named_parameters()) | dict(self.actor.named_parameters()))
        display(graph)
        
        coordinates_outputs = action_outputs[:,:2]
        self.coordinates_distributions = distributions.MultivariateNormal(coordinates_outputs, self.coordinates_cov)
        self.coordinates_chosen = torch.round(self.coordinates_distributions.sample())
        
        tile_output_distribution = F.softmax(action_outputs[:,2:], dim=1)
        self.tile_choice_distributions = distributions.Multinomial(total_count=1, probs=tile_output_distribution)
        self.tile_chosen = self.tile_choice_distributions.sample()
                
        
        action = torch.column_stack((self.coordinates_chosen,self.tile_chosen.nonzero()[:,1]))
        
        return action
            

    def give_reward(self, rewards):
        self.rewards = torch.tensor(rewards)

