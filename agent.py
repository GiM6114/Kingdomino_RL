import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch

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


class PlayerAC:
    
    def __init__(self, n_players, gamma, lr_a, lr_c,
                 coordinate_std, _id):
        
        self.gamma = gamma
        
        self.shared_rep_size = 200
        self.shared = Shared(
            n_players, 
            board_rep_size=100,
            shared_rep_size=self.shared_rep_size)
        self.actor = NeuralNetwork(
            input_size=self.shared_rep_size, 
            output_size=2+4, 
            # coordinates to place previous tile + which tile to take
            l=5, 
            n=150)
        self.critic = NeuralNetwork(
            input_size=self.shared_rep_size, 
            output_size=1, 
            l=5, 
            n=150)
        # params = list(self.shared.parameters) + \
        #          list(self.actor.parameters) + \
        #          list(self.critic.parameters)
   
        self.coordinates_cov = torch.eye(2) * coordinate_std**2
   
        self.optimizer_c = torch.optim.AdamW(
            list(self.shared.parameters()) + list(self.critic.parameters()),
            amsgrad=True,
            lr=0.001)
        self.criterion_c = nn.SmoothL1Loss()
        
        self.optimizer_a = torch.optim.AdamW(
            list(self.shared.parameters()) + list(self.actor.parameters()),
            amsgrad=True,
            lr=0.0005)
        self.criterion_a = nn.SmoothL1Loss()
        
        self.reset()
        
    def reset(self):
        super().reset()
        self.reward = None
        self.tile_choice_distribution = None
        self.tile_chosen = None
        self.coordinates_distribution = None
        self.coordinates_chosen = None
        self.value = None
    
    def action(self, state):
        shared_rep = self.shared(self.id, state)
        
        # Here we have the previous state value, action, reward and state is the next state
        # -> Update
        self.prev_value = self.value
        self.value = self.critic(shared_rep)

        if self.prev_value is not None:
            target = self.reward + self.gamma*self.value
            
            # Critic update
            loss_c = self.criterion_c(
                target,
                self.prev_value)
            self.optimizer_c.zero_grad()
            loss_c.backward()
            self.optimizer_c.step()
            
            # Actor update
            # product of proba as tile choice and coordinates chosen for previous tiles
            # assumeed independant (they are not)
            loss_a = (-self.coordinates_distribution.log_prob(self.coordinates_chosen) \
                     -self.tile_choice_distribution.log_prob(self.tile_chosen)) * loss_c
            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()
        
        action_output = self.actor(shared_rep)
        
        coordinates_output = action_output[:2]
        self.coordinates_distribution = distributions.MultivariateNormal(coordinates_output, self.coordinates_cov)
        self.coordinates_chosen = torch.round(self.coordinates_distribution.sample())
        
        tile_output_distribution = F.softmax(action_output[2:], dim=0)
        self.tile_choice_distribution = distributions.Multinomial(tile_output_distribution)
        self.tile_chosen = self.tile_choice_distribution.sample()
                
        
        action = (self.coordinates_chosen,self.tile_chosen)
        
        return action
            

    def give_reward(self, reward):
        self.reward = reward

