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

class Player:
    def __init__(self, _id):
        self.id = _id
        self.reset()
        
    def reset(self):
        self.board = Board()

    
    def __str__(self):
        return 'Player ' + str(self.id)
    
    def give_scores(self, scores):
        pass
    
    def action(self, state):
        pass


class HumanPlayer(Player):      
    def action(self, state):
        tile_id = input("Which tile do you choose ?")
        x1 = int(input("x1 ? "))
        y1 = int(input("y1 ? "))      
        x2 = int(input("x2 ? "))
        y2 = int(input("y2 ? "))
        return tile_id, (x1,y1,x2,y2)


class RandomPlayer(Player):
    def action(self, state):
        return (self.kingdomino.selectTileRandom(),
                self.kingdomino.selectTilePositionRandom(self))


class PlayerAC(Player):
    
    def __init__(self, env, gamma, lr_a, lr_c,
                 coordinate_std):
        
        self.gamma = gamma
        
        self.shared_rep_size = 200
        self.shared = Shared(
            env.n_players, 
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
        params = list(self.shared.parameters) + \
                 list(self.actor.parameters) + \
                 list(self.critic.parameters)
   
        self.coordinates_cov = torch.eye(2) * coordinate_std**2
   
        self.optimizer_c = torch.optim.AdamW(
            self.shared.parameters() + self.critic.parameters(),
            amsgrad=True,
            lr=0.001)
        self.criterion_c = nn.SmoothL1Loss()
        
        self.optimizer_a = torch.optim.AdamW(
            self.shared.parameters() + self.actor.parameters(),
            amsgrad=True,
            lr=0.0005)
        self.criterion_a = nn.SmoothL1Loss()
        
        self.reset()
        
    def reset(self):
        self.reward = None
        self.action = None
        self.value = None
    
    def action(self, state):
        self.prev_value = self.value
        shared_rep = self.shared(state)
        action_output = self.actor(shared_rep)
        
        coordinates_output = action_output[:2]
        coordinates_distribution = distributions.MultivariateNormal(coordinates_output, self.coordinates_cov)
        coordinates = coordinates_distribution.sample()
        
        tile_output_distribution = F.softmax(action_output[2:], dim=0)
        tile = torch.multinomial(tile_output_distribution, num_sample=1)
                
        self.value = self.critic(shared_rep)
        
        # Updates
        if self.prev_value is not None:
            target = self.reward + self.gamma*self.value
            
            loss_c = self.criterion_c(
                target, 
                self.value)
            self.optimizer_c.zero_grad()
            loss_c.backward
            self.optimizer_c.step()
            
            loss_a = -torch.log() * loss_c
            self.optimizer_c.zero_grad()
            loss_c.backward
            self.optimizer_c.step()
            
        return 
            
            

    def give_reward(self, reward):
        self.reward = reward

