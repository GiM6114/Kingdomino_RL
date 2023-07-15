import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    
    def __init__(self, env, gamma, lr_a, lr_c):
        
        self.gamma = gamma
        
        self.shared_rep_size = 200
        self.shared = Shared(
            env.n_players, 
            board_rep_size=100,
            shared_rep_size=self.shared_rep_size)
        self.actor = NeuralNetwork(
            input_size=self.shared_rep_size, 
            output_size=2+4, 
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
        self.optimizer = torch.optim.Adam(params)
    
    def action(self, state):
        shared_rep = self.shared(state)
        self.ac
 

