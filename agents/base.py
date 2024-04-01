import numpy as np
import random
from abc import ABC, abstractmethod
import torch

from agents.encoding import state_encoding
from prioritized_experience_replay import PrioritizedReplayBuffer

class Player(ABC):
    @abstractmethod
    def action(self, state, kingdomino):
        pass

    def processReward(self, reward, next_state, done):
        pass
    
class HumanPlayer(Player):      
    def action(self, state, kingdomino):
        tile_id = int(input("Which tile do you choose ?"))
        x1 = int(input("x1 ? "))
        y1 = int(input("y1 ? "))      
        x2 = int(input("x2 ? "))
        y2 = int(input("y2 ? "))
        return tile_id, np.array([[x1,y1],[x2,y2]])


class RandomPlayer(Player):
    def action(self, state, kingdomino):
        return random.choice(kingdomino.getPossibleActions())
    
class LearningAgent(Player, ABC):
    # Factorizable in LearningAgent
    def reset(self):
        self.encoded_prev_s = None
        self.prev_a = None
    
    # Factorizable in LearningAgent
    # state : dict['Boards','Previous tiles','Current tiles']
    # state_encoding : same but one_hot_encoded
    @torch.no_grad()
    def action(self, s, kingdomino):
        encoded_s = state_encoding(s, self.n_players, self.device)
        self.policy.filter_encoded_state(encoded_s)
        if self.training:
            self.encoded_prev_s = encoded_s
        with torch.no_grad():
            a,a_torch = self.select_action(s, kingdomino)
        self.prev_a = a_torch
        return a
    
    # Factorizable in LearningAgent
    def PER(self):
        return isinstance(self.memory, PrioritizedReplayBuffer)
    
    @abstractmethod
    def select_action(self, s, kingdomino):
        raise NotImplementedError()
    @abstractmethod
    def optimize(self):
        raise NotImplementedError()

         
    # Factorizable in LearningAgent
    def process_reward(self, r, n_s, d, p_a):
        encoded_n_s = state_encoding(n_s, self.n_players, self.device)
        self.policy.filter_encoded_state(encoded_n_s)
        self.memory.add((
            self.encoded_prev_s['Boards'].squeeze(),
            self.encoded_prev_s['Current tiles'].squeeze(),
            self.encoded_prev_s['Previous tiles'].squeeze(),
            self.prev_a.unsqueeze(0),
            torch.tensor([r], device=self.device, dtype=torch.float32),
            encoded_n_s['Boards'].squeeze(),
            encoded_n_s['Current tiles'].squeeze(),
            encoded_n_s['Previous tiles'].squeeze(),
            d,
            self.action_interface.encode(p_a)))
        self.optimize()
        if d:
            self.reset()