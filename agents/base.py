import numpy as np
import random
from abc import ABC, abstractmethod
import torch
from itertools import product

from agents.encoding import state_encoding
from agents.prioritized_experience_replay.prioritized_experience_replay import PrioritizedReplayBuffer

class Player(ABC):
    @abstractmethod
    def action(self, state, kingdomino):
        pass

    def process_reward(self, reward, done):
        pass
    def train(self):
        pass
    def eval(self):
        pass
    def test(self):
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
    def process_reward(self, r, d):
        pass


class LearningAgent(Player, ABC):
    def __init__(self, log_dir, log_file, encode_state=True):
        self.log_dir = log_dir
        self.log_file = log_file
        self.encode_state = encode_state
    
    def reset(self):
        self.prev_s = None
        self.prev_a = None
        self.prev_r = None
    
    def add_to_memory(self, next_s, p_a):
        # Check if first state of the episode
        if self.prev_s is None:
            return
        # Check if last state of the episode
        if next_s is None:
            next_s = self.prev_s
            p_a = (self.action_interface.default_action,)
            
        s = self.prev_s
        a = self.prev_a
        r = torch.tensor([self.prev_r], dtype=torch.float32)
        d = self.prev_d
        self.memory.add((
            s['Boards'].squeeze(),
            s['Current tiles'].squeeze(),
            s['Previous tiles'].squeeze(),
            s.get('Actions', None),
            a,
            r,
            next_s['Boards'].squeeze(),
            next_s['Current tiles'].squeeze(),
            next_s['Previous tiles'].squeeze(),
            next_s.get('Actions', None),
            d,
            self.action_interface.encode(p_a)))
    
    # state : dict['Boards','Previous tiles','Current tiles']
    # state_encoding : same but one_hot_encoded
    @torch.no_grad()
    def action(self, s, p_a):
        encoded_s = s
        if self.encode_state:
            encoded_s = state_encoding(s, self.n_players, self.device)
        s = self.policy.filter_encoded_state(encoded_s)
        p_a = self.action_interface.expand_possible_actions(p_a)
        if self.training:
            self.add_to_memory(encoded_s, p_a)
            self.prev_s = encoded_s
        with torch.no_grad():
            # first output is what is returned to the environment
            # second is what is stored in the replay memory
            a,a_torch = self.select_action(encoded_s, p_a)
        self.prev_a = a_torch
        return a
    
    def PER(self):
        return isinstance(self.memory, PrioritizedReplayBuffer)
    
    @abstractmethod
    def select_action(self, s, p_a):
        pass
    @abstractmethod
    def optimize(self):
        pass
    @abstractmethod     
    def load(self, path):
        pass
    @abstractmethod
    def save(self, log_dir=None, log_name=None):
        pass
    
    def process_reward(self, r, d):
        self.prev_r = r
        self.prev_d = d
        loss = self.optimize()
        if d:
            self.add_to_memory(next_s=None, p_a=None)
            self.reset()
        return loss