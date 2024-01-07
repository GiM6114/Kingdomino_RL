import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch
torch.autograd.set_detect_anomaly(True)
from collections import namedtuple,deque
# from torchviz import make_dot
# from IPython.display import display
from time import time
from copy import deepcopy
import itertools
import math
from IPython.core.display import Image, display

from board import Board
from setup import N_TILE_TYPES
from networks import Shared, NeuralNetwork, PlayerFocusedACNN
from printer import Printer
from graphics import draw_obs
from prioritized_experience_replay import PrioritizedReplayBuffer
from encoding import boards2onehot, get_current_tiles_vector, tile2onehot, actions2onehot

class HumanPlayer:      
    def action(self, state, kingdomino):
        tile_id = int(input("Which tile do you choose ?"))
        x1 = int(input("x1 ? "))
        y1 = int(input("y1 ? "))      
        x2 = int(input("x2 ? "))
        y2 = int(input("y2 ? "))
        return tile_id, np.array([[x1,y1],[x2,y2]])


class RandomPlayer:
    def action(self, state, kingdomino):
        return kingdomino.getRandomAction()


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'possible_actions'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN_Agent:
    def __init__(self, 
                 n_players,
                 exploration_batch_size,
                 batch_size,
                 eps_scheduler,
                 tau,
                 lr,
                 gamma,
                 id,
                 network_name=None,
                 hp_archi=None,
                 replay_memory_size=None,
                 device='cpu',
                 policy=None,
                 target=None,
                 memory=None):
        # should be 1 but let's keep it general
        self.exploration_batch_size = exploration_batch_size
        self.batch_size = batch_size
        self.eps_scheduler = eps_scheduler
        self.tau = tau
        self.lr = lr
        self.device = device
        self.n_players = n_players
        self.id = id
        self.gamma = gamma
        
        if network_name == 'PlayerFocusedACNN':
            self.network = PlayerFocusedACNN
        
        self.policy = policy
        if self.policy is None:
            self.policy = self.network(
                n_players=self.n_players,
                network_info=hp_archi,
                device=self.device).to(self.device)
        self.target = target
        if self.target is None:
            self.target = self.network(
                n_players=self.n_players,
                network_info=hp_archi,
                device=self.device).to(self.device)
            self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        self.memory = memory
        self.learning = True
        self.reset()
    
    def reset(self):
        self.n_steps = 0
        self.prev_state = None
        self.prev_action = None
    
    # state : dict['Boards','Previous tiles','Current tiles']
    def action(self, state, kingdomino):
                
        state = self.convert_state_torch(state)
        self.prev_state = state
        action,action_torch = self.select_action(state, kingdomino)
        self.prev_action = action_torch
        return action
    
    # TODO: stuff to do here
    def select_action(self, state, kingdomino):
        actions = kingdomino.getPossibleActions()
        actions_torch = actions2onehot(actions, self.n_players, self.device)
        boards,aux = self.policy.state2tensors(state, actions_torch.shape[0])
        if state is None:
            return None,None
        if self.learning:
            sample = random.random()
            eps_threshold = self.eps_scheduler.eps() 
        if not self.learning or sample > eps_threshold:
            with torch.no_grad():
                qvalues = self.policy(state, actions_torch).squeeze()
                argmax = qvalues.argmax()
                return actions[argmax],actions_torch[argmax]
        i = random.randint(0, len(actions)-1)
        return actions[i],actions_torch[i]
    

        
    def give_reward(self, reward, next_state, possible_actions):
        # print('Previous state:')
        # display(draw_obs(self.prev_state))
        # print('State:')
        # display(draw_obs(next_state))
        self.memory.push((
            self.prev_state, 
            self.prev_action.unsqueeze(0),
            torch.tensor([reward], device=self.device, dtype=torch.float32),
            self.convert_state_torch(next_state),
            self.convert_actions_torch(possible_actions)))
        self.optimize()
        
    # might need to rearrange this a bit
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # non_final_mask = torch.tensor(
        #     tuple(map(lambda s: s is not None, 
        #               batch.next_state)), device=self.device, dtype=torch.bool)
        # non_final_next_states = [s for s in batch.next_state if s is not None]
        next_states = batch.next_state
        state_batch_listdict = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_batch = self.listdict2dicttensor(state_batch_listdict)
        possible_actions_batch = batch.possible_actions

        state_action_values = self.policy(state_batch, action_batch).squeeze()
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # problem states list of dict, cant apply non_final_mask directly
            # next_state_values_non_final = torch.zeros(torch.sum(non_final_mask), device=self.device)
            # TODO: god awful loop, but problem bcs possible actions take different shapes you see
            # Perhaps should fill in with 0s and ignore the results of those
            for i,(state,actions) in enumerate(zip(next_states,possible_actions_batch)):
                # print('STATE :')
                # display(draw_obs(state_batch_listdict[i]))
                # print('ACTION :')
                # print(action_batch[i])
                # print('REWARD :')
                # print(reward_batch[i])
                if state is None:
                    # print('Next state is None')
                    continue
                values = self.target(state, actions).squeeze()
                next_state_values[i] = values.max()
                # print('NEXT STATE :')
                # display(draw_obs(state))
                # print('POSSIBLE ACTIONS :', actions)
                # print('VALUES :', values)
                # print('MAX VALUES :', values.max())
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()
        self.update_target()
        
    def update_target(self):
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target.load_state_dict(target_net_state_dict)
        
    def listdict2dicttensor(self, state_batch):
        state_batch_dict = {key:value for key,value in state_batch[0].items()}
        for state in state_batch[1:]:
            for key in state_batch_dict.keys():
                state_batch_dict[key] = torch.cat((state_batch_dict[key], state[key]), dim=0)
        return state_batch_dict

#%%

class MPC_Agent:
    
    def __init__(self, n_rollouts, player_id):
        self.n_rollouts = n_rollouts
        self.id = player_id
    
    def action(self, state, kingdomino):
        actions = kingdomino.getPossibleActions()
        best_action = None
        best_result = -np.inf
        for action in actions:
            print(action)
            # do the action in deepcopied env
            # unroll with random policy
            result = 0
            for i in range(self.n_rollouts):
                print(i)
                kingdomino_copy = deepcopy(kingdomino)
                result += self.rollout(kingdomino_copy, action)
            if result > best_result:
                best_action = action
                
        return best_action

    def rollout(self, kingdomino, action):
        done = False
        Printer.print('Copied KDs order :', kingdomino.order)
        Printer.print('Copied KDs current player id :', kingdomino.current_player_id)
        while not done:
            terminated = kingdomino.step(action)
            done = terminated
            if not done:
                action = kingdomino.getRandomAction()
        scores = kingdomino.scores()
        final_result = self.id == np.argmax(scores)
        return final_result