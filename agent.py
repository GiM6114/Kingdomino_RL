import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch
from torch.nn.utils.rnn import pad_sequence
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
from graphics import draw_encoded_state
from prioritized_experience_replay import PrioritizedReplayBuffer
from encoding import state_encoding, actions_encoding

from kingdomino import Kingdomino

#%%

class Player:
    def action(self, state, kingdomino):
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
    
    
class DQN_Agent(Player):
    def __init__(self, 
                 n_players,
                 batch_size,
                 eps_scheduler,
                 tau,
                 lr,
                 gamma,
                 id,
                 exploration_batch_size=1,
                 network_name=None,
                 hp_archi=None,
                 replay_memory_size=None,
                 device='cpu',
                 policy=None,
                 target=None,
                 memory=None):
        # should be 1
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
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.memory = memory
        self.learning = True
        self.reset()
    
    def reset(self):
        self.n_steps = 0
        self.encoded_prev_state = None
        self.prev_action = None        
    
    # state : dict['Boards','Previous tiles','Current tiles']
    # state_encoding : same but one_hot_encoded
    def action(self, state, kingdomino):
        encoded_state = state_encoding(state, self.n_players, self.device)
        self.policy.filter_encoded_state(encoded_state)
        self.encoded_prev_state = encoded_state
        action,action_torch = self.select_action(encoded_state, kingdomino)
        self.prev_action = action_torch
        return action
    
    def select_action(self, encoded_state, kingdomino):
        actions = kingdomino.getPossibleActions()
        actions_torch = actions_encoding(actions, self.n_players, self.device)
        if encoded_state is None:
            return None,None
        if self.learning:
            sample = random.random()
            eps_threshold = self.eps_scheduler.eps() 
        if not self.learning or sample > eps_threshold:
            with torch.no_grad():
                qvalues = self.policy(
                    (encoded_state['Boards'],encoded_state['Current tiles'],encoded_state['Previous tiles']), 
                    actions_torch).squeeze()
                argmax = qvalues.argmax()
                return actions[argmax],actions_torch[argmax]
        i = random.randint(0, len(actions)-1)
        return actions[i],actions_torch[i]
    
    # TODO: kingdomino env should never output None reward and next state
    # TODO: convert next statehere and reuse in select action possible
    def give_reward(self, reward, next_state, done, possible_actions):
        # print('Previous state:')
        # display(draw_obs(self.prev_state))
        # print('State:')
        # display(draw_obs(next_state))
        encoded_next_state = state_encoding(next_state, self.n_players, self.device)
        self.policy.filter_encoded_state(encoded_next_state)
        self.memory.add((
            self.encoded_prev_state['Boards'].squeeze(),
            self.encoded_prev_state['Current tiles'].squeeze(),
            self.encoded_prev_state['Previous tiles'].squeeze(),
            self.prev_action.unsqueeze(0),
            torch.tensor([reward], device=self.device, dtype=torch.float32),
            encoded_next_state['Boards'].squeeze(),
            encoded_next_state['Current tiles'].squeeze(),
            encoded_next_state['Previous tiles'].squeeze(),
            done,
            actions_encoding(possible_actions, self.n_players)))
        self.optimize()
        
    # states: (boards, cur_tiles, prev_tiles)
    # boards: bs x boards_shape
    # cur_tiles: bs x cur_tiles_shape
    # prev_tiles: bs x prev_tiles_shape
    # possible_actions: bs x varying sizes x action shape
    # def get_expected_values(self, state, possible_actions):
    #     # possible_actions: bs x varying x action shape --> bs x max size x action shape
    #     # BUT can't loop over possible actions to get max :(
    #     # Good middle gorund: memory remembers the biggest possible actions and it can be used
    #     # here to augment possible actions
    #     state = boards,cur_tiles,prev_tiles
    #     # max_n_possible_actions = self.memory.max_n_possible_actions
    #     possible_actions = torch.zeros(self.batch_size, max_n_possible_actions, *possible_actions.shape)
    #     padded_possible_actions = pad_sequence(data, batch_first=True)
    #     pass
        
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        if isinstance(self.memory, PrioritizedReplayBuffer):
            batch, weights, tree_idxs = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
        
        boards_state,cur_tiles_state,prev_tiles_state, \
        action, reward, \
        boards_next_state, cur_tiles_next_state, prev_tiles_next_state, \
        done, possible_actions = batch
        
        state_action_values = self.policy(
            (boards_state.to(self.device), cur_tiles_state.to(self.device), prev_tiles_state.to(self.device)), 
            action.to(self.device)).squeeze()
        
        with torch.no_grad():
            # get size of possible actions
            # useful later when reconstruction
            possible_actions_sizes = torch.zeros(
                self.batch_size, dtype=torch.int, device=self.device)
            for i in range(self.batch_size):
                possible_actions_sizes[i] = len(possible_actions[i])

                
            possible_actions_torch = torch.cat(possible_actions)
    
            boards_next_state = boards_next_state.repeat_interleave(
                possible_actions_sizes,
                dim=0)
            cur_tiles_next_state = cur_tiles_next_state.repeat_interleave(
                possible_actions_sizes,
                dim=0)
            prev_tiles_next_state = prev_tiles_next_state.repeat_interleave(
                possible_actions_sizes,
                dim=0)
    
            all_next_state_action_values = self.target(
                (boards_next_state.to(self.device), 
                 cur_tiles_next_state.to(self.device),
                 prev_tiles_next_state.to(self.device)), 
                possible_actions_torch.to(self.device)).squeeze()
            
            possible_actions_idxs = torch.zeros(
                self.batch_size+1,
                dtype=torch.int,
                device=self.device)
            possible_actions_idxs[1:] = torch.cumsum(
                possible_actions_sizes, 0)
            # next state values b*sum possible actions in batch x *action
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            for i in range(self.batch_size):
                next_state_values[i] = torch.max(
                    all_next_state_action_values[possible_actions_idxs[i]:possible_actions_idxs[i+1]])
            next_state_values = (1 - done.to(self.device)) * next_state_values
            
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        # with torch.no_grad():
        #     # TODO: god awful loop, but problem bcs possible actions take different shapes you see
        #     # Perhaps should fill in with 0s and ignore the results of those
        #     for i in range(self.batch_size):
        #         if done[i]:
        #             # print('Episode is done')
        #             continue
        #         # print('STATE :')
        #         # display(draw_encoded_state(
        #         #     (boards_state[i],cur_tiles_state[i],prev_tiles_state[i])))
        #         # print('ACTION :')
        #         # print(action[i])
        #         # print('REWARD :')
        #         # print(reward[i])
        #         values = self.target(
        #             (boards_next_state[i].to(self.device), 
        #              cur_tiles_next_state[i].to(self.device),
        #              prev_tiles_next_state[i].to(self.device)), 
        #             possible_actions[i].to(self.device)).squeeze()
        #         next_state_values[i] = values.max()
        #         # print('NEXT STATE :')
        #         # display(draw_encoded_state((boards_next_state[i],cur_tiles_next_state[i],prev_tiles_next_state[i])))
        #         # print('POSSIBLE ACTIONS :', possible_actions[i])
        #         # print('VALUES :', values)
        #         # print('MAX VALUES :', values.max())
        expected_state_action_values = (next_state_values * self.gamma) + reward.to(self.device)
        #loss = self.criterion(state_action_values, expected_state_action_values)
        if isinstance(self.memory, PrioritizedReplayBuffer):
            td_error = self.criterion(state_action_values - expected_state_action_values).detach()
            self.memory.update_priorities(tree_idxs, td_error.cpu().numpy())
        if isinstance(self.memory, PrioritizedReplayBuffer):
            loss = torch.mean(self.criterion(state_action_values, expected_state_action_values) * weights.to(self.device))
        else:
            loss = torch.mean(self.criterion(state_action_values, expected_state_action_values))
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

class AggressivePlayer(Player):
    def action(self, state, kingdomino):
        actions = kingdomino.getPossibleActions()
    
def around_center(position):
    for point in position:
        if list(point) in [[4,3],[3,4],[4,5],[5,4]]:
            return True
    return False
    
class SelfCenteredPlayer(Player):
    # Heuristic: min nb of territories
    # + when equality favor immediate tile placement not near center
    # could also favor next play not near center for further finetuning
    def action(self, state, kingdomino):
        if self.previous_tile is None:
            return random.choice(kingdomino.getPossibleActions())
        best_action = None
        lowest_n_territories = 999
        best_near_center = False
        best_near_center_2 = False
        actions = kingdomino.getPossibleActions()
        for action in actions:
            tile_choice,position = action
            positionT = position.T
            if not (position == Kingdomino.discard_tile).all():
                board_values = self.board.board[positionT[0],positionT[1]]
                self.board.board[positionT[0],positionT[1]] = self.previous_tile[:2]
            next_tile = kingdomino.current_tiles[tile_choice].tile
            next_tile_positions = kingdomino.getPossiblePositions(next_tile, every_pos=True)
            # print('Player', kingdomino.current_player_id, 'playing according to kingdomino')
            # print('Action', action)
            # print('Next tile', next_tile)
            for next_tile_position in next_tile_positions:
                next_tile_positionT = next_tile_position.T
                if not (next_tile_position == Kingdomino.discard_tile).all():
                    board_new_values = self.board.board[next_tile_positionT[0],next_tile_positionT[1]]
                    self.board.board[next_tile_positionT[0],next_tile_positionT[1]] = next_tile[:2]
                territories = self.board.getTerritories()
                n_territories = len(territories)
                # print(territories)
                # print('Next pos', next_tile_position)
                # print('n territories', n_territories)
                if (n_territories < lowest_n_territories) or (n_territories == lowest_n_territories and ((best_near_center and not around_center(position)) or (not best_near_center and best_near_center_2 and not around_center(next_tile_position)))):
                    best_action = action
                    lowest_n_territories = n_territories
                    best_near_center = around_center(position)
                    best_near_center_2 = around_center(next_tile_position)
                if not (next_tile_position == Kingdomino.discard_tile).all():
                    self.board.board[next_tile_positionT[0],next_tile_positionT[1]] = board_new_values
            if not (position == Kingdomino.discard_tile).all():
                self.board.board[positionT[0],positionT[1]] = board_values
        return best_action
        
    
    
#%%

class MPC_Agent(Player):
    
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