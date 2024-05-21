import random
import torch.nn as nn
import torch
from abc import ABC, abstractmethod

from networks import PlayerFocusedFC, PlayerFocusedACNN
from prioritized_experience_replay import PrioritizedReplayBuffer
from agents.encoding import state_encoding
from agents.base import LearningAgent
from agents.encoding import ActionInterface  
from utils import cartesian_product

class DQN_Agent(LearningAgent):
    def __init__(self, 
                 n_players,
                 board_size,
                 batch_size,
                 eps_scheduler,
                 tau,
                 lr,
                 gamma,
                 id,
                 action_interface,
                 Network,
                 exploration_batch_size=1,
                 network_name=None,
                 network_hp=None,
                 replay_memory_size=None,
                 device='cpu',
                 policy=None,
                 target=None,
                 memory=None):
        # should be 1 if env on cpu
        self.exploration_batch_size = exploration_batch_size
        self.batch_size = batch_size
        self.eps_scheduler = eps_scheduler
        self.tau = tau
        self.lr = lr
        self.device = device
        self.n_players = n_players
        self.board_size = board_size
        self.id = id
        self.gamma = gamma
        self.memory = memory
        self.action_interface = action_interface
        self.network_hp = network_hp
        self.Network = Network
    
        self.policy = policy
        self.target = target
        self.setup_networks()
        
        self.target.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.train()
        self.reset()
    
    def setup_networks(self):
        n_actions = self.action_interface.n_actions
        if self.policy is None:
            self.policy = self.Network(self.n_players, self.board_size, self.network_hp, self.device, False, n_actions).to(self.device)
        if self.target is None:
            self.target = self.Network(self.n_players, self.board_size, self.network_hp, self.device, False, n_actions).to(self.device)

    def select_action(self, s, p_a):
        t,p = p_a # list of possible tiles, possible positions
        if self.training:
            sample = random.random()
            eps_threshold = self.eps_scheduler.eps() 
        if not self.training or sample > eps_threshold:
            # pick action with max qvalue among possible actions
            possible_a_mask = self.action_interface.encode(t, p)
            qvalues = self.policy(s).squeeze() # policy should output qvalues for EVERY action
            # 2 pbs:
                # returns id of the MASKED array so idx is wrong
                # if several same qvalue, then always the same taken
            masked_qvalues = qvalues.masked_fill(torch.from_numpy(~possible_a_mask).cuda(), float('-inf'))
            max_val = masked_qvalues.max()
            max_idxs = torch.where(masked_qvalues == max_val)[0]
            idx = int(random.choice(max_idxs))
            env_action = self.action_interface.decode(idx)
            torch_action = torch.zeros(possible_a_mask.shape[0], dtype=bool)
            torch_action[idx] = 1
            return env_action,torch_action # actual action,action to register in buffer
        t_idx = random.randint(0, len(t)-1)
        p_idx = random.randint(0, len(p)-1)
        return ((t[t_idx],p[p_idx]),
                self.action_interface.encode(
                    t[t_idx][None,],
                    p[p_idx][None,]))
    
    def train(self):
        self.training = True
        self.policy.train()
        self.target.train()
        
    def eval(self):
        self.training = False
        self.policy.eval()
        self.target.eval()
    
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        # Get batch
        if self.PER():
            batch, weights, tree_idxs = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
        
        boards_s,cur_tiles_s,prev_tiles_s, \
        a, r, \
        boards_next_s, cur_tiles_next_s, prev_tiles_next_s, \
        d, possible_a = batch
        
        # Transfer on accelerator
        boards_s = boards_s.to(self.device).float()
        cur_tiles_s = cur_tiles_s.to(self.device).float()
        prev_tiles_s = prev_tiles_s.to(self.device).float()
        a = a.to(self.device).bool()
        r = r.to(self.device)
        boards_next_s = boards_next_s.to(self.device).float()
        cur_tiles_next_s = cur_tiles_next_s.to(self.device).float()
        prev_tiles_next_s = prev_tiles_next_s.to(self.device).float()
        
        q = self.policy(
            {'Boards':boards_s, 
             'Current tiles':cur_tiles_s, 
             'Previous tiles': prev_tiles_s}, a).squeeze()
        q = q[a.nonzero(as_tuple=True)]
        
        next_s = {
            'Boards':boards_next_s, 
            'Current tiles':cur_tiles_next_s, 
            'Previous tiles':prev_tiles_next_s}
        with torch.no_grad():
            q_target = self.get_q_target(next_s, r, d, possible_a)
        
        with torch.no_grad():
            if self.PER():
                td_error = torch.abs(q - q_target)
                self.memory.update_priorities(tree_idxs, td_error.detach().cpu().numpy())
        if self.PER():
            loss = torch.mean(self.criterion(q, q_target) * weights.to(self.device))
        else:
            loss = torch.mean(self.criterion(q, q_target))
            
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()
        
        self.update_target()
        return {'DQN Loss':loss.item()}
        
    def get_q_target(self, next_s, r, d, p_a):
        next_q_exhaustive = self.target(next_s)
        masked_next_q = torch.masked_fill(next_q_exhaustive, ~p_a, float('-inf'))
        next_q = masked_next_q.max(1).values
        target = r + (1-d)*self.gamma*next_q
        return target
    
    @torch.no_grad()
    def update_target(self):
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target.load_state_dict(target_net_state_dict)
        

    

    
 #%%   
 
# class DQN_Agent_Loop(DQN_Agent_Base):
#     @torch.no_grad()
#     def select_action(self, s, kingdomino):
#         p_a = kingdomino.getPossibleActions()
#         p_a_torch = self.action_encoder.encode(p_a)
#         if self.training:
#             sample = random.random()
#             eps_threshold = self.eps_scheduler.eps() 
#         if not self.training or sample > eps_threshold:
#             n_p_a = actions_torch.shape[0]
#             qvalues = self.policy(
#                 (encoded_s['Boards'].repeat(repeats=(n_p_a,1,1,1)),
#                  encoded_s['Current tiles'].repeat(repeats=(n_p_a,1)),
#                  encoded_s['Previous tiles'].repeat(repeats=(n_p_a,1))), 
#                 p_a_torch).squeeze()
#             argmax = qvalues.argmax()
#             return actions[argmax],actions_torch[argmax]
#         i = random.randint(0, len(actions)-1)
#         return actions[i],actions_torch[i] 
    
#     @torch.no_grad()
#     def get_q_target(self, next_state, reward, done, possible_actions):
#         boards_next_state, cur_tiles_next_state, prev_tiles_next_state = next_state
#         # get size of possible actions
#         # useful later when reconstruction
#         possible_actions_sizes = torch.zeros(
#             self.batch_size, dtype=torch.int, device=self.device)
#         for i in range(self.batch_size):
#             possible_actions_sizes[i] = len(possible_actions[i])
            
#         possible_actions_torch = torch.cat(possible_actions)

#         boards_next_state = boards_next_state.repeat_interleave(
#             possible_actions_sizes,
#             dim=0)
#         cur_tiles_next_state = cur_tiles_next_state.repeat_interleave(
#             possible_actions_sizes,
#             dim=0)
#         prev_tiles_next_state = prev_tiles_next_state.repeat_interleave(
#             possible_actions_sizes,
#             dim=0)

#         all_next_state_action_values = self.target(
#             (boards_next_state.to(self.device), 
#               cur_tiles_next_state.to(self.device),
#               prev_tiles_next_state.to(self.device)), 
#             possible_actions_torch.to(self.device)).squeeze()
        
#         possible_actions_idxs = torch.zeros(
#             self.batch_size+1,
#             dtype=torch.int,
#             device=self.device)
#         possible_actions_idxs[1:] = torch.cumsum(
#             possible_actions_sizes, 0)
#         # next state values b*sum possible actions in batch x *action
#         next_state_values = torch.zeros(self.batch_size, device=self.device)
#         for i in range(self.batch_size):
#             next_state_values[i] = torch.max(
#                 all_next_state_action_values[possible_actions_idxs[i]:possible_actions_idxs[i+1]])
#         next_state_values = (1 - done.to(self.device)) * next_state_values
        
#         expected_state_action_values = (next_state_values * self.gamma) + reward.to(self.device)
#         return expected_state_action_values
    
if __name__ == '__main__':
    pass