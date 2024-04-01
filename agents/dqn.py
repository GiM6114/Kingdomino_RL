import random
import torch.nn as nn
import torch

from networks import PlayerFocusedFC, PlayerFocusedACNN
from prioritized_experience_replay import PrioritizedReplayBuffer
from agents.encoding import state_encoding
from agents.base import Player  
from agents.encoding import ActionInterface  
from utils import cartesian_product

class DQN_Agent_Base(Player):
    def __init__(self, 
                 n_players,
                 batch_size,
                 eps_scheduler,
                 tau,
                 lr,
                 gamma,
                 id,
                 action_interface,
                 double=False,
                 exploration_batch_size=1,
                 network_name=None,
                 hp_archi=None,
                 replay_memory_size=None,
                 device='cpu',
                 policy=None,
                 policy2=None,
                 target=None,
                 target2=None,
                 memory=None):
        # should be 1 if env on cpu
        self.exploration_batch_size = exploration_batch_size
        self.batch_size = batch_size
        self.eps_scheduler = eps_scheduler
        self.tau = tau
        self.lr = lr
        self.device = device
        self.n_players = n_players
        self.id = id
        self.gamma = gamma
        self.memory = memory
        self.action_interface = action_interface
        self.hp_archi = hp_archi
        self.double = double
    
        self.policy = policy
        if self.double:
            self.policy2 = policy2
        self.target = target
        if self.double:
            self.target2 = target2
        self.setup_networks()
        
        self.target.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        if self.double:
            self.optimizer2 = torch.optim.AdamW(self.policy2.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.train()
        self.reset()
        
    def setup_networks(self):
        raise NotImplementedError()
    
    def train(self):
        self.training = True
        self.policy.train()
        self.target.train()
        if self.double:
            self.policy2.train()
            self.target2.train()
        
    def eval(self):
        self.training = False
        self.policy.eval()
        self.target.eval()
        if self.double:
            self.policy2.eval()
            self.target2.eval()
    
    def select_action(self, s, kingdomino):
        raise NotImplementedError()
        
    def get_q_target(self, next_s, r, d, p_a):
        raise NotImplementedError()

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
        boards_s = boards_s.ti(self.device)
        cur_tiles_s = cur_tiles_s.to(self.device)
        prev_tiles_s = prev_tiles_s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        
        q = self.policy(
            (boards_s, cur_tiles_s, prev_tiles_s), a).squeeze()
        if self.double:
            q2 = self.policy2(
                (boards_s, cur_tiles_s, prev_tiles_s), a).squeeze()
        
        next_s = (boards_next_s, cur_tiles_next_s, prev_tiles_next_s)
        with torch.no_grad():
            q_target = self.get_q_target(next_s, r, d, possible_a)
        
        with torch.no_grad():
            if self.PER():
                td_error = torch.abs(q - q_target)
                self.memory.update_priorities(tree_idxs, td_error.detach().cpu().numpy())
        if self.PER():
            loss = torch.mean(self.criterion(q, q_target) * weights.to(self.device))
            if self.double:
                loss2 = torch.mean(self.criterion(q2, q_target) * weights.to(self.device))
        else:
            loss = torch.mean(self.criterion(q, q_target))
            if self.double:
                loss2 = torch.mean(self.criterion(q, q_target))
            
        self.optimizer.zero_grad()
        if self.double:
            self.optimizer2.zero_grad()
        loss.backward()
        if self.double:
            loss2.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()
        if self.double:
            self.optimizer2.step()
        
        self.update_target()
        
    @torch.no_grad()
    def update_target(self):
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        if self.double:
            target_net_state_dict2 = self.target2.state_dict()
            policy_net_state_dict2 = self.policy2.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            if self.double:
                target_net_state_dict2[key] = policy_net_state_dict2[key]*self.tau + target_net_state_dict2[key]*(1-self.tau)
        self.target.load_state_dict(target_net_state_dict)
        if self.double:
            self.target2.load_state_dict(target_net_state_dict2)
        

    
class DQN_Agent_FC(DQN_Agent_Base):
    def setup_networks(self):
        self.Network = PlayerFocusedFC
        if self.policy is None:
            self.policy = self.Network(self.n_players, self.hp_archi, self.device)
        if self.target is None:
            self.target = self.Network(self.n_players, self.hp_archi, self.device)
        if self.double:
            if self.policy2 is None:
                self.policy2 = self.Network(self.n_players, self.hp_archi, self.device)
            if self.target2 is None:
                self.target2 = self.Network(self.n_players, self.hp_archi, self.device)
    
    def select_action(self, s, kingdomino):
        t,p = kingdomino.getPossibleTilesPositions() # list of possible tiles, possible positions
        if self.training:
            sample = random.random()
            eps_threshold = self.eps_scheduler.eps() 
        if not self.training or sample > eps_threshold:
            # pick action with max qvalue among possible actions
            possible_a_mask = self.action_interface.encode_batch(t, p)
            qvalues = self.policy(s) # policy should output qvalues for EVERY action
            idx = qvalues[possible_a_mask].max(1).indices.view(1, 1)
            return self.action_interface.decode(idx),possible_a_mask[idx] # actual action,action to register in buffer
        t_idx = random.randint(0, t.shape[0]-1)
        p_idx = random.randint(0, p.shape[0]-1)
        return (t[t_idx],p[p_idx]),self.action_interface.encode(t[t_idx],p[p_idx])
    
    def get_q_target(self, next_s, r, d, p_a):
        next_q_exhaustive = self.target(next_s)
        if self.double:
            next_q_exhaustive2 = self.target2(next_s)
        for i in range(self.batch_size):
            next_q = torch.max(next_q_exhaustive[i,p_a])
            if self.double:
                next_q = torch.min(next_q, torch.max(next_q_exhaustive2[i,p_a]))
        expected_next_q = r + (1-d)*self.gamma*next_q
        return expected_next_q

    
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