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
                 exploration_batch_size=1,
                 network_name=None,
                 hp_archi=None,
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
        self.id = id
        self.gamma = gamma
        self.memory = memory
        self.action_interface = action_interface
        self.hp_archi = hp_archi

        self.policy = policy
        self.target = target
        self.setup_networks()
        
        self.target.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.train()
        self.reset()
        
    def setup_networks(self):
        raise NotImplementedError()
    
    def train(self):
        self.training = True
        self.policy.train()
        self.target.train()
        
    def eval(self):
        self.training = False
        self.policy.eval()
        self.target.eval()
    
    def reset(self):
        self.encoded_prev_s = None
        self.prev_a = None
        
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
    
    def select_action(self, s, kingdomino):
        raise NotImplementedError()
        
    def get_q_target(self, next_s, r, d, p_a):
        raise NotImplementedError()
    
    
    def PER(self):
        return isinstance(self.memory, PrioritizedReplayBuffer)

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        if self.PER():
            batch, weights, tree_idxs = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
        
        boards_state,cur_tiles_state,prev_tiles_state, \
        action, reward, \
        boards_next_state, cur_tiles_next_state, prev_tiles_next_state, \
        done, possible_actions = batch
        
        q = self.policy(
            (boards_state.to(self.device), cur_tiles_state.to(self.device), prev_tiles_state.to(self.device)), 
            action.to(self.device)).squeeze()
        
        next_state = (boards_next_state, cur_tiles_next_state, prev_tiles_next_state)
        with torch.no_grad():
            q_target = self.get_q_target(next_state, reward, done, possible_actions)
        
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
        
    
    @torch.no_grad()
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
    
class DQN_Agent_FC(DQN_Agent_Base):
    def setup_networks(self):
        self.Network = PlayerFocusedFC
        if self.policy is None:
            self.policy = self.Network(self.n_players, self.hp_archi, self.device)
        if self.target is None:
            self.target = self.Network(self.n_players, self.hp_archi, self.device)
    
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
        next_s_a_values_exhaustive = self.policy(next_s)
        expected_next_s_a_values = torch.zerso(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            next_s_value = (1 - d[i]) * torch.max(next_s_a_values_exhaustive[i,p_a])
            expected_next_s_a_values[i] = (next_s_value * self.gamma) + r[i].to(self.device)
        return expected_next_s_a_values
    
 #%%   
 
class DQN_Agent_Loop(DQN_Agent_Base):
    @torch.no_grad()
    def select_action(self, s, kingdomino):
        p_a = kingdomino.getPossibleActions()
        p_a_torch = self.action_encoder.encode(p_a)
        if self.training:
            sample = random.random()
            eps_threshold = self.eps_scheduler.eps() 
        if not self.training or sample > eps_threshold:
            n_p_a = actions_torch.shape[0]
            qvalues = self.policy(
                (encoded_s['Boards'].repeat(repeats=(n_p_a,1,1,1)),
                 encoded_s['Current tiles'].repeat(repeats=(n_p_a,1)),
                 encoded_s['Previous tiles'].repeat(repeats=(n_p_a,1))), 
                p_a_torch).squeeze()
            argmax = qvalues.argmax()
            return actions[argmax],actions_torch[argmax]
        i = random.randint(0, len(actions)-1)
        return actions[i],actions_torch[i] 
    
    @torch.no_grad()
    def get_q_target(self, next_state, reward, done, possible_actions):
        boards_next_state, cur_tiles_next_state, prev_tiles_next_state = next_state
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
        
        expected_state_action_values = (next_state_values * self.gamma) + reward.to(self.device)
        return expected_state_action_values
    
if __name__ == '__main__':
    pass