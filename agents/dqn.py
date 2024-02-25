import random
import torch.nn as nn
import torch

from networks import PlayerFocusedFC, PlayerFocusedACNN
from prioritized_experience_replay import PrioritizedReplayBuffer
from encoding import state_encoding, actions_encoding
from agents.base import Player    

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
        elif network_name == 'PlayerFocusedFC':
            self.network = PlayerFocusedFC
        
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
        self.train()
        self.reset()
    
    def train(self):
        self.training = True
        self.policy.train()
        self.target.train()
        
    def eval(self):
        self.training = False
        self.policy.eval()
        self.target.eval()
    
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
        if self.training:
            sample = random.random()
            eps_threshold = self.eps_scheduler.eps() 
        if not self.training or sample > eps_threshold:
            with torch.no_grad():
                actions_size = actions_torch.shape[0]
                qvalues = self.policy(
                    (encoded_state['Boards'].repeat(repeats=(actions_size,1,1,1)),
                     encoded_state['Current tiles'].repeat(repeats=(actions_size,1)),
                     encoded_state['Previous tiles'].repeat(repeats=(actions_size,1))), 
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