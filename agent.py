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

from board import Board
from setup import N_TILE_TYPES
from networks import Shared, NeuralNetwork
from printer import Printer

class Pipeline:
    def __init__(self, player, env):
        self.player = player
        self.env = env


class HumanPlayer:      
    def action(self, state, kingdomino):
        tile_id = int(input("Which tile do you choose ?"))
        x1 = int(input("x1 ? "))
        y1 = int(input("y1 ? "))      
        x2 = int(input("x2 ? "))
        y2 = int(input("y2 ? "))
        return tile_id, (x1,y1,x2,y2)


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

class DQN(nn.Module):
    def __init__(self,
                 n_players,
                 network_info,
                 output_size,
                 device):
        super().__init__()
        self.shared = Shared(
            n_players=n_players,
            network_info=network_info,
            output_size=network_info.shared_rep_size,
            device=device).to(device)
        self.action_and_shared = NeuralNetwork(
            input_size=network_info.shared_rep_size + 2 + 2 + n_players,
            output_size=1,
            l=network_info.shared_l,
            n=network_info.shared_n)
        
    # state : batch_size * messy (batch_size = 1 ==> forward with multiple actions)
    # action: batch_size * 6 OR #possible_actions * 6
    def forward(self, state, action):
        state = self.shared(state) # batch_size * shared_rep_size
        if state.shape[0] == 1:
            # case where several actions applied to a single state
            state = state.repeat(repeats=(action.size(dim=0),1))
        x = torch.cat((state,action), dim=1)
        x = self.action_and_shared(x)
        return x
        

class DQN_Agent:
    
    def __init__(self, 
                 n_players, 
                 batch_size,
                 eps_scheduler,
                 tau,
                 lr,
                 gamma,
                 id,
                 network_info=None,
                 replay_memory_size=None,
                 device='cpu',
                 policy=None,
                 target=None,
                 memory=None):
        self.batch_size = batch_size
        self.eps_scheduler = eps_scheduler
        self.tau = tau
        self.lr = lr
        self.device = device
        self.n_players = n_players
        self.id = id
        self.gamma = gamma
        
        self.policy = policy
        if self.policy is None:
            self.policy = DQN(
                n_players=self.n_players,
                network_info=network_info,
                output_size=1,
                device=self.device).to(self.device)
        self.target = target
        if self.target is None:
            self.target = DQN(
                n_players=self.n_players,
                network_info=network_info,
                output_size=1,
                device=self.device).to(self.device)
            self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        self.memory = memory
        if self.memory is None:
            self.memory = ReplayMemory(replay_memory_size)
        self.learning = True
        self.reset()
    
    def reset(self):
        self.n_steps = 0
        
    def convert_state_torch(self, state):
        if state is None or torch.is_tensor(state['Boards']):
            return state
        for k in state:
            state[k] = torch.tensor(
                state[k], device=self.device, dtype=torch.int64).unsqueeze(0)
        return state
    
    def convert_actions_torch(self, actions):
        actions_conc = [np.concatenate(([action[0]],action[1].reshape(1,-1).squeeze())) for action in actions]
        # convert first action nb to one_hot
        actions_conc = torch.tensor(np.array(actions_conc), device=self.device, dtype=torch.int64)
        actions_conc_one_hot_tile_id = \
            torch.zeros((actions_conc.size()[0], self.n_players+2+2), device=self.device, dtype=torch.int64)
        actions_conc_one_hot_tile_id[:,self.n_players:] = actions_conc[:,1:]
        actions_conc_one_hot_tile_id[:,:self.n_players] = F.one_hot(actions_conc[:,0], num_classes=self.n_players)
        return actions_conc_one_hot_tile_id
    
    def select_action(self, state, kingdomino):
        actions = kingdomino.getPossibleActions()
        actions_torch = self.convert_actions_torch(actions)
        self.prev_possible_actions = actions_torch
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
    
    def action(self, state, kingdomino):
        state = self.convert_state_torch(state)
        self.prev_state = state
        action,action_torch = self.select_action(state, kingdomino)
        self.prev_action = action_torch
        return action
        
    def give_reward(self, reward, next_state):
        self.memory.push(
            self.prev_state, 
            self.prev_action.unsqueeze(0),
            torch.tensor([reward], device=self.device),
            self.convert_state_torch(next_state),
            self.prev_possible_actions)
        self.optimize()
        
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, 
                      batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]
        state_batch = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_batch = self.listdict2dicttensor(state_batch)
        #non_final_next_states = self.listdict2dicttensor(non_final_next_states)
        possible_actions_batch = batch.possible_actions
        
        state_action_values = self.policy(state_batch, action_batch).squeeze()
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        with torch.no_grad():
            # TODO: problem states list of dict, cant apply non_final_mask directly
            next_state_values_non_final = torch.zeros(torch.sum(non_final_mask), device=self.device)
            # TODO: awful loop, but problem bcs possible actions take different shapes you see
            for i,(state,actions) in enumerate(zip(non_final_next_states,possible_actions_batch)):
                if state is None:
                    continue
                values = self.target(state, actions).squeeze()
                next_state_values_non_final[i] = values.max()
            next_state_values[non_final_mask] = next_state_values_non_final
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
    
    def action(self, kingdomino):
        actions = kingdomino.getPossibleActions()
        best_action = None
        best_result = -np.inf
        for action in actions:
            # do the action in deepcopied env
            # unroll with random policy
            result = 0
            for i in range(self.n_rollouts):
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
      


NetworksInfo = namedtuple('NetworksInfo', 
                          ['in_channels', 
                           'output_size',
                           'conv_channels',
                           'conv_l',
                           'conv_kernel_size', 
                           'conv_stride',
                           'pool_place',
                           'pool_kernel_size', 
                           'pool_stride',
                           'board_rep_size',
                           'board_fc_n',
                           'board_fc_l',
                           'player_rep_size',
                           'board_prev_tile_fc_l',
                           'shared_rep_size',
                           'shared_l',
                           'shared_n',
                           'actor_l',
                           'actor_n',
                           'critic_l',
                           'critic_n'])
      
#%%
class PlayerAC:
    def __init__(self, 
                 n_players, batch_size, 
                 gamma, lr_a, lr_c, coordinate_std,
                 network_info,
                 device = 'cpu',
                 critic = None,
                 actor = None):
        self.n_players = n_players
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        
        self.critic = critic
        if self.critic is None:
            self.critic = Shared(
                n_players = n_players, 
                network_info = network_info,
                output_size = 1,
                device = self.device).to(device)
        self.actor = actor
        if self.actor is None:
            self.actor = Shared(
                n_players = n_players,
                network_info = network_info,
                output_size = 3+4+n_players,
                device=self.device).to(device)
        # 2 : mean pos, 1 : std pos, 4 : direction other tile, n_players : which tile
    
        self.criterion_c = nn.SmoothL1Loss()
        
        self.optimizer_c = torch.optim.AdamW(
            self.critic.parameters(),
            amsgrad=True,
            lr=lr_c)
        self.optimizer_a = torch.optim.AdamW(
            self.actor.parameters(),
            amsgrad=True,
            lr=lr_a)
        
        self.learning = True
        self.reset()
        
    def reset(self):
        self.rewards = 0
        self.tile_choice_distributions = None
        self.tiles_chosen = None
        self.coordinates_distributions = None
        self.coordinates_chosen = None
        self.values = torch.zeros(self.batch_size, device=self.device)
        self.first_step = True
    
    def action(self, states, dones):
        with torch.set_grad_enabled(self.learning):
            not_dones = np.invert(dones)
            all_dones = (dones == True).all()
            with torch.no_grad():
                for key in states:
                    states[key] = torch.tensor(states[key], dtype=torch.int64, device=self.device)
                    batch_size = states[key].size()[0]
                    states[key] = states[key][not_dones]
    
            if self.learning:        
                self.prev_values = self.values
                self.values = torch.zeros(batch_size, device=self.device)
                if not all_dones:
                    self.values[not_dones] = self.critic(states).squeeze()
    
            # check necessary bcs np gradient first step
            # when game restarts afterwards, there will be gradient
            # that will be useless in the transition, but no updates
            # as target = prev_value = 0
            if self.learning and not self.first_step:
                self.optimize()
            
            if not all_dones:
                action_outputs = self.actor(states)
                action = self.network_outputs_to_action(action_outputs)
                return action
            else:
                return -torch.ones((batch_size, 9))
    
    def optimize(self):
        # like DQL, "half-gradient"
        with torch.no_grad():
            targets = self.rewards + self.gamma*self.values
        # Critic update
        loss_c = self.criterion_c(
            targets,
            self.prev_values)
        self.optimizer_c.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_([p for g in self.optimizer_c.param_groups for p in g["params"]], 0.5) # gradient clipping
        self.optimizer_c.step()

        # Actor update
        # product of proba as tile choice and coordinates chosen for previous tiles
        # assumeed independant (they are not)
        loss_a = (
            (
            -self.coordinates_distributions.log_prob(self.coordinates_chosen) \
            -self.tile_choice_distributions.log_prob(self.tile_chosen) \
            -self.tile_choice_distributions.log_prob(self.tile_chosen)
            )
            * (targets - self.prev_values)).mean()
        self.optimizer_a.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_([p for g in self.optimizer_a.param_groups for p in g["params"]], 0.5) # gradient clipping
        self.optimizer_a.step()
        
    def network_outputs_to_action(self, outputs):
        batch_size = outputs.size()[0]
        
        coordinates_outputs_mean = outputs[:,:2]
        if self.learning:
            coordinates_outputs_std = outputs[:,2]
            coordinates_outputs_cov = torch.eye(2).to(self.device).repeat(batch_size,1,1) * (torch.exp(coordinates_outputs_std)**2).unsqueeze(1).repeat(1,4).reshape(batch_size,2,2)
            self.coordinates_distributions = distributions.MultivariateNormal(
                torch.exp(coordinates_outputs_mean),
                coordinates_outputs_cov)
            self.coordinates_chosen = torch.round(self.coordinates_distributions.sample())
        else:
            self.coordinates_chosen = torch.round(coordinates_outputs_mean)
       
        # prendre en compte self.learning
        if self.learning:
            direction_output_distribution = F.softmax(outputs[:,3:7], dim=1)
            self.direction_output_distribution = distributions.Multinomial(total_count=1, probs=direction_output_distribution)
            self.direction_chosen = self.direction_output_distribution.sample()
        else:
            self.direction_chosen = outputs[:,3:7].argmax(dim=1)
        
        if self.learning:
            tile_output_distribution = F.softmax(outputs[:,7:], dim=1)
            self.tile_choice_distributions = distributions.Multinomial(total_count=1, probs=tile_output_distribution)
            self.tile_chosen = self.tile_choice_distributions.sample()
        else:
            self.tile_chosen = outputs[:,7:].argmax(dim=1)
        
        if self.learning:
            return torch.column_stack((
                self.coordinates_chosen,
                self.direction_chosen.nonzero()[:,1],
                self.tile_chosen.nonzero()[:,1]))
        else:
            return torch.column_stack((
                self.coordinates_chosen,
                self.direction_chosen,
                self.tile_chosen))
        
    def give_reward(self, rewards):
        self.rewards = torch.tensor(rewards)

#%%

class PlayerAC_base:
    def __init__(self, 
                 n_players, 
                 gamma, lr_a, lr_c, coordinate_std,
                 network_info):
        
        self.gamma = gamma
        self.shared = Shared(
            n_players, 
            network_info)

        self.critic = NeuralNetwork(
            input_size=network_info.shared_rep_size, 
            output_size=1, 
            l=network_info.critic_l, 
            n=network_info.critic_n)
        self.actor = NeuralNetwork(
            input_size=network_info.shared_rep_size, 
            # 2 continuous outputs for first tile
            # 4 other possible positions (n,w,s,e)
            # n_players : which tile to pick
            output_size=2+4+n_players, 
            # coordinates to place previous tile + which tile to take
            l=network_info.actor_l, 
            n=network_info.actor_n)
   
        self.criterion_c = nn.SmoothL1Loss()
        self.reset()    
        
    def optimize(self):
        targets = self.rewards + self.gamma*self.values
        # Critic update
        loss_c = self.criterion_c(
            targets,
            self.prev_values)
        self.optimizer_c.zero_grad()
        loss_c.backward()
        self.optimizer_c.step()

        # Actor update
        # product of proba as tile choice and coordinates chosen for previous tiles
        # assumeed independant (they are not)
        loss_a = torch.mean((-self.coordinates_distributions.log_prob(self.coordinates_chosen) \
                  -self.tile_choice_distributions.log_prob(self.tile_chosen))) * loss_c
        self.optimizer_a.zero_grad()
        loss_a.backward()
        self.optimizer_a.step()
    
    def network_outputs_to_action(self, outputs):
        batch_size = outputs.size()[0]
        self.coordinates_cov = torch.eye(2).repeat(batch_size,1,1) * self.coordinate_std**2
        
        coordinates_outputs = outputs[:,:2]
        self.coordinates_distributions = distributions.MultivariateNormal(coordinates_outputs, self.coordinates_cov)
        self.coordinates_chosen = torch.round(self.coordinates_distributions.sample())
        direction_output_distribution = F.softmax(outputs[:,2:6], dim=1)
        self.direction_output_distribution = distributions.Multinomial(total_count=1, probs=direction_output_distribution)
        self.direction_chosen = self.direction_output_distribution.sample()
        tile_output_distribution = F.softmax(outputs[:,6:], dim=1)
        self.tile_choice_distributions = distributions.Multinomial(total_count=1, probs=tile_output_distribution)
        self.tile_chosen = self.tile_choice_distributions.sample()
        
        return torch.column_stack((
            self.coordinates_chosen, 
            self.direction_chosen.nonzero()[:,1], 
            self.tile_chosen.nonzero()[:,1]))
    
    def reset(self):
        self.rewards = None
        self.tile_choice_distributions = None
        self.tiles_chosen = None
        self.coordinates_distributions = None
        self.coordinates_chosen = None
        self.values = None
        self.first_step = True
        
    def give_reward(self, rewards):
        self.rewards = torch.tensor(rewards)
 
class PlayerAC_shared(PlayerAC_base):
    
    def __init__(self, 
                 batch_size,
                 n_players, 
                 gamma, lr_a, lr_c, coordinate_std,
                 network_info):
        
        super().__init__(
            n_players, 
            gamma, lr_a, lr_c, coordinate_std,
            network_info)

        self.optimizer_c = torch.optim.AdamW(
            list(self.shared.parameters()) + list(self.critic.parameters()),
            amsgrad=True,
            lr=lr_c)
        self.optimizer_a = torch.optim.AdamW(
            list(self.shared.parameters()) + list(self.actor.parameters()),
            amsgrad=True,
            lr=lr_a)
    
    def action(self, states, dones):
        with torch.no_grad():
            for key in states:
                states[key] = torch.tensor(states[key], dtype=torch.int64)

        shared_rep_c = self.shared(states)
        
        if not self.first_step:
            self.prev_values = self.values
        self.values = self.critic(shared_rep_c).squeeze()

        if not self.first_step:
            targets = self.rewards + self.gamma*self.values
            # Critic update
            loss_c = self.criterion_c(
                targets,
                self.prev_values)
            self.optimizer_c.zero_grad()
            loss_c.backward()
            self.optimizer_c.step()

            # Actor update
            # product of proba as tile choice and coordinates chosen for previous tiles
            # assumeed independant (they are not)
            loss_a = torch.mean((-self.coordinates_distributions.log_prob(self.coordinates_chosen) \
                      -self.tile_choice_distributions.log_prob(self.tile_chosen))) * loss_c
            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()
        
        shared_rep_a = self.shared(states)
        action_outputs = self.actor(shared_rep_a)
        action = self.network_outputs_to_action(action_outputs)
        return action
            



    
 
    
class PlayerAC_shared_critic_trained(PlayerAC_base):
    
    def __init__(self, 
                 batch_size,
                 n_players, 
                 gamma, lr_a, lr_c, coordinate_std,
                 network_info):
        
        super().__init__(
            batch_size, 
            n_players, 
            gamma, lr_a, lr_c, coordinate_std,
            network_info)
        self.optimizer_c = torch.optim.AdamW(
            list(self.shared.parameters()) + list(self.critic.parameters()),
            amsgrad=True,
            lr=lr_c)
        self.optimizer_a = torch.optim.AdamW(
            self.actor.parameters(),
            amsgrad=True,
            lr=lr_a)
        self.coordinate_std = coordinate_std
                
    
    def action(self, states, dones):
        with torch.no_grad():
            for key in states:
                states[key] = torch.tensor(states[key], dtype=torch.int64)
                batch_size = states[key].size()[0]
          
        
        shared_rep_c = self.shared(states)
        # if states['Boards'] is None:
        #     self.values = 
        
        if not self.first_step:
            with torch.no_grad():
                self.prev_values = self.values.clone()
        self.values = self.critic(shared_rep_c).squeeze()

        if not self.first_step:
            
            targets = self.rewards + self.gamma*self.values
            # Critic update
            loss_c = self.criterion_c(
                targets,
                self.prev_values)
            self.optimizer_c.zero_grad()
            loss_c.backward()
            self.optimizer_c.step()

            with torch.no_grad():
                loss_c_a = loss_c.clone()
            # Actor update
            # product of proba as tile choice and coordinates chosen for previous tiles
            # assumed independant (they are not)
            loss_a = torch.mean((-self.coordinates_distributions.log_prob(self.coordinates_chosen) \
                      -self.tile_choice_distributions.log_prob(self.tile_chosen) \
                      -self.direction_output_distribution.log_prob(self.direction_chosen))) * loss_c_a
            try:
                for p in self.actor.parameters():
                    print('Before zero grad :', p.grad.norm())
            except:
                pass
            self.optimizer_a.zero_grad()
            try:
                for p in self.actor.parameters():
                    print('After zero grad :', p.grad.norm())
            except:
                print('No grad')
            loss_a.backward()
            self.optimizer_a.step()
            for p in self.actor.parameters():
                print('After backward :', p.grad.norm())

        self.first_step = False
        
        with torch.no_grad():
            shared_rep_a = shared_rep_c.clone()
        action_outputs = self.actor(shared_rep_a)
        action = self.network_outputs_to_action(action_outputs)
        return action


class PlayerAC_shared_actor_trained(PlayerAC_base):
    
    def __init__(self, 
                 batch_size,
                 n_players, 
                 gamma, lr_a, lr_c, coordinate_std,
                 network_info):
        
        super().__init__(
            batch_size, 
            n_players, 
            gamma, lr_a, lr_c, coordinate_std,
            network_info)
        self.optimizer_c = torch.optim.AdamW(
            self.critic.parameters(),
            amsgrad=True,
            lr=lr_c)
        self.optimizer_a = torch.optim.AdamW(
            list(self.shared.parameters()) + list(self.actor.parameters()),
            amsgrad=True,
            lr=lr_a)
        
    def action(self, states):
        with torch.no_grad():
            for key in states:
                states[key] = torch.tensor(states[key], dtype=torch.int64)
            
        shared_rep_a = self.shared(states)
        action_outputs = self.actor(shared_rep_a)
        coordinates_outputs = action_outputs[:,:2]
        coordinates_distributions = distributions.MultivariateNormal(coordinates_outputs, self.coordinates_cov)
        coordinates_chosen = torch.round(coordinates_distributions.sample())
        tile_output_distribution = F.softmax(action_outputs[:,2:], dim=1)
        tile_choice_distributions = distributions.Multinomial(total_count=1, probs=tile_output_distribution)
        tile_chosen = tile_choice_distributions.sample()
        action = torch.column_stack((coordinates_chosen,tile_chosen.nonzero()[:,1]))
        
        self.prev_values = self.values
        with torch.no_grad():
            shared_rep_c = shared_rep_a.clone()
        self.values = self.critic(shared_rep_c).squeeze()

        if self.prev_values is not None:
            targets = self.rewards + self.gamma*self.values
            # Critic update
            loss_c = self.criterion_c(
                targets,
                self.prev_values)
            self.optimizer_c.zero_grad()
            loss_c.backward()
            self.optimizer_c.step()

            # Actor update
            # product of proba as tile choice and coordinates chosen for previous tiles
            # assumeed independant (they are not)
            loss_a = torch.mean((-self.coordinates_distributions.log_prob(self.coordinates_chosen) \
                      -self.tile_choice_distributions.log_prob(self.tile_chosen))) * loss_c
            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()
            
        self.coordinates_chosen = coordinates_chosen
        self.tile_chosen = tile_chosen
        self.coordinates_distributions = coordinates_distributions
        self.tile_choice_distributions = tile_choice_distributions
        
        with torch.no_grad():
            shared_rep_a = shared_rep_c.clone()

        return action


