import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch
torch.autograd.set_detect_anomaly(True)
from collections import namedtuple
# from torchviz import make_dot
# from IPython.display import display

from board import Board
from setup import N_TILE_TYPES
from networks import Shared, NeuralNetwork


class Pipeline:
    def __init__(self, player, env):
        self.player = player
        self.env = env


class HumanPlayer:      
    def action(self, state):
        tile_id = input("Which tile do you choose ?")
        x1 = int(input("x1 ? "))
        y1 = int(input("y1 ? "))      
        x2 = int(input("x2 ? "))
        y2 = int(input("y2 ? "))
        return tile_id, (x1,y1,x2,y2)


class RandomPlayer:
    def action(self, state, kingdomino):
        return (kingdomino.selectTileRandom(),
                kingdomino.selectTilePositionRandom())

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

class PlayerAC:
    def __init__(self, 
                 n_players, batch_size, 
                 gamma, lr_a, lr_c, coordinate_std,
                 network_info):
        self.batch_size = batch_size
        self.gamma = gamma
        self.critic = Shared(
            n_players, 
            network_info,
            1)
        self.actor = Shared(
            n_players,
            network_info,
            3+4+n_players) 
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
        
        self.reset()
        
    def reset(self):
        self.rewards = 0
        self.tile_choice_distributions = None
        self.tiles_chosen = None
        self.coordinates_distributions = None
        self.coordinates_chosen = None
        self.values = torch.zeros(self.batch_size)
        self.first_step = True
    
    def action(self, states, dones):
        not_dones = np.invert(dones)
        with torch.no_grad():
            for key in states:
                states[key] = torch.tensor(states[key], dtype=torch.int64)
                batch_size = states[key].size()[0]
                states[key] = states[key][not_dones]

                
        self.prev_values = self.values
        self.values = torch.zeros(batch_size)
        self.values[not dones] = self.critic(states).squeeze()

        # check necessary bcs np gradient first step
        # when game restarts afterwards, there will be gradient
        # that will be useless in the transition, but no updates
        # as target = prev_value = 0
        if not self.first_step:
            self.optimize()
        
        action_outputs = self.actor(states)
        action = self.network_outputs_to_action(action_outputs)
        return action
    
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
        coordinates_outputs_std = outputs[:,2]
        print('mean :', torch.exp(coordinates_outputs_mean))
        print('std :', torch.exp(coordinates_outputs_std))
        coordinates_outputs_cov = torch.eye(2).repeat(batch_size,1,1) * torch.exp(coordinates_outputs_std)**2
        self.coordinates_distributions = distributions.MultivariateNormal(
            torch.exp(coordinates_outputs_mean),
            coordinates_outputs_cov)
        self.coordinates_chosen = torch.round(self.coordinates_distributions.sample())
        direction_output_distribution = F.softmax(outputs[:,3:7], dim=1)
        self.direction_output_distribution = distributions.Multinomial(total_count=1, probs=direction_output_distribution)
        self.direction_chosen = self.direction_output_distribution.sample()
        tile_output_distribution = F.softmax(outputs[:,7:], dim=1)
        self.tile_choice_distributions = distributions.Multinomial(total_count=1, probs=tile_output_distribution)
        self.tile_chosen = self.tile_choice_distributions.sample()
        
        return torch.column_stack((
            self.coordinates_chosen, 
            self.direction_chosen.nonzero()[:,1], 
            self.tile_chosen.nonzero()[:,1]))
    
    def give_reward(self, rewards):
        self.rewards = torch.tensor(rewards)

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


