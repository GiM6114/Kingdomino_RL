import os
import torch
import json
import numpy as np

from kingdomino.utils import compute_n_actions
import agents.dqn
from agents.encoding import TileCoordinateInterface

from config import experiment, network_types, opponents, action_types

def find_available_directory_name(path, base_name):
    i = 1
    while os.path.exists(os.path.join(path, f'{base_name}_{i}')):
        i += 1
    return f'{base_name}_{i}'

def create_directory(path, base_directory_name):
    directory_name = find_available_directory_name(path, base_directory_name)
    directory_path = os.path.join(path, directory_name)
    os.makedirs(directory_path)
    return directory_path

def write_experiment(directory_path, parameters_dict):
    file_path = os.path.join(directory_path, 'experiment.txt')
    json.dump(parameters_dict, open(file_path, 'w'))
    print(f'Experiment file created at path {file_path}')

def read_experiment(file_path):
    experiment = json.load(open(file_path))
    return experiment

def find_highest_numbered_directory(directory):
    highest_number = 0
    highest_directory = ''
    for directory_name in os.listdir(f'{directory}/.'):
            try:
                number = int(directory_name)
                if number > highest_number:
                    highest_number = number
                    highest_directory = directory_name
            except ValueError:
                # Ignore directories that don't match the expected pattern
                pass
    return os.path.join(directory, highest_directory), highest_number



def load_player(
        n_players, 
        board_size,
        device, 
        hp, 
        continue_training,
        log_dir=None, 
        file_name=None, 
        player_2=False):
    
    n_episodes = 0
    best_avg_reward = -np.inf   
    
    n_actions = compute_n_actions(board_size, n_players)
    
    # NETWORK
    Network = network_types[hp['network_type']]
    
    if hp['action_type'] == 'singular':
        # ACTION INTERFACE
        action_interface = TileCoordinateInterface(
            board_size=board_size, 
            n_players=n_players)
        
        player_1 = agents.dqn.DQN_Agent(
            log_dir=log_dir,
            n_players=n_players,
            board_size=experiment['board_size'],
            Network=Network,
            n_actions=n_actions,
            hp=hp,
            action_interface=action_interface,
            device=device,
            id=0)
    elif hp['action_type'] == 'sequential':
        player_1 = agents.dqn.SequentialDQN_AgentInterface(
            log_dir=log_dir,
            n_players=n_players,
            board_size=experiment['board_size'],
            Network=Network,
            hp=hp,
            device=device,
            id=0)
        
    if continue_training:
        print(f'Loading models, memory and optimizers from {log_dir}')
        path = os.path.join(log_dir, file_name)
        player_1.load(path)
        checkpt = torch.load(path)
        n_episodes = checkpt['n_episodes']
        best_avg_reward = checkpt['best_avg_reward']
    
    if player_2:
        if hp['opponent_type'] == 'self':
            if action_types[hp['action_type']] == 'singular':
                player_2 = agents.dqn.DQN_Agent(
                    n_players=n_players,
                    board_size=experiment['board_size'],
                    n_actions=n_actions,
                    Network=Network,
                    eps_scheduler=player_1.eps_scheduler,
                    action_interface=action_interface,
                    device=device,
                    policy=player_1.policy,
                    target=player_1.target,
                    memory=player_1.memory,
                    id=1)
            elif hp['action_type'] == 'sequential':
                player_2 = agents.dqn.SequentialDQN_AgentInterface(
                    n_players=n_players,
                    board_size=experiment['board_size'],
                    Network=Network,
                    eps_scheduler=player_1.eps_scheduler,
                    device=device,
                    policy=player_1.policy,
                    target=player_1.target,
                    memory=player_1.memory,
                    id=1)                
        else:
            player_2 = agents.base.RandomPlayer()
        return player_1,player_2,n_episodes,best_avg_reward
    return player_1,n_episodes,best_avg_reward