import os
import torch
import json
import numpy as np

from epsilon_scheduler import EpsilonDecayRestart
from kingdomino.utils import get_n_actions
import agents.dqn
from prioritized_experience_replay import ReplayBuffer, PrioritizedReplayBuffer
from agents.encoding import ActionInterface, BOARD_CHANNELS, TILE_ENCODING_SIZE, CUR_TILES_ENCODING_SIZE

from config import experiment, network_types, opponents

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



def load_player(n_players, board_size, device, hp, log_dir=None, file_name=None, player_2=False):
    
    n_episodes = 0
    best_avg_reward = -np.inf   
    
    continue_training = log_dir is not None
    n_actions = get_n_actions(board_size, n_players)
    
    Memory = PrioritizedReplayBuffer if hp['PER'] else ReplayBuffer
    memory = Memory(
        boards_state_size=(n_players,BOARD_CHANNELS,board_size,board_size),
        cur_tiles_state_size=(CUR_TILES_ENCODING_SIZE)*n_players,
        prev_tiles_state_size=(n_players, TILE_ENCODING_SIZE),
        action_size=n_actions,
        buffer_size=hp['replay_memory_size'],
        fixed_possible_actions_size=n_actions,
        device=device)    
    
    # EPS SCHEDULER
    eps_scheduler = EpsilonDecayRestart(
        eps_start=hp['eps_start'],
        eps_end=hp['eps_end'],
        eps_decay=hp['eps_decay'],
        eps_restart=hp['eps_restart'],
        eps_restart_threshold=hp['eps_restart_threshold'])
    
    # ACTION INTERFACE
    action_interface = ActionInterface(
        board_size=board_size, 
        n_players=n_players)
    
    # NETWORK
    Network = network_types[hp['network_type']]
    
    player_1 = agents.dqn.DQN_Agent(
        n_players=n_players,
        board_size=experiment['board_size'],
        Network=Network,
        double=hp['double'],
        network_hp=hp['network_hp'],
        batch_size=hp['batch_size'], 
        eps_scheduler=eps_scheduler,
        action_interface=action_interface,
        tau=hp['tau'], 
        gamma=hp['gamma'],
        lr=hp['lr'],
        memory=memory,
        device=device,
        id=0)
    
    if continue_training:
        print(f'Loading models, memory and optimizers from {log_dir}')
        checkpt = torch.load(os.path.join(log_dir, file_name))
        player_1.policy.load_state_dict(checkpt['policy'])
        player_1.target.load_state_dict(checkpt['target'])
        player_1.optimizer.load_state_dict(checkpt['optimizer'])
        n_episodes = checkpt['n_episodes']
        best_avg_reward = checkpt['best_avg_reward']
        memory = checkpt['memory']
        eps_scheduler = checkpt['eps_scheduler']
        player_1.memory = memory
        player_1.eps_scheduler = eps_scheduler
    
    if player_2:
        if opponents[hp['opponent_type_id']] == 'self':
            player_2 = agents.dqn.DQN_Agent(
                n_players=n_players,
                board_size=experiment['board_size'],
                batch_size=hp['batch_size'], 
                Network=Network,
                network_hp=hp['network_hp'],
                tau=hp['tau'], 
                gamma=hp['gamma'],
                lr=hp['lr'],
                eps_scheduler=eps_scheduler,
                action_interface=action_interface,
                device=device,
                policy=player_1.policy,
                target=player_1.target,
                memory=player_1.memory,
                id=1)
        else:
            player_2 = agents.base.RandomPlayer()
        return player_1,player_2,n_episodes,best_avg_reward
    return player_1,n_episodes,best_avg_reward