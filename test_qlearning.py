import numpy as np
import gym
import torch
import os
from datetime import datetime
import pickle
from IPython.core.display import Image, display

from epsilon_scheduler import EpsilonDecayRestart
import kingdomino
import agent
from printer import Printer
import run
from graphics import draw_obs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device used: {device}')

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

def write_hyperparameters(directory_path, parameters_dict):
    file_path = os.path.join(directory_path, 'hyperparameters.txt')
    with open(file_path, 'w') as file:
        for key, value in parameters_dict.items():
            file.write(f'{key}: {value}\n')
    print(f'Hyperparameter file created at path {file_path}')

def read_hyperparameters(file_path):
    hyperparameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            hyperparameters[key] = eval(value)
    return hyperparameters

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

#%%

# hp = {'batch_size':128,
#       'tau':0.005,
#       'gamma':0.99999,
#       'lr':1e-4,
#       'replay_memory_size':20000,
#       # Exploration
#       'eps_start':0.9,
#       'eps_end':0.01,
#       'eps_decay':2500,
#       'eps_restart_threshold':0.05,
#       'eps_restart':0.1,
#       # Architecture
#       'conv_channels':[32,16,5],
#       'conv_l':3,
#       'conv_kernel_size':[3,3,3],
#       'conv_stride':[1,1,1],
#       'pool_place':[0,0,0],
#       'pool_kernel_size':None,
#       'pool_stride':None,
#       'board_rep_size':100,
#       'board_fc_n':100,
#       'board_fc_l':1, 
#       'player_rep_size':100,
#       'board_prev_tile_fc_l':1,
#       'shared_rep_size':100,
#       'shared_l':3, 
#       'shared_n':100
#       }

network_names = ['PlayerFocusedACNN']
reward_fns = [kingdomino.player_focused_reward]

hp = {'batch_size':128,
      'tau':0.005,
      'gamma':0.99999,
      'lr':1e-4,
      'replay_memory_size':30000,
      'reward_name_id':0,
      # Exploration
      'eps_start':0.9,
      'eps_end':0.01,
      'eps_decay':2500,
      'eps_restart_threshold':0.05,
      'eps_restart':0.2,
      # Architecture
      'network_name_id':0,
      'conv_channels':[16,8,4],
      'conv_l':3,
      'conv_kernel_size':[3,3,3],
      'conv_stride':[1,1,1],
      'pool_place':[1,1,1],
      'pool_kernel_size':[2,2,2],
      'pool_stride':[1,1,1],
      'FMN_l':2,
      'FMN_n':[64,64],
      'fc_l':2,
      'fc_n':[128,128]
      }

# TODO : adaptive cnn, learn network taking input context, output convolutional layer weights  
  
# player_1.policy.load_state_dict(torch.load(os.getcwd()+'/Models/20_100'))
# player_1.target.load_state_dict(torch.load(os.getcwd()+'/Models/20_100'))
# Model_1:
#   hyperparameters.txt
#   Training_1:
#       100:
#           policy_parameters
#           target_parameters
#           memory.pkl
#           train_rewards.npy
#           scores_vs_random.npy
#       200:
#           policy_parameters
#           target_parameters
#           memory.pkl
#           train_rewards.npy
#           scores_vs_random.npy
if __name__ == '__main__':
    continue_training = False
    n_episodes_done = 0
    models_dir = 'Models'
    model_id = int(input('Id of model to train ?'))
    if os.path.exists(os.path.join(models_dir, f'Model_{model_id}')):
        model_dir_path = os.path.join(models_dir, f'Model_{model_id}')
        hp = read_hyperparameters(model_dir_path + '/hyperparameters.txt')
        choice = int(input('Create new model (1) or continue training (2) ?'))
        if choice == 1:
            training_dir_path = create_directory(model_dir_path, 'Training')
        elif choice == 2:
            training_id = int(input('Which training to continue ?'))
            training_dir_path = os.path.join(model_dir_path, 'Training_'+str(training_id))
            trained_dir_path,n_episodes_done = find_highest_numbered_directory(training_dir_path)
            if n_episodes_done != 0:
                continue_training = True
    else:
        model_dir_path = create_directory(models_dir, 'Model')
        write_hyperparameters(model_dir_path, hp)
        training_dir_path = os.path.join(model_dir_path, 'Training_1')

    
    eps_scheduler = EpsilonDecayRestart(
        eps_start=hp['eps_start'],
        eps_end=hp['eps_end'],
        eps_decay=hp['eps_decay'],
        eps_restart=hp['eps_restart'],
        eps_restart_threshold=hp['eps_restart_threshold'])
    n_players = 2
    player_1 = agent.DQN_Agent(
        n_players=n_players,
        network_name=network_names[hp['network_name_id']],
        batch_size=hp['batch_size'], 
        hp_archi=hp, 
        eps_scheduler=eps_scheduler,
        tau=hp['tau'], 
        gamma=hp['gamma'],
        lr=hp['lr'],
        replay_memory_size=hp['replay_memory_size'],
        device=device,
        id=0)
    
    if continue_training:
        print(f'Loading parameters and memory from {trained_dir_path}')
        player_1.policy.load_state_dict(torch.load(os.path.join(trained_dir_path, 'policy_parameters')))
        player_1.target.load_state_dict(torch.load(os.path.join(trained_dir_path, 'target_parameters')))
        with open(os.path.join(trained_dir_path, 'memory.pkl'), 'rb') as file:
            memory = pickle.load(file)
        with open(os.path.join(trained_dir_path, 'eps_scheduler.pkl'), 'rb') as file:
            eps_scheduler = pickle.load(file)
        player_1.memory = memory
        player_1.eps_scheduler = eps_scheduler
        
    player_2 = agent.DQN_Agent(
        n_players=n_players, 
        batch_size=hp['batch_size'], 
        tau=hp['tau'], 
        gamma=hp['gamma'],
        lr=hp['lr'],
        eps_scheduler=eps_scheduler,
        device=device,
        policy=player_1.policy,
        target=player_1.target,
        memory=player_1.memory,
        id=1)
    players = [player_1, player_2]
    env = kingdomino.Kingdomino(
        players=players, 
        reward_fn=reward_fns[hp['reward_name_id']])

    #%%

    Printer.activated = False
    n_itrs = 50
    n_train_episodes = 100
    n_test_episodes = 100
    train_rewards = np.zeros((n_itrs,n_train_episodes,n_players))
    test_scores = np.zeros((n_itrs,n_test_episodes,n_players))
    for i in range(n_itrs):
        print('Itr', i)
        train_rewards[i] = run.train(env, players, n_train_episodes, 10)
        test_scores[i] = run.test_random(env, player_1, n_test_episodes)
        
        
        
        total_n_train_episodes = n_episodes_done + (i+1)*n_train_episodes
        path = os.path.join(training_dir_path, str(total_n_train_episodes))
        os.makedirs(path)
        print(f'Saving data at path {path}...')
        with open(os.path.join(path, 'memory.pkl'), 'wb') as file:
            pickle.dump(player_1.memory, file)
        with open(os.path.join(path, 'eps_scheduler.pkl'), 'wb') as file:
            pickle.dump(player_1.eps_scheduler, file)
        torch.save(player_1.policy.state_dict(), os.path.join(path, 'policy_parameters'))
        torch.save(player_1.target.state_dict(), os.path.join(path, 'target_parameters'))
        np.save(os.path.join(path, 'train_rewards.npy'), train_rewards[i])
        np.save(os.path.join(path, 'scores_vs_random.npy'), test_scores[i])
        print('Data saved !')
        # TODO: code to remove memory.pkl from previous folders
        for i,name in enumerate(os.listdir(training_dir_path)):
            if name == str(total_n_train_episodes):
                continue
            try:
                os.remove(os.path.join(os.getcwd(),training_dir_path,name,'memory.pkl'))
            except:
                pass
        
        player_test = agent.RandomPlayer()
        players_test = [player_1, player_test]
        state = env.reset()
        done = False
        reward = None
        while not done:
            for player_id in env.order:
                if player_id == 0:
                    print("Trained player's turn")
                elif player_id == 1:
                    print("Random player's turn")
                display(draw_obs(state))
                if reward is not None:
                    print('Reward:', reward)
                else:
                    print('Reward is None.')
                action = players[player_id].action(state, env)
                print('Action:', action)
                state,reward,done,info = env.step(action)
                if done:
                    break
        print('Scores :', info['Scores'])