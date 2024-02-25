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
from models_directory_utils import start
from networks import TILE_ENCODING_SIZE
from prioritized_experience_replay import ReplayBuffer, PrioritizedReplayBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device used: {device}')

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

network_names = ['PlayerFocusedACNN','PlayerFocusedFC']
reward_fns = [kingdomino.player_focused_reward]

hp = {'batch_size':256,
      'tau':0.005,
      'gamma':0.99999,
      'lr':1e-4,
      'replay_memory_size':100000,
      'PER':True,
      'reward_name_id':0,
      # Exploration
      'eps_start':0.9,
      'eps_end':0.01,
      'eps_decay':1000,
      'eps_restart_threshold':0.0175,
      'eps_restart':0.5,
      # Architecture
      'network_name_id':1,
      'conv_channels':[16,8,4],
      'conv_l':3,
      'conv_kernel_size':[3,3,3],
      'conv_stride':[1,1,1],
      'pool_place':[0,0,0],
      'pool_kernel_size':[2,2,2],
      'pool_stride':[1,1,1],
      'FMN_l':2,
      'FMN_n':[32,64],
      'fc_l':3,
      'fc_n':[128,128,128],
      # Game
      'grid_size':5
      }

  
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
    n_players = 2
    continue_training, trained_dir_path,\
        training_dir_path, n_episodes_done, hp = start(hp)

    n_players_input = 1 if 'PlayerFocused' in network_names[hp['network_name_id']] else n_players
    # first dim depends on player focused or not
    # previous tiles + current_tiles
    action_size = n_players+4
    
    if n_players_input == 1:
        boards_state_size = 8,hp['grid_size'],hp['grid_size']
    else:
        boards_state_size = n_players_input,8,hp['grid_size'],hp['grid_size']
    
    if hp['PER']:
        memory = PrioritizedReplayBuffer(
            boards_state_size=boards_state_size,
            cur_tiles_state_size=(TILE_ENCODING_SIZE+1)*n_players,
            prev_tiles_state_size=TILE_ENCODING_SIZE*n_players_input,
            action_size=action_size,
            buffer_size=hp['replay_memory_size'],
            device=device)
    else:
        memory = ReplayBuffer(
            boards_state_size=(n_players_input,8,hp['grid_size'],hp['grid_size']),
            cur_tiles_state_size=(TILE_ENCODING_SIZE+1)*n_players,
            prev_tiles_state_size=TILE_ENCODING_SIZE*n_players_input,
            action_size=action_size,
            buffer_size=hp['replay_memory_size'],
            device=device)


    eps_scheduler = EpsilonDecayRestart(
        eps_start=hp['eps_start'],
        eps_end=hp['eps_end'],
        eps_decay=hp['eps_decay'],
        eps_restart=hp['eps_restart'],
        eps_restart_threshold=hp['eps_restart_threshold'])
    
    player_1 = agent.DQN_Agent(
        n_players=n_players,
        network_name=network_names[hp['network_name_id']],
        batch_size=hp['batch_size'], 
        hp_archi=hp, 
        eps_scheduler=eps_scheduler,
        tau=hp['tau'], 
        gamma=hp['gamma'],
        lr=hp['lr'],
        memory=memory,
        device=device,
        id=0)
    
    if continue_training:
        print(f'Loading parameters and memory from {trained_dir_path}')
        model = torch.load(os.path.join(trained_dir_path, 'model'))
        player_1.policy.load_state_dict(model['policy'])
        player_1.target.load_state_dict(model['target'])
        player_1.optimizer.load_state_dict(model['optimizer'])
        with open(os.path.join(trained_dir_path, 'memory.pkl'), 'rb') as file:
            memory = pickle.load(file)
        player_1.memory = memory
        with open(os.path.join(trained_dir_path, 'eps_scheduler.pkl'), 'rb') as file:
            eps_scheduler = pickle.load(file)
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
        grid_size=hp['grid_size'],
        reward_fn=reward_fns[hp['reward_name_id']])

    #%%

    Printer.activated = False
    n_itrs = 150
    n_train_episodes = 100
    n_test_episodes = 50
    train_rewards = np.zeros((n_itrs,n_train_episodes,n_players))
    test_scores = np.zeros((n_itrs,n_test_episodes,n_players))
    for i in range(n_itrs):
        print('Itr', i)
        train_rewards[i] = run.train(env, players, n_train_episodes, 10)
        test_scores[i] = run.test_random(env, player_1, n_test_episodes)
        
        print(test_scores[i,:,0].mean())
        print(test_scores[i,:,1].mean())
        
        total_n_train_episodes = n_episodes_done + (i+1)*n_train_episodes
        path = os.path.join(training_dir_path, str(total_n_train_episodes))
        os.makedirs(path)
        print(f'Saving data at path {path}...')
        with open(os.path.join(path, 'memory.pkl'), 'wb') as file:
            pickle.dump(player_1.memory, file)
        with open(os.path.join(path, 'eps_scheduler.pkl'), 'wb') as file:
            pickle.dump(player_1.eps_scheduler, file)
        torch.save({
            'policy':player_1.policy.state_dict(), 
            'target':player_1.target.state_dict(),
            'optimizer':player_1.optimizer.state_dict()},
            os.path.join(path, 'model'))
        np.save(os.path.join(path, 'train_rewards.npy'), train_rewards[i])
        np.save(os.path.join(path, 'scores_vs_random.npy'), test_scores[i])
        print('Data saved !')
        
        # Remove memory.pkl from previous folders (takes space)
        for i,name in enumerate(os.listdir(training_dir_path)):
            if name == str(total_n_train_episodes):
                continue
            try:
                os.remove(os.path.join(os.getcwd(),training_dir_path,name,'memory.pkl'))
            except:
                pass
        
        # with torch.no_grad():
        #     player_1.eval()
        #     player_2.eval()
        #     state = env.reset()
        #     done = False
        #     reward = None
        #     while not done:
        #         for player_id in env.order:
        #             if player_id == 0:
        #                 print("Trained player's turn")
        #             elif player_id == 1:
        #                 print("Random player's turn")
        #             display(draw_obs(state))
        #             if reward is not None:
        #                 print('Reward:', reward)
        #             else:
        #                 print('Reward is None.')
        #             action = players[player_id].action(state, env)
        #             print('Action:', action)
        #             state,reward,done,info = env.step(action)
        #             if done:
        #                 break
        #     print('Scores :', info['Scores'])