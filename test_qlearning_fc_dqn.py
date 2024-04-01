import numpy as np
import torch
import os
import pickle
from IPython.core.display import Image, display
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from epsilon_scheduler import EpsilonDecayRestart
import env.kingdomino
import env.rewards
from env.utils import get_n_actions
import agents.dqn
from printer import Printer
import run
from graphics import draw_obs
from models_directory_utils import read_hyperparameters, write_hyperparameters
from networks import TILE_ENCODING_SIZE
from prioritized_experience_replay import ReplayBuffer, PrioritizedReplayBuffer
from agents.encoding import ActionInterface

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device used: {device}')

#%%

input_names = ['PlayerFocused','AllPlayers']
method_names = ['Loop','NoLoop']
reward_fns = [env.rewards.player_focused_reward]

hp = {'batch_size':256,
      'tau':0.005,
      'gamma':0.99,
      'lr':3e-5,
      'replay_memory_size':50000,
      'PER':True,
      'reward_name_id':0,
      # Exploration
      'eps_start':0.9,
      'eps_end':0.01,
      'eps_decay':1000,
      'eps_restart_threshold':0.0175,
      'eps_restart':0.5,
      # Architecture
      'input_name_id':0,
      'method_name_id':1,
      'n_hidden_layers':3,
      'hidden_layers_width':[128,128,128],
      # Game
      'grid_size':5
      }

  
# player_1.policy.load_state_dict(torch.load(os.getcwd()+'/Models/20_100'))
# player_1.target.load_state_dict(torch.load(os.getcwd()+'/Models/20_100'))
# runs
#   DQN_FC
#       240324_0210
#       250324_1154...
#   DQN_Conv...
if __name__ == '__main__':
    
    log_dir = None # 'runs/DQN_FC/240324_0210'
    continue_training = log_dir is not None
    if log_dir is not None:
        hp = read_hyperparameters(log_dir)
    else:
        log_dir = os.path.join('runs', 'DQN_FC', datetime.now().strftime('%d%m%y_%H%M%S'))
        os.makedirs(log_dir)
        write_hyperparameters(directory_path=log_dir, parameters_dict=hp)
    
    n_players = 2
    
    method_name = method_names[hp['method_name_id']]
    input_name = input_names[hp['input_name_id']]
    writer = SummaryWriter(
        log_dir=log_dir)
    
    n_players_input = 1 if 'PlayerFocused' in input_name else n_players
    # first dim depends on player focused or not
    # previous tiles + current_tiles
    action_size = n_players+4
    
    if n_players_input == 1:
        boards_state_size = 8,hp['grid_size'],hp['grid_size']
    else:
        boards_state_size = n_players_input,8,hp['grid_size'],hp['grid_size']
    
    Memory = PrioritizedReplayBuffer if hp['PER'] else ReplayBuffer
    memory = Memory(
        boards_state_size=(n_players_input,8,hp['grid_size'],hp['grid_size']),
        cur_tiles_state_size=(TILE_ENCODING_SIZE+1)*n_players,
        prev_tiles_state_size=TILE_ENCODING_SIZE*n_players_input,
        action_size=action_size,
        buffer_size=hp['replay_memory_size'],
        fixed_possible_actions_size=False if method_name == 'Loop' else get_n_actions(hp['grid_size']),
        device=device)

    eps_scheduler = EpsilonDecayRestart(
        eps_start=hp['eps_start'],
        eps_end=hp['eps_end'],
        eps_decay=hp['eps_decay'],
        eps_restart=hp['eps_restart'],
        eps_restart_threshold=hp['eps_restart_threshold'])
    
    action_interface = ActionInterface(
        board_size=hp['grid_size'], 
        n_players=n_players)
    
    player_1 = agents.dqn.DQN_Agent_FC(
        n_players=n_players,
        batch_size=hp['batch_size'], 
        hp_archi=hp,
        eps_scheduler=eps_scheduler,
        action_interface=action_interface,
        tau=hp['tau'], 
        gamma=hp['gamma'],
        lr=hp['lr'],
        memory=memory,
        device=device,
        id=0)
    
    best_avg_reward = 0
    n_episodes = 0
    if continue_training:
        print(f'Loading models, memory and optimizers from {log_dir}')
        checkpt = torch.load(os.path.join(log_dir, 'checkpt.pt'))
        player_1.policy.load_state_dict(checkpt['policy'])
        player_1.target.load_state_dict(checkpt['target'])
        player_1.optimizer.load_state_dict(checkpt['optimizer'])
        n_episodes = checkpt['n_episodes']
        best_avg_reward = checkpt['best_avg_reward']
        memory = checkpt['memory']
        eps_scheduler = checkpt['eps_scheduler']
        player_1.memory = memory
        player_1.eps_scheduler = eps_scheduler
        
    player_2 = agents.dqn.DQN_Agent_FC(
        n_players=n_players, 
        batch_size=hp['batch_size'], 
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
    
    players = [player_1, player_2]
    env = env.kingdomino.Kingdomino(
        players=players,
        board_size=hp['grid_size'],
        reward_fn=reward_fns[hp['reward_name_id']])

    #%%

    Printer.activated = False
    n_itrs = 150
    n_train_episodes = 100
    n_test_episodes = 20
    train_rewards = np.zeros((n_itrs,n_train_episodes,n_players))
    test_scores = np.zeros((n_itrs,n_test_episodes,n_players))
    for i in tqdm(range(n_itrs)):
        train_rewards[i] = run.train(env, players, n_train_episodes) 
        test_scores[i] = run.test_random(env, player_1, n_test_episodes)
        score_player_1 = test_scores[i,:,0].mean()
        score_random = test_scores[i,:,1].mean()
        print(f'Mean test score {n_episodes}-episodes-trained agent:', score_player_1)
        print('Mean test score random player:', score_random)
        writer.add_scalars('test_scores', 
                           {'PLayer 1':score_player_1,
                            'Player 2':score_random},
                           global_step=n_episodes)
        
        total_n_train_episodes = n_episodes + (i+1)*n_train_episodes
        path = os.path.join(log_dir, str(total_n_train_episodes))
        os.makedirs(path)
        if score_random > best_avg_reward:
            best_avg_reward = score_random
            torch.save({
                'policy':player_1.policy.state_dict(), 
                'target':player_1.target.state_dict(),
                'optimizer':player_1.optimizer.state_dict(),
                'n_episodes':n_episodes,
                'best_avg_reward':best_avg_reward,
                'memory':player_1.memory,
                'eps_scheduler':player_1.eps_scheduler},
                os.path.join(path, 'best_checkpt.pt'))
        print(f'Saving data at path {log_dir}...')
        torch.save({
            'policy':player_1.policy.state_dict(), 
            'target':player_1.target.state_dict(),
            'optimizer':player_1.optimizer.state_dict(),
            'n_episodes':n_episodes,
            'best_avg_reward':best_avg_reward,
            'memory':player_1.memory,
            'eps_scheduler':player_1.eps_scheduler},
            os.path.join(path, 'checkpt.pt'))
            
        print('Data saved !')
    
        
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