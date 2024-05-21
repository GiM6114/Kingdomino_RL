import numpy as np
import torch
import os
import pickle
from IPython.core.display import Image, display
from datetime import datetime
from tqdm.auto import tqdm

from epsilon_scheduler import EpsilonDecayRestart
import kingdomino.kingdomino
import kingdomino.rewards
from kingdomino.utils import get_n_actions
import agents.dqn
from printer import Printer
import run
from graphics import draw_obs
from models_directory_utils import read_experiment, write_experiment, load_player
from prioritized_experience_replay import ReplayBuffer, PrioritizedReplayBuffer
from agents.encoding import ActionInterface, BOARD_CHANNELS, TILE_ENCODING_SIZE, CUR_TILES_ENCODING_SIZE
from log import Logger

from config import experiment, network_types, reward_fns, opponents

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device used: {device}')

#%%
  
# player_1.policy.load_state_dict(torch.load(os.getcwd()+'/Models/20_100'))
# player_1.target.load_state_dict(torch.load(os.getcwd()+'/Models/20_100'))
# runs
#   DQN_FC
#       240324_0210
#           best_chkpt.pt
#           progress.txt
#           event
#       250324_1154...
#   DQN_Conv...


if __name__ == '__main__':
    initial_log_dir = None
    file_name = None
    continue_training = initial_log_dir is not None
    if continue_training:
        experiment = read_experiment(initial_log_dir)
        n_episodes = experiment['n_episodes']
        best_avg_reward = experiment['best_avg_reward']
    else:
        log_dir = os.path.join('runs', 'DQN_FC', datetime.now().strftime('%d%m%y_%H%M%S'))
        os.makedirs(log_dir)
        write_experiment(directory_path=log_dir, parameters_dict=experiment)
     
    hp = experiment['hp']
    board_size = experiment['board_size']
    n_players = experiment['n_players']
    
    logger = Logger(log_dir)
    
    player_1,player_2,n_episodes,best_avg_reward = load_player(
        n_players,
        board_size,
        device,
        hp,
        initial_log_dir,
        file_name,
        player_2=True)

    players = [player_1, player_2]
    env = kingdomino.kingdomino.Kingdomino(
        n_players=len(players),
        board_size=board_size,
        reward_fn=reward_fns[hp['reward_name_id']])
        
    #%%

    Printer.activated = False
    n_itrs = 150
    n_train_episodes = 100
    n_test_episodes = 20
    for i in range(n_itrs):
        logger.log(f'Iteration {i}')
        train_rewards = run.train(env, players, n_train_episodes)
        n_episodes += n_train_episodes
        test_scores = run.test_random(env, player_1, n_test_episodes)    
        
        # Logging
        rewards_train_player_1 = train_rewards[:,0].mean()        
        score_test_player_1 = test_scores[:,0].mean()
        score_random = test_scores[:,1].mean()
        logger.log('Train rewards after {n_episodes}: {score_train_player_1} ')
        logger.log(f'Test score against random after {n_episodes}: {score_test_player_1}')
        logger.log(f'Score of random: {score_random}')
        logger.add_scalars('test_scores', 
                           {'agent':score_test_player_1,
                            'random':score_random},
                           global_step=n_episodes)
        
        # Recording info of current model
        print(f'Saving checkpt at path {log_dir}...')
        torch.save({
            'policy':player_1.policy.state_dict(), 
            'target':player_1.target.state_dict(),
            'optimizer':player_1.optimizer.state_dict(),
            'n_episodes':n_episodes,
            'best_avg_reward':best_avg_reward,
            'memory':player_1.memory,
            'eps_scheduler':player_1.eps_scheduler},
            os.path.join(log_dir, 'checkpt.pt'))
        print('Checkpt saved !')

        # Special recording if best model so far
        if score_test_player_1 > best_avg_reward:
            print(f'Saving best checkpt at path {log_dir}...')
            best_avg_reward = score_test_player_1
            torch.save({
                'policy':player_1.policy.state_dict(), 
                'target':player_1.target.state_dict(),
                'optimizer':player_1.optimizer.state_dict(),
                'n_episodes':n_episodes,
                'best_avg_reward':best_avg_reward,
                'memory':player_1.memory,
                'eps_scheduler':player_1.eps_scheduler},
                os.path.join(log_dir, 'best_checkpt.pt'))
            print('Best checkpt saved !')
        
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