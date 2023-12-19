import numpy as np
import gym
import torch
import os

from epsilon_scheduler import EpsilonDecayRestart
import kingdomino
import agent
from printer import Printer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 128
N_PLAYERS = 2                                                                   

def test(env, players, n_episodes):
    for player in players:
        player.learning = False
    n_players = len(players)
    done = False
    scores = np.zeros((n_episodes,n_players))
    for i in range(n_episodes):
        print('Episode :', i)
        state = env.reset()
        j = 0
        done = False
        while not done:
            j += 1
            for player_id in env.order:
                action = players[player_id].action(state, env)
                state,reward,done,info = env.step(action)
                if done:
                    scores[i] = info['Scores']
                    break
    Printer.activated = False
    return scores

def train(env, players, n_episodes):
    for player in players:
        player.learning = True
    n_players = len(players)
    rewards_tracker = np.zeros((n_episodes,n_players))
    for i in range(n_episodes):
        state = env.reset()
        done = False
        sum_rewards = np.zeros(n_players)
        reward_next_player = None
        j = 0
        while not done:
            j += 1
            for player_id in env.order:
                # Give reward of previous move
                if reward_next_player is not None:
                    players[player_id].give_reward(reward_next_player, state)
                    sum_rewards[player_id] += reward_next_player
                action = players[player_id].action(state, env)
                state,reward_next_player,done,info = env.step(action) # reward none for first turn
        for id,player in enumerate(players):
            player.give_reward(env.reward(True, id=id), None)
        rewards_tracker[i] = sum_rewards
    return rewards_tracker

#%%

import agent
def human_test(env, player):
    player.learning = False
    Printer.activated=True
    human = agent.HumanPlayer()
    players = [player, human]
    state = env.reset()
    done = False
    while not done:
        for player_id in env.order:
            Printer.print(state['Boards'][player_id])
            action = players[player_id].action(state, env)
            state,reward,done,info = env.step(action)
            
def random_test(env, player):
    player.learning = False
    Printer.activated = False
    random_player = agent.RandomPlayer()
    players = [player, random_player]
    state = env.reset()
    done = False
    while not done:
        for player_id in env.order:
            action = players[player_id].action(state, env)
            state,reward,done,info = env.step(action)
            
def test_random(env, player, n_episodes):
    player.learning = False
    Printer.activated = False
    random_player = agent.RandomPlayer()
    players = [player, random_player]
    n_players = len(players)
    done = False
    scores = np.zeros((n_episodes,n_players))
    for i in range(n_episodes):
        print('Episode :', i)
        state = env.reset()
        j = 0
        done = False
        while not done:
            j += 1
            for player_id in env.order:
                action = players[player_id].action(state, env)
                state,reward,done,info = env.step(action)
                if done:
                    scores[i] = info['Scores']
                    break
    Printer.activated = False
    return scores

#%%

network_info = agent.NetworksInfo(
    in_channels=10, 
    output_size=10, 
    conv_channels=10, 
    conv_l=5,
    conv_kernel_size=3,
    conv_stride=1,
    pool_place=[0,1,0,1,0,1,0,1,0,1],
    pool_kernel_size=2,
    pool_stride=1,
    board_rep_size=50,
    board_fc_n=50,
    board_fc_l=3, 
    player_rep_size=50,
    board_prev_tile_fc_l=2,
    shared_rep_size=50,
    shared_l=10, 
    shared_n=50,
    actor_l=-1,
    actor_n=-1,
    critic_l=-1,
    critic_n=-1)
    
# player_1.policy.load_state_dict(torch.load(os.getcwd()+'/Models/20_100'))
# player_1.target.load_state_dict(torch.load(os.getcwd()+'/Models/20_100'))        
if __name__ == '__main__':
    
    print('main')
    
    #%%
    
    eps_scheduler = EpsilonDecayRestart(0.9,0.01,2000,0.05)
    n_players = 2
    player_1 = agent.DQN_Agent(
        n_players=n_players, 
        batch_size=BATCH_SIZE, 
        network_info=network_info, 
        eps_scheduler=eps_scheduler,
        tau=0.005, 
        gamma=0.99,
        lr=1e-4,
        replay_memory_size=50000,
        device=device,
        id=0)

    player_2 = agent.DQN_Agent(
        n_players=n_players, 
        batch_size=BATCH_SIZE, 
        eps_scheduler=eps_scheduler,
        tau=0.005, 
        gamma=0.99,
        lr=1e-4,
        device=device,
        policy=player_1.policy,
        target=player_1.target,
        memory=player_1.memory,
        id=1)
    players = [player_1, player_2]
    env = kingdomino.SimpleKingdomino(
        n_players=2, 
        reward_fn=kingdomino.reward_last_quantitative)

    #%%

    Printer.activated = False
    n_itrs = 20
    n_train_episodes = 100
    n_test_episodes = 50
    train_rewards = np.zeros((n_itrs,n_train_episodes,n_players))
    test_scores = np.zeros((n_itrs,n_test_episodes,n_players))
    for i in range(n_itrs):
        print('Itr', i)
        train_rewards[i] = train(env, players, n_train_episodes)
        test_scores[i] = test_random(env, player_1, n_test_episodes)
    