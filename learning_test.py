import numpy as np
import gym
import torch

from kingdomino import Kingdomino
import agent
from printer import Printer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 1
N_PLAYERS = 2

# network_info = agent.NetworksInfo(
#     in_channels=50, 
#     output_size=50, 
#     conv_channels=50, 
#     conv_l=2,
#     conv_kernel_size=5,
#     conv_stride=1,
#     pool_place=[0,1,0,1,0,1,0,1,0,1],
#     pool_kernel_size=2,
#     pool_stride=1,
#     board_rep_size=220,
#     board_fc_n=220,
#     board_fc_l=2, 
#     player_rep_size=210,
#     board_prev_tile_fc_l=2,
#     shared_rep_size=240,
#     shared_l=10, 
#     shared_n=210,
#     actor_l=2,
#     actor_n=200,
#     critic_l=2,
#     critic_n=240)

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
    shared_rep_size=-1,
    shared_l=10, 
    shared_n=50,
    actor_l=-1,
    actor_n=-1,
    critic_l=-1,
    critic_n=-1)

#%%

def test(env, players, n_episodes):
    n_players = len(players)
    dones = [False]
    scores = np.zeros(n_episodes)
    Printer.activated = False
    for i in range(n_episodes):
        print('Episode :', i)
        state,_ = env.reset()
        j = 0
        dones = np.array([False],dtype=bool)
        while not dones[0]:
            j += 1
            print('Turn :', j)
            for player in players:
                for key in state:
                    state[key] = np.expand_dims(state[key], axis=0)
                action = player.action(state, dones)
                state,reward,terminated,truncated,info = env.step(np.array(action.to('cpu')))
                dones = np.array([terminated or truncated], dtype=bool)
                if dones[0]:
                    scores[i] = info['Scores'].mean()
                    break
    Printer.activated = False
    return scores

def train(envs, players, n_turns):
    batch_size = envs.num_envs
    n_players = len(players)
    dones = np.zeros(batch_size, dtype=bool)
    states,_ = envs.reset()
    rewards_tracker = np.zeros((n_players, n_turns))
    for i in range(n_turns):
        print('Turn', i)
        for j,player in enumerate(players):
            actions = player.action(states, dones)
            states,rewards,terminateds,truncateds,info = envs.step(np.array(actions.to('cpu')))
            player.give_reward(rewards)
            rewards_tracker[j,i] = rewards[0]
            dones = terminateds | truncateds

if __name__ == '__main__':
    player_1 = agent.PlayerAC(
        batch_size = BATCH_SIZE,
        n_players = N_PLAYERS,
        gamma = 0.99,
        lr_a = 1e-4,
        lr_c = 5e-4,
        coordinate_std = 1,
        network_info = network_info,
        device = device
    )
    player_2 = agent.PlayerAC(
        n_players = N_PLAYERS,
        batch_size = BATCH_SIZE,
        gamma = 0.99,
        lr_a = 1e-4,
        lr_c = 5e-4,
        coordinate_std = 1,
        network_info = network_info,
        device = device,
        critic = player_1.critic,
        actor = player_1.actor)
    players = [player_1, player_2]
    
    env = gym.make('KingDomino-v0', n_players=N_PLAYERS, test=True, disable_env_checker=True)
    envs = gym.vector.AsyncVectorEnv([
        lambda: gym.make("KingDomino-v0", disable_env_checker=True, n_players=N_PLAYERS) for i in range(BATCH_SIZE)
    ])
    
    #%%
    Printer.activated = True
    n_itrs = 20
    scores = np.zeros(n_itrs)
    for i in range(n_itrs):
        for player in players:
            player.learning = True
        train(envs=envs, players=players, n_turns=100)
        for player in players:
            player.learning = False
        scores[i] = test(env=env, players=players, n_episodes=20).mean()
        
    
    # env = gym.make('KingDomino-v0', n_players=2)
    # Printer.activated = False
    # n_steps = 11000
    # state,_ = env.reset()
    # dones = np.array([False],dtype=bool)
    # rewards = np.zeros(n_steps)
    # dones = [False]
    # for i in range(n_steps):
    #     print('Turn :', i)
        
    #     for player in players:
    #         for key in state:
    #             state[key] = np.expand_dims(state[key], axis=0)
    #         action = player.action(state, dones)
    #         state,reward,terminated,truncated,info = env.step(np.array(action))
    #         rewards[i] = reward
    #         player.give_reward(reward)
    
    #         dones = [terminated or truncated]

    #%%        
            
    # envs = gym.vector.AsyncVectorEnv([
    #     lambda: gym.make("KingDomino-v0", disable_env_checker=True, n_players=N_PLAYERS) for i in range(BATCH_SIZE)
    # ])
    
    # Printer.activated = False
    # dones = np.zeros(BATCH_SIZE, dtype=bool)
    # states,_ = envs.reset()
    # nb_turns = 20000
    # rewards_tracker = np.zeros((N_PLAYERS, nb_turns))
    # for i in range(nb_turns):
    #     print('Step', i)
    #     for j,player in enumerate(players):
    #         actions = player.action(states, dones)
    #         states,rewards,terminateds,truncateds,info = envs.step(np.array(actions.to('cpu')))
    #         player.give_reward(rewards)
    #         rewards_tracker[j,i] = rewards[0]
    #         dones = terminateds | truncateds
    #np.save('Rewards', rewards_tracker, allow_pickle=True)
    
    #%%
    
    