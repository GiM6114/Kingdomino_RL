import numpy as np
import gym

from kingdomino import Kingdomino
from agent import RandomPlayer, PlayerAC, NetworksInfo
from printer import Printer

BATCH_SIZE = 3
N_PLAYERS = 2

network_info = NetworksInfo(
    in_channels=50, 
    output_size=50, 
    conv_channels=50, 
    conv_l=2,
    conv_kernel_size=5,
    conv_stride=1,
    pool_place=[0,1,0,1,0,1,0,1,0,1],
    pool_kernel_size=2,
    pool_stride=1,
    board_rep_size=220,
    board_fc_n=220,
    board_fc_l=2, 
    player_rep_size=210,
    board_prev_tile_fc_l=2,
    shared_rep_size=240,
    shared_l=10, 
    shared_n=210,
    actor_l=2,
    actor_n=200,
    critic_l=2,
    critic_n=240)

if __name__ == '__main__':
    agent = PlayerAC(
        batch_size = BATCH_SIZE,
        n_players = N_PLAYERS,
        gamma = 0.99,
        lr_a = 1e-4,
        lr_c = 5e-4,
        coordinate_std = 2,
        network_info = network_info
    )
    
    # env = gym.make('KingDomino-v0', n_players=2)
    # Printer.activated = True
    # done = False
    # state,_ = env.reset()
    # for e in range(100):
    #     while True:
    #         action = agent.action(state)
    #         print(action)
    #         state,reward,terminated,truncated,info = env.step(np.array(action))
    #         agent.give_reward([reward])
    #         if terminated or truncated:
    #             break
        
    envs = gym.vector.AsyncVectorEnv([
        lambda: gym.make("KingDomino-v0", n_players=N_PLAYERS) for i in range(BATCH_SIZE)
    ])
    
    Printer.activated = False
    done = False
    states,_ = envs.reset()
    for i in range(500):
        print('Step', i)
        actions = agent.action(states)
        print(actions)
        states,rewards,terminateds,truncateds,info = envs.step(np.array(actions))
        agent.give_reward(rewards)