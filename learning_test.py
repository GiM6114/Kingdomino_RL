import numpy as np
import gym
from inputimeout import inputimeout 
from threading import Thread
import time
import sys

from kingdomino import Kingdomino
import agent
from printer import Printer

import msvcrt
import time

class TimeoutExpired(Exception):
    pass

def input_with_timeout(prompt, timeout, timer=time.monotonic):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    endtime = timer() + timeout
    result = []
    while timer() < endtime:
        if msvcrt.kbhit():
            result.append(msvcrt.getwche()) #XXX can it block on multibyte characters?
            if result[-1] == '\r':
                return ''.join(result[:-1])
        time.sleep(0.04) # just to yield to other processes/threads
    raise TimeoutExpired

a = None
def check():
    time.sleep(4)
    if a != None:
        return
    print("Resume training for 1000 episodes...")

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
    conv_kernel_size=5,
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

if __name__ == '__main__':
    player = agent.PlayerAC(
        n_players = N_PLAYERS,
        gamma = 0.99,
        lr_a = 1e-4,
        lr_c = 5e-4,
        coordinate_std = 2,
        network_info = network_info
    )
    
    env = gym.make('KingDomino-v0', n_players=2)
    Printer.activated = False
    n_episodes = 11250
    scores_episode = np.zeros(n_episodes)
    sum_rewards_episode = np.zeros(n_episodes)
    for e in range(n_episodes):
        print('Episode :', e)
        i = 0
        state,_ = env.reset()
        dones = [False]
        total_reward = 0
        while True:
            for key in state:
                state[key] = np.expand_dims(state[key], axis=0)
            action = player.action(state, dones)
            state,reward,terminated,truncated,info = env.step(np.array(action))
            print('Reward :', reward)
            player.give_reward(reward)
            total_reward += reward
            dones = [terminated or truncated]
            if dones[0]:
                scores_episode[e] = np.mean(info['Scores'])
                sum_rewards_episode[e] = total_reward
                break
        # if e % 1000 == 0:
        #     Thread(target = check).start()
        #     a = input('Input to stop')
                    
    # envs = gym.vector.AsyncVectorEnv([
    #     lambda: gym.make("KingDomino-v0", n_players=N_PLAYERS) for i in range(BATCH_SIZE)
    # ])
    
    # Printer.activated = False
    # done = False
    # states,_ = envs.reset()
    # for i in range(500):
    #     print('Step', i)
    #     actions = player.action(states)
    #     print(actions)
    #     print(envs.step(np.array(actions)))
    #     states,rewards,terminateds,truncateds,info = envs.step(np.array(actions))
    #     player.give_reward(rewards)