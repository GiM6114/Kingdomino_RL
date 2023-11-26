from agent import RandomPlayer, MPC_Agent
from kingdomino import Kingdomino
from printer import Printer

import gym
import numpy as np

def test(env, players, n_episodes):
    n_players = len(players)
    scores = np.array((n_episodes,len(players)))
    for i in range(n_episodes):
        print('Episode :', i)
        state,_ = env.reset()
        done = False
        while not done:
            for j,player in enumerate(players):
                action = player.action(env)
                state,reward,terminated,truncated,info = env.step(action)
                done = terminated or truncated
                if done:
                    scores[i] = info['Scores']
                    break
    Printer.activated = False
    return scores

if __name__ == '__main__':
    player_1 = RandomPlayer()
    player_2 = MPC_Agent(3, 1)
    players = [player_1, player_2]
    
    env = gym.make('KingDomino-v0', n_players=len(players))

    Printer.activated = False
    n_episodes = 10
    scores = test(env=env, players=players, n_episodes=n_episodes)