from agent import RandomPlayer, MPC_Agent
from kingdomino import SimpleKingdomino
from printer import Printer

import gym
import numpy as np

def test(env, players, n_episodes):
    n_players = len(players)
    scores = np.zeros((n_episodes,len(players)))
    for i in range(n_episodes):
        print('Episode :', i)
        env.reset()
        Printer.print('Real env current player id:', env.current_player_id)
        done = False
        j = 0
        while not done:
            print('Turn', j)
            j += 1
            for k,player_id in enumerate(env.order):
                print('Player', k)
                action = players[player_id].action(env)
                terminated = env.step(action)
                done = terminated
                if done:
                    scores[i] = env.scores()
                    break
    Printer.activated = False
    return scores

if __name__ == '__main__':
    player_1 = RandomPlayer()
    player_2 = MPC_Agent(5, 1)
    players = [player_1, player_2]
    
    env = SimpleKingdomino(n_players=2)

    Printer.activated = False
    n_episodes = 100
    scores = test(env=env, players=players, n_episodes=n_episodes)