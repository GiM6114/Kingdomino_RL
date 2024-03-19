import agent
import numpy as np

from IPython.core.display import Image, display
from graphics import draw_obs

import torch


def test(env, players, n_episodes, verbose=0, print_every=10):
    with torch.no_grad():
        for player in players:
            player.eval()
        n_players = len(players)
        done = False
        scores = np.zeros((n_episodes,n_players))
        for i in range(n_episodes):
            if verbose == 1 and i % print_every == 0:
                print('Test episode :', i)
            state = env.reset()
            done = False
            while not done:
                for player_id in env.order:
                    if verbose == 2:
                        print(f"Player {player_id}'turn")
                        display(draw_obs(state))
                    action = players[player_id].action(state, env)
                    if verbose == 2:
                        print(action)
                    state,reward,done,info = env.step(action)
                    if verbose == 2:
                        display(draw_obs(state))
                    if done:
                        scores[i] = info['Scores']
                        break
        return scores

def train(env, players, n_episodes, print_every=10):
    for player in players:
        player.train()
    n_players = len(players)
    rewards_tracker = np.zeros((n_episodes,n_players))
    for i in range(n_episodes):
        if i%print_every == 0:
            print('Train episode :', i)
        s = env.reset()
        d = False
        sum_rewards = np.zeros(n_players)
        while not d:
            for p_id in env.order:
                if not env.first_turn:
                    r = env.getReward(p_id)
                    p_a = env.getPossibleActions(encode=True)
                    players[p_id].processReward(r, s, False, p_a)
                    sum_rewards[p_id] += r
                a = players[p_id].action(s, env)
                s,d,info = env.step(a)
            if d:
                for i,player in enumerate(players):
                    players[p_id].processReward(r, s, d, p_a)
                    sum_rewards[p_id] += r
        rewards_tracker[i] = sum_rewards
    return rewards_tracker

def human_test(env, player):
    player.eval()
    human = agent.HumanPlayer()
    players = [player, human]
    state = env.reset()
    done = False
    while not done:
        for player_id in env.order:
            display(draw_obs(state))
            action = players[player_id].action(state, env)
            state,reward,done,info = env.step(action)
            if done:
                break
            
def test_random(env, player, n_episodes, print_every=10):
    player.eval()
    random_player = agent.RandomPlayer()
    players = [player, random_player]
    n_players = len(players)
    env.players = players
    scores = np.zeros((n_episodes,n_players))
    for i in range(n_episodes):
        if i%print_every == 0:
            print('Test random episode :', i)
        state = env.reset()
        done = False
        while not done:
            for player_id in env.order:
                action = players[player_id].action(state, env)
                state,reward,done,info = env.step(action)
                if done:
                    scores[i] = info['Scores']
                    break
    return scores

