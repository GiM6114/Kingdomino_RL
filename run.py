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
        state = env.reset()
        done = False
        sum_rewards = np.zeros(n_players)
        reward_next_player = None
        while not done:
            for player_id in env.order:
                # Give reward of previous move
                if not env.first_turn:
                    # if reward_next_player != 0:
                    #     print('Reward:', reward_next_player)
                    players[player_id].give_reward(reward_next_player, state, env.empty_end_turn, env.getPossibleActions())
                    sum_rewards[player_id] += reward_next_player
                action = players[player_id].action(state, env)
                state,reward_next_player,done,info = env.step(action) # reward none for first turn
                if done:
                    break
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

