import agents.base
import numpy as np
import torch
from IPython.core.display import Image, display
from tqdm import tqdm

from graphics import draw_obs

def test(env, players, n_episodes, verbose=-1):
    with torch.no_grad():
        for player in players:
            player.eval()
        n_players = len(players)
        done = False
        scores = np.zeros((n_episodes,n_players))
        for i in range(n_episodes):
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
                        print(scores)
                        break
        return scores

def train(env, players, n_episodes):
    for player in players:
        player.train()
    n_players = len(players)
    rewards_tracker = np.zeros((n_episodes,n_players))
    pbar = tqdm(range(n_episodes))
    for i in pbar:
        s = env.reset()
        d = False
        sum_rewards = np.zeros(n_players)
        while not d:
            for p_id in env.order:
                if not env.first_turn:
                    r = env.getReward(p_id)
                    players[p_id].process_reward(r, False)
                    sum_rewards[p_id] += r
                a = players[p_id].action(s, env)
                s,d,info = env.step(a)
            if d:
                for i,player in enumerate(players):
                    loss = players[p_id].process_reward(r, d)
                    if loss is not None:
                        pbar.set_postfix(**loss)
                    sum_rewards[p_id] += r
        rewards_tracker[i] = sum_rewards
    return rewards_tracker
      
def test_random(env, player, n_episodes):
    with torch.no_grad():
        player.eval()
        random_player = agents.base.RandomPlayer()
        players = [player, random_player]
        n_players = len(players)
        scores = np.zeros((n_episodes,n_players))
        for i in tqdm(range(n_episodes)):
            s = env.reset()
            d = False
            while not d:
                for p_id in env.order:
                    # print('State:')
                    # display(draw_obs(s))
                    a = players[p_id].action(s, env)
                    s,d,info = env.step(a)
                    # print('Action:', a)
                    # display(draw_obs(s))
                    # print('Reward:', env.getReward(p_id))
                    # print('Done:', d)
                    # print('Info', info)
                    if d:
                        scores[i] = info['Scores']
                        break
        return scores

def human_test(env, player):
    player.eval()
    human = agents.base.HumanPlayer()
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