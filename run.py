import agent
import numpy as np

from IPython.core.display import Image, display
from graphics import draw_obs

def test(env, players, n_episodes, print_every=10):
    for player in players:
        player.learning = False
    n_players = len(players)
    done = False
    scores = np.zeros((n_episodes,n_players))
    for i in range(n_episodes):
        if i % print_every == 0:
            print('Test episode :', i)
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

def train(env, players, n_episodes, print_every=10):
    for player in players:
        player.learning = True
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
                if reward_next_player is not None:
                    # if reward_next_player != 0:
                    #     print('Reward:', reward_next_player)
                    players[player_id].give_reward(reward_next_player, state)
                    sum_rewards[player_id] += reward_next_player
                action = players[player_id].action(state, env)
                state,reward_next_player,done,info = env.step(action) # reward none for first turn
                if done:
                    break
        rewards_tracker[i] = sum_rewards
    return rewards_tracker

def human_test(env, player):
    player.learning = False
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
    player.learning = False
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
