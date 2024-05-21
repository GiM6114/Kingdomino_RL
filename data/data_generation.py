# Use MPC agents to generate good actions to pre-train networks

import pickle
import numpy as np

from agents.no_learning import MPC_Agent
from agents.encoding import BOARD_CHANNELS, CUR_TILES_ENCODING_SIZE, TILE_ENCODING_SIZE
from kingdomino.kingdomino import Kingdomino

n_rollouts = 100
n_episodes = 1e6

player_1 = MPC_Agent(n_rollouts=n_rollouts, player_id=0)
player_2 = MPC_Agent(n_rollouts=n_rollouts, player_id=1)

players = [player_1, player_2]

n_players = len(players)
board_size = 5
horizon = 26 # max length of game is 13

done = False

boards_shape = (n_players, BOARD_CHANNELS, board_size, board_size)
prev_tiles_shape = (n_players, TILE_ENCODING_SIZE)
cur_tiles_shape = (n_players, CUR_TILES_ENCODING_SIZE)
action_shape = (4 + n_players,) # (x,y) coordinates of both points + tile selected

boards_states = np.zeros((n_episodes, horizon, *boards_shape))
prev_tiles_states = np.zeros((n_episodes, horizon, *prev_tiles_shape))
cur_tiles_states = np.zeros((n_episodes, horizon, *cur_tiles_shape))
actions = np.zeros((n_episodes, horizon, *action_shape))
rewards = np.zeros((n_episodes, horizon))


env = Kingdomino(
    n_players=len(players),
    board_size=board_size,
    reward_fn=reward_fns[hp['reward_name_id']])

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

data = {
        'states': states, 
        'actions': actions, 
        'rewards': rewards,
        'rewards_to_go': rewards_to_go}
with open(f'kingdomino_MPC_{n_rollouts}'):
    pickle.dump(data, file)