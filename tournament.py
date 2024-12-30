import os
from tqdm import tqdm
import numpy as np

from kingdomino.kingdomino import Kingdomino
from agents.base import RandomPlayer
from agents.no_learning import MPC_Agent
from setup import TILE_SIZE

def run_data(kingdomino, players, n_episodes):
    board_size = kingdomino.board_size
    n_players = len(players)
    boards = np.zeros([n_episodes, 13, n_players, n_players, 2, board_size, board_size], dtype=np.int64)
    current_tiles = np.zeros([n_episodes, 13, n_players, n_players, TILE_SIZE+1], dtype=np.int64)
    previous_tiles = np.zeros([n_episodes, 13, n_players, n_players,TILE_SIZE], dtype=np.int64)
    actions = np.zeros([n_episodes, 13, n_players, 5], dtype=np.int64)
    scores = np.zeros((n_episodes, n_players), dtype=np.int64)
    turns = np.zeros((n_episodes, 13))
    for e in tqdm(range(n_episodes), position=0, leave=True):
        state = env.reset()
        print(kingdomino.current_tiles)
        print(state['Current tiles'])
        done = False
        s = 0
        while not done:
            for player_id in env.order:
                turns[e, s] = player_id
                boards[e, s, player_id] = state['Boards']
                current_tiles[e, s, player_id] = state['Current tiles']
                previous_tiles[e, s] = state['Previous tiles']
                action = players[player_id].action(state, env)
                actions[e, s, player_id] = np.array([action[0], *np.array(action[1]).flatten()])
                state, done = env.step(action)
                if done:
                    sc = kingdomino.getScores()
                    scores[e] = sc
                    break
            s += 1
    return {'Boards':boards, 'Current tiles':current_tiles, 'Previous tiles':previous_tiles,
            'Actions':actions, 'Scores':scores}

if __name__ == '__main__':
    MPC_rollouts = [1, 10]#, 100, 500, 1000, 10000]
    agents = [RandomPlayer()]# + [MPC_Agent(8*i) for i in MPC_rollouts]
    agents_names = ['Random']# + [f'MPC_{str(i)}' for i in MPC_rollouts]
    n_episodes = 10
    folder = 'tournament_results'
    os.makedirs(folder, exist_ok=True)
    for name_1,agent_1 in zip(agents_names, agents):
        print('->', name_1)
        for name_2,agent_2 in zip(agents_names, agents):
            path = os.path.join(folder, f'n_episodes_{n_episodes}_{name_1}_{name_2}.npy')
            # if os.path.exists(path):
            #     print(name_2, 'already done.')
            #     continue
            print(name_2)
            env = Kingdomino(5, random_start_order=False, n_players=2)
            agent_1.id = 0
            agent_2.id = 1
            data = run_data(env, [agent_1, agent_2], n_episodes)
            np.save(path, data)
            
#%%
import numpy as np
from IPython.core.display import display
from graphics import draw_obs

a = np.load('tournament_results/n_episodes_10_Random_Random.npy', allow_pickle=True).item()
for i in range(13):
    for j in range(2):
        display(draw_obs({'Boards':a['Boards'][0,i,j], 'Current tiles':a['Current tiles'][0,i,j],
                          'Previous tiles':a['Previous tiles'][0,i,j]}))
        print(a['Actions'][0,i,j])
print(a['Scores'][0])