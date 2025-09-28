import multiprocessing
import os
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from kingdomino.kingdomino import Kingdomino
from agents.base import RandomPlayer
from agents.no_learning import MPC_Agent
from setup import TILE_SIZE

def run_data(env, players, n_episodes, seed):
    np.random.seed(seed)
    board_size = env.board_size
    n_players = len(players)
    boards = np.zeros([n_episodes, 13, n_players, n_players, 2, board_size, board_size], dtype=np.int64)
    current_tiles = np.zeros([n_episodes, 13, n_players, n_players, TILE_SIZE+1], dtype=np.int64)
    previous_tiles = np.zeros([n_episodes, 13, n_players, n_players,TILE_SIZE], dtype=np.int64)
    actions = np.zeros([n_episodes, 13, n_players, 5], dtype=np.int64)
    scores = np.zeros((n_episodes, n_players), dtype=np.int64)
    turns = np.zeros((n_episodes, 13, n_players))
    for e in tqdm(range(n_episodes), position=0, leave=True):
    # for e in range(n_episodes):
        state = env.reset()
        done = False
        s = 0
        while not done:
            for i,player_id in enumerate(env.order):
                turns[e, s, i] = player_id
                boards[e, s, player_id] = state['Boards']
                current_tiles[e, s, player_id] = state['Current tiles']
                previous_tiles[e, s] = state['Previous tiles']
                action = players[player_id].action(state, env)
                actions[e, s, player_id] = np.array([action[0], *np.array(action[1]).flatten()])
                state, done = env.step(action)
                if done:
                    sc = env.getScores()
                    scores[e] = sc
                    break
            s += 1
    return {'Boards':boards, 'Current tiles':current_tiles, 'Previous tiles':previous_tiles,
            'Actions':actions, 'Scores':scores, 'Turns':turns}

if __name__ == '__main__':
    # seed = 0
    # n_episodes = 100
    # rollouts = [0, 1, 10, 100]
    # agents_names = ['Random'] + [f'MPC_{str(i)}' for i in rollouts[:1]]
    # folder = 'tournament_results'
    # os.makedirs(folder, exist_ok=True)
    # for name_1,rollout_1 in zip(agents_names, rollouts):
    #     print('->', name_1)
    #     for name_2,rollout_2 in zip(agents_names, rollouts):
    #         path = os.path.join(folder, f'n_episodes_{n_episodes}_{name_1}_{name_2}.npy')
    #         if os.path.exists(path):
    #             print(name_2, 'already done.')
    #             continue
    #         print(name_2)
    #         agent_1 = RandomPlayer() if rollout_1 == 0 else MPC_Agent(8*rollout_1, id=0)
    #         agent_2 = RandomPlayer() if rollout_2 == 0 else MPC_Agent(8*rollout_2, id=1)
    #         with multiprocessing.Pool(processes=4) as pool:
    #             print('OUI ALLO')
    #             data = pool.starmap(run_data, 
    #                 [(Kingdomino(5, random_start_order=False, n_players=2),
    #                 [agent_1, agent_2], 1, seed) for seed in range(n_episodes)])
    #         final_data = {}
    #         for d in data:
    #             for key in d:
    #                 final_data.getattr(key, []).append(d[key])
    #         for k,v in final_data.items():
    #             final_data[k] = np.array(v)
    #         np.save(path, data)
    
    seed = 0
    MPC_rollouts = [8, 80]#, 500, 1000, 10000]
    agents_heuristic_rollouts = [MPC_Agent(i, n_zones_only=True) for i in MPC_rollouts]
    agents_full_rollouts = [MPC_Agent(i, n_zones_only=False) for i in MPC_rollouts]
    agents = [RandomPlayer()] + agents_heuristic_rollouts + agents_full_rollouts
    agents_names = ['Random'] + [f'MPC_heuristic_{str(i)}' for i in MPC_rollouts] + \
        [f'MPC_full_{str(i)}' for i in MPC_rollouts]
    n_episodes = 100
    folder = 'tournament_results'
    os.makedirs(folder, exist_ok=True)
    env = Kingdomino(5, random_start_order=False, n_players=2)
    for name_1,agent_1 in zip(agents_names, agents):
        print('->', name_1)
        for name_2,agent_2 in zip(agents_names, agents):
            path = os.path.join(folder, f'n_episodes_{n_episodes}_{name_1}_{name_2}.npy')
            if os.path.exists(path):
                print(name_2, 'already done.')
                print(np.load(path, allow_pickle=True).item()['Scores'].mean(axis=0))
                print(np.load(path, allow_pickle=True).item()['Scores'].std(axis=0))
                continue
            print(name_2)
            agent_1.id = 0
            agent_2.id = 1
            data = run_data(env, [agent_1, agent_2], n_episodes, seed)
            np.save(path, data)
            print(data['Scores'].mean(axis=0))
            
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

#%%

from os import listdir
from os.path import isfile, join

path = 'tournament_results/'
for f in listdir(path):
    print(f)
    if isfile(join(path, f)):
        print(np.load(path+f, allow_pickle=True).item()['Scores'].mean(axis=0))