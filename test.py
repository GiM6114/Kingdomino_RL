import gym
import numpy as np

from agent import HumanPlayer, RandomPlayer
from printer import Printer
from kingdomino import Kingdomino


agent = RandomPlayer()
kingdomino = gym.make('KingDomino-v0', n_players=2)

Printer.activated = True
done = False
state,_ = kingdomino.reset()
for i in range(90):
    print(state['Boards'])
    action = list(agent.action(state, kingdomino))
    if (action[1] == Kingdomino.discard_tile).all():
        if action[0] is None:
            action = np.array((-1,-1,4,0))
        else:
            action = np.array((-1,-1,4,action[0]))
    elif action[0] is None:
        action = np.array((
            action[1][0][0],
            action[1][0][1],
            np.where(np.all(np.array(list(Kingdomino.direction.values())) == (action[1][1]-action[1][0]), axis=1))[0][0],
            0))
    elif action[1] is None:
        action = np.array((0,0,0,action[0]))
    else:
        action = np.array((
                    action[1][0][0],
                    action[1][0][1],
                    np.where(np.all(np.array(list(Kingdomino.direction.values())) == (action[1][1]-action[1][0]), axis=1))[0][0],
                    action[0]))
    state,reward,terminated,truncated,infos = kingdomino.step(action)
    print('Reward :', reward)

#%%

import graphics
from IPython.core.display import Image, display

m = player_1.memory
samples = m.sample(min(100,len(m)))
for sample in samples:
    print('------- NEW SAMPLE --------')
    obs,action,reward,next_obs,possible_actions = sample
    display(graphics.draw_obs(obs))
    print(action)
    print(reward)
    print(possible_actions)
    display(graphics.draw_obs(next_obs))
#%%
m = player_1.memory
samples = m.sample(len(m))
for sample in samples:
    obs,action,reward,next_obs,possible_actions = sample
    if reward == 0:
        continue
    display(graphics.draw_obs(obs))
    print(action)
    print(reward)
    print(possible_actions)
    display(graphics.draw_obs(next_obs))
    
#%%
import os
import numpy as np
import matplotlib.pyplot as plt

direc = 'C:/Users/hugom/OneDrive/Documents/ProjetProg/KingdominoAgent/Kingdomino_RL/Models/Model_3/Training_1'
n = 20
s = np.zeros((n,100,2))
r = np.zeros((n,100,2))
for i in range(1,n+1):
    s[i-1] = np.load(os.path.join(direc, str(i*100), 'scores_vs_random.npy'))
    r[i-1] = np.load(os.path.join(direc, str(i*100), 'train_rewards.npy'))
plt.plot(s.mean(1)[:,0], color='blue')
plt.plot(r.mean(1)[:,0], color='red')

#%%

from run import test
from kingdomino import Kingdomino
import agent

player_1 = agent.SelfCenteredPlayer()
player_2 = agent.SelfCenteredPlayer()
players = [player_1, player_2]
env = Kingdomino(
    players=players)
test(env, players, 1, 2)

scores = test(env, players, 100, 1, 10)

#%%
import pickle
import graphics
from IPython.core.display import Image, display
from einops import rearrange
s = 'C:/Users/hugom/OneDrive/Documents/ProjetProg/KingdominoAgent/Kingdomino_RL/Models/Model_4/Training_1/1300/memory.pkl'
# s = s.replace('/','\\')
with open(s,'rb') as f:
    m = pickle.load(f)
samples = m.sample(min(100,len(m)))
for i in range(100):
    print('------- NEW SAMPLE --------')
    board = samples[0][i]
    cur_tiles = samples[1][i]
    prev_tile = samples[2][i]
    action = samples[3][i]
    reward = samples[4][i]
    next_board = samples[5][i]
    next_cur_tiles = samples[6][i]
    next_prev_tile = samples[7][i]
    done = samples[8][i]
    next_actions = samples[9][i]

    obs = (board, cur_tiles, prev_tile)
    display(graphics.draw_encoded_state(obs))
    print(action)
    print(reward)
    next_obs = (next_board, next_cur_tiles, next_prev_tile)
    display(graphics.draw_encoded_state(next_obs))
    print(next_actions)


