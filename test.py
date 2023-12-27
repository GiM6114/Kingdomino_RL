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