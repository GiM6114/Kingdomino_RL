import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from IPython.core.display import display

from agents.base import RandomPlayer
from kingdomino.kingdomino import Kingdomino
from kingdomino.rewards import reward_score
from graphics import draw_board

board_size = 5

players = [RandomPlayer(), RandomPlayer()]

boards = []

env = Kingdomino(
    n_players=len(players),
    board_size=board_size,
    reward_fn=reward_score)

n_epochs = 1000
n_episodes_per_epoch = 1000

path = os.path.join('predict_reward', 'data')
if not os.path.exists(path):
    os.mkdir(path)
    
#%%

for e in tqdm(range(n_epochs), position=0, leave=True):
    boards = np.zeros((n_episodes_per_epoch, 13, len(players), 2, board_size, board_size), dtype=int)
    scores = np.zeros((n_episodes_per_epoch, 13, len(players)), dtype=int)
    for i in tqdm(range(n_episodes_per_epoch), position=0, leave=False):
        state = env.reset()
        done = False
        s = 0
        while not done:
            for player_id in env.order:
                action = players[player_id].action(state, env)
                boards[i, s, player_id] = state['Boards'][0]
                reward = env.getReward(player_id)
                scores[i, s, player_id] = reward
                state,done,info = env.step(action)
                if done:
                    break
            s += 1
    np.save(os.path.join(path, f'boards_{e}.npy'), boards)
    np.save(os.path.join(path, f'scores_{e}.npy'), scores)

#%%    
# Test if data makes sense :

boards = np.load('predict_reward/data/boards_0.npy')
scores = np.load('predict_reward/data/scores_0.npy')

img_size = 400

for p in [0,1]:
    for i in range(13):
        img = Image.new("RGB", (img_size,img_size), "white")
        board = boards[0, i, p]
        draw_board(board=board[0], crown=board[1], img=img, board_size=400, top_left=(0,0), grid_size=board_size)
        display(img)
        print(scores[0, i, p])