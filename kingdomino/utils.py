import numpy as np

from kingdomino.kingdomino import Kingdomino

def get_n_positions(board_size):
    # + 1: discard tile
    return 4*board_size**2 - 4*board_size - 8

def get_n_actions(board_size, n_players):
    n_positions = get_n_positions(board_size)
    return (n_players+1)*(n_positions+1)

def action2id2action(board_size, n_players):
    middle = board_size // 2
    action2id = {}
    id2action = []
    n_positions = get_n_positions(board_size)
    for i in range(board_size):
        for j in range(board_size):
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if i+ii < 0 or i+ii > board_size-1 or j+jj < 0 or j+jj > board_size - 1:
                        continue
                    if (abs(ii) == abs(jj)):
                        continue
                    if (i == middle and j == middle) or (i+ii == middle and j+jj == middle):
                        continue
                    pos = ((i,j),(i+ii,j+jj))
                    
                    for id,p in enumerate([-1] + list(range(n_players))):
                        action2id[(p,pos)] = len(id2action)
                        id2action.append((p,np.array(pos)))
    # Discard tile
    pos = tuple(map(tuple, Kingdomino.discard_tile))
    for id,p in enumerate([-1] + list(range(n_players))):
        action2id[(p,pos)] = len(id2action)
        id2action.append((p,pos))
    assert (len(action2id.keys()) == (n_players+1)*(n_positions+1))
    assert(len(id2action) == (n_players+1)*(n_positions+1))
    return action2id,id2action

