import torch
import torch.nn.functional as F
import numpy as np
from operator import itemgetter
from itertools import product

from setup import N_TILE_TYPES
from kingdomino.utils import compute_n_positions, compute_n_actions
from utils import cartesian_product
from kingdomino.kingdomino import Kingdomino

# 2*N_TILE_TYPES + 2 + 1 : one hot encoded tiles + crowns + value of tile
TILE_ENCODING_SIZE = 2*(N_TILE_TYPES+1) + 2 + 1
CUR_TILES_ENCODING_SIZE = TILE_ENCODING_SIZE + 1
BOARD_CHANNELS = N_TILE_TYPES+2

class ActionInterface:
    def encode(self, action):
        print('Action',action)
        if len(action) == 1:
            actions_id = self.action2id[action[0]]
        else:
            actions_id = list(itemgetter(*action)(self.action2id))
        actions = np.zeros(self.n_actions, dtype=bool)
        print(actions_id)
        actions[actions_id] = 1
        return actions
    
    def expand_possible_actions(self, p_a):
        return p_a
    
    # network --> Kingdomino
    def decode_action_id(self, action_id):
        return self.id2action[action_id]
    
    # returns the list of actions where actions_id == 1
    def decode_action_vector(self, actions_id):
        ids = np.where(actions_id.cpu().numpy() == 1)
        actions = list(itemgetter(*list(ids[0]))(self.id2action))
        return actions

class CoordinateInterface(ActionInterface):
    def __init__(self, board_size):
        self.board_size = board_size
        self.n_actions = compute_n_positions(board_size) + 1
        self.action2id,self.id2action = CoordinateInterface.coordinate2id2coordinate(board_size)
    
    def coordinate2id2coordinate(board_size):
        middle = board_size // 2
        action2id = {}
        id2action = []
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
                        action2id[pos] = len(id2action)
                        id2action.append(np.array(pos))
        # Discard tile
        pos = tuple(map(tuple, Kingdomino.discard_tile))
        action2id[pos] = len(id2action)
        id2action.append(pos)               
        return action2id,id2action
    
class TileInterface(ActionInterface):
    def __init__(self, n_players):
        self.n_players = n_players
        self.n_actions = self.n_players
        self.action2id,self.id2action = TileInterface.tile2id2tile(n_players)

    def tile2id2tile(n_players):
        id2action = [(i,) for i in range(-1,n_players)]
        action2id = {(i-1,):i for i in range(n_players)}
        return id2action,action2id        

class TileCoordinateInterface(ActionInterface):
    def __init__(self, n_players, board_size):
        self.n_players = n_players
        self.board_size = board_size
        self.n_actions = (n_players+1)*(compute_n_positions(board_size)+1)
        self.action2id,self.id2action = TileCoordinateInterface.action2id2action(n_players, board_size)
    
    def expand_possible_actions(self, p_a):
        print(p_a)
        return list(product(*p_a))
    
    # tiles: np array of possible tiles (ex: 0,2)
    # positions: np array of possible positions
    def encode(self, action):
        print('Action to encode', action)
        if len(action) == 1:
            actions_id = self.action2id[(action[0][0], action[0][1])]
        else:
            actions_id = list(itemgetter(*action)(self.action2id))
        actions = np.zeros(self.n_actions, dtype=bool)
        actions[actions_id] = 1
        return actions
    
    def action2id2action(n_players, board_size):
        middle = board_size // 2
        action2id = {}
        id2action = []
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
                        
                        for p in list(range(-1,n_players)):
                            action2id[(p,pos)] = len(id2action)
                            id2action.append((p,np.array(pos)))
        # Discard tile
        pos = tuple(map(tuple, Kingdomino.discard_tile))
        for p in list(range(-1,n_players)):
            action2id[(p,pos)] = len(id2action)
            id2action.append((p,pos))
        return action2id,id2action
   


# Assumes the observation at 0 is current player
# Additional information : taken or not (TODO: and by whom)
def current_tiles_encoding(current_tiles, n_players, device='cpu'):
    current_tiles = torch.as_tensor(current_tiles)
    batch_size = current_tiles.shape[0]
    current_tiles_info = torch.zeros([
        batch_size,
        n_players,
        CUR_TILES_ENCODING_SIZE],
        dtype=torch.int8,
        device=device)
    current_tiles_info[:,:,:-1] = tiles_encoding(current_tiles[:,:,:-1], n_players, device)
    current_tiles_info[:,:,-1] = current_tiles[:,:,-1]
    return current_tiles_info.reshape(batch_size,-1)

def tiles_encoding(tiles, n_players, device='cpu'):
    batch_size = tiles.shape[0]
    tiles_info = torch.zeros([
        batch_size,
        n_players,
        TILE_ENCODING_SIZE],
        dtype=torch.int8,
        device=device)
    tiles_info[:,:,-1] = tiles[:,:,-1] # Value
    # +1 for empty (previous tile first round and current tiles last round)
    tiles_info[:,:,-3:-1] = tiles[:,:,-3:-1] # Crowns
    tiles_info[:,:,:N_TILE_TYPES+1] = F.one_hot(tiles[:,:,0], num_classes=N_TILE_TYPES+1)
    tiles_info[:,:,N_TILE_TYPES+1:-3] = F.one_hot(tiles[:,:,1], num_classes=N_TILE_TYPES+1)
    return tiles_info
    #return prev_tiles_info.reshape(self.batch_size, -1)

# boards : (batch_size, n_players, 2, 9, 9)
# Returns : (batch_size, n_players, N_TILE_TYPES+2, 9, 9)
def boards_encoding(boards, n_players, device='cpu'):
    boards = torch.as_tensor(boards, device=device)
    batch_size = boards.shape[0]
    board_size = boards.shape[-1]
    # (N_TILE_TYPES) + 3 : crown + empty tiles + center
    boards_one_hot = torch.zeros([
        batch_size,
        n_players,
        BOARD_CHANNELS+1,
        board_size,board_size],
        dtype=torch.int8,
        device=device)
    boards_one_hot.scatter_(2, (boards[:,:,0]+2).unsqueeze(2), 1)
    boards_one_hot[:,:,-1,:,:] = boards[:,:,1] # Place crowns at the end
    return boards_one_hot[:,:,1:]

# positions_onehot: whether to also onehot encode the position part of the action
def actions_encoding(actions, n_players, device='cpu', positions_onehot=False):
    actions_conc = [np.concatenate(([action[0]],action[1].reshape(1,-1).squeeze())) for action in actions]
    # convert first action nb to one_hot
    actions_conc = torch.tensor(
        np.array(actions_conc), 
        device=device, 
        dtype=torch.int64)
    actions_conc_one_hot_tile_id = \
        torch.zeros((
            actions_conc.shape[0], 
            n_players+2+2), 
            device=device, 
            dtype=torch.int8)
    actions_conc_one_hot_tile_id[:,n_players:] = actions_conc[:,1:]
    actions_conc_one_hot_tile_id[:,:n_players] = F.one_hot(actions_conc[:,0], num_classes=n_players)
    return actions_conc_one_hot_tile_id


# only works for no batch, single input...
# (it is ok because only needed when environment used,
# but will be annoying if parallelization several envs)
def state_encoding(state, n_players, device='cpu'):
    boards = boards_encoding(torch.as_tensor(state['Boards'], device=device).unsqueeze(0), n_players, device).float()
    current_tiles = current_tiles_encoding(torch.as_tensor(state['Current tiles'], device=device).unsqueeze(0), n_players, device).float()
    previous_tiles = tiles_encoding(torch.as_tensor(state['Previous tiles'], device=device).unsqueeze(0), n_players, device).float()  
    state['Boards'] = boards
    state['Current tiles'] = current_tiles
    state['Previous tiles'] = previous_tiles
    return state