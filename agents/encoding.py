import torch
import torch.nn.functional as F
import numpy as np
from operator import itemgetter

from setup import N_TILE_TYPES
from env.utils import position2id2position, get_n_actions
from utils import cartesian_product

# 2*N_TILE_TYPES + 2 + 1 : one hot encoded tiles + crowns + value of tile
TILE_ENCODING_SIZE = 2*(N_TILE_TYPES+1) + 2 + 1

class ActionEncoding:
    def __init__(self, board_size, n_players):
        self.n_players = n_players
        self.n_actions = get_n_actions(board_size)
        self.position2id,self.id2position = position2id2position(board_size)
        
    def encode(self, actions):
       tiles,positions = actions
       actions_id = itemgetter(positions)(self.position2id)
       actions = np.zeros(self.n_actions * self.n_players)
       actions[actions_id] = 1
       actions[-self.n_players:] = tiles
       return actions
   
    def action_decoding(self, action_id):
        return self.id2action(action_id)

# Assumes the observation at 0 is current player
# Additional information : taken or not (TODO: and by whom)
def current_tiles_encoding(current_tiles, n_players, device='cpu'):
    current_tiles = torch.as_tensor(current_tiles)
    batch_size = current_tiles.shape[0]
    current_tiles_info = torch.zeros([
        batch_size,
        n_players,
        TILE_ENCODING_SIZE+1],
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
        N_TILE_TYPES+3,
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


def state_encoding(state, n_players, device='cpu'):
    boards = boards_encoding(torch.as_tensor(state['Boards'], device=device).unsqueeze(0), n_players, device)
    current_tiles = current_tiles_encoding(torch.as_tensor(state['Current tiles'], device=device).unsqueeze(0), n_players, device)
    previous_tiles = tiles_encoding(torch.as_tensor(state['Previous tiles'], device=device).unsqueeze(0), n_players, device)
    
    return {'Boards':boards, 'Current tiles':current_tiles, 'Previous tiles':previous_tiles}