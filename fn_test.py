import torch
import torch.nn.functional as F

batch_size = 3
n_tile_types = 6
n_players = 2
board_size = 3

# TEST BOARD TO NETWORK INPUT

boards = torch.zeros([
    batch_size,
    n_players,
    2, # board tiles, crowns
    board_size,
    board_size]
    )
boards[:,:,0] = torch.randint(
    low=-1,
    high=5,
    size=[batch_size,n_players,board_size,board_size])
boards[:,:,0,board_size//2,board_size//2] = -2
boards[:,:,1] = (torch.rand([batch_size,n_players,board_size,board_size]) >= 0.5)
boards = boards.to(torch.int64)
boards[:,:,1,board_size//2,board_size//2] = 0

print('Tile types :', boards[:,:,0])
print('Crowns :', boards[:,:,1])
# Objective : turn the tile types in one-hot encoded matrices

# +2 : crowns,empty tiles
boards_one_hot = torch.zeros([
    batch_size,
    n_players,
    n_tile_types+2,
    board_size,
    board_size])
# +2 because center is -2, needs to be converted to an index
boards_one_hot.scatter_(2, (boards[:,:,0]+2).unsqueeze(2), 1)
boards_one_hot = boards_one_hot[:,:,1:]

boards_one_hot[:,:,-1,:,:] = boards[:,:,1] # place crowns at the end

print('Boards one hot :', boards_one_hot)

# switch rows so that player's board output is always associated
# to the first input neurons
# maybe pb : player_id vector
# boards_output[:,[0,player_id]] = boards_output[:,[player_id,0]]


#%%

# TEST PREV TILES

prev_tiles = torch.tensor([[[1,3,2,0],[3,2,5,5]],
                           [[5,3,4,2],[3,5,0,1]],
                           [[1,3,5,0],[3,2,5,5]]])
prev_tiles_info = torch.zeros([
    3, # batch_size
    2, # n_players
    2*(n_tile_types+1)])
prev_tiles_info[:,:,-2:] = prev_tiles[:,:,-2:] # Crowns
prev_tiles_info[:,:,:n_tile_types] = F.one_hot(prev_tiles[:,:,0], num_classes=n_tile_types)
prev_tiles_info[:,:,n_tile_types:-2] = F.one_hot(prev_tiles[:,:,1], num_classes=n_tile_types)

print(prev_tiles_info)
prev_tiles_info[:,[(0,1,0),(1,0,1)]] = prev_tiles_info[:,[(1,0,1),(0,1,0)]]
print(prev_tiles_info)