# Assumes the observation at 0 is current player
# Additional information : taken or not, and by whom
def get_current_tiles_vector(current_tiles, n_players, device):
    batch_size = current_tiles.shape[0]
    current_tiles_info = torch.zeros([
        batch_size,
        n_players,
        TILE_ENCODING_SIZE+1],
        device=device)
    current_tiles_info[:,:,:-1] = tile2onehot(current_tiles[:,:,:-1], n_players, device)
    current_tiles_info[:,:,-1] = current_tiles[:,:,-1]
    return current_tiles_info

def tile2onehot(tiles, n_players, device):
    batch_size = tiles.size()[0]
    tiles_info = torch.zeros([
        batch_size,
        n_players,
        TILE_ENCODING_SIZE],
        device=device)
    tiles_info[:,:,-1] = tiles[:,:,-1] # Value
    # +1 for empty (previous tile first round and current tiles last round)
    tiles_info[:,:,-3:-1] = tiles[:,:,-3:-1] # Crowns
    tiles_info[:,:,:N_TILE_TYPES+1] = F.one_hot(tiles[:,:,0], num_classes=N_TILE_TYPES+1)
    tiles_info[:,:,N_TILE_TYPES+1:-3] = F.one_hot(tiles[:,:,1], num_classes=N_TILE_TYPES+1)
    return tiles_info
    #return prev_tiles_info.reshape(self.batch_size, -1)

# x['Boards'] : (batch_size, n_players, 2, 9, 9)
# Returns : (batch_size, n_players, N_TILE_TYPES+2, 9, 9)
def boards2onehot(boards, n_players, device):
    if state is None:
        return None
    batch_size = boards.shape[0]
    board_size = boards.shape[-1]
    ## BOARDS
    # (N_TILE_TYPES) + 3 : crown + empty tiles + center
    boards_one_hot = torch.zeros([
        batch_size,
        n_players,
        N_TILE_TYPES+3,
        board_size,board_size],
        device=device)
    boards_one_hot.scatter_(2, (boards[:,:,0]+2).unsqueeze(2), 1)
    boards_one_hot[:,:,-1,:,:] = boards[:,:,1] # Place crowns at the end
    return boards_one_hot

# positions_onehot: whether to also onehot encode the position part of the action
def actions2onehot(actions, n_players, device, positions_onehot=False):
    if actions is None:
        return None
    actions_conc = [np.concatenate(([action[0]],action[1].reshape(1,-1).squeeze())) for action in actions]
    # convert first action nb to one_hot
    actions_conc = torch.tensor(np.array(actions_conc), device=device, dtype=torch.int64)
    actions_conc_one_hot_tile_id = \
        torch.zeros((actions_conc.size()[0], n_players+2+2), device=device, dtype=torch.int64)
    actions_conc_one_hot_tile_id[:,n_players:] = actions_conc[:,1:]
    actions_conc_one_hot_tile_id[:,:n_players] = F.one_hot(actions_conc[:,0], num_classes=n_players)
    return actions_conc_one_hot_tile_id