import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from setup import N_TILE_TYPES, TILE_SIZE

# 2*N_TILE_TYPES + 2 + 1 : one hot encoded tiles + crowns + value of tile
TILE_ENCODING_SIZE = 2*N_TILE_TYPES + 2 + 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, l, n):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        
        layers = [nn.Linear(self.input_size, n)]
        for i in range(l):
            layers.append(nn.SELU())  # Activation function
            layers.append(nn.Linear(n, n))  # Hidden layers
        layers.append(nn.Linear(n, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BoardNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(BoardNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=n_inputs, out_channels=20,
                               kernel_size=(5,5)) # 1620
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(1,1)) # 405

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
                               kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(1,1))

        self.fc1 = nn.Linear(in_features=450, out_features=500) # not 100    
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=500, out_features=n_outputs)    
        self.relu4 = nn.ReLU()
        
    # x : [batch_size, 2, 9, 9]
    def forward(self, x):
        x = F.pad(x, (2,2,2,2), 'constant', -1) # pad each dimensions
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        x = self.relu4(x)
        
        return x

# /        Board network        /
# Player board -> convo -> fc ->  fc -> Player specific vector
#               Previous tile ->
class PlayerNetwork(nn.Module):
    def __init__(self, board_rep_size, player_rep_size):
        super(PlayerNetwork, self).__init__()
        
        self.board_network = BoardNetwork(
            n_inputs=N_TILE_TYPES+1,
            n_outputs=board_rep_size)
        
        self.join_info_network = NeuralNetwork(
            input_size=board_rep_size + 2*TILE_ENCODING_SIZE,
            output_size=player_rep_size, 
            l=4,
            n=player_rep_size)

    def forward(self, board, previous_tile):
        board_rep = self.board_network(board)
        x = torch.cat((board_rep,previous_tile), dim=1)
        x = self.join_info_network(x)
        return x
        

class Shared(nn.Module):
    def __init__(self, batch_size, n_players, board_rep_size, player_rep_size, shared_rep_size):
        super(Shared, self).__init__()
        self.batch_size = batch_size
        self.n_players = n_players
        self.board_rep_size = board_rep_size
        self.player_rep_size = player_rep_size
        self.shared_rep_size = shared_rep_size
        
        self.player_network = PlayerNetwork(
            board_rep_size=self.board_rep_size,
            player_rep_size=self.player_rep_size)
        
        self.shared_input_size = self.player_rep_size * self.n_players \
            + TILE_ENCODING_SIZE
        
        self.shared_network = NeuralNetwork(
            input_size=self.shared_input_size,
            output_size=self.shared_rep_size,
            l=4, 
            n=200)

    # x['Boards'] : (batch_size, n_players, 2, 9, 9)
    # Returns : (batch_size, n_players, N_TILE_TYPES+2, 9, 9)
    def boards2onehot(self, x):
        board_size = x['Boards'].size()[-1]
        ## BOARDS
        # (N_TILE_TYPES) + 2 : crown + empty tiles
        boards_one_hot = torch.zeros([
            self.batch_size,
            self.n_players,
            N_TILE_TYPES+2,
            board_size,board_size])
        boards_one_hot.scatter_(2, (x['Boards'][:,:,0]+2).unsqueeze(2), 1)
        boards_one_hot = boards_one_hot[:,:,1:] # Exclude center matrices
        boards_one_hot[:,:,-1,:,:] = x['Boards'][:,:,1] # Place crowns at the end
        return boards_one_hot

    # x['Previous tiles'] : (batch_size, n_players, TILE_SIZE (5))
    def tile2onehot(self, tiles):
        ## PREV TILES
        prev_tiles_info = torch.zeros([
            self.batch_size,
            self.n_players,
            TILE_ENCODING_SIZE])
        prev_tiles_info[:,:,-1] = tiles[:,:,-1] # Value
        prev_tiles_info[:,:,-3:-1] = tiles[:,:,-3:-1] # Crowns
        prev_tiles_info[:,:,:N_TILE_TYPES] = F.one_hot(tiles[:,:,0], num_classes=N_TILE_TYPES)
        prev_tiles_info[:,:,N_TILE_TYPES:-3] = F.one_hot(tiles[:,:,1], num_classes=N_TILE_TYPES)
        return prev_tiles_info
        #return prev_tiles_info.reshape(self.batch_size, -1)
    
    # Assumes the observation at 0 is current player
    # Additional information : taken or not, and by whom
    def current_tiles_vector(self, x):
        current_tiles_info = torch.zeros([
            self.batch_size,
            self.n_players,
            TILE_ENCODING_SIZE+1])
        current_tiles_info[:,:,:-1] = self.tile2onehot(x['Current tiles'][:,:,:-1])
        current_tiles_info[:,:,-1] = x['Current tiles'][:,:,-1]
        return current_tiles_info
    
    # Assumes the observation at 0 is current player
    def players_vector(self, x):
        boards_one_hot = self.boards2onehot(x)
        previous_tile_one_hot = self.tile2onehot(x['Previous tiles'])
        
        players_output = torch.zeros([
            self.batch_size,
            self.n_players,
            self.board_rep_size])
        # TODO : parallelize this
        for i in range(self.n_players):
            players_output[:,i] = self.player_network(
                board=boards_one_hot[:,i],
                previous_tile=previous_tile_one_hot[:,i])
        return players_output.reshape(self.batch_size, -1)
    

    


    
    # def curr_tiles_vector(self, player_id, x):
    #     curr_tiles_info = torch.zeros([self.n_players, 2*(N_TILE_TYPES+2+1)+self.n_players-1])
    #     # + n_players : corresponds to one_hot_encoded player_id
    #     curr_tiles_info[:,:N_TILE_TYPES] = F.one_hot(x['Current tiles'][:,0], num_classes=N_TILE_TYPES)
    #     curr_tiles_info[:,N_TILE_TYPES:2*(N_TILE_TYPES+2)] = F.one_hot(x['Current tiles'][:,1], num_classes=N_TILE_TYPES)
    #     curr_tiles_info[:,2*N_TILE_TYPES] = x['Current tiles'][:,2]
    #     curr_tiles_info[:,2*N_TILE_TYPES+1] = x['Current tiles'][:,3]
    #     curr_tiles_info[:,2*(N_TILE_TYPES+2)+1:2*(N_TILE_TYPES+2)+1+self.n_players-1] = \
    #         F.one_hot(x['Current tiles'][:,5], num_classes=self.n_players-1) \
    #             * (x['Current tiles'][:,5] != -1)
    #     return curr_tiles_info.flatten()

    # equivalence of players not taken into account (should share weights) 
    def forward(self, player_id, x):
        print(x)
        for key in x:
            x[key] = torch.tensor(x[key], dtype=torch.int64)
        players_vector = self.players_vector(player_id, x).reshape(self.batch_size, -1)
        current_tiles_vector = self.current_tiles_vector(player_id, x)
        
        x = torch.cat([players_vector, current_tiles_vector], dim=1)
        x = self.shared_network(x)
        return x

#%%

if __name__ == "__main__":

    s = Shared(3,2,10,20,30)
    boards = torch.zeros([3, 2, 2, 3, 3])
    boards[:,:,0] = torch.randint(
        low=-1,
        high=5,
        size=[3,2,3,3])
    boards[:,:,0,3//2,3//2] = -2
    boards[:,:,1] = (torch.rand([3,2,3,3]) >= 0.5)
    boards = boards.to(torch.int64)
    boards[:,:,1,3//2,3//2] = 0
    prev_tiles = torch.tensor([[[1,3,0,0,20],[3,2,0,2,35]],
                               [[5,3,0,1,55],[3,5,0,2,10]],
                               [[1,3,1,5,59],[3,2,0,0,75]]])
    curr_tiles = torch.tensor([[[1,3,0,0,20,3],[3,2,0,2,35,0]],
                               [[5,3,0,1,55,2],[3,5,0,2,10,2]],
                               [[1,3,1,5,59,1],[3,2,0,0,75,3]]])
    x = {'Boards':boards,
         'Previous tiles':prev_tiles,
         'Current tiles':curr_tiles}
    
    print(x['Boards'])
    print(s.boards2onehot(x))
    print(x['Previous tiles'])
    print(s.tile2onehot(x['Previous tiles']))
    print(x['Current tiles'])
    print(s.current_tiles_vector(x))