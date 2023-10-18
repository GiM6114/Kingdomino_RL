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
    
class CNN(nn.Module):
    def __init__(self,
                 in_channels,
                 conv_channels, conv_kernel_size, conv_stride, l,
                 pool_place, pool_kernel_size, pool_stride
                 ):
        super(CNN, self).__init__()
        
        layers = [nn.Conv2d(
            in_channels = in_channels,
            out_channels = conv_channels,
            kernel_size = conv_kernel_size,
            stride = conv_stride)
            ]
        for i in range(l):
            layers.append(nn.ReLU())
            if pool_place[i] != 0:
                layers.append(nn.MaxPool2d(
                    kernel_size = pool_kernel_size,
                    stride = pool_stride))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BoardNetwork(nn.Module):
    def __init__(self, n_inputs, network_info):
        super(BoardNetwork, self).__init__()
        
        self.cnn = CNN(
            in_channels = n_inputs,
            conv_channels = network_info.conv_channels,
            conv_kernel_size = network_info.conv_kernel_size,
            conv_stride = network_info.conv_stride,
            l = network_info.conv_l,
            pool_place = network_info.pool_place,
            pool_kernel_size = network_info.pool_kernel_size,
            pool_stride = network_info.pool_stride
            )
        self.fc = NeuralNetwork(
            input_size = 3200, # modify according to error...or compute conv accordingly
            output_size = network_info.board_rep_size,
            n = network_info.board_fc_n,
            l = network_info.board_fc_l)
        
        
    # x : [batch_size, 2, 9, 9]
    def forward(self, x):
        x = F.pad(x, (2,2,2,2), 'constant', -1) # pad each dimensions
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# /        Board network        /
# Player board -> convo -> fc ->  fc -> Player specific vector
#               Previous tile ->
class PlayerNetwork(nn.Module):
    def __init__(self, network_info):
        super(PlayerNetwork, self).__init__()
        
        self.board_network = BoardNetwork(
            n_inputs=N_TILE_TYPES+1,
            network_info=network_info)
        
        self.join_info_network = NeuralNetwork(
            input_size = network_info.board_rep_size + TILE_ENCODING_SIZE,
            output_size = network_info.player_rep_size, 
            l = network_info.board_prev_tile_fc_l,
            n = network_info.player_rep_size)

    def forward(self, board, previous_tile):
        board_rep = self.board_network(board)
        x = torch.cat((board_rep,previous_tile), dim=1)
        x = self.join_info_network(x)
        return x
        

class Shared(nn.Module):
    def __init__(self, n_players, network_info):
        super(Shared, self).__init__()
        self.n_players = n_players
        self.network_info = network_info
        
        self.player_network = PlayerNetwork(network_info)
        
        shared_input_size = \
            (self.network_info.player_rep_size + TILE_ENCODING_SIZE+1) * self.n_players
            
        self.shared_network = NeuralNetwork(
            input_size=shared_input_size,
            output_size=self.network_info.shared_rep_size,
            l=self.network_info.shared_l, 
            n=self.network_info.shared_n)

    # x['Boards'] : (batch_size, n_players, 2, 9, 9)
    # Returns : (batch_size, n_players, N_TILE_TYPES+2, 9, 9)
    def boards2onehot(self, x):
        batch_size = x['Boards'].size()[0]
        board_size = x['Boards'].size()[-1]
        ## BOARDS
        # (N_TILE_TYPES) + 2 : crown + empty tiles
        boards_one_hot = torch.zeros([
            batch_size,
            self.n_players,
            N_TILE_TYPES+2,
            board_size,board_size])
        boards_one_hot.scatter_(2, (x['Boards'][:,:,0]+2).unsqueeze(2), 1)
        boards_one_hot = boards_one_hot[:,:,1:] # Exclude center matrices
        boards_one_hot[:,:,-1,:,:] = x['Boards'][:,:,1] # Place crowns at the end
        return boards_one_hot

    # x['Previous tiles'] : (batch_size, n_players, TILE_SIZE (5))
    def tile2onehot(self, tiles):
        batch_size = tiles.size()[0]
        prev_tiles_info = torch.zeros([
            batch_size,
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
        batch_size = x['Boards'].size()[0]
        current_tiles_info = torch.zeros([
            batch_size,
            self.n_players,
            TILE_ENCODING_SIZE+1])
        current_tiles_info[:,:,:-1] = self.tile2onehot(x['Current tiles'][:,:,:-1])
        current_tiles_info[:,:,-1] = x['Current tiles'][:,:,-1]
        return current_tiles_info
    
    # Assumes the observation at 0 is current player
    def players_vector(self, x):
        with torch.no_grad():
            batch_size = x['Boards'].size()[0]
            boards_one_hot = self.boards2onehot(x)
            previous_tile_one_hot = self.tile2onehot(x['Previous tiles'])
            
        players_output = torch.zeros([
            batch_size,
            self.n_players,
            self.network_info.player_rep_size])
        # TODO : parallelize this ?
        # for i in range(self.n_players):
        #     players_output[:,i] = self.player_network(
        #         board=boards_one_hot[:,i],
        #         previous_tile=previous_tile_one_hot[:,i])
        return players_output.reshape(batch_size, -1)

    # equivalence of players not taken into account (ideally should share weights) 
    def forward(self, x):
        batch_size = x['Boards'].size()[0]
        players_vector = self.players_vector(x).reshape(batch_size, -1)
        with torch.no_grad():
            current_tiles_vector = self.current_tiles_vector(x)
        x = torch.cat([players_vector, current_tiles_vector.reshape(batch_size,-1)], dim=1)
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