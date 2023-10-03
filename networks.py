import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from setup import N_TILE_TYPES


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

class Shared(nn.Module):
    def __init__(self, n_players, board_rep_size, shared_rep_size):
        super(Shared, self).__init__()
        self.n_players = n_players
        self.board_rep_size = board_rep_size
        self.shared_rep_size = shared_rep_size
        
        self.board_network = BoardNetwork(N_TILE_TYPES,self.board_rep_size)
        
        shared_input_size = self.board_rep_size \
            + self.n_players*(board_rep_size + 2*(N_TILE_TYPES+2+1))
        
        self.fc1 = nn.Linear(shared_input_size, 200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,200)
        self.fc4 = nn.Linear(200,self.shared_rep_size)

    def board_vector(self, player_id, x):
        ## BOARDS        
        boards_info = torch.zeros([self.n_players,N_TILE_TYPES+1,9,9])
        # +1 : crown
        board_types_info = torch.zeros(self.n_players, N_TILE_TYPES+2, 9, 9)
        board_types_info.scatter_(1, (x['Boards'][:,0]+2).unsqueeze(1), 1)[2:]
        
        boards_info[:,-1,:,:] = x['Boards'][:,1]
        boards_info[:,:-1,:,:] = board_types_info
        
        boards_output = self.board_network(boards_info)
        # switch rows so that player's board output is always associated
        # to the first input neurons
        boards_output[[0,player_id]] = boards_output[[player_id,0]]
        return boards_output.flatten() 

    def prev_tiles_vector(self, player_id, x):
        ## PREV TILES
        prev_tiles_info = torch.zeros([self.n_players, 2*(N_TILE_TYPES+2+1)])
        prev_tiles_info[:,-2] = x['Previous tiles'][:,-3:-1]
        prev_tiles_info[:,:N_TILE_TYPES+2] = F.one_hot(x['Previous tiles'][:,0], num_classes=N_TILE_TYPES)
        prev_tiles_info[:,N_TILE_TYPES+2:-2] = F.one_hot(x['Previous tiles'][:,1], num_classes=N_TILE_TYPES)
        # switch rows so player is first
        prev_tiles_info[[0,player_id]] = prev_tiles_info[[player_id,0]]
        return prev_tiles_info.flatten()        

    def curr_tiles_vector(self, player_id, x):
        curr_tiles_info = torch.zeros([self.n_players, 2*(N_TILE_TYPES+2+1)+self.n_players-1])
        # + n_players : corresponds to one_hot_encoded player_id
        curr_tiles_info[:,:N_TILE_TYPES+2] = F.one_hot(x['Current tiles'][:,0], num_classes=N_TILE_TYPES)
        curr_tiles_info[:,N_TILE_TYPES+2:2*(N_TILE_TYPES+2)] = F.one_hot(x['Current tiles'][:,1], num_classes=N_TILE_TYPES)
        curr_tiles_info[:,2*(N_TILE_TYPES+2)] = x['Current tiles'][:,2]
        curr_tiles_info[:,2*(N_TILE_TYPES+2)+1] = x['Current tiles'][:,3]
        curr_tiles_info[:,2*(N_TILE_TYPES+2)+1:2*(N_TILE_TYPES+2)+1+self.n_players-1] = \
            F.one_hot(x['Current tiles'][:,5], num_classes=self.n_players-1) \
                * (x['Current tiles'][:,5] != -1)
        return curr_tiles_info.flatten()

    # equivalence of players not taken into account (should share weights) 
    def forward(self, player_id, x):
        boards_output = self.board_vector(player_id, x)
        prev_tiles_vector = self.prev_tiles_vector(player_id, x)
        curr_tiles_vector = self.curr_tiles_vector(player_id, x)
        # TODO : donner ordre de jeu aussi, rajouter à obs
        
        x = torch.cat([boards_output, prev_tiles_vector, curr_tiles_vector])
        
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        x = F.selu(x)
        x = self.fc3(x)
        x = F.selu(x)
        x = self.fc4(x)
        x = F.selu(x)
        
        return x


class BoardNetwork(nn.Module):
    def __init__(self, n_channels, n_outputs):
        super(BoardNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=20,
                               kernel_size=(5,5)) # 1620
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) # 405

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
                               kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = nn.Linear(in_features=100, out_features=500) # not 100    
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=500, out_features=n_outputs)    
        self.relu4 = nn.ReLU()
        
    # x : [n_players, 2, 9, 9]
    def forward(self, x):
        x = F.pad(x, (2,2,2,2), 'constant', -1) # pad each board
        
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
        