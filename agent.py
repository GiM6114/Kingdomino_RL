import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from board import Board

class Player:
    def __init__(self, _id):
        self.id = _id
        self.reset()
        
    def reset(self):
        self.board = Board()

    
    def __str__(self):
        return 'Player ' + str(self.id)
    
    def give_scores(self, scores):
        pass
    
    def action(self, state):
        pass


class HumanPlayer(Player):      
    def action(self, state):
        tile_id = input("Which tile do you choose ?")
        x1 = int(input("x1 ? "))
        y1 = int(input("y1 ? "))      
        x2 = int(input("x2 ? "))
        y2 = int(input("y2 ? "))
        return tile_id, (x1,y1,x2,y2)


class RandomPlayer(Player):
    def action(self, state):
        return (self.kingdomino.selectTileRandom(),
                self.kingdomino.selectTilePositionRandom(self))


class PlayerAC(Player):
    
    def __init__(self, gamma, lr_a, lr_c):
        
        self.gamma = gamma
        
        self.actor = ...
        self.critic = ...
    
    # obs : dictionary of np.arrays
    # state : 1D tensor concatenating these arrays
    # useless : need to treat boards in CNN and other values separately
    def obs_to_state(self, obs):
        state = np.concatenate([v.flatten() for v in obs.values()])
        return torch.from_numpy(state)
    
    def action(self, obs):
        state = self.obs_to_state(obs)
 

class Shared(nn.Module):
    def __init__(self, n_players, board_rep_size, prev_tiles_rep_size, shared_rep_size):
        super(Shared, self).__init__()
        self.n_players = n_players
        self.board_rep_size = board_rep_size
        self.prev_tiles_rep_size = prev_tiles_rep_size
        self.shared_rep_size = shared_rep_size
        
        self.cnn = BoardNetwork(2,self.board_rep_size)
        self.tiles_network = TilesNetwork(self.prev_tiles_rep_size)
        
        self.fc1 = nn.Linear(self.board_rep_size+self.prev_tiles_rep_size, 200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,self.shared_rep_size)

        
    def forward(self, x):
        output = np.zeros((self.n_players,self.shared_rep_size))
        boards_outputs = 
        # loop over players
        for i,board in enumerate(x['Boards']):
            output[i] = self.cnn(board)
            
        
        

class TilesNetwork(nn.Module):
    def __init__(self):
        
        self.

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
        
        
# Preliminary neural net that reduces information from a board (160) + a previous tile (4) to 10 variables
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.nb_outputs = 10
        self.fc1 = nn.Linear(164, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, self.nb_outputs)
        
    def forward(self, x):
        x = func.selu(self.fc1(x))
        x = func.selu(self.fc2(x))
        x = func.selu(self.fc3(x))
        return x

class NeuralNet(nn.Module):
    def __init__(self):
        self.lsm = nn.LogSoftmax(dim=0)
        super(NeuralNet, self).__init__()
        self.players_data_neural_net = PlayersDataNeuralNet()
        self.fc1 = nn.Linear(self.players_data_neural_net.nb_outputs*4 + 5*4, 20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,20)
        self.fc4 = nn.Linear(20,20)
        
        # one-hot encoding for the output of tile to choose
        self.tile_to_take_layer = nn.Linear(20, 4)
        
        # one-hot encoding for the output of position to place previous_tile
        # 8 pos in a line of 9, *2 for inverted tile, *9 for each line = 
        # - 2 for around the castle
        # everything *2 because horizontal and vertical
        self.pos_to_place_layer = nn.Linear(20, 284)
        
    def forward(self, players_data, current_tiles_data):
        players_data = np.apply_along_axis(self.players_data_neural_net.forward, 1, players_data)
        x = np.append(players_data, current_tiles_data)
        
        x = func.selu(self.fc1(x))
        x = func.selu(self.fc2(x))
        x = func.selu(self.fc3(x))   
        x = func.selu(self.fc4(x))
        
        tile_to_take = torch.argmax(self.lsm(func.selu(self.tile_to_take_layer(x))))        
        pos_to_place = torch.argmax(self.lsm(func.selu(self.pos_to_place_layer(x))))
        
        return tile_to_take, pos_to_place

class NNAgent(LearningPlayer):
    
    def __init__(self, device, _id):
        super(NNAgent, self).__init__(_id)
    
    def gameToState(self, kingdomino):
        player_data = -2 * np.ones([4,164]) # -2 : no player 3/4, -1 : empty but there is a player
        player_data[0,:160] = self.boardToState(kingdomino.boards[self.id].board)
        player_data[0,160:164] = self.tilePlayerToState(kingdomino.previous_tiles[self.id])
        
        other_ids = [i for i in range(kingdomino.nb_players) if i!=self.id]
        idx = 160
        for _id in other_ids:
            player_data[_id,idx:idx+160] = self.boardToState(kingdomino.boards[_id].board)
            idx += 160
            player_data[_id,idx:idx+4] = self.tilePlayerToState(kingdomino.previous_tiles[_id])
            idx += 4
        
        current_tiles_data = -2 * np.ones(20)
        idx = 0
        for i,current_tile in enumerate(kingdomino.current_tiles):
            current_tiles_data[i*5, (i+1)*5] = self.tilePlayerToState(current_tile, skip_id=False)
            idx += 5
        
        return player_data, current_tiles_data
        
        
    def playRandom(self, kingdomino):
        pass
    
    def play(self, kingdomino):
        
        if random.rand() < self.epsilon:
            return self.playRandom(kingdomino)
        
        player_data,current_tiles_data = self.gameToState(kingdomino)