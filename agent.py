import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as func

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


class LearningPlayer(Player):
    
    def __init__(self, _id, epsilon=0.1):
        super(LearningPlayer, self).__init__(_id)
        self.epsilon = 0.1
        self.previous_scores = np.zeros(4, dtype='int16')
    
    def tilePlayerToState(self, tile_player, skip_id=True):
        base = np.array(
            [tile_player.tile.tile1.type, 
             tile_player.tile.tile2.type, 
             tile_player.tile.tile2.nb_crown,
             tile_player.tile.value])
        return base if skip_id else np.append(base, tile_player.player_id)
    
    
    def boardToState(self, board):
        relevant_columns = [0,1,2,3,5,6,7,8]
        env_state = board[self.id].board[:,relevant_columns].flatten()
        crown_state = board[self.id].crown[:,relevant_columns].flatten()
        state = np.append(env_state, crown_state)
        return state
        
    
    # "Rough state" : all data in one big np array
    def gameToState(self, kingdomino):
        # First 160 inputs : the agent's boards (env and crowns (without center))
        state = -np.ones(676)
        state[:160] = self.boardToState(kingdomino.boards[self.id])
        
        
        # Next 160*3 inputs : other player's data or -1s
        other_ids = [i for i in range(kingdomino.nb_players) if i!=self.id]

        idx = 160
        for _id in other_ids:
            state[idx, idx+160] = self.boardToState(kingdomino.boards[self._id])
            idx += 160     
        idx += (4-kingdomino.nb_players) * 160
        
        # Next 4 inputs : the agent's previous tiles data
        state[idx, idx+4] = self.tilePlayerToState(kingdomino.previous_tiles[self.id])
        
        # Next 4*3 inputs : others' previous tiles data
        for _id in other_ids:
            state[idx, idx+4] = self.tilePlayerToState(kingdomino.previous_tiles[_id])
            idx += 4
        idx += (4-kingdomino.nb_players) * 4
       
        # Next 5*4 inputs : current tiles + 1 saying which player owns it
        for current_tile in kingdomino.current_tiles:
            state[idx, idx+5] = self.tilePlayerToState(current_tile, skip_id=False)
            idx += 5
        idx += (4-kingdomino.nb_players) * 5
        
        return state

# Preliminary neural net that reduces information from a board (160) + a previous tile (4) to 10 variables
class PlayersDataNeuralNet(nn.Module):
    def __init__(self):
        super(PlayersDataNeuralNet, self).__init__()
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