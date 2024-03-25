import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange

from setup import N_TILE_TYPES, TILE_SIZE
from agents.encoding import TILE_ENCODING_SIZE, state_encoding



# use adaptive CNN
# maybe use different network for current player and other players...
# ...because current player is not just any player !
# give as input to the current player network not just board
# and previous tile, but also action (as auxiliary info of the ACNN)


    
class FC(nn.Module):
    def __init__(self, input_size, output_size, l, n):
        super(FC, self).__init__()
        prev_n = input_size
        layers = []
        for i in range(l):
            layers.append(nn.Linear(prev_n, n[i]))  # Hidden layers
            layers.append(nn.SELU())  # Activation function
            prev_n = n[i]
        layers.append(nn.Linear(prev_n, output_size))
        layers.append(nn.SELU())  # Activation function
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class AdaptiveCNN(nn.Module):
    def __init__(
            self, aux_info_size, in_channels,
            conv_channels, conv_kernel_size, conv_stride, l,
            pool_place, pool_kernel_size, pool_stride,
            FMN_l, FMN_n):
        super(AdaptiveCNN, self).__init__()
        
        prev_n_channels = in_channels
        FMNs = []
        for i in range(l):
            parameters_size = (prev_n_channels*conv_channels[i]*(conv_kernel_size[i]**2))+conv_channels[i]
            FMNs.append(FC(aux_info_size, parameters_size, FMN_l, FMN_n))
            prev_n_channels = conv_channels[i]
        self.FMNs = nn.ModuleList(FMNs)
        
        self.conv_stride = conv_stride
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_channels = conv_channels
        self.pool_place = pool_place
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.l = l
        
    def forward(self, main, side):
        prev_n_channels = self.in_channels
        for i in range(self.l):
            wb = self.FMNs[i](side) # (batch,(kernel_size**2)*prev_n_channels*n_channels + n_channels)
            w = wb[:,:-self.conv_channels[i]].reshape(main.shape[0], self.conv_channels[i], prev_n_channels, self.conv_kernel_size[i], self.conv_kernel_size[i])
            b = wb[:,-self.conv_channels[i]:]
            pad = self.conv_kernel_size[i]//2
            original_main_shape = main.shape
            main = F.pad(main, (pad,pad,pad,pad), 'constant', -1) # pad each dimensions
            main = F.conv2d(
                input=main.reshape((1,-1)+main.shape[2:]), 
                weight=w.reshape((-1,)+w.shape[2:]),
                bias=b.reshape((-1,)+b.shape[2:]),
                groups=w.shape[0]).reshape((original_main_shape[0], w.shape[1])+original_main_shape[2:])
            if self.pool_place[i] != 0:
                main = F.max_pool2d(main, self.pool_kernel_size[i], self.pool_stride[i])
            main = F.selu(main)
            prev_n_channels = self.conv_channels[i]
        return main


class DQN(nn.Module):
    def __init__(self,
                 n_players,
                 network_info,
                 output_size,
                 device):
        super().__init__()
        self.shared = Shared(
            n_players=n_players,
            network_info=network_info,
            output_size=network_info['shared_rep_size'],
            device=device).to(device)
        self.action_and_shared = FC(
            input_size=network_info['shared_rep_size'] + 2 + 2 + n_players,
            output_size=1,
            l=network_info['shared_l'],
            n=network_info['shared_n'])
        
    # state : batch_size * messy (batch_size = 1 ==> forward with multiple actions)
    # action: batch_size * 6 OR #possible_actions * 6
    def forward(self, state, action):
        state = self.shared(state) # batch_size * shared_rep_size
        if state.shape[0] == 1:
            # case where several actions applied to a single state
            state = state.repeat(repeats=(action.size(dim=0),1))
        x = torch.cat((state,action), dim=1)
        x = self.action_and_shared(x)
        return x
    
# Takes the whole board as input of a FC
# One Qvalue output
class PlayerFocusedFC(nn.Module):
    def __init__(self, n_players, network_info, device):
        super().__init__()
        self.n_players = n_players
        self.device = device
        grid_size = network_info['grid_size']
        # board + cur_tiles_info + cur_tiles_belong_player + prev_tile
        state_size = 8*(grid_size**2) + TILE_ENCODING_SIZE*(n_players+1) + n_players
        action_size = n_players + 4
        # action_size = 4*grid_size**2  + 2*grid_size - 11
        self.network = FC(
            input_size=state_size+action_size,
            output_size=1, 
            l=network_info['n_hidden_layers'],
            n=network_info['hidden_layers_width'])
        
    def filter_encoded_state(self, encoded_state):
        encoded_state['Boards'] = encoded_state['Boards'][:,0]
        encoded_state['Previous tiles'] = encoded_state['Previous tiles'][:,0]
        
    def forward(self, state, action):
        board_state,cur_tiles,prev_tile = state
        board_state = rearrange(board_state, 'b c g1 g2 -> b (c g1 g2)')
        x = torch.cat((board_state, cur_tiles, prev_tile, action), dim=1)
        x = self.network(x.float())
        return x
    
# Action of player, previous tile of player, current tiles in FC networks
# outputting the weights of each convolutional layer on the board
class PlayerFocusedACNN(nn.Module):
    def __init__(self, n_players, network_info, device):
        super().__init__()
        self.n_players = n_players
        self.device = device
        action_size = n_players + 4
        self.acnn = AdaptiveCNN(
            aux_info_size = action_size + TILE_ENCODING_SIZE + n_players*(TILE_ENCODING_SIZE+1), 
            in_channels = 9, 
            conv_channels = network_info['conv_channels'], 
            conv_kernel_size = network_info['conv_kernel_size'], 
            conv_stride = network_info['conv_stride'], 
            l = network_info['conv_l'], 
            pool_place = network_info['pool_place'], 
            pool_kernel_size = network_info['pool_kernel_size'], 
            pool_stride = network_info['pool_stride'], 
            FMN_l = network_info['FMN_l'], 
            FMN_n = network_info['FMN_n'])
        self.fc = FC(324, 1, network_info['fc_l'], network_info['fc_n'])
    
    # modifies encoded state so that useless information (related to other players) is not saved into the replay buffer
    def filter_encoded_state(self, encoded_state):
        encoded_state['Boards'] = encoded_state['Boards'][:,0]
        encoded_state['Previous tiles'] = encoded_state['Previous tiles'][:,0]
    
    # state: (boards,current_tiles,previous_tiles)
    # state is (b*varying x state_shape) with action (b x action_shape)
    def forward(self, state, action):
        boards,cur_tiles,prev_tiles = state
        actions_size = action.shape[0] # equal to state_batch_size if optimizing
        aux = torch.cat([action, cur_tiles, prev_tiles],dim=1)
        # actions_size: batch size or number of possible actions if we are choosing the best action
        # print('Aux:',aux)
        x = self.acnn(boards.float().squeeze(1), aux.float()).reshape(actions_size, -1)
        x = self.fc(x)
        return x

class CNN(nn.Module):
    def __init__(self,
                 in_channels,
                 conv_channels, conv_kernel_size, conv_stride, l,
                 pool_place, pool_kernel_size, pool_stride
                 ):
        super(CNN, self).__init__()
        
        layers = []
        for i in range(l):
            layers.append(nn.Conv2d(
                in_channels = in_channels,
                out_channels = conv_channels[i],
                kernel_size = conv_kernel_size[i],
                stride = conv_stride[i]))
            layers.append(nn.SELU())
            if pool_place[i] != 0:
                layers.append(nn.MaxPool2d(
                    kernel_size = pool_kernel_size,
                    stride = pool_stride))
            in_channels = conv_channels[i]    
            
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                pad = layer.kernel_size[0]//2
                x = F.pad(x, (pad,pad,pad,pad), 'constant', -1) # pad each dimensions
            x = layer(x)
        return x

class BoardNetwork(nn.Module):
    def __init__(self, n_inputs, network_info):
        super(BoardNetwork, self).__init__()
        
        self.cnn = CNN(
            in_channels = n_inputs,
            conv_channels = network_info['conv_channels'],
            conv_kernel_size = network_info['conv_kernel_size'],
            conv_stride = network_info['conv_stride'],
            l = network_info['conv_l'],
            pool_place = network_info['pool_place'],
            pool_kernel_size = network_info['pool_kernel_size'],
            pool_stride = network_info['pool_stride']
            )
        self.fc = FC(
            input_size = 405, # modify according to error...or compute conv accordingly
            output_size = network_info['board_rep_size'],
            n = network_info['board_fc_n'],
            l = network_info['board_fc_l'])
        
    # x : [batch_size, 2, 9, 9]
    def forward(self, x):
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
            n_inputs=N_TILE_TYPES+3,
            network_info=network_info)
        
        self.join_info_network = FC(
            input_size = network_info['board_rep_size'] + TILE_ENCODING_SIZE,
            output_size = network_info['player_rep_size'], 
            l = network_info['board_prev_tile_fc_l'],
            n = network_info['player_rep_size'])

    def forward(self, board, previous_tile):
        board_rep = self.board_network(board)
        x = torch.cat((board_rep,previous_tile), dim=1)
        x = self.join_info_network(x)
        return x
        


class Shared(nn.Module):
    def __init__(
            self, n_players, device, network_info, output_size):
        super(Shared, self).__init__()
        self.n_players = n_players
        self.network_info = network_info
        self.device = device
        
        self.other_players_network = PlayerNetwork(network_info)
        self.player_network = PlayerNetwork(network_info)
        
        shared_input_size = \
            (self.network_info['player_rep_size'] + TILE_ENCODING_SIZE+1) * self.n_players
            
        self.shared_network = FC(
            input_size=shared_input_size,
            output_size=output_size,
            l=self.network_info['shared_l'], 
            n=self.network_info['shared_n'])


    
    # Assumes the observation at 0 is current player
    def players_vector(self, x):
        with torch.no_grad():
            batch_size = x['Boards'].size()[0]
            boards_one_hot = self.boards2onehot(x)
            previous_tile_one_hot = tile2onehot(x['Previous tiles'], self.n_players, self.device)
            
        players_output = torch.zeros([
            batch_size,
            self.n_players,
            self.network_info['player_rep_size']],
            device=self.device)
        # TODO : parallelize this ?
        network = self.player_network
        for i in range(self.n_players):
            players_output[:,i] = network(
                board=boards_one_hot[:,i],
                previous_tile=previous_tile_one_hot[:,i])
            network = self.other_players_network
        return players_output.reshape(batch_size, -1)

    # equivalence of players not taken into account (ideally should share weights) 
    def forward(self, x):
        batch_size = x['Boards'].size()[0]
        players_vector = self.players_vector(x).reshape(batch_size, -1)
        with torch.no_grad():
            current_tiles_vector = get_current_tiles_vector(x, self.n_players, self.device)
        x = torch.cat([players_vector, current_tiles_vector.reshape(batch_size,-1)], dim=1)
        x = self.shared_network(x)
        return x

#%%

if __name__ == "__main__":
    hp = {'player_rep_size':50,
          'shared_l':3,
          'shared_n':100,
          'conv_channels':[32,16,5],
          'conv_l':3,
          'conv_kernel_size':[3,3,3],
          'conv_stride':[1,1,1],
          'pool_place':[0,0,0],
          'pool_kernel_size':None,
          'pool_stride':None,
          'board_rep_size':100,
          'board_fc_n':100,
          'board_fc_l':1, 
          'player_rep_size':100,
          'board_prev_tile_fc_l':1,
          'shared_rep_size':100}
    s = Shared(n_players=2,output_size=10, network_info=hp, device='cpu')
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