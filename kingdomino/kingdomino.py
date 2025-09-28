import random
import numpy as np
from itertools import product
from copy import deepcopy
from einops import rearrange,repeat
from operator import itemgetter
from copy import deepcopy

from setup import GET_TILE_DATA, TILE_SIZE, N_TILE_TYPES
from kingdomino.board import Boards
from printer import Printer
from graphics import draw_obs
from utils import switch, arr_except, cartesian_product, arr2tuple

class TileDeck:      
    def __init__(self):
        self.tiles = GET_TILE_DATA()
        self.reset()
        
    def reset(self):
        self.tiles = np.random.permutation(self.tiles)
        self.idx = 0
        
    def shuffle_remaining_tiles(self):
        self.tiles[self.idx:] = np.random.permutation(self.tiles[self.idx:])
    
    def draw(self, n):
        if self.idx + n > self.tiles.shape[0]:
            return np.repeat(rearrange(Kingdomino.empty_tile, 't -> 1 t'), repeats=n, axis=0)
        tiles = self.tiles[self.idx:self.idx+n]
        self.idx += n
        return tiles

#%%
# from gym.envs.registration import register

# register(
#     id='KingDomino-v0',
#     entry_point='kingdomino:Kingdomino')

#%%

class GameException(Exception):
    def __init__(self, msg):
        self.msg = msg

def sort_rows(a, column):
    return a[a[:, column].argsort()]

# would be nice to turn as many function in pure functions as possible to jax it
# also no additions to player objects
class Kingdomino:
    
    # self.current_tiles_player[tile_id] = prev_current_tiles_player_tile_id
    # current_tiles_player[tile_id] = player who chose this tile id (or -1)
    
    empty_tile = np.array([N_TILE_TYPES,N_TILE_TYPES,-1,-1,-1], dtype='int64') 
    # 0 not perfect because 0 represents a tile type...
    # but cant put -1 bcs will be flagged as incorrect index
    discard_tile = np.array([[-1,-1],[-1,-1]], dtype='int64')
    
    def __init__(self, 
                 board_size,
                 random_start_order=True,
                 n_players=None, 
                 render_mode=None,
                 reward_fn=None,
                 compute_obs=True):
        self.compute_obs = compute_obs
        self.random_start_order = random_start_order
        self.board_size = board_size

        if reward_fn is None:
            reward_fn = self.default_reward
        self.reward_fn = reward_fn

        self.tile_deck = TileDeck()
        self.n_players = n_players
        self.n_turns = 13
        
    def getReward(self, p_id):
        return self.reward_fn(self, p_id)

    @staticmethod
    def default_reward(env, p_id):
        return 0

    def reset(self, seed=None, options=None):
        # last position in list : plays last
        self.obs = {
            'Boards'         : np.zeros([self.n_players,2,self.board_size,self.board_size], dtype=np.int64),
            'Current tiles'  : np.zeros([self.n_players,TILE_SIZE+1], dtype=np.int64),
            'Previous tiles' : np.zeros([self.n_players,TILE_SIZE], dtype=np.int64)}
        self.reset_scores()
        self.prev_scores = np.zeros(self.n_players)
        self.tile_deck.reset()
        self.players_current_tile_id = -np.ones(self.n_players, dtype='int64')
        self.players_previous_tile = repeat(Kingdomino.empty_tile, 't -> p t', p=self.n_players)
        self.current_tiles = -np.ones((self.n_players, TILE_SIZE+1), dtype='int64')
        self.boards = Boards(size=self.board_size, n_players=self.n_players)
        if self.random_start_order:
            self.new_order = np.random.permutation(self.n_players)
        else:
            self.new_order = np.array([0, 1])
        self.order = self.new_order.copy()
        self.current_player_itr = 0
        self.turn_id = 0
        self.previous_scores = None
        self._startTurn()
        return self._get_obs()
    
    @property
    def first_turn(self):
        return self.turn_id == 1
    @property
    def last_turn(self):
        return self.turn_id == self.n_turns
    
    @property
    def current_player_itr(self):
        return self._current_player_itr
    @current_player_itr.setter
    def current_player_itr(self, v):
        self._current_player_itr = v
        if self._current_player_itr == self.n_players:
            self._current_player_itr = 0
            self._startTurn()
        self.current_player_id = self.order[self._current_player_itr]
    
    # Requirements for score:
        # - Computed only when necessary (some reward function don't need it every turn)
        # - Has to log previous values as well
        # - Has to be reseted to 0 when game is reset
    # Solution:
        # Private attribute _scores, np.array, value is -1 when not up to date for a player
        # Property scores, when called

    def getScores(self):
        return np.array([self.getScore(p_id) for p_id in range(self.n_players)])
    def getScore(self, p_id):
        if self.scores[p_id] != -1:
            return self.scores[p_id]
        # parallelize ?
        self.scores[p_id] = self.boards.count(p_id)
        return self.scores[p_id]
    
    def reset_scores(self):
        self.scores = np.zeros(self.n_players)
        self.prev_scores = np.zeros(self.n_players)
    def reset_score(self, p_id):
        self.prev_scores[p_id] = self.scores[p_id]
        self.scores[p_id] = -1

    # Automatically called when current_player_itr is set to 0
    def _startTurn(self):
        self.turn_id += 1
        self.order = self.new_order.copy()
        if not self.first_turn:
            self.players_previous_tile = sort_rows(self.current_tiles, -1)[:,:-1] # sort by player
        if not self.last_turn:
            self.current_tiles[:,-1] = -1
            self.current_tiles[:,:-1] = self.tile_deck.draw(self.n_players)
            self.current_tiles = sort_rows(self.current_tiles,-2)        

    # Board : self.board_size*self.board_size tile type + self.board_size*self.board_size crowns
    # Current/Previous tiles : 2 tile type, 2 crowns, 1 which player has chosen it, 1 value of tile, self.n_players whose player
    # Previous tiles : 2 tile type, 2 crowns*
    # If get_obs: no current_tiles
    def _get_obs(self):
        if not self.compute_obs:
            return None
        player_order = switch(np.arange(self.n_players), 0, self.current_player_id)
        self.obs['Boards'][:,0] = self.boards.boards[player_order]
        self.obs['Boards'][:,1] = self.boards.crowns[player_order]
        self.obs['Previous tiles'] = self.players_previous_tile[player_order]
        self.obs['Current tiles'][:,:-1] = self.current_tiles[:,:-1]
        self.obs['Current tiles'][:,-1] = self.current_tiles[:,-1] == -1
        for i,tile in enumerate(self.current_tiles):
            self.obs['Current tiles'][i,:-1] = tile[:-1]
            if tile[-1] != -1:
                self.obs['Current tiles'][i,-1] = player_order[tile[-1]]
            else:
                self.obs['Current tiles'][i,-1] = -1
        
        for i,p_id in enumerate(player_order):
            # if self.current_tiles[self.players_current_tile_id[p_id],-1] != -1:
            #     obs['Current tiles'][i] = self.current_tiles[self.players_current_tile_id[p_id]]
            #     obs['Current tiles'][i,-1] = p_id
            # else:
            #     obs['Current tiles'][i] = self.current_tiles[i]
 
            self.obs['Previous tiles'][i] = self.players_previous_tile[p_id] # previous tiles should be ordered by player
        return self.obs
        

    def step(self, action):
        tile_id, position = action
        position = np.array(position)
        # Select tile
        if not self.last_turn:
            self._pickTile(tile_id)

        if not self.first_turn:
            if not (position == Kingdomino.discard_tile).all():
                self._placeTile(position, self.players_previous_tile[self.current_player_id])
        self.reset_score(self.current_player_id)

        done = self.last_turn and \
            self.current_player_itr == (self.n_players-1)
            
        self.current_player_itr += 1
        return self._get_obs(), done#, {'Scores':self.getScores()}
       

    def _pickTile(self, tile_id):
        if 0 > tile_id or tile_id >= self.n_players or self.current_tiles[tile_id,-1] != -1:
            raise GameException(f'Tile id {tile_id} erroneous, might be already selected !')
        self.new_order[tile_id] = self.current_player_id
        self.current_tiles[tile_id,-1] = self.current_player_id
        self.players_current_tile_id[self.current_player_id] = tile_id

    def _placeTile(self, position, tile):
        if (position < 0).any() or (position > self.board_size).any():
            return
        checkPlacementValid(self.current_player_id, position, tile, self.boards)
        self.boards.placeTile(self.current_player_id, position, tile)
        
    # Does not output cartesian product of possible tiles and actions
    # Because some method might want to handle positions and then tile selection for instance
    def getPossibleTilesPositions(self):
        if self.last_turn:
            tiles_possible = (-1,)
        else:
            tiles_possible = arr2tuple(np.where(self.current_tiles[:,-1] == -1)[0])
            
        if self.first_turn:
            positions_possible = (arr2tuple(Kingdomino.discard_tile),)
        elif self.turn_id == 2:
            # Quick positions on first round
            mid = self.board_size // 2
            positions_possible = [[(mid,mid-1),(mid,mid-2)], [(mid,mid-1),(mid-1,mid-1)]]
            if not tile_symmetrical(tile=self.players_previous_tile[self.current_player_id]):
                positions_possible.extend([[(mid,mid-2),(mid,mid-1)], [(mid-1,mid-1),(mid,mid-1)]])
        else:
            # print('Previous tiles of current player:', self.players_previous_tile[self.current_player_id])
            # print('Player id:', self.current_player_id)
            positions_possible = getPossiblePositions(
                tile=self.players_previous_tile[self.current_player_id],
                every_pos=True,
                boards=self.boards,
                player_id=self.current_player_id)
        return tiles_possible, positions_possible
    
    def getPossibleActions(self):
        return list(product(*self.getPossibleTilesPositions()))

# position : TilePosition
# inversed false : position.p1 corresponds to first tile
# TODO : perhaps speed this up with position as a numpy array
# TODO : this should not work with exceptions (a lot slower)
def checkPlacementValid(p_id, position, tile, boards):
    # Check five square and no overlapping
    for point in position:
        if (not boards.size > point[0] >= 0) or (not boards.size > point[1] >= 0):
            return False
        if not boards.isInFiveSquare(p_id, point):
            return False
        if boards.getBoard(p_id, point[0], point[1]) != -1 or \
            boards.getBoard(p_id, point[0], point[1]) == -2:
            return False
    
    # Check that at least one position is near castle or near same type of env
    for i,point in enumerate(position):
        if isNeighbourToCastleOrSame(p_id, point, tile[i], boards):
            return True
    return False  
    

def computeZone(x, y, board, board_seen, env_type):
    s = board.shape[-1]
    if 0 > x or x >= s or 0 > y or y >= s or board[0,x,y] != env_type or board_seen[x,y] == True:
        return 0,0
    board_seen[x,y] = True
    add_squares = 1
    add_crowns  = board[1,x,y]
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            # Get rid of diagonals
            if i != x and j != y:
                continue
            add_squares_temp,add_crowns_temp = computeZone(
                i, j, board, board_seen, env_type
                )
            add_squares += add_squares_temp
            add_crowns += add_crowns_temp
            
    return add_squares, add_crowns



def count(board):
    s = board.shape[-1]
    score = 0
    board_seen = np.zeros((s,s), dtype=bool)
    for x in range(s):
        for y in range(s):
            if 0 > x or x >= s or 0 > y or y >= s or board[0,x,y] in [-1,-2]:
                continue
            n_squares, n_crowns = computeZone(x, y, board, board_seen, board[0,x,y])
            score += n_squares * n_crowns
    return score

# def boards_encoding(boards, device='cpu'):
#     # boards : batch_size, 2 (tile type and crown), 5, 5
#     boards = torch.as_tensor(boards, device=device, dtype=torch.int64)
#     batch_size = boards.shape[0]
#     board_size = boards.shape[-1]
#     boards_one_hot = torch.zeros([
#         batch_size,
#         BOARD_CHANNELS+1, # (env + center + empty) + crowns 
#         board_size,
#         board_size],
#         dtype=torch.int8,
#         device=device)
#     boards_one_hot.scatter_(1, (boards[:,0]+2).unsqueeze(1), 1)
#     boards_one_hot[:,-1,:,:] = boards[:,1] # Place crowns at the end
#     return boards_one_hot[:,1:]

def isNeighbourToCastleOrSame(p_id, point, tile_type, boards):
    middle = boards.size // 2
    castle_neighbors = np.array(
        [(middle-1,middle),
         (middle,middle-1),
         (middle,middle+1),
         (middle+1,middle)])
    # print('Point:', point)
    # Check if castle next
    if np.any(np.all(point == castle_neighbors, axis=1)):
        # print('Next to castle')
        return True
    # Check if same next
    for i in range(point[0]-1, point[0]+2):
        for j in range(point[1]-1, point[1]+2):
            # Exclude diagonals and center
            if (i != point[0] and j != point[1]) or (i == point[0] and j == point[1]):
                continue

            if boards.getBoard(p_id, i,j) == tile_type:
                # print('Next to same tile type', tile_type, 'at', i,j)
                return True
    # print('Not next to')    
    return False 

def tile_symmetrical(tile):
    return tile[0] == tile[1] and tile[2] == tile[3]

# odd shaped array only I think ? (board size always odd anyways)
def array_symmetrical(array, axis):
    first = np.take(array, indices=range(array.shape[axis]//2), axis=axis)
    second = np.take(array, indices=range(array.shape[axis]//2+1, array.shape[axis]), axis=axis)
    return (first == second).all()

def board_symmetrical(boards, p_id, axis):
    tiles = array_symmetrical(boards.boards[p_id], axis=axis)
    crowns = array_symmetrical(boards.crowns[p_id], axis=axis)
    return tiles and crowns

# TODO : actually should check for symmetries in a consistent manner by comparing
# board ouputs maybe (should be an option at least, so not too many actions possible)
def getPossiblePositions(tile, boards, every_pos, player_id):
    '''
        I will NOT understand any of this in 5 days but it gives
        a minimal amount of possible actions taking into account
        reflective symmetries (h, v) in a fairly optimized way ! 
    '''
    h_sym = board_symmetrical(boards, player_id, axis=0)
    h_range = range(boards.size//2 + 1 if h_sym else boards.size)
    v_sym = board_symmetrical(boards, player_id, axis=1)
    v_range = range(boards.size//2 + 1 if v_sym else boards.size)
    available_pos = []
    symmetrical = tile_symmetrical(tile)
    for i in h_range:
        for j in v_range:
            if boards.getBoard(player_id, i, j) == -1:
                for _i in range(i-1,i+2):
                    for _j in range(j-1,j+2):
                        if (_i != i and _j != j) or (_i==i and _j==j):
                            continue
                        # If symmetry and beyond axis, ignore
                        if (h_sym and _i > boards.size//2) or (v_sym and _j > boards.size//2):
                            continue
                        pos = ((i,j),(_i,_j))
                        valid = checkPlacementValid(
                            p_id=player_id, 
                            position=pos, 
                            tile=tile, 
                            boards=boards)
                        if valid and not (symmetrical and ((_i,_j),(i,j)) in available_pos):
                            available_pos.append(pos)
                            if not every_pos:
                                return available_pos
    if len(available_pos) == 0:
        return [arr2tuple(Kingdomino.discard_tile)]
    return available_pos



# TODO : check if first element is discard tile
# def isTilePlaceable(tile, board):
#     return getPossiblePositions(tile, board, every_pos=False)

#%%

if __name__ == '__main__':
    import agents.base as agent
    from IPython.core.display import Image, display
    from kingdomino.rewards import player_focused_reward
    
    player_1 = agent.RandomPlayer()
    player_2 = agent.RandomPlayer()
    players = [player_1,player_2]
    Printer.activated = False
    done = False
    env = Kingdomino(
        n_players=2,
        board_size=5,
        reward_fn=player_focused_reward)

    for i in range(1):
        state = env.reset()
        done = False
        j = 0
        while not done:
            j += 1
            print(f'=========== Turn {j} ==============')
            for p_id in env.order:
                print(f'--- Player {p_id} playing ---')
                display(draw_obs(state))
                if not env.first_turn:
                    reward = env.getReward(p_id)
                    print(f'Reward: {reward}')
                    players[p_id].process_reward(reward, False)
                action = players[p_id].action(state, env)
                print('Action:', action)
                state,done = env.step(action)
                print(state['Current tiles'])
                print(env.current_tiles)
                display(draw_obs(state))
                print('Done:', done)
                print('Player id:', p_id)
                print('Possible actions:', env.getPossibleTilesPositions())
                if done:
                    print('Last turn rewards')
                    for p_id,player in enumerate(players):
                        reward = env.getReward(p_id)
                        player.process_reward(reward=reward, next_state=state, done=True)
                        print(f'Player {p_id}: {reward}')
        
    Printer.activated = False