import random
import numpy as np
from itertools import product
from copy import deepcopy
from einops import rearrange,repeat
from operator import itemgetter

from setup import GET_TILE_DATA, TILE_SIZE, N_TILE_TYPES
from kingdomino.board import Boards
from printer import Printer
from graphics import draw_obs
from utils import switch, arr_except, cartesian_product


class TileDeck:      
    def __init__(self):
        self.tiles = GET_TILE_DATA()
        self.reset()
        
    def reset(self):
        self.tiles = np.random.permutation(self.tiles)
        self.idx = 0
    
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
                 n_players=None, 
                 render_mode=None,
                 kingdomino=None,
                 reward_fn=lambda x,y: 0):
        self.board_size = board_size
        self.getReward = lambda p_id: reward_fn(self, p_id)
        self.tile_deck = TileDeck()
        self.n_players = n_players
        self.n_turns = 13

    def reset(self, seed=None, options=None):
        # last position in list : plays last
        self.reset_scores()
        self.prev_scores = np.zeros(self.n_players)
        self.tile_deck.reset()
        self.players_current_tile_id = -np.ones(self.n_players, dtype='int64')
        self.players_previous_tile = repeat(Kingdomino.empty_tile, 't -> p t', p=self.n_players)
        self.current_tiles = -np.ones((self.n_players, TILE_SIZE+1), dtype='int64')
        self.boards = Boards(size=self.board_size, n_players=self.n_players)
        self.new_order = np.random.permutation(self.n_players)
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
        obs = {'Boards'         : np.zeros([self.n_players,2,self.board_size,self.board_size], dtype=np.int64),
               'Current tiles'  : np.zeros([self.n_players,TILE_SIZE+1], dtype=np.int64),
               'Previous tiles' : np.zeros([self.n_players,TILE_SIZE], dtype=np.int64)}
        
        player_order = switch(np.arange(self.n_players), 0, self.current_player_id)
        obs['Boards'][:,0] = self.boards.boards[player_order]
        obs['Boards'][:,1] = self.boards.crowns[player_order]
        obs['Previous tiles'] = self.players_previous_tile[player_order]
        obs['Current tiles'][:,:-1] = self.current_tiles[:,:-1]
        obs['Current tiles'][:,-1] = self.current_tiles[:,-1] == -1
        for i,tile in enumerate(self.current_tiles):
            obs['Current tiles'][i,:-1] = tile[:-1]
            if tile[-1] != -1:
                obs['Current tiles'][i,-1] = player_order[tile[-1]]
            else:
                obs['Current tiles'][i,-1] = -1
        
        for i,p_id in enumerate(player_order):
            # if self.current_tiles[self.players_current_tile_id[p_id],-1] != -1:
            #     obs['Current tiles'][i] = self.current_tiles[self.players_current_tile_id[p_id]]
            #     obs['Current tiles'][i,-1] = p_id
            # else:
            #     obs['Current tiles'][i] = self.current_tiles[i]
 
            obs['Previous tiles'][i] = self.players_previous_tile[p_id] # previous tiles should be ordered by player
        return obs
        

    def step(self, action):
        tile_id, position = action
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
        return self._get_obs(), done, {'Scores':self.getScores()}
       

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
            tiles_possible = np.array([-1])
        else:
            tiles_possible = np.where(self.current_tiles[:,-1] == -1)[0]
        if self.first_turn:
            positions_possible = np.array([Kingdomino.discard_tile])
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
def checkPlacementValid(p_id, position, tile, boards):
    # Check five square and no overlapping
    for point in position:
        if (not boards.size > point[0] >= 0) or (not boards.size > point[1] >= 0):
            raise GameException(f'{point[0]} or {point[1]} wrong index')
        if not boards.isInFiveSquare(p_id, point):
            raise GameException(f'{point} is not in a five square.')
        if boards.getBoard(p_id, point[0], point[1]) != -1 or \
            boards.getBoard(p_id, point[0], point[1]) == -2:
            raise GameException(f'Overlapping at point {point}.')
    
    # print('Tile', tile)
    # print('Position', position)
    # Check that at least one position is near castle or near same type of env
    for i,point in enumerate(position):
        if isNeighbourToCastleOrSame(p_id, point, tile[i], boards):
            return
    raise GameException(f'{position} {tile} \n {boards.boards[p_id]} not next to castle or same type of env.')    
    

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

def getPossiblePositions(tile, boards, every_pos, player_id):
    available_pos = []
    for i in range(boards.size):
        for j in range(boards.size):
            if boards.getBoard(player_id, i, j) == -1:
                for _i in range(i-1,i+2):
                    for _j in range(j-1,j+2):
                        if (_i != i and _j != j) or (_i==i and _j==j) or boards.getBoard(player_id, _i, _j) != -1:
                            continue
                        try:
                            pos = np.array([[i,j],[_i,_j]])
                            checkPlacementValid(
                                p_id=player_id, 
                                position=pos, 
                                tile=tile, 
                                boards=boards)
                            available_pos.append(pos)
                            if not every_pos:
                                return available_pos
                        except GameException:
                            pass
    if len(available_pos) == 0:
        return [Kingdomino.discard_tile]
    return np.array(available_pos)

def isTilePlaceable(tile, board):
    return getPossiblePositions(tile, board, every_pos=False)

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
                    players[p_id].process_reward(reward=reward, next_state=state, done=False)
                action = players[p_id].action(state, env)
                print('Action:', action)
                state,done,info = env.step(action)
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