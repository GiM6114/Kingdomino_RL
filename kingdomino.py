import random
import numpy as np
from collections import namedtuple
from itertools import permutations,product
import gym
from gym import spaces
from copy import deepcopy

from setup import GET_TILE_DATA, TILE_SIZE, N_TILE_TYPES
from board import Board
from printer import Printer

import torch.nn.functional as F


class GameException(Exception):
    def __init__(self, msg):
        self.msg = msg

class TileDeck:      
    def __init__(self):
        self.tile_data = GET_TILE_DATA() # array of tiles
        self.nb_tiles = self.tile_data.shape[0]
        self.reset()
        
    def reset(self):
        self.available_tiles = np.ones(self.nb_tiles, dtype=bool)
        self.open_tiles = self.tile_data
    
    
    
    def draw(self, nb):
        if self.open_tiles.shape[0] < nb:
            return np.tile(Kingdomino.empty_tile, nb).reshape(nb, TILE_SIZE)
        tiles = np.zeros((nb,TILE_SIZE))
        for i in range(nb):
            tile_id = random.randint(0, self.open_tiles.shape[0]-1)
            tiles[i] = self.open_tiles[tile_id]
            self.available_tiles[tile_id] = False
            self.open_tiles = self.open_tiles[self.available_tiles]
            self.available_tiles = self.available_tiles[self.available_tiles]
        Printer.print('Open tiles left :', self.open_tiles.shape[0])
        return tiles

#%%
from gym.envs.registration import register

register(
    id='KingDomino-v0',
    entry_point='kingdomino:Kingdomino')

#%%
class Kingdomino(gym.Env):
    
    
    # self.current_tiles_player[tile_id] = prev_current_tiles_player_tile_id
    # current_tiles_player[tile_id] = player who chose this tile id (or -1)
    
    empty_tile = np.array([N_TILE_TYPES,N_TILE_TYPES,-1,-1,-1]) 
    # 0 not perfect because 0 represents a tile type...
    # but cant put -1 bcs will be flagged as incorrect index
    discard_tile = np.array([[-1,-1],[-1,-1]])
    direction = {0:np.array([1,0]),
                 1:np.array([0,1]),
                 2:np.array([-1,0]),
                 3:np.array([0,-1]),
                 4:np.array([0,0])}
        
    def __init__(self, 
                 n_players=None, 
                 test=False, 
                 players=None, 
                 render_mode=None,
                 kingdomino=None):
        self.tile_deck = TileDeck()
        if players is not None:
            self.players = players
            self.n_players = len(self.players)
        else:
            self.n_players = n_players
        self.observation_space = spaces.Dict(
            {
                'Boards': spaces.Box(low=-2, high=N_TILE_TYPES-1, shape=(self.n_players, 2, 9, 9), dtype=float),
                'Previous tiles': spaces.Box(low=0, high=100, shape=(self.n_players, TILE_SIZE), dtype=float),
                'Current tiles': spaces.Box(low=0, high=100, shape=(self.n_players, TILE_SIZE+1), dtype=float)
            }
        )
        # self.action_space = spaces.Dict(
        #     {
        #         'Position': spaces.Box(low=0, high=9, shape=(2,), dtype=int),
        #         'Selected tile': spaces.Box(low=0, high=self.n_players-1, shape=(1,), dtype=int) 
        #     }
        # )
        
        self.just_reseted = None
        
        self.action_space = spaces.Box(low=0, high=9, shape=(3,), dtype=float)
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # last position in list : plays last
        self.tile_deck.reset()
        self.first_turn = True
        self.last_turn = False
        self.just_reseted = False
        self.current_tiles = np.tile(Kingdomino.empty_tile, self.n_players).reshape(self.n_players, TILE_SIZE)
        self.current_tiles_player = -np.ones(self.n_players)
        self.previous_tiles = np.tile(Kingdomino.empty_tile, self.n_players).reshape(self.n_players, TILE_SIZE)
        self.order = np.arange(2)
        self.new_order = np.arange(2) # order of first turn
        # could be #np.random.permutation(self.n_players)
        self.boards = []
        for i in range(self.n_players):
            self.boards.append(Board())
        self.current_player_itr = 0
        self.startTurn()
        return self._get_obs(), {}
    
    
    @property
    def current_player_itr(self):
        return self._current_player_itr
    @current_player_itr.setter
    def current_player_itr(self, v):
        self._current_player_itr = v
        if self._current_player_itr == self.n_players:
            if self.last_turn:
                self.reset()
                self.just_reseted = True
                self.players_aware_reset = 0
            else:
                self.first_turn = False
                self._current_player_itr = 0
                self.startTurn()
        self.current_player_id = self.order[self._current_player_itr]
            

    def scores(self):
        return np.array([board.count() for board in self.boards])
        
    def tilesEmpty(self, tiles):
        return any(np.equal(tiles,Kingdomino.empty_tile).all(1))
    
    # Automatically called when current_player_itr is set to 0
    def startTurn(self):
        Printer.print('--------------Starting new turn-------------')
        self.order = self.new_order.copy()
        Printer.print('Player order :', self.order)
        self.draw()

        if self.tilesEmpty(self.current_tiles):
            Printer.print('This is the last turn.')
            self.last_turn = True
        else:
            self.current_tiles_player[:] = -1


    # Pulls out tiles from the stack
    def draw(self):
        self.previous_tiles = self.current_tiles.copy()[self.order]
        self.current_tiles = self.tile_deck.draw(self.n_players)
        if not self.tilesEmpty(self.current_tiles):
            self.current_tiles = self.current_tiles[self.current_tiles[:,-1].argsort()]
        Printer.print('Current tiles :', self.current_tiles)
        Printer.print('Previous tiles :', self.previous_tiles)
        
    def reward(self, terminated):
        pass

    def step(self, action):
        
        tile_id, position = action
        # Select tile
        if not self.last_turn:
            self.pickTile(tile_id)

        if not self.first_turn:
            if not (position == Kingdomino.discard_tile).all():
                self.placeTile(position, self.previous_tiles[self.current_player_id])
            else:
                Printer.print('Tile discarded')
 
 
        terminated = self.last_turn and \
            self.current_player_itr == (self.n_players-1)
        reward = self.reward(terminated)
        self.current_player_itr += 1

        return {}, reward, terminated, False, {'Scores': self.scores()} if terminated else {}
        #return self._get_obs(), reward, terminated, False, {'Scores': self.scores()}

    # def step(self, action):
        
    #     if self.just_reseted:
    #         Printer.print('Just reseted, empty step.')
    #         self.players_aware_reset += 1
    #         if self.players_aware_reset == self.n_players:
    #             self.players_aware_reset = 0
    #             self.just_reseted = False
    #         return self._get_obs(), 0, False, False, {'Scores':np.zeros(self.n_players)}
        
    #     Printer.print('---',self.current_player_id, "'s turn-----")
    #     Printer.print('Action :')

    #     action = action.flatten()
    #     position = np.stack((action[:2], 
    #                          action[:2] + Kingdomino.direction[action[2]]
    #                         )).astype(int)
    #     tile_id = action[-1].astype(int)
        
    #     Printer.print('    Position :', position)
    #     Printer.print('    Tile chosen :', tile_id)

        
    #     truncated = False
    #     try:
    #         if self.current_tiles_player[tile_id] != -1:
    #             tile_id = 1 - tile_id
                
    #         prev_new_order_tile_id = self.new_order[tile_id]
    #         prev_current_tiles_player_tile_id = self.current_tiles_player[tile_id]
            
    #         # Select tile
    #         if not self.last_turn:
    #             # GameException possible
    #             self.pickTile(tile_id)

    #         if not self.first_turn:
    #             if not (position == Kingdomino.discard_tile).all():
    #                 # GameException possible
    #                 self.placeTile(position, self.previous_tiles[self.current_player_id])
    #             else:
    #                 Printer.print('Tile discarded')
                    
    #         scores = self.scores()
    #         tot_scores = scores.sum()
    #         score = scores[self.current_player_id]
    #         reward = (score / max(tot_scores,0.01))
    #         reward += 0.5 #managed to reach this point, no out of game decision
            
    #         Printer.print('Scores :', scores)
    #         Printer.print('Reward :', reward)
 
    #         terminated = self.last_turn and \
    #             self.current_player_itr == (self.n_players-1)
    #         self.current_player_itr += 1

    #         state = self._get_obs()
    #     except GameException as e:
    #         # send same states as output before, with negative reward !!!!
    #         self.new_order[tile_id] = prev_new_order_tile_id
    #         self.current_tiles_player[tile_id] = prev_current_tiles_player_tile_id
    #         scores = self.scores()
    #         if self.test:
    #             truncated = True
    #             print('OuiOuiOuiOuiOuiOuiOuiOuiOuiOui')
    #         return self._get_obs(), -0.001, False, truncated, {'Scores': scores}

    #     return state, reward, terminated, truncated, {'Scores': scores}
        
    # Board : 9*9 tile type + 9*9 crowns
    # Current/Previous tiles : 2 tile type, 2 crowns, 1 which player has chosen it, 1 value of tile
    # Previous tiles : 2 tile type, 2 crowns
    # TODO : ORDER PLAYERS NICELY !!!! (player playing first)
    def _get_obs(self):
        obs = {'Boards'         : np.zeros([self.n_players,2,9,9]),
               'Current tiles'  : np.zeros([self.n_players,TILE_SIZE+1]),
               'Previous tiles' : np.zeros([self.n_players,TILE_SIZE])}
        for i in self.order:     
            obs['Boards'][i,0] = self.boards[i].board
            obs['Boards'][i,1] = self.boards[i].crown

        obs['Boards'][[0,self.current_player_id]] = obs['Boards'][[self.current_player_id,0]]
        obs['Previous tiles'] = self.previous_tiles
        obs['Current tiles'][:,:-1] = self.current_tiles
        obs['Current tiles'][:,-1] = (self.current_tiles_player == -1).astype(float)

        # same order as boards
        obs['Previous tiles'] = obs['Previous tiles'][self.order]
        obs['Current tiles'] = obs['Current tiles'][self.order]

        # player as first
        obs['Previous tiles'][[0,self.current_player_id]] = obs['Previous tiles'][[self.current_player_id,0]]
        
        return obs
        

    def pickTile(self, tile_id):
        Printer.print('Tile id :', tile_id)
        if 0 > tile_id or tile_id >= self.n_players or self.current_tiles_player[tile_id] != -1:
            raise GameException(f'Error : Player {self.current_player_id} cannot choose tile {tile_id} because it is already selected !')
        self.new_order[tile_id] = self.current_player_id
        self.current_tiles_player[tile_id] = self.current_player_id
        Printer.print('New order :', self.new_order)
        Printer.print('Current tiles player :', self.current_tiles_player)

    def placeTile(self, position, tile):
        Printer.print(f'Trying to place tile {tile} at {position}.')
        if (position < 0).any() or (position > 9).any():
            Printer.print('Tile discarded.')
            return
        self.checkPlacementValid(position, tile)   
        self.boards[self.current_player_id].placeTile(position, tile)
    
    def selectTileRandom(self):
        if self.last_turn:
            return None
        available = np.where(self.current_tiles_player == -1)[0]
        tile_id = available[random.randint(0, len(available)-1)]
        return tile_id
    
    def selectTilePositionRandom(self):
        if self.first_turn:
            return None
        
        tile = self.previous_tiles[self.current_player_id]
        
        positions = self.getPossiblePositions(tile)
        if positions:
            position = random.sample(positions, k=1)[0]
        else:
            position = Kingdomino.discard_tile
        return position

    def getRandomAction(self):
        return (self.selectTileRandom(), self.selectTilePositionRandom())
    
    def printCurrentTiles(self):
        print('Tiles to pick from : \n')
        [print(tileplayer) for tileplayer in self.current_tiles]       


    # position : TilePosition
    # inversed false : position.p1 corresponds to first tile
    # TODO : perhaps speed this up with position as a numpy array
    def checkPlacementValid(self, position, tile):
        board = self.boards[self.current_player_id] 
        # Check five square and no overlapping
        for point in position:
            if (not 9 > point[0] >= 0) or (not 9 > point[1] >= 0):
                raise GameException(f'{point[0]} or {point[1]} wrong index')
            if not board.isInFiveSquare(point):
                raise GameException(f'{point} is not in a five square.')
            if board.getBoard(point[0], point[1]) != -1 or \
                board.getBoard(point[0], point[1]) == -2:
                raise GameException(f'Overlapping at point {point}.')
        
        # Check that at least one position is near castle or near same type of env
        for i,point in enumerate(position):
            if self.isNeighbourToCastleOrSame(point, tile[i]):
                return
        raise GameException('Not next to castle or same type of env.')    
    
    
    castle_neighbors = np.array([(3,4),(4,3),(4,5),(5,4)])
    def isNeighbourToCastleOrSame(self, point, tile_type):
        # Check if castle next
        if np.any(np.all(point == Kingdomino.castle_neighbors, axis=1)):
            return True
                
        # Check if same next
        for i in range(point[0]-1, point[0]+2):
            for j in range(point[1]-1, point[1]+2):
                # Exclude diagonals
                if i != point[0] and j != point[1]:
                    continue

                if self.boards[self.current_player_id].getBoard(i,j) == tile_type:
                    return True
                
        return False
    
    def getPossibleActions(self):
        tiles_possible = self.getPossibleTileChoices()
        positions_possible = self.getPossiblePositions(
            tile=self.previous_tiles[self.current_player_id],
            every_pos=True)
        
        # TODO: speed up using numpy generated cartesian product ?
        return product(tiles_possible, positions_possible)
        
    
    def getPossibleTileChoices(self):
        if self.last_turn:
            return [None]
        return np.where(self.current_tiles_player == -1)[0]

    
    def getPossiblePositions(self, tile, every_pos=False):
        board = self.boards[self.current_player_id]
        available_pos = []
        for i in range(9):
            for j in range(9):
                if board.getBoard(i, j) == -1:
                    for _i in range(i-1,i+2):
                        for _j in range(j-1,j+2):
                            if (_i != i and _j != j) or (_i==i and _j==j) or board.getBoard(_i, _j) != -1:
                                continue
                            try:
                                pos = np.array([[i,j],[_i,_j]])
                                self.checkPlacementValid(pos, tile)
                                available_pos.append(pos)
                                if not every_pos:
                                    return available_pos
                            except GameException:
                                pass
                            try:
                                pos = np.array([[_i,_j],[i,j]])
                                self.checkPlacementValid(pos, tile)
                                available_pos.append(pos)
                                if not every_pos:
                                    return available_pos
                            except GameException:
                                pass
        if len(available_pos) == 0:
            available_pos = [self.discard_tile]
        return available_pos
    
    def isTilePlaceable(self, tile):
        return self.getPossiblePositions(tile, every_pos=False)
                            

