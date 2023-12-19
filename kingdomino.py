import random
import numpy as np
from itertools import product
from copy import deepcopy

from setup import GET_TILE_DATA, TILE_SIZE, N_TILE_TYPES
from board import Board
from printer import Printer

# TODO: add graphics to Printer.print
from graphics import

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
# from gym.envs.registration import register

# register(
#     id='KingDomino-v0',
#     entry_point='kingdomino:Kingdomino')

#%%

def reward_last_quantitative(kd, terminated):
    if kd.first_turn:
        return None
    if not kd.empty_end_turn:
        return 0
    scores = kd.scores()
    best_id = np.argmax(arr_except(scores, except_id=kd.current_player_id))
    reward = scores[kd.current_player_id] - scores[best_id]
    kd.prev_scores = scores
    return reward


def arr_except(arr, except_id):
    return np.concatenate((arr[:except_id],arr[except_id+1:]))

def switch(l, i, j):
    lc = l.copy()
    lc[i],lc[j] = lc[j],lc[i]
    return lc

class CurrentTile:
    def __init__(self, tile):
        self.tile = tile
        self.player_id = -1
        
    def __str__(self):
        return (self.tile, self.id)

class Kingdomino:
    
    # self.current_tiles_player[tile_id] = prev_current_tiles_player_tile_id
    # current_tiles_player[tile_id] = player who chose this tile id (or -1)
    
    empty_tile = np.array([N_TILE_TYPES,N_TILE_TYPES,-1,-1,-1]) 
    # 0 not perfect because 0 represents a tile type...
    # but cant put -1 bcs will be flagged as incorrect index
    discard_tile = np.array([[-1,-1],[-1,-1]])
        
    def __init__(self, 
                 players=None, 
                 render_mode=None,
                 kingdomino=None,
                 reward_fn=None):
        self.reward_fn = reward_fn
        self.tile_deck = TileDeck()
        self.players = players
        self.n_players = len(self.players)

    def reset(self, seed=None, options=None):
        # last position in list : plays last
        self.tile_deck.reset()
        self.first_turn = True
        self.last_turn = False
        self.empty_end_turn = False # necessary to give last rewards at the end
        for player in self.players:
            player.current_tile_id = -1
            player.previous_tile = np.copy(Kingdomino.empty_tile)
            player.board = Board()
        self.new_order = np.random.permutation(self.n_players)
        self.order = self.new_order.copy()
        self.current_player_itr = 0
        self.startTurn()
        return self._get_obs()
    
    
    @property
    def current_player_itr(self):
        return self._current_player_itr
    @current_player_itr.setter
    def current_player_itr(self, v):
        self._current_player_itr = v
        if self._current_player_itr == self.n_players:
            self.first_turn = False
            self._current_player_itr = 0
            self.startTurn()
        self.current_player_id = self.order[self._current_player_itr]
            

    def scores(self):
        return np.array([player.board.count() for player in self.players])
        
    def currentTilesEmpty(self):
        for current_tile in self.current_tiles:
            if (current_tile.tile == Kingdomino.empty_tile).all():
                return True
    
    # Automatically called when current_player_itr is set to 0
    def startTurn(self):
        Printer.print('--------------Starting new turn-------------')
        Printer.print('^^^^ Player', self.current_player_id, 'playing. ^^^^')
        Printer.print(self.players[self.current_player_id].board)
        self.order = self.new_order.copy()
        Printer.print('Player order :', self.order)
        self.draw()
        if self.currentTilesEmpty():
            Printer.print('This is the last turn.')
            self.last_turn = True
        for player in self.players:
            player.current_tile_id = -1


    # Pulls out tiles from the stack
    def draw(self):
        if not self.first_turn:
            for current_tile in self.current_tiles:
                self.players[current_tile.player_id].previous_tile = current_tile.tile.copy()
                Printer.print('Player', current_tile.player_id, "'s previous tile :", current_tile.tile)
        self.current_tiles = [CurrentTile(tile) for tile in self.tile_deck.draw(self.n_players)]
        if not self.currentTilesEmpty():
            self.current_tiles.sort(key=lambda x: x.tile[-1])
        Printer.print('Current tiles :', [
            (current_tile.tile,current_tile.player_id) 
            for current_tile in self.current_tiles])
        

    # Board : 9*9 tile type + 9*9 crowns
    # Current/Previous tiles : 2 tile type, 2 crowns, 1 which player has chosen it, 1 value of tile
    # Previous tiles : 2 tile type, 2 crowns
    # TODO : ORDER PLAYERS NICELY !!!! (player playing first)
    def _get_obs(self):
        obs = {'Boards'         : np.zeros([self.n_players,2,9,9]),
               'Current tiles'  : np.zeros([self.n_players,TILE_SIZE+1]),
               'Previous tiles' : np.zeros([self.n_players,TILE_SIZE])}
        
        obs['Current tiles'][:,:-1] = np.array([current_tile.tile for current_tile in self.current_tiles])
        obs['Current tiles'][:,-1] = 0
        players = switch(self.players, self.current_player_id, 0)
        for i,player in enumerate(players):
            obs['Boards'][i,0] = player.board.board
            obs['Boards'][i,1] = player.board.crown
            obs['Previous tiles'][i] = player.previous_tile
            if player.current_tile_id != -1:
                obs['Current tiles'][player.current_tile_id,-1] = 1
        return obs
        

    def step(self, action):
        if not self.empty_end_turn:
            tile_id, position = action
            Printer.print('Tile chosen:', tile_id)
            Printer.print('Position for their previous tile:', position)
            # Select tile
            if not self.last_turn:
                self.pickTile(tile_id)
    
            if not self.first_turn:
                if not (position == Kingdomino.discard_tile).all():
                    self.placeTile(position, self.players[self.current_player_id].previous_tile)
                else:
                    Printer.print('Tile discarded')
 
 
        terminated = self.empty_end_turn and \
            self.current_player_itr == (self.n_players-1)
        Printer.print('Game ended:', terminated)
        self.empty_end_turn = self.last_turn and \
            self.current_player_itr == (self.n_players-1)
        self.current_player_itr += 1
        reward = self.reward_fn(self, terminated)

        return self._get_obs(), reward, terminated, {'Scores': self.scores()} # TODO: scores function if already computed this player turn dont recompute it
       

    def pickTile(self, tile_id):
        if 0 > tile_id or tile_id >= self.n_players or self.current_tiles[tile_id].player_id != -1:
            raise GameException(f'Error : Player {self.current_player_id} cannot choose tile {tile_id} because it is already selected !')
        self.new_order[tile_id] = self.current_player_id
        self.current_tiles[tile_id].player_id = self.current_player_id
        Printer.print('New order after this play:', self.new_order)

    def placeTile(self, position, tile):
        Printer.print(f'Trying to place tile {tile} at {position}.')
        if (position < 0).any() or (position > 9).any():
            Printer.print('Tile discarded.')
            return
        self.checkPlacementValid(position, tile)   
        self.players[self.current_player_id].board.placeTile(position, tile)
    
    def selectTileRandom(self):
        if self.last_turn:
            return None
        available = [i for i,current_tile in enumerate(self.current_tiles) if current_tile.player_id == -1]
        return random.choice(available)
    
    def selectTilePositionRandom(self):
        if self.first_turn:
            return None
        tile = self.players[self.current_player_id].previous_tile
        positions = self.getPossiblePositions(tile)
        if positions:
            position = random.sample(positions, k=1)[0]
        else:
            position = Kingdomino.discard_tile
        return position

    def getRandomAction(self):
        return (self.selectTileRandom(), self.selectTilePositionRandom())
    

    # position : TilePosition
    # inversed false : position.p1 corresponds to first tile
    # TODO : perhaps speed this up with position as a numpy array
    def checkPlacementValid(self, position, tile):
        board = self.players[self.current_player_id].board 
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

                if self.players[self.current_player_id].board.getBoard(i,j) == tile_type:
                    return True
                
        return False
    
    def getPossibleActions(self):
        tiles_possible = self.getPossibleTileChoices()
        positions_possible = self.getPossiblePositions(
            tile=self.players[self.current_player_id].previous_tile,
            every_pos=True)
        
        # TODO: speed up using numpy generated cartesian product ?
        actions = list(product(tiles_possible, positions_possible))
        return actions
        
    
    def getPossibleTileChoices(self):
        if self.last_turn:
            return [0]
        return [i for i,current_tile in enumerate(self.current_tiles) if current_tile.player_id == -1]

    
    def getPossiblePositions(self, tile, every_pos=False):
        if self.first_turn:
            return [self.discard_tile]
        board = self.players[self.current_player_id].board
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
            return [self.discard_tile]
        return available_pos
    
    def isTilePlaceable(self, tile):
        return self.getPossiblePositions(tile, every_pos=False)

#%%

if __name__ == '__main__':
    import agent
    player_1 = agent.HumanPlayer()
    player_2 = agent.RandomPlayer()
    players = [player_1,player_2]
    Printer.activated = True
    done = False
    env = Kingdomino(
        players=players,
        reward_fn=reward_last_quantitative)
    state = env.reset()
    done = False
    while not done:
        for player_id in env.order:
            action = players[player_id].action(state, env)
            state,reward,done,info = env.step(action)
            if done:
                break
    Printer.activated = False