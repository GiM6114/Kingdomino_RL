import random
import numpy as np

from setup import GET_TILE_DATA
from board import Board


class GameException(Exception):
    def __init__(self, msg):
        self.msg = msg
        
class TilePosition:
    def __init__(self, x1, y1, x2, y2):
        coords = x1,y1,x2,y2
        for coord in coords:
            if coord < 0 or coord > 8:
                raise GameException('Coords should be between 0 and 8 (included).')
        if abs(x1 - x2 + y1 - y2) != 1:
            raise GameException('Coordinates should be next to each other.')
        self.p1 = x1,y1
        self.p2 = x2,y2
        self.points = self.p1,self.p2


class Kingdomino:   
    
    class TilePlayer:
        def __init__(self, tile, player_id=-1):
            self.tile = tile
            self.player_id = player_id
        def isSelected(self):
            return self.player_id != -1
        def __str__(self):
            return str(self.tile) + ' ' + str(self.player_id)
    
    def __init__(self, nb_players=4, compute_scores_every_round=True, log=False):
        self.compute_scores_every_round = compute_scores_every_round
        self.log = log
        self.nb_players = nb_players
        self.tiles = GET_TILE_DATA()
        self.reset()
    
    def reset(self):
        # last position in list : plays last
        self.last_turn = False
        self.current_tiles = []
        self.turnNb = 0
        self.order = np.arange(0, self.nb_players)
        self.new_order = self.order.copy()
        self.boards = [Board() for i in range(self.nb_players)]
        self.player_can_play = np.ones(4, dtype='bool')
        self.player_previous_tiles = [None, None, None, None]
        self.player_current_tiles = [None, None, None, None]
        self.player_placed_all = np.ones(4, dtype='bool')
        self.current_player_itr = 0
        
    @property
    def current_player_itr(self):
        return self._current_player_itr
    @current_player_itr.setter
    def current_player_itr(self, v):
        self._current_player_itr = v
        if self._current_player_itr == self.nb_players:
            self._current_player_itr = 0
    
    
    def scores(self):
        scores = np.zeros(self.nb_players)
        for i,board in enumerate(self.boards):
            scores[i] = board.count()
        return scores
        
        
    def startTurn(self):
        self.order = self.new_order.copy()
        self.player_previous_tiles = self.player_current_tiles.copy()
        if len(self.tiles) >= self.nb_players:
            self.draw()
        else:
            self.last_turn = True        

    
    # player_id         : player who wants to play
    # tile_id           : id of tile he wants to pick if there is still tiles to pick
    # position_inversed : tuple (TilePosition, bool) if there is a tile the player has
    def play(self, player_id, tile_id=None, position=None):
        if player_id != self.order[self.current_player_itr]:
            raise GameException(f'Error : Player {self.current_player} should play, not player {player_id} !')
        # Select tile
        if tile_id is not None:
            self.pickTile(player_id, tile_id)
        print('Tile picked playable ? ' + str(self.player_can_play[player_id]))
        if position is not None and self.player_can_play[player_id]:
            print('tile : ',  self.player_previous_tiles[player_id].tile)
            self.placeTile(player_id, position, self.player_previous_tiles[player_id].tile)
        self.current_player_itr += 1
        self.player_can_play[player_id] = True
        
        return self.scores()


    def pickTile(self, player_id, tile_id):
        if not self.current_tiles:
            raise GameException('Error : No tiles left !')
        if self.current_tiles[tile_id].isSelected():
            raise GameException(f'Error : Player {player_id} cannot choose tile {tile_id} because it is already selected !')
        self.new_order[tile_id] = player_id
        self.current_tiles[tile_id].player_id = player_id
        self.player_current_tiles[player_id] = self.current_tiles[tile_id]
        # if player selects a tile they won't be able to place
        self.player_can_play[player_id] = self.isTilePlaceable(player_id, self.current_tiles[tile_id].tile)
        self.player_placed_all[player_id] = False
    
    def pickTileRandom(self, player_id):
        open_tiles = []
        print(self.current_tiles)
        for tile in self.current_tiles:
            if not tile.isSelected():
                open_tiles.append(tile)
        self.pickTile(player_id, random.sample(open_tiles, k=1)[0])
        

    def placeTile(self, player_id, position, tile):
        self.checkPlacementValid(player_id, position, tile)     

        self.boards[player_id].setBoard(position.points[0], v=tile.tile1.type)
        self.boards[player_id].setBoard(position.points[1], v=tile.tile2.type)
        self.boards[player_id].setBoardCrown(position.points[0], v=tile.tile1.nb_crown)
        self.boards[player_id].setBoardCrown(position.points[1], v=tile.tile2.nb_crown)
        
        if self.log:
            print(self.boards[player_id])
            
        return True
    
    def placeTileRandom(self, player_id, tile):
        positions = self.getPossiblePositions(player_id, tile)
        if positions:
            position = random.sample(positions, k=1)[0]
        else:
            self.player_can_play[player_id] = False
        self.placeTile(player_id, position, tile)
       
    # Pulls out 4 tiles from the stack
    def draw(self):
        self.previous_tiles = self.current_tiles.copy()
        self.current_tiles = [Kingdomino.TilePlayer(tile) for tile in random.sample(self.tiles,4)]
        self.tiles = [tile for tile in self.tiles if tile not in self.current_tiles]
        self.current_tiles = sorted(self.current_tiles, key=lambda x: x.tile.value)
    
    def printCurrentTiles(self):
        print('Tiles to pick from : \n')
        [print(tileplayer) for tileplayer in self.current_tiles]       


    # position : TilePosition
    # inversed false : position.p1 corresponds to first tile
    def checkPlacementValid(self, player_id, position, tile):   
        # Add for check in 5x5 square TODO
        
        # Check no overlapping
        for point in position.points:
            if self.boards[player_id].getBoard(point[0], point[1]) != -1:
                raise GameException(f'Overlapping at point {point}.')
        
        # Check that at least one position is near castle or near same type of env
        for point in position.points:
            if self.isNeighbourToCastleOrSame(player_id, point, tile):
                return
        raise GameException('Not next to castle or same type of env.')    
    
    
    def isNeighbourToCastleOrSame(self, player_id, point, tile):
        # Check if castle next
        if point in [(3,4),(4,3),(4,5),(5,4)]:
            return True
        
        tileToCompareType = tile.tile1.type
        
        # Check if same next
        for i in range(point[0]-1, point[0]+2):
            for j in range(point[1]-1, point[1]+2):
                # Exclude diagonals
                if i != point[0] and j != point[1]:
                    continue

                if self.boards[player_id].getBoard(i,j) == tileToCompareType:
                    return True
                
        return False
    
    def getPossiblePositions(self, player_id, tile, every_pos=False):
        available_pos = []
        for i in range(9):
            for j in range(9):
                if self.boards[player_id].getBoard(i, j) == -1:
                    for _i in range(i-1,i+2):
                        for _j in range(j-1,j+2):
                            if (_i != i and _j != j) or (_i==i and _j==j) or self.boards[player_id].getBoard(_i, _j) != -1:
                                continue
                            try:
                                pos = TilePosition(i,j,_i,_j)
                                self.checkPlacementValid(player_id, pos, tile)
                                available_pos.append(pos)
                                if not every_pos:
                                    return available_pos
                            except GameException:
                                pass
        return available_pos
    
    def isTilePlaceable(self, player_id, tile):
        return self.getPossiblePositions(player_id, tile, every_pos=False)
                            

