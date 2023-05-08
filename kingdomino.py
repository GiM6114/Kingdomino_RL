import random
import numpy as np
from collections import namedtuple
from itertools import permutations


from setup import GET_TILE_DATA
from board import Board
from agent import Player


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



class TileDeck:      
    def __init__(self):
        self.tile_data = GET_TILE_DATA() #list of Tile
        self.nb_tiles = len(self.tile_data)
        self.open_tiles = []
        self.reset()
        
    def reset(self):
        self.open_tiles.clear()
        self.open_tiles.extend(self.tile_data) # extend does a copy
            
    def draw(self, nb):
        tiles = []
        for i in range(nb):
            tile_id = random.randint(0, len(self.open_tiles)-1)
            tiles.append(self.open_tiles.pop(tile_id))
        return tiles
        
        
class TilePlayer:
    def __init__(self, tile, order=None, player=None):
        self.order = order
        self.tile = tile
        self.player = player
    def isSelected(self):
        return self.player != None
    def __str__(self):
        return str(self.tile) + ' ' + str(self.player)

class Kingdomino:
    
    discard_tile = np.array([-1,-1])
    
    def __init__(self, players, compute_scores_every_round=True, log=False):
        self.players = players
        self.nb_players = len(self.players)
        self.compute_scores_every_round = compute_scores_every_round
        self.log = log

        self.tile_deck = TileDeck()
        self.reset()
    
    def reset(self):
        # last position in list : plays last
        self.tile_deck.reset()
        self.last_turn = False
        self.current_tiles = []
        self.turnNb = 0
        self.order = np.random.permutation(self.nb_players)
        self.new_order = self.order.copy()
        for player in self.players:
            player.reset()
        self.current_player_itr = 0
    
    
    @property
    def current_player_itr(self):
        return self._current_player_itr
    @current_player_itr.setter
    def current_player_itr(self, v):
        self._current_player_itr = v
        if self._current_player_itr == self.nb_players:
            self._current_player_itr = 0
        if self._current_player_itr == 0:
            self.startTurn()
            
            
    def scores(self):
        return [p.board.count() for p in self.players]
        
    
    # Automatically called when current_player_itr is set to 0
    def startTurn(self):
        self.order = self.new_order.copy()
        for player in self.players:
            player.startTurn()
        self.draw()
        if len(self.current_tiles) < self.nb_players:
            self.last_turn = True    

    
    # player_id         : player who wants to play
    # tile_id           : id of tile he wants to pick if there is still tiles to pick
    # position_inversed : tuple (TilePosition, bool) if there is a tile the player has
    def play(self, player, tile_id, position=None):
        if player.id != self.order[self.current_player_itr]:
            raise GameException(f'Error : Player {self.current_player} should play, not {player} !')
        
        # Select tile
        if not self.last_turn:
            self.pickTile(player, tile_id)

        if position is not None:
            if not (position == Kingdomino.discard_tile).all():
                print('tile : ',  player.previous_tile.tile)
                self.placeTile(player, position, player.previous_tile.tile)
            else:
                print('tile discarded')
        self.current_player_itr += 1
        
        if self.compute_scores_every_round:
            return self.scores()


    def pickTile(self, player, tile_id):
        if not self.current_tiles:
            raise GameException('Error : No tiles left !')

        tile = self.current_tiles[tile_id]

        if tile.isSelected():
            raise GameException(f'Error : {player} cannot choose tile {player.current_tile} because it is already selected !')
        
        self.new_order[tile.order] = player.id
        player.current_tile = tile
        tile.player = player

    
    def selectTileRandom(self):
        open_tiles = []
        for i,tile in enumerate(self.current_tiles):
            if not tile.isSelected():
                open_tiles.append(i)
        return random.sample(open_tiles, k=1)[0]
        

    def placeTile(self, player, position, tile):
        self.checkPlacementValid(player, position, tile)     

        player.board.setBoard(position.points[0], v=tile.tile1.type)
        player.board.setBoard(position.points[1], v=tile.tile2.type)
        player.board.setBoardCrown(position.points[0], v=tile.tile1.nb_crown)
        player.board.setBoardCrown(position.points[1], v=tile.tile2.nb_crown)
        
        if self.log:
            print(player.board)
            
        return True
    
    def selectTilePositionRandom(self, player, tile):
        positions = self.getPossiblePositions(player, tile)
        if positions:
            position = random.sample(positions, k=1)[0]
        else:
            position = Kingdomino.discard_tile
        return position
       
    # Pulls out tiles from the stack
    def draw(self):
        self.current_tiles = [TilePlayer(tile) 
                              for tile 
                              in self.tile_deck.draw(self.nb_players)]
        self.current_tiles = sorted(self.current_tiles, key=lambda x: x.tile.value)
        for order,tile in enumerate(self.current_tiles):
            tile.order = order
    
    def printCurrentTiles(self):
        print('Tiles to pick from : \n')
        [print(tileplayer) for tileplayer in self.current_tiles]       


    # position : TilePosition
    # inversed false : position.p1 corresponds to first tile
    # TODO : perhaps speed this up with position as a numpy array
    def checkPlacementValid(self, player, position, tile):           
        # Check five square and no overlapping
        for point in position.points:
            if not player.board.isInFiveSquare(point):
                raise GameException(f'{point} is not in a five square.')
            if player.board.getBoard(point[0], point[1]) != -1:
                raise GameException(f'Overlapping at point {point}.')
        
        # Check that at least one position is near castle or near same type of env
        for point in position.points:
            if self.isNeighbourToCastleOrSame(player, point, tile):
                return
        raise GameException('Not next to castle or same type of env.')    
    
    
    def isNeighbourToCastleOrSame(self, player, point, tile):
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

                if player.board.getBoard(i,j) == tileToCompareType:
                    return True
                
        return False
    
    def getPossiblePositions(self, player, tile, every_pos=False):
        available_pos = []
        for i in range(9):
            for j in range(9):
                if player.board.getBoard(i, j) == -1:
                    for _i in range(i-1,i+2):
                        for _j in range(j-1,j+2):
                            if (_i != i and _j != j) or (_i==i and _j==j) or player.board.getBoard(_i, _j) != -1:
                                continue
                            try:
                                pos = TilePosition(i,j,_i,_j)
                                self.checkPlacementValid(player, pos, tile)
                                available_pos.append(pos)
                                if not every_pos:
                                    return available_pos
                            except GameException:
                                pass
        return available_pos
    
    def isTilePlaceable(self, player, tile):
        return self.getPossiblePositions(player, tile, every_pos=False)
                            

