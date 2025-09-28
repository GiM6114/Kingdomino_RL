from printer import Printer

import numpy as np

class Zone:
    def __init__(self):
        pass

class Boards:
    '''
        Holds all the boards of a kingdomino env
    '''
    def __init__(self, size, n_players):
        self.max_planks = 12
        self.size = size
        self.middle = self.size // 2
        self.n_players = n_players
        self.reset()

    def reset(self):
        self.nb_planks = np.zeros(self.n_players)
        
        self.boards = -np.ones(
            [self.n_players, self.size,self.size], 
            dtype='int8')
        self.boards[:, self.middle,self.middle] = -2
        self.crowns = np.zeros(
            [self.n_players, self.size,self.size], 
            dtype='uint8')
        
        self.left_most   = self.middle * np.ones(self.n_players)
        self.right_most  = self.middle * np.ones(self.n_players)
        self.bottom_most = self.middle * np.ones(self.n_players)
        self.top_most    = self.middle * np.ones(self.n_players)
        
    def getBoard(self, p, x, y):
        if self.size > x >= 0 and self.size > y >= 0:
            return self.boards[p,x,y]
        return -1
    
    # position : np.array([pos halftile 1, pos halftile 2])
    def placeTile(self, p, position, tile):
        position = position.T
        self.boards[p,position[0],position[1]] = tile[:2]
        if tile[0] != -1:
            self.nb_planks[p] += 1
            self.left_most[p] = min(self.left_most[p], np.min(position[0,:]))
            self.right_most[p] = max(self.right_most[p], np.max(position[0,:]))
            self.bottom_most[p] = max(self.bottom_most[p], np.max(position[1,:]))
            self.top_most[p] = min(self.top_most[p], np.min(position[1,:])) # inverted for display
        self.crowns[p,position[0],position[1]] = tile[2:4]

    def count(self, p):
        _,x_size,y_size = self.boards.shape
        score = 0
        board_seen = np.zeros_like(self.boards[p], dtype=bool)
        for x in range(x_size):
            for y in range(y_size):
                if self.getBoard(p, x,y) in [-1,-2]:
                    continue
                n_squares,n_crowns = self.computeZone(p, x, y, board_seen, self.boards[p,x,y])
                score += n_squares * n_crowns
        # The Middle Kingdom
        # if self.isCastleCentered(p):
        #     score += 10
        # Harmony
        # if self.nb_planks[p] == self.max_planks:
        #     score += 5
        
        return score
    
    def computeZone(self, p, x, y, board_seen, env_type):
        if self.getBoard(p, x, y) != env_type or board_seen[x,y] == True:
            return 0,0
        board_seen[x,y] = True
        
        add_squares = 1
        add_crowns  = self.crowns[p,x,y]
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                # Get rid of diagonals
                if i != x and j != y:
                    continue
                add_squares_temp,add_crowns_temp = self.computeZone(
                    p, i, j, board_seen, env_type
                    )
                add_squares += add_squares_temp
                add_crowns += add_crowns_temp
                
        return add_squares, add_crowns
    
    def getTerritory(self, x, y, board_seen, env_type):
        if self.getBoard(x, y) != env_type or board_seen[x,y] == 1:
            return []
        board_seen[x,y] = 1
        
        add_points = [(x,y)]
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                # Get rid of diagonals
                if i != x and j != y:
                    continue
                adjacent_points = self.getTerritory(
                    i, j, board_seen, env_type)
                add_points += adjacent_points
        
        return add_points
    
    def isInFiveSquare(self, p, point):
        return not (self.right_most[p] - point[0] >= 5
                    or self.bottom_most[p] - point[1] >= 5 
                    or point[0] - self.left_most[p] >= 5
                    or point[1] - self.top_most[p] >= 5)
    
    
    def isCastleCentered(self, p):
        print(self.left_most[p])
        return self.left_most[p] - self.right_most[p] == 0 and self.top_most[p] - self.bottom_most[p] == 0
        # return self.left_most[p] == (self.middle-2) and self.right_most[p] == (self.middle+2) and self.bottom_most[p] == (self.middle+2) and self.top_most[p] == (self.middle-2)
      
        
def get_territories(board):
    x_size,y_size = board.shape
    territories = []
    board_seen = np.zeros_like(board, dtype='int8')
    for x in range(x_size):
        for y in range(y_size):
            if get_board(board, x, y) in [-1,-2]:
                continue
            territory = get_territory(board, x, y, board_seen, board[x,y])
            if len(territory) != 0:           
                territories.append((territory, board[x,y]))
    return territories  

def get_territory(board, x, y, board_seen, env_type):
    if get_board(board, x, y) != env_type or board_seen[x,y] == 1:
        return []
    board_seen[x,y] = 1
    
    add_points = [(x,y)]
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            # Get rid of diagonals
            if i != x and j != y:
                continue
            adjacent_points = get_territory(board, i, j, board_seen, env_type)
            add_points += adjacent_points
    
    return add_points

def get_board(board, x, y):
    size = board.shape[0]
    if size > x >= 0 and size > y >= 0:
        return board[x,y]
    return -1      

if __name__ == '__main__':

    test_board = Board()

    grass_tiles  = [[4,5],[4,6],[5,5],[5,4],[3,5],[3,6],[6,2]]
    forest_tiles = [[4,3],[5,3],[6,3],[6,4],[6,5],[6,6],[5,6]]
    water_tiles  = [[2,6],[2,5],[2,4],[3,4],[3,3],[3,2]]
    mine_tiles   = [[2,3],[2,2]]

    crown_1 = [(4,5),(4,6),(2,2)]
    crown_2 = [(5,4)]

    for tiles in [(grass_tiles,0), (water_tiles,3), (mine_tiles,5), (forest_tiles,1)]:
        v = tiles[1]
        for tile in tiles[0]:
            test_board.setBoard(tuple(tile), v)

    for i,crown in enumerate([crown_1, crown_2]):
        for pos in crown:
            test_board.setBoardCrown(pos, i+1)

    print('Should be 36 : ', test_board.count())

    test_board = Board()

    grass_tiles  = [[4,5],[4,6],[5,5],[5,4],[3,5],[3,6],[6,2]]
    forest_tiles = [[4,3],[5,3],[6,3],[6,4],[6,5],[6,6],[5,6]]
    water_tiles  = [[3,4],[3,3],[3,2]]
    mine_tiles   = []

    crown_1 = [(4,5),(4,6),(2,2)]
    crown_2 = [(5,4)]

    for tiles in [(grass_tiles,0), (water_tiles,3), (mine_tiles,5), (forest_tiles,1)]:
        v = tiles[1]
        for tile in tiles[0]:
            test_board.setBoard(tuple(tile), v)

    for i,crown in enumerate([crown_1, crown_2]):
        for pos in crown:
            test_board.setBoardCrown(pos, i+1)

    print('Should be 24 : ', test_board.count())

    
#%%
# class Board:
#     def __init__(self, size):
#         self.max_planks = 12
#         self.size = size
#         self.middle = self.size // 2
#         self.reset()
        
    
#     def reset(self):
#         self.nb_planks = 0
        
#         self.board = -np.ones([self.size,self.size], dtype='int8')
#         self.board[self.middle,self.middle] = -2
#         self.crown = np.zeros([self.size,self.size], dtype='uint8')
        
#         self.left_most   = self.middle
#         self.right_most  = self.middle
#         self.bottom_most = self.middle
#         self.top_most    = self.middle  
        
    
#     def getBoard(self, x, y):
#         if self.size > x >= 0 and self.size > y >= 0:
#             return self.board[x,y]
#         return -1
    
    
#     # position : np.array([pos halftile 1, pos halftile 2])
#     def placeTile(self, position, tile):
#         position = position.T
#         self.board[position[0],position[1]] = tile[:2]
#         if tile[0] != -1:
#             self.nb_planks += 1
#             self.left_most = min(self.left_most, np.min(position[0,:]))
#             self.right_most = max(self.right_most, np.max(position[0,:]))
#             self.bottom_most = max(self.bottom_most, np.max(position[1,:]))
#             self.top_most = min(self.top_most, np.min(position[1,:])) # inverted for display
#         self.crown[position[0],position[1]] = tile[2:4]    
           
    
#     def count(self):
#         x_size,y_size = self.board.shape
#         score = 0
#         board_seen = np.zeros_like(self.board)
#         Printer.print('Left, right, bottom, top :', self.left_most, self.right_most, self.bottom_most, self.top_most)
#         for x in range(x_size):
#             for y in range(y_size):
#                 if self.getBoard(x,y) in [-1,-2]:
#                     continue
#                 nb_squares,nb_crowns = self.computeZone(x, y, board_seen, self.board[x,y])
#                 score += nb_squares * nb_crowns

#         # The Middle Kingdom
#         if self.isCastleCentered():
#             score += 10
            
#         # Harmony
#         if self.nb_planks == self.max_planks:
#             score += 5
        
#         return score
                
        
#     def getTerritories(self):
#         x_size,y_size = self.board.shape
#         territories = []
#         board_seen = np.zeros_like(self.board, dtype='int8')
#         Printer.print('Left, right, bottom, top :', self.left_most, self.right_most, self.bottom_most, self.top_most)
#         for x in range(x_size):
#             for y in range(y_size):
#                 if self.getBoard(x,y) in [-1,-2]:
#                     continue
#                 territory = self.getTerritory(x, y, board_seen, self.board[x,y])
#                 if len(territory) != 0:           
#                     territories.append((territory,self.board[x,y]))
#         return territories
    
#     def getTerritory(self, x, y, board_seen, env_type):
#         if self.getBoard(x, y) != env_type or board_seen[x,y] == 1:
#             return []
#         board_seen[x,y] = 1
        
#         add_points = [(x,y)]
#         for i in range(x-1, x+2):
#             for j in range(y-1, y+2):
#                 # Get rid of diagonals
#                 if i != x and j != y:
#                     continue
#                 adjacent_points = self.getTerritory(
#                     i, j, board_seen, env_type)
#                 add_points += adjacent_points
        
#         return add_points
        
#     def computeZone(self, x, y, board_seen, env_type):
#         if self.getBoard(x, y) != env_type or board_seen[x,y] == 1:
#             return 0,0
#         board_seen[x,y] = 1
        
#         add_squares = 1
#         add_crowns  = self.crown[x,y]
#         for i in range(x-1, x+2):
#             for j in range(y-1, y+2):
#                 # Get rid of diagonals
#                 if i != x and j != y:
#                     continue
#                 add_squares_temp,add_crowns_temp = self.computeZone(
#                     i, j, board_seen, env_type
#                     )
#                 add_squares += add_squares_temp
#                 add_crowns += add_crowns_temp
                
#         return add_squares, add_crowns
    
    
#     def isInFiveSquare(self, point):
#         return not (self.right_most - point[0] >= 5
#                     or self.bottom_most - point[1] >= 5 
#                     or point[0] - self.left_most >= 5
#                     or point[1] - self.top_most >= 5)
    
    
#     def isCastleCentered(self):
#         return self.left_most == 2 and self.right_most == 6 and self.bottom_most == 6 and self.top_most == 2
        
    
#     def __str__(self):     
#         string = ''
#         for j in range(self.size):
#             for i in range(self.size):
#                 addition = ( (f'{self.board[i,j]}' if self.board[i,j] != -1 else '') + ('' if self.crown[i,j] == 0 else f'+{self.crown[i,j]}')).center(5,' ')
#                 string += addition + '|'
#             string += '\n'
#         return string
    
    


