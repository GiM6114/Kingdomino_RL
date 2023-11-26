from printer import Printer

import numpy as np

class Zone:
    def __init__(self):
        pass

class Board:
    def __init__(self):
        self.max_planks = 12
        self.reset()
        
    
    def reset(self):
        self.nb_planks = 0
        
        self.board = -np.ones([9,9], dtype='int8')
        self.board[4,4] = -2
        self.crown = np.zeros([9,9], dtype='uint8')
        
        self.left_most   = 4
        self.right_most  = 4
        self.bottom_most = 4
        self.top_most    = 4  
        
    
    def getBoard(self, x, y):
        if 9 > x >= 0 and 9 > y >= 0:
            return self.board[x,y]
        return -1
    
    
    # position : np.array([pos halftile 1, pos halftile 2])
    def placeTile(self, position, tile):
        position = position.T
        self.board[position[0],position[1]] = tile[:2]
        if tile[0] != -1:
            self.nb_planks += 1
            self.left_most = min(self.left_most, np.min(position[1,:]))
            self.right_most = max(self.right_most, np.max(position[1,:]))
            self.bottom_most = max(self.bottom_most, np.max(position[0,:]))
            self.top_most = min(self.top_most, np.min(position[0,:])) # inverted for display
        self.crown[position[0],position[1]] = tile[2:4]    
           
    
    def count(self):
        x_size,y_size = self.board.shape
        score = 0
        board_seen = np.zeros_like(self.board)
        Printer.print('Left, right, bottom, top :', self.left_most, self.right_most, self.bottom_most, self.top_most)
        for x in range(x_size):
            for y in range(y_size):
                if self.getBoard(x,y) == -1:
                    continue
                nb_squares,nb_crowns = self.computeZone(x, y, board_seen, self.board[x,y])
                score += nb_squares * nb_crowns

        # The Middle Kingdom
        if self.isCastleCentered():
            score += 10
            
        # Harmony
        if self.nb_planks == self.max_planks:
            score += 5
        
        return score
                
                
    def computeZone(self, x, y, board_seen, env_type):
        if self.getBoard(x, y) != env_type or board_seen[x,y] == 1:
            return 0,0
        board_seen[x,y] = 1
        
        add_squares = 1
        add_crowns  = self.crown[x,y]
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                # Get rid of diagonals
                if i != x and j != y:
                    continue
                add_squares_temp,add_crowns_temp = self.computeZone(
                    i, j, board_seen, env_type
                    )
                add_squares += add_squares_temp
                add_crowns += add_crowns_temp
                
        return add_squares, add_crowns
    
    
    def isInFiveSquare(self, point):
        return not (self.right_most - point[0] >= 5
                    or self.bottom_most - point[1] >= 5 
                    or point[0] - self.left_most >= 5
                    or point[1] - self.top_most >= 5)
    
    
    def isCastleCentered(self):
        return self.left_most == 2 and self.right_most == 6 and self.bottom_most == 6 and self.top_most == 2
        
    
    def __str__(self):     
        string = ''
        for j in range(9):
            for i in range(9):
                addition = ( (f'{self.board[i,j]}' if self.board[i,j] != -1 else '') + ('' if self.crown[i,j] == 0 else f'+{self.crown[i,j]}')).center(5,' ')
                string += addition + '|'
            string += '\n'
        return string
    
    


