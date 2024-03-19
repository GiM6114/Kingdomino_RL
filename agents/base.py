import numpy as np
import random

class Player:
    def action(self, state, kingdomino):
        pass
    def processReward(self, reward, next_state, done):
        pass
    
class HumanPlayer(Player):      
    def action(self, state, kingdomino):
        tile_id = int(input("Which tile do you choose ?"))
        x1 = int(input("x1 ? "))
        y1 = int(input("y1 ? "))      
        x2 = int(input("x2 ? "))
        y2 = int(input("y2 ? "))
        return tile_id, np.array([[x1,y1],[x2,y2]])


class RandomPlayer(Player):
    def action(self, state, kingdomino):
        return random.choice(kingdomino.getPossibleActions())