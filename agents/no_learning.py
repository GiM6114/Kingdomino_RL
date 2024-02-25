import random
import numpy as np
from copy import deepcopy

from agent.base import Player
from kingdomino import Kingdomino
from printer import Printer

class AggressivePlayer(Player):
    def action(self, state, kingdomino):
        actions = kingdomino.getPossibleActions()
    
def around_center(position):
    for point in position:
        if list(point) in [[4,3],[3,4],[4,5],[5,4]]:
            return True
    return False
    
class SelfCenteredPlayer(Player):
    # Heuristic: min nb of territories
    # + when equality favor immediate tile placement not near center
    # could also favor next play not near center for further finetuning
    def action(self, state, kingdomino):
        if self.previous_tile is None:
            return random.choice(kingdomino.getPossibleActions())
        best_action = None
        lowest_n_territories = 999
        best_near_center = False
        best_near_center_2 = False
        actions = kingdomino.getPossibleActions()
        for action in actions:
            tile_choice,position = action
            positionT = position.T
            if not (position == Kingdomino.discard_tile).all():
                board_values = self.board.board[positionT[0],positionT[1]]
                self.board.board[positionT[0],positionT[1]] = self.previous_tile[:2]
            next_tile = kingdomino.current_tiles[tile_choice].tile
            next_tile_positions = kingdomino.getPossiblePositions(next_tile, every_pos=True)
            # print('Player', kingdomino.current_player_id, 'playing according to kingdomino')
            # print('Action', action)
            # print('Next tile', next_tile)
            for next_tile_position in next_tile_positions:
                next_tile_positionT = next_tile_position.T
                if not (next_tile_position == Kingdomino.discard_tile).all():
                    board_new_values = self.board.board[next_tile_positionT[0],next_tile_positionT[1]]
                    self.board.board[next_tile_positionT[0],next_tile_positionT[1]] = next_tile[:2]
                territories = self.board.getTerritories()
                n_territories = len(territories)
                # print(territories)
                # print('Next pos', next_tile_position)
                # print('n territories', n_territories)
                if (n_territories < lowest_n_territories) or (n_territories == lowest_n_territories and ((best_near_center and not around_center(position)) or (not best_near_center and best_near_center_2 and not around_center(next_tile_position)))):
                    best_action = action
                    lowest_n_territories = n_territories
                    best_near_center = around_center(position)
                    best_near_center_2 = around_center(next_tile_position)
                if not (next_tile_position == Kingdomino.discard_tile).all():
                    self.board.board[next_tile_positionT[0],next_tile_positionT[1]] = board_new_values
            if not (position == Kingdomino.discard_tile).all():
                self.board.board[positionT[0],positionT[1]] = board_values
        return best_action
        
    
    
#%%

class MPC_Agent(Player):
    
    def __init__(self, n_rollouts, player_id):
        self.n_rollouts = n_rollouts
        self.id = player_id
    
    def action(self, state, kingdomino):
        actions = kingdomino.getPossibleActions()
        best_action = None
        best_result = -np.inf
        for action in actions:
            print(action)
            # do the action in deepcopied env
            # unroll with random policy
            result = 0
            for i in range(self.n_rollouts):
                print(i)
                kingdomino_copy = deepcopy(kingdomino)
                result += self.rollout(kingdomino_copy, action)
            if result > best_result:
                best_action = action
                
        return best_action

    def rollout(self, kingdomino, action):
        done = False
        Printer.print('Copied KDs order :', kingdomino.order)
        Printer.print('Copied KDs current player id :', kingdomino.current_player_id)
        while not done:
            terminated = kingdomino.step(action)
            done = terminated
            if not done:
                action = kingdomino.getRandomAction()
        scores = kingdomino.scores()
        final_result = self.id == np.argmax(scores)
        return final_result