import random
import numpy as np
from copy import deepcopy
from itertools import product

from agents.base import Player
from kingdomino.kingdomino import Kingdomino
from kingdomino.board import get_territories
from printer import Printer


    
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

import multiprocessing
import random
import numpy as np
from copy import deepcopy
from itertools import product

from agents.base import Player
from kingdomino.kingdomino import Kingdomino
from kingdomino.board import get_territories
from printer import Printer
from utils import arr_except

class MPC_Agent(Player):
    '''
        Model Predictive Control agent
        Does rollouts until end of game for each action
    '''
    def __init__(self, n_rollouts, n_processes=None, id=None, n_zones_only=False):
        '''
            n_zones only : only applies MPC on position placements that minimize nb of regions
            Example : if 2 open tiles, and only one placements lead to 0 new regions, then two actions considered in rollouts
        '''
        self.n_zones_only = n_zones_only
        self.id = id
        self.n_rollouts = n_rollouts
        if n_processes is None:
            self.n_processes = multiprocessing.cpu_count()

    def get_opt_positions(self, kingdomino):
        tiles_possible, positions_possible = kingdomino.getPossibleTilesPositions()
        lowest_n_regions = np.inf
        opt_positions = []
        for position in positions_possible:
            position = np.array(position)
            tile = kingdomino.players_previous_tile[self.id]
            board = kingdomino.boards.boards[self.id].copy()
            board[position.T[0],position.T[1]] = tile[:2]
            n_regions = len(get_territories(board))
            if n_regions < lowest_n_regions:
                lowest_n_regions = n_regions
                opt_positions = [position]
            elif n_regions == lowest_n_regions:
                opt_positions.append(position)
        return list(product(tiles_possible, opt_positions))

    def action(self, state, kingdomino):
        if self.n_zones_only:
            actions = self.get_opt_positions(kingdomino)
        else:
            actions = kingdomino.getPossibleActions()
        if len(actions) == 1:
            return actions[0]
        best_action = None
        best_result = -np.inf
        for action in actions:
            result = self.parallel_rollout(kingdomino, action, self.n_rollouts, self.n_processes)
            if result > best_result:
                best_action = action
                best_result = result
        return best_action
    
    def parallel_rollout(self, kingdomino, action, n_rollouts, n_processes):
        results = []
        for i in range(n_rollouts):
            results.append(self.rollout(deepcopy(kingdomino), action))
        # with multiprocessing.Pool(processes=n_processes) as pool:
        #     print('starting rollouts')
        #     results = pool.starmap(
        #         self.rollout, 
        #         [(copy_kingdomino(kingdomino), action) for _ in range(n_rollouts)]
        #     )
        return sum(results)

    def rollout(self, kingdomino, action):
        '''
            In the rollout:
                This player minimizes n_territories
                Other player does random moves
        '''
        kingdomino.compute_obs = False
        done = False
        _,done = kingdomino.step(action)
        while not done:
            for player_id in kingdomino.order:
                if player_id == self.id:
                    action = random.choice(self.get_opt_positions(kingdomino))
                else:    
                    action = random.choice(kingdomino.getPossibleActions())
                _,done = kingdomino.step(action)
                if done:
                    break
        scores = kingdomino.getScores()
        delta = scores[self.id] - np.mean(arr_except(scores, except_id=self.id))
        return delta
    
def copy_kingdomino(kingdomino):
    '''
        Returns a kingdomino in the same game state as the given kingdomino
    '''
    k_copy = deepcopy(kingdomino)
    k_copy.tile_deck.shuffle_remaining_tiles() # no advantages of knowing rest of the cards
    return k_copy
    
if __name__ == '__main__':
    from tqdm import tqdm
    from agents.base import RandomPlayer
    from kingdomino.rewards import reward_last_delta_quantitative
    from IPython.core.display import Image, display
    from graphics import draw_obs

    env = Kingdomino(5, random_start_order=False, n_players=2, reward_fn=None)
    players = [MPC_Agent(8*1, id=0, n_zones_only=True), RandomPlayer()]
    n_episodes = 1
    for i in tqdm(range(n_episodes), position=0, leave=False):
        state = env.reset()
        done = False
        s = 0
        while not done:
            for player_id in env.order:
                print(f"PLAYER {player_id} TURN")
                display(draw_obs(state))
                action = players[player_id].action(state, env)
                print(action)
                state,done = env.step(action)
                if done:
                    break
    print('Final State')
    display(draw_obs(state))
    print(env.getScores())