import numpy as np

class Player:
    def __init__(self, _id):
        self.id = _id
    
    def chooseTile(self):
        pass
    
    def placeTile(self):
        pass



class HumanPlayer(Player):  
    
    def chooseTile(self, kingdomino):
        try:
            print('Tile previously chosen : ', kingdomino.previous_tiles[self.id])
        except:
            pass
        tile_id = input("Which tile do you choose ?")
        return int(tile_id)
        
    def placeTile(self, kingdomino):
        x1 = int(input("x1 ? "))
        y1 = int(input("y1 ? "))      
        x2 = int(input("x2 ? "))
        y2 = int(input("y2 ? "))
        return x1,y1,x2,y2
    

class LearningPlayer(Player):
    
    def tilePlayerToState(self, tile_player, skip_id=True):
        base = np.array(
            [tile_player.tile.tile1.type, 
             tile_player.tile.tile2.type, 
             tile_player.tile.tile2.nb_crown,
             tile_player.tile.value])
        return base if skip_id else np.append(base, tile_player.player_id)
    
    
    def boardToState(self, board):
        relevant_columns = [0,1,2,3,5,6,7,8]
        env_state = board[self.id].board[:,relevant_columns].flatten()
        crown_state = board[self.id].crown[:,relevant_columns].flatten()
        state = np.append(env_state, crown_state)
        return state
        
    
    def gameToState(self, kingdomino):
        # First 160 inputs : the agent's boards (env and crowns (without center))
        state = -np.ones(676)
        state[:160] = self.boardToState(kingdomino.boards[self.id])
        
        
        # Next 160*3 inputs : other player's data or -1s
        other_ids = [i for i in range(kingdomino.nb_players) if i!=self.id]

        idx = 160
        for _id in other_ids:
            state[idx, idx+160] = self.boardToState(kingdomino.boards[self._id])
            idx += 160     
        idx += (4-kingdomino.nb_players) * 160
        
        # Next 4 inputs : the agent's previous tiles data
        state[idx, idx+4] = self.tilePlayerToState(kingdomino.previous_tiles[self.id])
        
        # Next 4*3 inputs : others' previous tiles data
        for _id in other_ids:
            state[idx, idx+4] = self.tilePlayerToState(kingdomino.previous_tiles[_id])
            idx += 4
        idx += (4-kingdomino.nb_players) * 4
       
        # Next 5*4 inputs : current tiles + 1 saying which player owns it
        for current_tile in kingdomino.current_tiles:
            state[idx, idx+5] = self.tilePlayerToState(current_tile, skip_id=False)
            idx += 5
        idx += (4-kingdomino.nb_players) * 5
        
        return state   