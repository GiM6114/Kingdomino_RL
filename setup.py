import re

ENV_TYPES = ['Grass','Forest','Field','Water','Swamp','Mine']
IDX_TO_TYPES = {i:env for i,env in enumerate(ENV_TYPES)}
TYPES_TO_IDX = {env:i for i,env in enumerate(ENV_TYPES)}

def GET_TILE_DATA():
    tiles = []
    types_or = '|'.join(ENV_TYPES)
    regexp = f'({types_or}):*([0-9])*,({types_or}):*([0-9])*;([0-9]+)'
    with open('tiles_data.txt') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            
            # Get rid of comments
            if '#' in line:
                continue
            
            # Remove spaces in the line
            line = line.replace(' ','')
            
            # Apply regexp to get the info of the tile
            m = re.search(regexp, line)
            try:
                half_tile1 = HalfTile(TYPES_TO_IDX[m.group(1)], m.group(2))
                half_tile2 = HalfTile(TYPES_TO_IDX[m.group(3)], m.group(4))
                
                # Add the tile to our stack
                newTile = Tile(half_tile1, half_tile2, m.group(5))
                tiles.append(newTile)
            except Exception as e:
                print(f'Error at line {i}')
                print(e)
    return tiles

class Tile:
    def __init__(self, tile1, tile2, value):
        self.tile1 = tile1
        self.tile2 = tile2
        self.value = int(value)
    
    def __str__(self):
        return f'{self.tile1} {self.tile2} {self.value}'


class HalfTile:
    def __init__(self, env_type, nb_crown):
        self.type = int(env_type)
        self.nb_crown = 0 if nb_crown is None else int(nb_crown)
    
    def __str__(self):
        return f'{IDX_TO_TYPES[self.type]}:{self.nb_crown}'