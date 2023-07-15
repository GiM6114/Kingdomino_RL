import re
import numpy as np

ENV_TYPES = ['Grass','Forest','Field','Water','Swamp','Mine']
IDX_TO_TYPES = {i:env for i,env in enumerate(ENV_TYPES)}
TYPES_TO_IDX = {env:i for i,env in enumerate(ENV_TYPES)}
TILE_SIZE = 5
N_TILE_TYPES = len(ENV_TYPES)

def GET_TILE_DATA():
    types_or = '|'.join(ENV_TYPES)
    regexp = f'({types_or}):*([0-9])*,({types_or}):*([0-9])*;([0-9]+)'
    with open('tiles_data.txt') as f:
        lines = f.readlines()
        tiles = np.zeros((len(lines),5))
        for i,line in enumerate(lines):
            
            # Get rid of comments
            if '#' in line:
                continue
            
            # Remove spaces in the line
            line = line.replace(' ','')
            
            # Apply regexp to get the info of the tile
            m = re.search(regexp, line)
            try:
                types = np.array([TYPES_TO_IDX[m.group(1)],
                                  TYPES_TO_IDX[m.group(3)]],
                                 dtype='int64')
                crowns = np.array([m.group(2) or 0,
                                   m.group(4) or 0],
                                  dtype='int64')
                crowns = np.nan_to_num(crowns)
                value = np.array([m.group(5)],
                                 dtype='int64')
                
                tile = np.concatenate((types, crowns, value))                
                tiles[i] = tile
            except Exception as e:
                print(f'Error at line {i}')
                print(e)
    return tiles

# class Tile:
#     def __init__(self, tile1, tile2, value):
#         self.tile1 = tile1
#         self.tile2 = tile2
#         self.value = int(value)
    
#     def __str__(self):
#         return f'{self.tiles[0]} {self.tiles[1]} {self.value}'


# class HalfTile:
#     def __init__(self, env_type, nb_crown):
#         self.type = int(env_type)
#         self.nb_crown = 0 if nb_crown is None else int(nb_crown)
    
#     def __str__(self):
#         return f'{IDX_TO_TYPES[self.type]}:{self.nb_crown}'