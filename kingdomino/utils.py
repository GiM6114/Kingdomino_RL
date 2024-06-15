import numpy as np

from kingdomino.kingdomino import Kingdomino

def compute_n_positions(board_size):
    # + 1: discard tile
    return 4*board_size**2 - 4*board_size - 8

def compute_n_actions(board_size, n_players):
    n_positions = compute_n_positions(board_size)
    return (n_players+1)*(n_positions+1)

