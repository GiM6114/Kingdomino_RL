import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import jax
import matplotlib
matplotlib.use("inline")   # or "TkAgg" depending on system

from setup import GET_TILE_DATA
from kingdomino.JAX.env import reset, step
from kingdomino.JAX.action import legal_placements_mask, legal_tile_choices_masks, random_action
from kingdomino.JAX.utils import generate_placements
from kingdomino.JAX.render import render_state

def run_random_game(state, n_players, reward_fn, max_steps=20):
    state, obs = state, None

    fig, axes = plt.subplots(1, n_players, figsize=(4 * n_players, 4))

    if n_players == 1:
        axes = [axes]

    step_id = 0
    clear_output(wait=True)
    fig = render_state(state)
    display(fig)
    plt.close(fig)
    print(state.deck)
    while not bool(state.done) and step_id < max_steps:
        print('deck_idx', state.deck_idx)
        action = random_action(key, state, PLACEMENTS)
        state, obs, done, info = step(state, action)


        clear_output(wait=True)
        fig = render_state(state)
        display(fig)
        plt.close(fig)
        print('done:', done)
        print(state.turn_id)

        step_id += 1

    print(state.scores)
    return state

BOARD_SIZE = 5
PLACEMENTS = generate_placements(5)

if __name__ == "__main__":
    tile_data = GET_TILE_DATA()
    n_players = 2
    key = jax.random.key(0)
    start_state,initial_obs = reset(key, tile_data, n_players, BOARD_SIZE)
    run_random_game(start_state, n_players, lambda x,y,z:0, max_steps=30)
