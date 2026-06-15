import os
N_PARALLEL = 7
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_PARALLEL}'
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import clear_output, display
matplotlib.use("inline")   # or "TkAgg" depending on system

from setup import GET_TILE_DATA
from kingdomino.JAX.env import reset, step
from kingdomino.JAX.utils import generate_placements
from agents.MPC_JAX import mpc_action_jittable
from kingdomino.JAX.action import random_action, generate_actions
from kingdomino.JAX.render import render_state

def run_game(key, state, action_fns, actions_kwargs):
    state, obs = state, None
    
    fig, axes = plt.subplots(1, N_PLAYERS, figsize=(4 * N_PLAYERS, 4))

    if N_PLAYERS == 1:
        axes = [axes]

    step_id = 0
    clear_output(wait=True)
    fig = render_state(state)
    display(fig)
    plt.close(fig)
    
    step_id = 0
    while not bool(state.done):
        key, subkey = jax.random.split(key)
        print(state.current_player_id)
        action = action_fns[state.current_player_id](subkey, state, **actions_kwargs[state.current_player_id])
        # action = [tile_id, PLACEMENTS[placement_id]]
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
N_PLAYERS = 2
ACTIONS = generate_actions(N_PLAYERS, PLACEMENTS)

if __name__ == "__main__":
    tile_data = GET_TILE_DATA()
    key = jax.random.key(1)
    key_reset,key_game = jax.random.split(key)
    start_state,initial_obs = reset(key_reset, tile_data, N_PLAYERS, BOARD_SIZE)
    # state = run_game(
    #     start_state, 
    #     [mpc_action, random_action], 
    #     [{'n_rollouts':500, 'n_processes':7, 'placements':PLACEMENTS, 
    #          # 'maximization_fn':lambda s: jnp.max(s[:,0])},
    #       'maximization_fn':lambda s:(jnp.argmax(s, axis=1) == 0).mean()},
    #      {'placements':PLACEMENTS}
    #     ])
    
    random_player = False
    if random_player:
        player = random_action
        player_args = {'placements':PLACEMENTS}
    else:
        player = mpc_action_jittable
        player_args = {'n_rollouts':4000, 'n_processes':N_PARALLEL, 'placements':PLACEMENTS, 'actions':ACTIONS,  
             # 'maximization_fn':lambda s: jnp.max(s[:,0])},
          'maximization_fn':lambda s:(jnp.argmax(s, axis=1) == 0).mean()}
    
    random_opponent = True
    if random_opponent:
        opponent = random_action
        opponent_args = {'placements':PLACEMENTS}
    else:
        opponent = mpc_action_jittable
        opponent_args = {
            'n_rollouts':1, 'n_processes':N_PARALLEL, 'placements':PLACEMENTS, 'actions':ACTIONS, 
             # 'maximization_fn':lambda s: jnp.max(s[:,0])},
            'maximization_fn':lambda s:(jnp.argmax(s, axis=1) == 1).mean()}
        
    state = run_game(
        key_game,
        start_state, 
        [player, opponent], 
        [player_args, opponent_args])
