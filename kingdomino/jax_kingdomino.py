# NO JAX FOR WINDOWS+NVIDIA GPU :))))))) FML

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random

from env.base_env import BaseEnv
from setup import N_TILES, TILE_SIZE, GET_TILE_DATA

#%%

# Class attributes are constant that will not change after instantiation

# Numpy prng is stateful: bad
# key = random.PRNGKey(seed)
# random.normal(key, shape=(1,))
#☺ key,subkey = random.split(key)
# subkeys = random.split(key,int)
# use subkey for random generation
# key can be reused again after with split

# numpy not ok with parallelization and reproducibility

# f_jit = jit(f, static_argnums=(0,)) if conditioning on first argment in f
# make_jaxpr(f_jit, static_argnums=(0,)(x))
# also if loop over this argument
# static args should not change too much otw recompilation
# better than for loop: lax.fori_loop(0,n,fn,input), jittable
# allowed to condition on x dimensionality

# from jax.config import config
# config.update("jax_debug_nans", True)

class TileDeck:
        
    # state: tiles, available_mask, nb of remaining tiles
    def __init__(self):
        self.tiles = GET_TILE_DATA()
    
    @partial(jit, static_argnums=(0,))
    def reset(self, deck_state, key):
        key, subkey = random.split(key)
        tiles = random.permutation(subkey, self.tiles)
        idx = 0
        return (tiles,idx), key
        
    @partial(jit, static_argnums=(0,1))
    def draw(self, tiles, state, n):
        tiles, idx = state
        new_idx = idx + n
        drawn_tiles = tiles[idx:new_idx]
        new_state = tiles,new_idx
        return drawn_tiles, new_state
        

class Kingdomino(BaseEnv):
    def __init__(self, 
                 grid_size,
                 players=None, 
                 render_mode=None,
                 kingdomino=None,
                 reward_fn=lambda x,y: 0):
        pass
    
    def _get_obs(self, state):
        return state
    
    def _reset(self, key):
        return self.initial_state, key
    
    def _reset_if_done(self, env_state, done):
        key = env_state[1]
        return lax.cond(
            done,
            self._reset,
            lambda key: env_state,
            key,
        )
    
    def _get_reward_done(self, new_state):
        done = jnp.all(new_state == self.goal_state)
        reward = jnp.int32(done)
        return reward, done

    @partial(jit, static_argnums=(0))
    def step(self, env_state, action):
        state, key = env_state
        action = self.movements[action]
        new_state = jnp.clip(jnp.add(state, action), jnp.array([0, 0]), self.grid_size)
        reward, done = self._get_reward_done(new_state)
    
        env_state = new_state, key
        env_state = self._reset_if_done(env_state, done)
        new_state = env_state[0]
        return env_state, self._get_obs(new_state), reward, done

    def reset(self, key):
        env_state = self._reset(key)