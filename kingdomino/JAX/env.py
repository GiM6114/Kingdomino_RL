# NO JAX FOR WINDOWS+NVIDIA GPU :))))))) FML

import jax
import jax.numpy as jnp
from flax import struct

from kingdomino.JAX.utils import draw_tiles, sort_tiles, EMPTY_TILE
from kingdomino.JAX.action import pick_tile, place_tile
from kingdomino.JAX.scores import compute_score

#%%

N_TURNS = 13

@struct.dataclass
class State:
    boards: jnp.ndarray             # (P, S, S)
    crowns: jnp.ndarray             # (P, S, S)

    current_tiles: jnp.ndarray      # (P, TILE_SIZE+1)
    previous_tiles: jnp.ndarray     # (P, TILE_SIZE+1)

    scores: jnp.ndarray             # (P,)
    prev_scores: jnp.ndarray        # (P,)

    order: jnp.ndarray              # (P,)
    new_order: jnp.ndarray          # (P,)

    current_player_itr: jnp.int32
    current_player_id: jnp.int32
    turn_id: jnp.int32

    deck: jnp.ndarray
    deck_idx: jnp.int32

    done: jnp.bool_

def reset(key, tile_data, n_players, board_size):
    """
    Pure JAX reset function.

    Inputs:
        key               : jax PRNGKey
        tile_data         : (N_TILES, TILE_SIZE) array
        n_players         : int

    Returns:
        state, observation
    """

    key_deck, key_order = jax.random.split(key)
    
    deck = jax.random.permutation(key_deck, tile_data)

    boards = (
        jnp.full((n_players, board_size, board_size), -1, dtype=jnp.int32)
        .at[:, board_size // 2, board_size // 2]
        .set(-2)
    )
    crowns = jnp.zeros((n_players, board_size, board_size), dtype=jnp.int32)
    scores = jnp.zeros((n_players,), dtype=jnp.int32)

    prev_scores = jnp.zeros((n_players,), dtype=jnp.int32)

    new_order = -jnp.ones(n_players, dtype=jnp.int32)
    order = jax.random.permutation(key_order, jnp.arange(n_players))

    current_player_itr = jnp.int32(0)
    current_player_id = order[current_player_itr]

    previous_tiles = EMPTY_TILE.reshape(1,-1).repeat(n_players, axis=0)

    deck_idx = jnp.int32(0)
    current_tiles = draw_tiles(deck, deck_idx, n_players)
    deck_idx = deck_idx + n_players

    # sort by tile value
    current_tiles = sort_tiles(current_tiles, 4)

    # mark as unclaimed
    current_tiles = current_tiles.at[:, -1].set(-1)

    state = State(
        boards=boards,
        crowns=crowns,

        current_tiles=current_tiles,
        previous_tiles=previous_tiles,

        scores=scores,
        prev_scores=prev_scores,

        order=order,
        new_order=new_order,

        current_player_itr=current_player_itr,
        current_player_id=current_player_id,

        turn_id=jnp.int32(1),

        deck=deck,
        deck_idx=deck_idx,

        done=jnp.bool_(False),
    )

    obs = build_observation(state)
    return state, obs

def step(state, action):
    """
    Pure JAX environment step.

    Args:
        state:
            current State

        action:
            scalar int action

        reward_fn:
            callable(state_before, action, state_after) -> reward

    Returns:
        next_state,
        obs,
        reward,
        done,
        info
    """

    next_state = transition(state, action)
    done = next_state.done
    obs = build_observation(next_state)
    info = {}

    return next_state, obs, done, info

def transition(state, action):
    # tile_id, placement_id = decode_action(action)
    tile_id, position = action[0], action[1:]
    # position = all_placements[placement_id]
    player_id = state.current_player_id

    next_state = pick_tile(state, player_id, tile_id)
    next_state = place_tile(next_state, player_id, position)
    next_state = advance_turn(next_state)
    return next_state

def advance_turn(state):
    n_players = state.boards.shape[0]

    next_itr = state.current_player_itr + 1
    new_round = next_itr == n_players
    # Next player
    current_player_itr = next_itr % n_players
    order = jax.lax.select(new_round, state.new_order, state.order)
    new_order = jax.lax.select(new_round, -jnp.ones(n_players, dtype=jnp.int32), state.new_order)
    current_player_id = order[current_player_itr]

    previous_tiles = jax.lax.select(
        new_round,
        sort_tiles(state.current_tiles, 5),
        state.previous_tiles,
    )

    current_tiles = jax.lax.select(
        new_round,
        sort_tiles(draw_tiles(state.deck, state.deck_idx, n_players), 4),
        state.current_tiles,
    )

    deck_idx = jax.lax.select(
        new_round,
        state.deck_idx + n_players,
        state.deck_idx,
    )

    turn_id = state.turn_id + new_round.astype(jnp.int32)
    done = turn_id > N_TURNS
    
    scores = jax.lax.cond(
        done,
        lambda _: compute_scores_jax(state.boards, state.crowns),
        lambda _: state.scores,
        operand=None,
    )

    return state.replace(
        previous_tiles=previous_tiles,
        current_tiles=current_tiles,
        order=order,
        new_order=new_order,
        scores=scores,
        current_player_itr=current_player_itr,
        current_player_id=current_player_id,
        deck_idx=deck_idx,
        turn_id=turn_id,
        done=done,
    )

def compute_scores_jax(boards, crowns):
    return jax.vmap(compute_score)(boards, crowns)

def build_observation(state):
    """
    Builds per-player observation.

    Output shape is typically:
        (P, ...)

    Everything must remain static-shape.
    """

    return {
        "boards": state.boards,                      # (P, S, S)
        "crowns": state.crowns,                      # (P, S, S)

        "current_tiles": state.current_tiles,        # (P, TILE_SIZE+1)
        "previous_tiles": state.previous_tiles,      # (P, TILE_SIZE)

        "scores": state.scores,                      # (P,)

        "turn_id": jnp.broadcast_to(
            state.turn_id,
            state.scores.shape
        ),

        "done": jnp.broadcast_to(
            state.done,
            state.scores.shape
        ),
    }