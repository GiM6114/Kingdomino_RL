# NO JAX FOR WINDOWS+NVIDIA GPU :))))))) FML

import jax
import jax.numpy as jnp
from flax import struct

from kingdomino.JAX.utils import draw_tiles, sort_tiles, EMPTY_TILE

#%%

@struct.dataclass
class State:
    boards: jnp.ndarray              # (P, S, S)
    crowns: jnp.ndarray              # (P, S, S)

    current_tiles: jnp.ndarray      # (P, TILE_SIZE+1)
    previous_tiles: jnp.ndarray     # (P, TILE_SIZE)

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



def reset(
    key,
    *,
    tile_data,
    board_size,
    n_players,
    random_start_order=True,
):
    """
    Pure JAX reset function.

    Inputs:
        key               : jax PRNGKey
        tile_data         : (N_TILES, TILE_SIZE) array
        board_size        : int
        n_players         : int

    Returns:
        state, observation
    """

    key_deck, key_order = jax.random.split(key)
    
    deck = jax.random.permutation(key_deck, tile_data)

    boards = -jnp.ones((n_players, board_size, board_size), dtype=jnp.int32)
    crowns = jnp.zeros((n_players, board_size, board_size), dtype=jnp.int32)
    scores = jnp.zeros((n_players,), dtype=jnp.int32)

    prev_scores = jnp.zeros((n_players,), dtype=jnp.int32)

    if random_start_order:
        new_order = jax.random.permutation(key_order, jnp.arange(n_players))
    else:
        new_order = jnp.arange(n_players)

    order = new_order

    current_player_itr = jnp.int32(0)
    current_player_id = order[current_player_itr]

    previous_tiles = EMPTY_TILE.repeat(n_players, axis=0)

    deck_idx = jnp.int32(0)
    current_tiles = draw_tiles(deck, deck_idx, n_players)
    deck_idx = deck_idx + n_players

    # sort by tile value
    current_tiles = sort_tiles(current_tiles)

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

def step(state, action, reward_fn,
):
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

    # ======================================================
    # Decode action
    # ======================================================

    tile_id, placement_id = decode_action(action)
    position = ALL_PLACEMENTS[placement_id]
    player_id = state.current_player_id

    # ======================================================
    # Validate action
    # ======================================================

    valid = is_action_valid(
        state=state,
        tile_id=tile_id,
        placement_id=placement_id,
    )

    # Invalid actions indicate a bug in masking/training
    if not valid:
        raise ValueError("Invalid action selected.")

    # ======================================================
    # Pick next tile
    # ======================================================

    next_state = jax.lax.cond(
        is_last_turn(state),

        # no tile selection on last turn
        lambda s: s,

        # normal tile selection
        lambda s: pick_tile(
            state=s,
            tile_id=tile_id,
        ),

        state,
    )

    # ======================================================
    # Place previous tile
    # ======================================================

    next_state = jax.lax.cond(
        is_first_turn(state),

        # no placement on first turn
        lambda s: s,

        # normal placement
        lambda s: place_tile(
            state=s,
            player_id=player_id,
            position=position,
        ),

        next_state,
    )

    # ======================================================
    # Advance player iterator
    # ======================================================

    next_state = advance_turn(next_state)

    # ======================================================
    # Reward
    # ======================================================

    reward = reward_fn(
        state,
        action,
        next_state,
    )

    # ======================================================
    # Done
    # ======================================================

    done = next_state.done

    # ======================================================
    # Observation
    # ======================================================

    obs = build_observation(next_state)

    # ======================================================
    # Optional diagnostics
    # ======================================================

    info = {}

    return next_state, obs, reward, done, info

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