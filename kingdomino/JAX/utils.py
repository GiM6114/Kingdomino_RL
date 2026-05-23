import jax.numpy as jnp

from setup import N_TILE_TYPES

EMPTY_TILE = jnp.array([N_TILE_TYPES,N_TILE_TYPES,-1,-1,-1], dtype=jnp.int32) 
ORIENTATIONS = jnp.array([[0,1],[1,0],[0,-1],[-1,0]], dtype=jnp.int32)

def sort_tiles(tiles):
    """
    Sort tiles by last column (ascending) to determine player order.
    
    tiles: (P, TILE_SIZE)
    """
    sort_key = tiles[:, 4]
    idx = jnp.argsort(sort_key, stable=True)
    return tiles[idx]

def draw_tiles(
    deck,
    deck_idx,
    n,
):
    """
    JIT-safe fixed-shape tile draw.

    Always returns shape:
        (n, TILE_FEATURES)

    Missing tiles are replaced with EMPTY_TILE.
    """

    n_tiles = deck.shape[0]
    indices = jnp.arange(deck_idx, deck_idx + n)
    valid = indices < n_tiles
    safe_indices = jnp.minimum(indices, n_tiles - 1)
    drawn_tiles = deck[safe_indices]
    empty_tiles = jnp.repeat(
        EMPTY_TILE[None],
        n,
        axis=0,
    )
    tiles = jnp.where(
        valid[:, None],
        drawn_tiles,
        empty_tiles,
    )
    return tiles