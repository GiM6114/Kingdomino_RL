import jax.numpy as jnp

EMPTY_TILE = jnp.array([-1,-1,-1,-1,-1,-1], dtype=jnp.int32) 
PASS_POSITION = jnp.array([-1, -1, -1, -1], dtype=jnp.int32)
ORIENTATIONS = jnp.array([[0,1],[1,0],[0,-1],[-1,0]], dtype=jnp.int32)

def sort_tiles(tiles, id):
    """
    Sort tiles ascending, 4: by value, 5: by player id
    
    tiles: (P, TILE_SIZE)
    """
    sort_key = tiles[:, id]
    idx = jnp.argsort(sort_key, stable=True)
    return tiles[idx]

def draw_tiles(deck, deck_idx, n):
    """
    Always returns shape:
        (n, TILE_FEATURES)

    Missing tiles are replaced with EMPTY_TILE.
    """

    n_tiles = deck.shape[0]
    indices = deck_idx + jnp.arange(n)
    valid = indices < n_tiles
    safe_indices = jnp.minimum(indices, n_tiles - 1)
    drawn_tiles = jnp.copy(deck[safe_indices])
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

def generate_placements(board_size):
    """
    Returns array of valid placements:
        (x1, y1, x2, y2)
    Called only once at the beginning

    Constraints:
        - both cells in bounds
        - neither cell is the center
    """

    center = (board_size // 2, board_size // 2)

    coords = jnp.arange(board_size)
    xs, ys = jnp.meshgrid(coords, coords, indexing="ij")

    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    placements = [PASS_POSITION.reshape(1,-1)]

    for o in ORIENTATIONS:

        x2 = xs + o[0]
        y2 = ys + o[1]

        in_bounds = (
            (x2 >= 0) & (x2 < board_size) &
            (y2 >= 0) & (y2 < board_size)
        )

        # exclude center for both endpoints
        not_center_1 = ~((xs == center[0]) & (ys == center[1]))
        not_center_2 = ~((x2 == center[0]) & (y2 == center[1]))

        valid = in_bounds & not_center_1 & not_center_2

        placements.append(
            jnp.stack(
                [
                    xs[valid],
                    ys[valid],
                    x2[valid],
                    y2[valid],
                ],
                axis=1,
            )
        )

    return jnp.concatenate(placements, axis=0)