import jax.numpy as jnp

from kingdomino.JAX.env import ORIENTATIONS

def generate_placements(board_size):
    """
    Returns array of valid placements:
        (x1, y1, x2, y2)

    Constraints:
        - both cells in bounds
        - neither cell is the center
    """

    center = (board_size // 2, board_size // 2)

    coords = jnp.arange(board_size)
    xs, ys = jnp.meshgrid(coords, coords, indexing="ij")

    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    # direction offsets (4-neighborhood)
    dx = jnp.array([0, 1, 0, -1], dtype=jnp.int32)
    dy = jnp.array([1, 0, -1, 0], dtype=jnp.int32)

    placements = []

    for o in range(4):

        x2 = xs + dx[o]
        y2 = ys + dy[o]

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

def legal_placements_mask(board, placements, tile, center):
    """
    board: (S, S)
    placements: (N_P, 4) -> x1,y1,x2,y2
    tile: (4,) -> t1,t2,c1,c2
    center: (cx,cy)
    """

    S = board.shape[0]
    cx, cy = center

    x1 = placements[:, 0]
    y1 = placements[:, 1]
    x2 = placements[:, 2]
    y2 = placements[:, 3]

    # -------------------------------------------------------
    # 1. occupancy constraint
    # -------------------------------------------------------

    free = (
        (board[x1, y1] == -1) &
        (board[x2, y2] == -1)
    )

    # -------------------------------------------------------
    # 2. tile terrain types (THIS WAS MISSING)
    # -------------------------------------------------------

    t1 = tile[0]
    t2 = tile[1]

    t1_vec = jnp.full(x1.shape, t1)
    t2_vec = jnp.full(x1.shape, t2)

    # -------------------------------------------------------
    # 3. adjacency using ORIENTATIONS
    # -------------------------------------------------------

    def has_match(x, y, terrain):

        xn = x[:, None] + ORIENTATIONS[:, 0]
        yn = y[:, None] + ORIENTATIONS[:, 1]

        in_bounds = (
            (xn >= 0) & (xn < S) &
            (yn >= 0) & (yn < S)
        )

        neighbor_vals = jnp.where(
            in_bounds,
            board[xn, yn],
            -9999
        )

        return jnp.any(neighbor_vals == terrain[:, None], axis=1)

    adj1 = has_match(x1, y1, t1_vec)
    adj2 = has_match(x2, y2, t2_vec)

    # -------------------------------------------------------
    # 4. center adjacency
    # -------------------------------------------------------

    def touches_center(x, y):

        xn = x[:, None] + ORIENTATIONS[:, 0]
        yn = y[:, None] + ORIENTATIONS[:, 1]

        return ((xn == cx) & (yn == cy)).any(axis=1)

    center_adj1 = touches_center(x1, y1)
    center_adj2 = touches_center(x2, y2)

    # -------------------------------------------------------
    # 5. rule: at least one endpoint must connect
    # -------------------------------------------------------

    connected = (
        (adj1 | center_adj1) |
        (adj2 | center_adj2)
    )

    return free & connected

def test():
    """
        Test the valid actions mask for an artifical board
    """
    
    def row_in_set(included, includer):
        """
        checks if each row in `included` exists in `includer`
        """
        return jnp.any(
            jnp.all(included[:, None, :] == includer[None, :, :], axis=-1),
            axis=1
        )
        
    board_size = 5

    board = -jnp.ones((board_size, board_size), dtype=jnp.int32)

    board = board.at[2, 1].set(1)
    board = board.at[3, 1].set(3)
    board = board.at[1, 1].set(2)
    board = board.at[1, 2].set(2)

    center = (board_size // 2, board_size // 2)
    board = board.at[center[0], center[1]].set(-2)

    tile = jnp.array([3, 2, 0, 0, 0, 0])

    valid_placements = jnp.array([
        [0,0,1,0],
        [2,0,1,0],
        [3,0,2,0],
        [3,0,4,0],
        [4,1,4,0],
        [4,1,4,2],
        [3,2,4,2],
        [4,2,3,2],
        [3,2,3,3],
        [3,3,3,2],
        [2,3,3,3],
        [3,3,2,3],
        [2,3,2,4],
        [2,4,2,3],
        [2,3,1,3],
        [1,3,2,3],
        [1,4,1,3],
        [0,3,1,3],
        [0,3,0,2],
        [0,1,0,2],
        [0,2,0,1],
        [0,0,0,1]
        ])

    placements = generate_placements(board_size)
    mask = legal_placements_mask(board, placements, tile, center)
    putative_valid_placements = placements[mask]
    
    assert(len(putative_valid_placements) == len(valid_placements))
    
    in_ref = row_in_set(putative_valid_placements, valid_placements)
    assert jnp.all(in_ref), "Some generated placements are not in expected set"

    in_gen = row_in_set(valid_placements, putative_valid_placements)
    assert jnp.all(in_gen), "Some expected placements are missing from generated set"

    print("OK: exact match")

if __name__ == "__main__":
    test()