import jax
import jax.numpy as jnp

from kingdomino.JAX.utils import ORIENTATIONS, PASS_POSITION

def pick_tile(state, player_id, tile_id):
    """
    Attributes a tile to a player and updates next turn order.
    Assumes:
        - tile_id is valid
        - tile is not already taken
        - current_tiles are sorted by ascending tile value
    """
    current_tiles = state.current_tiles.at[tile_id, -1].set(player_id)
    new_order = state.new_order.at[tile_id].set(player_id)
    return state.replace(
        current_tiles=current_tiles,
        new_order=new_order,
    )

def place_tile(state, player_id, position):
    is_pass = jnp.all(position == PASS_POSITION)
    def do_nothing(_):
        return state
    def do_place(_):
         tile = state.previous_tiles[player_id]
         boards = state.boards.at[player_id, position[0], position[1]].set(tile[0])
         boards = boards.at[player_id, position[2], position[3]].set(tile[1])
         crowns = state.crowns.at[player_id, position[0], position[1]].set(tile[2])
         crowns = crowns.at[player_id, position[2], position[3]].set(tile[3])
         return state.replace(
             boards=boards,
             crowns=crowns,
         )

    return jax.lax.cond(
        is_pass,
        do_nothing,
        do_place,
        operand=None,
    )

def decode_action(action_id, placements):

    tile_id = action_id // placements.shape[0]
    placement_id = action_id % placements.shape[0]

    position = placements[placement_id]

    return jnp.concatenate(
        [
            jnp.array([tile_id], dtype=jnp.int32),
            position,
        ],
        axis=0,
    )
def generate_actions(n_players, placements):
    return jnp.concatenate((
        jnp.repeat(jnp.arange(n_players), placements.shape[0])[:, None],
        jnp.tile(placements, reps=(n_players,1))
        ),
        axis = 1)

def legal_placements_mask(board, placements, tile):
    """
    board: (S, S)
    placements: (N_P, 4) -> x1,y1,x2,y2
    tile: (4,) -> t1,t2,c1,c2
    center: (cx,cy)
    """

    S = board.shape[0]
    center = S//2

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
    # 2. tile terrain types
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

        return ((xn == center) & (yn == center)).any(axis=1)

    center_adj1 = touches_center(x1, y1)
    center_adj2 = touches_center(x2, y2)

    # -------------------------------------------------------
    # 5. rule: at least one endpoint must connect
    # -------------------------------------------------------

    connected = (
        (adj1 | center_adj1) |
        (adj2 | center_adj2)
    )

    placements_mask = free & connected
    
    # If no placement is possible, enable the first placement
    has_any = jnp.any(placements_mask)
    placements_mask = jax.lax.select(
        has_any,
        placements_mask.at[0].set(False),
        jnp.zeros_like(placements_mask).at[0].set(True)
    )    
    return placements_mask

def legal_tile_choices_masks(state):
    return state.current_tiles[:,-1] == -1

# Get random actions (very common across actors)

def random_action(key, state, placements):
    key1, key2 = jax.random.split(key)
    tile_mask = legal_tile_choices_masks(state)
    tile_probs = tile_mask.astype(jnp.float32)
    tile_probs = tile_probs / jnp.sum(tile_probs)
    tile_id = jax.random.choice(key1, tile_mask.shape[0], p=tile_probs)
    placement_mask = legal_placements_mask(
        state.boards[state.current_player_id],
        placements,
        state.previous_tiles[state.current_player_id])
       
    placement_probs = placement_mask.astype(jnp.float32)
    placement_probs = placement_probs / jnp.sum(placement_probs)
    placement_id = jax.random.choice(key2, placement_mask.shape[0], p=placement_probs)
    return jnp.concatenate(
        [
            jnp.array([tile_id], dtype=jnp.int32),
            placements[placement_id],
        ],
        axis=0,
    )

def get_legal_actions(state, placements):
    """
        Generate all tuples of legal actions (cartesian product of current tiles to pick and previous tile tile placement)
        Pass is considered only legal when no other move is possible
        Not jittable
    """
    tile_ids = jnp.where(legal_tile_choices_masks(state))[0]
    placement_ids = jnp.where(legal_placements_mask(
        state.boards[state.current_player_id],
        placements,
        state.previous_tiles[state.current_player_id]))[0]
    placements = jnp.take(placements, placement_ids, axis=0)
    A = tile_ids.shape[0]
    B = placements.shape[0]

    t_exp = jnp.repeat(tile_ids, B)
    p_exp = jnp.tile(placements, (A, 1))

    all_actions = jnp.concatenate([t_exp[:, None], p_exp], axis=1)
    return all_actions

def get_legal_action_mask(state, placements):
    """
        Same objective as get_legal_actions except it's jittable
    """

    tile_mask = legal_tile_choices_masks(state)

    placement_mask = legal_placements_mask(
        state.boards[state.current_player_id],
        placements,
        state.previous_tiles[state.current_player_id],
    )
    legal = tile_mask[:, None] & placement_mask[None, :]
    return legal.reshape(-1)

# Test the functions

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