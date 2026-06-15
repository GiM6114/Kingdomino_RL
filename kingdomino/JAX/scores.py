import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def compute_score(board, crowns, done=True):
    H, W = board.shape
    N = H * W

    flat_board = board.reshape(-1)
    flat_crowns = crowns.reshape(-1)

    valid_tiles = flat_board >= 0

    # Each valid tile starts as its own component
    labels = jnp.where(
        valid_tiles,
        jnp.arange(N, dtype=jnp.int32),
        -1,
    )

    # Build static adjacency edges
    grid = jnp.arange(N).reshape(H, W)

    up_a = grid[:-1, :].reshape(-1)
    up_b = grid[1:, :].reshape(-1)

    left_a = grid[:, :-1].reshape(-1)
    left_b = grid[:, 1:].reshape(-1)

    edge_a = jnp.concatenate([up_a, left_a])
    edge_b = jnp.concatenate([up_b, left_b])

    valid_edges = (
        valid_tiles[edge_a]
        & valid_tiles[edge_b]
        & (flat_board[edge_a] == flat_board[edge_b])
    )

    def cond_fn(state):
        labels, changed = state
        return changed

    def body_fn(state):
        labels, _ = state

        la = labels[edge_a]
        lb = labels[edge_b]

        m = jnp.minimum(la, lb)

        # Only propagate on valid edges
        prop_a = jnp.where(valid_edges, m, la)
        prop_b = jnp.where(valid_edges, m, lb)

        new_labels = labels
        new_labels = new_labels.at[edge_a].min(prop_a)
        new_labels = new_labels.at[edge_b].min(prop_b)

        changed = jnp.any(new_labels != labels)

        return new_labels, changed

    labels, _ = lax.while_loop(
        cond_fn,
        body_fn,
        (labels, jnp.array(True)),
    )

    # Path compression
    def compress(_, labels):
        return jnp.where(
            labels >= 0,
            labels[labels],
            -1,
        )

    labels = lax.fori_loop(0, N, compress, labels)

    # Compute score
    component_ids = jnp.arange(N)

    def component_score(cid):
        mask = labels == cid

        size = jnp.sum(mask)
        crown_sum = jnp.sum(
            jnp.where(mask, flat_crowns, 0)
        )

        return size * crown_sum

    total = jnp.sum(
        jax.vmap(component_score)(component_ids)
    )

    return jnp.where(done, total, 0).astype(jnp.int32)

if __name__ == '__main__':
    board = jnp.array([
        [-1,  0,  0, -1],
        [ 1, -2,  0,  2],
        [ 1,  1,  2,  2],
        [-1,  1, -1, -1],
    ])
    
    crowns = jnp.array([
        [0, 1, 0, 0],
        [1, 0, 2, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])
    
    print(compute_score(board, crowns))