import jax
import jax.numpy as jnp
from functools import partial

from kingdomino.JAX.action import get_legal_actions, random_action, get_legal_action_mask, decode_action
from kingdomino.JAX.env import transition

@partial(jax.jit, static_argnums=(2,3,6))
def mpc_action_jittable(key, state, n_rollouts, n_processes, placements, actions, maximization_fn):

    legal_mask = get_legal_action_mask(state, placements)
    action_ids = jnp.arange(legal_mask.shape[0])
    
    action_keys = jax.random.split(key, action_ids.shape[0])

    values = jax.vmap(
        lambda k, aid: evaluate_action(
            k,
            state,
            actions[aid],
            state.current_player_id,
            n_rollouts,
            placements,
            maximization_fn,
        )
    )(action_keys, action_ids)

    values = jnp.where(legal_mask, values, -jnp.inf)

    best_action_id = jnp.argmax(values)

    return decode_action(best_action_id, placements)

def evaluate_action(key, state, action, player_id, n_rollouts, placements, maximization_fn):
    keys = jax.random.split(key, n_rollouts)

    scores = jax.vmap(
        lambda k: rollout(
            k,
            state,
            action,
            player_id,
            placements,
        )
    )(keys)

    return maximization_fn(scores)
    # return wins.mean()

def rollout(key, state, action, player_id, placements):
    key_shuffle, key_rollout = jax.random.split(key)
    
    # shuffle deck so that agent has no information on hidden remaining tiles
    state = shuffle_unknown_deck(key_shuffle, state)
    
    state = transition(state, action)

    def cond_fn(carry):
        _, state = carry
        return ~state.done

    def body_fn(carry):
        key, state = carry
        key, subkey = jax.random.split(key)
        action = random_action(subkey, state, placements)
        state = transition(state, action)
        return key, state

    _, state = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (key, state),
    )
    return state.scores

    # return (jnp.argmax(state.scores) == player_id).astype(jnp.float32)

def shuffle_unknown_deck(key, state):

    deck = state.deck
    idx = state.deck_idx
    N = deck.shape[0]

    # random values
    rnd = jax.random.uniform(key, (N,))

    # force prefix to stay ordered
    keys = jnp.where(
        jnp.arange(N) < idx,
        -jnp.arange(N, 0, -1),  # deterministic ordered prefix
        rnd,
    )

    perm = jnp.argsort(keys)
    new_deck = deck[perm] # jnp.take(deck, perm)
    return state.replace(deck=new_deck)

# TODO: check where cards should be shuffled, currently could include previous_tiles or current_tiles and
# could cause errors because their player belonging is not -1 for instance......
# def shuffle_unknown_deck(key, state):
#     deck = state.deck
#     idx = state.deck_idx

#     N = deck.shape[0]

#     perm = jnp.arange(N)

#     perm = perm.at[idx:].set(
#         jax.random.permutation(key, N - idx) + idx
#     )

#     return state.replace(deck=deck[perm])
                                   
    # new_deck = jnp.where(
    #     known_mask[:, None],
    #     deck,
    #     shuffled_deck,
    # )
    # print(new_deck)

    # return state.replace(deck=new_deck)

# def rollout(key, state, action, player_id, placements):
#     state, _, _, _ = step(state, action)

#     while not bool(state.done):
#         key, subkey = jax.random.split(key)
#         action = random_action(subkey, state, placements)
#         state, _, _, _ = step(state, action)

#     winner = jnp.argmax(state.scores)

#     return (winner == player_id).astype(jnp.float32)

# def mpc_action(key, state, n_rollouts, n_processes, placements, maximization_fn):
#     player_id = state.current_player_id
#     legal_actions = get_legal_actions(state, placements)

#     action_values = jax.vmap(
#         lambda a: evaluate_action(
#             key,
#             state,
#             a,
#             player_id,
#             n_rollouts,
#             placements,
#             maximization_fn
#         )
#     )(legal_actions)

#     best_idx = jnp.argmax(action_values)

#     return legal_actions[best_idx]

