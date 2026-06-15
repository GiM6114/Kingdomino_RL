import os, json
N_PARALLEL = 7
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_PARALLEL}'
import jax
import jax.numpy as jnp
from functools import partial

from kingdomino.JAX.env import reset, step


def run_single_game(key, tile_data, placements, player_fns):
    state, _ = reset(key, tile_data, 2, 5)

    def body(carry):
        key, state = carry
        key, subkey = jax.random.split(key)

        action = jax.lax.switch(
            state.current_player_id,
            player_fns,
            *(subkey, state)
        )

        state, _, _, _ = step(state, action)
        return (key, state)

    def cond(carry):
        _, state = carry
        return ~state.done

    _, state = jax.lax.while_loop(cond, body, (key, state))

    return jnp.argmax(state.scores)

def run_batch(keys, tile_data, placements, player_fns):
    return jax.vmap(
        lambda k: run_single_game(k, tile_data, placements, player_fns)
    )(keys)


SAVE_PATH = "tournament.json"


def load():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            return json.load(f)
    return {}


def save(data):
    with open(SAVE_PATH, "w") as f:
        json.dump(data, f)


def pair_key(a, b):
    return f"{a}__vs__{b}"


def ensure(data, k):
    if k not in data:
        data[k] = []
    return data


def tournament(players, player_args, tile_data, placements,
               games_per_pair=1,
               symmetric=True,
               batch_size=N_PARALLEL):

    results = load()

    names = list(players.keys())

    base_key = jax.random.key(0)



    for i, p1 in enumerate(names):
        for j, p2 in enumerate(names):

            if (not symmetric) and j < i:
                continue

            k = pair_key(p1, p2)
            results = ensure(results, k)

            player_fns = {
                k: make_player_fn(players[k], args[k])
                for k in players
            }

            while len(results[k]) < games_per_pair:

                fns = list(player_fns.values())
                base_key, subkey = jax.random.split(base_key)
                keys = jax.random.split(subkey, batch_size)

                batch_winners = run_batch(
                    keys,
                    tile_data,
                    placements,
                    player_fns = fns
                )

                results[k].extend([int(x == 0) for x in batch_winners.tolist()])

                save(results)

                print(k, len(results[k]))

            # symmetric mirror matches
            if symmetric and p1 != p2:
                k2 = pair_key(p2, p1)
                results = ensure(results, k2)

                # invert results logically
                results[k2] = [1 - r for r in results[k][:games_per_pair]]

                save(results)

    return results

def make_player_fn(player_fn, kwargs):
    return lambda key, state: player_fn(key, state, **kwargs)


if __name__ == '__main__':

    from setup import GET_TILE_DATA
    from kingdomino.JAX.utils import generate_placements
    from kingdomino.JAX.action import generate_actions, random_action
    from agents.MPC_JAX import mpc_action_jittable

    tile_data = GET_TILE_DATA()
    placements = generate_placements(5)

    players = {
        "random": random_action,
        "mpc_5": mpc_action_jittable,
        "mpc_50": mpc_action_jittable,
        "mpc_100": mpc_action_jittable,
        "mpc_1000": mpc_action_jittable,
    }

    args = {
        "random": {
            "placements": placements
        },

        "mpc_5": {
            "n_rollouts": 5,
            "n_processes": N_PARALLEL,
            "placements": placements,
            "actions": generate_actions(2, placements),
            "maximization_fn": lambda s: (jnp.argmax(s, axis=1) == 0).mean(),
        },

        "mpc_50": {
            "n_rollouts": 50,
            "n_processes": N_PARALLEL,
            "placements": placements,
            "actions": generate_actions(2, placements),
            "maximization_fn": lambda s: (jnp.argmax(s, axis=1) == 0).mean(),
        },

        "mpc_100": {
            "n_rollouts": 100,
            "n_processes": N_PARALLEL,
            "placements": placements,
            "actions": generate_actions(2, placements),
            "maximization_fn": lambda s: (jnp.argmax(s, axis=1) == 0).mean(),
        },

        "mpc_1000": {
            "n_rollouts": 1000,
            "n_processes": N_PARALLEL,
            "placements": placements,
            "actions": generate_actions(2, placements),
            "maximization_fn": lambda s: (jnp.argmax(s, axis=1) == 0).mean(),
        },
    }


    results = tournament(
        players=players,
        player_args=args,
        tile_data=tile_data,
        placements=placements,
        games_per_pair=50,
        symmetric=True,
        batch_size=N_PARALLEL,
    )

    print(results)