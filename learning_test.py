import numpy as np
import gym

from kingdomino import Kingdomino
from agent import RandomPlayer, PlayerAC
from printer import Printer

BATCH_SIZE = 3
N_PLAYERS = 2

envs = gym.vector.AsyncVectorEnv([
    lambda: gym.make("KingDomino-v0", n_players=N_PLAYERS),
    lambda: gym.make("KingDomino-v0", n_players=N_PLAYERS),
    lambda: gym.make("KingDomino-v0", n_players=N_PLAYERS),
])

agent = PlayerAC(
    n_players = N_PLAYERS,
    gamma = 0.99,
    lr_a = 1e-4,
    lr_c = 5e-4,
    coordinate_std = 2
)

Printer.activated = True
done = False
state = envs.reset()
for i in range(500):
    action = agent.action(state)
    states,rewards,dones,_ = envs.step(action)
    agent.give_reward(rewards)