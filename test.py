import gym

from agent import HumanPlayer, RandomPlayer
from printer import Printer

agent = RandomPlayer()
kingdomino = gym.make('KingDomino-v0', n_players=2)

Printer.activated = True
done = False
state = kingdomino.reset()
while not done:
    for i in range(2):
        action = agent.action(state, kingdomino)
        state,reward,done,_ = kingdomino.step(action)
        if done:
            break
    scores = kingdomino.scores()