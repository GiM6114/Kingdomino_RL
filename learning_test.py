import numpy as np

from kingdomino import Kingdomino
from agent import RandomPlayer, PlayerAC
from printer import Printer


kingdomino = Kingdomino(2)
agents = [RandomPlayer(0)]
agents.append(PlayerAC(
    _id = 1,
    env=kingdomino,
    gamma = 0.99,
    lr_a = 1e-4,
    lr_c = 5e-4,
    coordinate_std = 2
    ))
for agent in agents:
    agent.kingdomino = kingdomino
kingdomino.players = agents
# first_turn = True
# while not donegdomse = None
#         if not kingdomino.last_turn:
#             tile_id = play: 
#     # Kingdomino logic
#     last_turn = kiner.chooseTile(kingdomino)
#         if nino.last_turn
#     for player_id in kingdomino.order:
#         player = agents[player_id]
#         tile_id = None
#         position = kingdomino

# done = Falot first_turn:
#             x1,y1,x2,y2 = player.placeTile(kingdomino)
#             position = TilePosition(x1,y1, x2,y2)
#         kingdomino.play(player, tile_id, position)
#     if kingdomino.last_turn and last_turn:
#         scores = kingdomino.scores
#         break
#     first_turn = False

Printer.activated = True
done = False
state = kingdomino.reset()
while not done:
    for player_id in kingdomino.order:
        player = agents[player_id]
        action = player.action(state)
        state,done = kingdomino.step(player, action)
        if done:
            break
    scores = kingdomino.scores()
    for player in agents:
        player.give_scores(scores)