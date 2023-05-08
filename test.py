from kingdomino import Kingdomino, TilePosition
from agent import HumanPlayer, RandomPlayer


agents = [RandomPlayer(0), RandomPlayer(1), RandomPlayer(2), RandomPlayer(3)]
kingdomino = Kingdomino(agents, log=True)

done = False
first_turn = True
while not done: 
    # Kingdomino logic
    last_turn = kingdomino.last_turn
    for player_id in kingdomino.order:
        player = agents[player_id]
        tile_id = None
        position = None
        if not kingdomino.last_turn:
            tile_id = player.chooseTile(kingdomino)
        if not first_turn:
            x1,y1,x2,y2 = player.placeTile(kingdomino)
            position = TilePosition(x1,y1, x2,y2)
        kingdomino.play(player, tile_id, position)
    if kingdomino.last_turn and last_turn:
        scores = kingdomino.scores
        break
    first_turn = False