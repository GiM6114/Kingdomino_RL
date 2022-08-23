from kingdomino import Kingdomino, TilePosition
from agent import HumanPlayer

agents = [HumanPlayer(0), HumanPlayer(1), HumanPlayer(2), HumanPlayer(3)]
kingdomino = Kingdomino(len(agents), log=True)

first_turn = True
while True:
    if kingdomino.last_turn:
        last_turn = True
    for player_id in kingdomino.order:
        print(kingdomino.printCurrentTiles())
        print(f"\nPlayer {player_id}'s turn.")
        print(kingdomino.boards[player_id])
        player = agents[player_id]
        tile_id = None
        position = None
        if not kingdomino.last_turn:
            tile_id = player.chooseTile(kingdomino)
        if not first_turn:
            x1,y1,x2,y2 = player.placeTile(kingdomino)
            position = TilePosition(x1,y1, x2,y2)
        kingdomino.play(player_id, tile_id, position)
    if kingdomino.last_turn and last_turn:
        scores = kingdomino.scores
        break
    first_turn = False