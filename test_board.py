from board import Board

test_board = Board()

grass_tiles  = [[4,5],[4,6],[5,5],[5,4],[3,5],[3,6],[6,2]]
forest_tiles = [[4,3],[5,3],[6,3],[6,4],[6,5],[6,6],[5,6]]
water_tiles  = [[2,6],[2,5],[2,4],[3,4],[3,3],[3,2]]
mine_tiles   = [[2,3],[2,2]]

crown_1 = [(4,5),(4,6),(2,2)]
crown_2 = [(5,4)]

for tiles in [(grass_tiles,0), (water_tiles,3), (mine_tiles,5), (forest_tiles,1)]:
    v = tiles[1]
    for tile in tiles[0]:
        test_board.setBoard(tuple(tile), v)

for i,crown in enumerate([crown_1, crown_2]):
    for pos in crown:
        test_board.setBoardCrown(pos, i+1)

print('Should be 36 : ', test_board.count())

test_board = Board()

grass_tiles  = [[4,5],[4,6],[5,5],[5,4],[3,5],[3,6],[6,2]]
forest_tiles = [[4,3],[5,3],[6,3],[6,4],[6,5],[6,6],[5,6]]
water_tiles  = [[3,4],[3,3],[3,2]]
mine_tiles   = []

crown_1 = [(4,5),(4,6),(2,2)]
crown_2 = [(5,4)]

for tiles in [(grass_tiles,0), (water_tiles,3), (mine_tiles,5), (forest_tiles,1)]:
    v = tiles[1]
    for tile in tiles[0]:
        test_board.setBoard(tuple(tile), v)

for i,crown in enumerate([crown_1, crown_2]):
    for pos in crown:
        test_board.setBoardCrown(pos, i+1)

print('Should be 24 : ', test_board.count())
