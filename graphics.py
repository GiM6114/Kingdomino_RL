from PIL import Image, ImageDraw

type2color = [
    'orange',
    'black',
    'lightgreen',
    'darkgreen',
    'yellow',
    'blue',
    'brown',
    'grey',
    'black']

img_size = 400
separation_size = 10

def draw_obs(obs):
    obs['Boards'] = obs['Boards'].squeeze()
    n_players = obs['Boards'].shape[0]
    img = Image.new("RGB", (img_size,img_size), "white")
    board_size = (img_size-(separation_size*(n_players+1))) // n_players

    for i,board in enumerate(obs['Boards'].boards):
        draw_board(board, img, board_size, top_left=(
                        10 + i*(board_size+separation_size),10))
    for i,tile in zip(kingdomino.order, kingdomino.previous_tiles):
        draw_tile(tile, img,
                        top_left=(
                            10 + i*(board_size+separation_size),
                            20+board_size))
    return img

# TODO: do other fnct where it is drawn from obs
def draw_game(kingdomino):
    n_players = kingdomino.n_players
    img = Image.new("RGB", (img_size,img_size), "white")
    board_size = (img_size-(separation_size*(n_players+1))) // n_players
    
    for i,board in enumerate(kingdomino.boards):
        draw_board(board, img, board_size, top_left=(
                        10 + i*(board_size+separation_size),10))
    for i,tile in zip(kingdomino.order, kingdomino.previous_tiles):
        draw_tile(tile, img,
                        top_left=(
                            10 + i*(board_size+separation_size),
                            20+board_size))
    return img


def draw_board(board, img, board_size, top_left):
    tile_size = board_size // 9
    draw = ImageDraw.Draw(img)
    pointer = top_left
    for y in range(9):
        for x in range(9):
            draw.rectangle((pointer[0], pointer[1],
                            pointer[0]+tile_size,pointer[1]+tile_size),
                           fill=type2color[board.getBoard(x,y)+2])
            crown_pointer = pointer
            for c in range(board.crown[x,y]):
                draw.circle((
                    crown_pointer[0], crown_pointer[1],
                    crown_pointer[0]+5, crown_pointer[1]+5,
                    ), "orange")
                crown_pointer = (crown_pointer[0]+5,crown_pointer[1])
            pointer = (pointer[0]+tile_size, pointer[1])
        pointer = (top_left[0], pointer[1]+tile_size)
        
def draw_tile(tile, img, top_left):
    draw = ImageDraw.Draw(img)
    tile_size = 20
    draw.rectangle((
        top_left[0],top_left[1],
        top_left[0]+tile_size, top_left[1]+tile_size),
        type2color[tile[0]+2])
    pointer = top_left
    for i in range(tile[2]):
        draw.circle((
            pointer[0], pointer[1],
            pointer[0]+5, pointer[1]+5),
            'orange')
        pointer = (pointer[0]+5,pointer[1])

    draw.rectangle((
        top_left[0]+tile_size,top_left[1],
        top_left[0]+2*tile_size, top_left[1]+tile_size),
        type2color[tile[1]+2])
    pointer = (top_left[0]+tile_size, top_left[1])
    for i in range(tile[3]):
        draw.circle((
            pointer[0], pointer[1],
            pointer[0]+5, pointer[1]+5),
            'orange')
        pointer = (pointer[0]+5,pointer[1])
    
    
    
    
    
    
    