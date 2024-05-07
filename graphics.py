from PIL import Image, ImageDraw
import torch

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

# only for single player as of yet
# e : {'Boards':(n_players*grid_size*grid_size*grid_size), 'Current tiles':(n_players,18)
def encoded_to_original(e):
    e = {'Boards':e[0],'Current tiles':e[1], 'Previous tiles':e[2]}
    grid_size = e['Boards'].shape[-1]
    n_players_board = e['Boards'].shape[0]
    board = torch.zeros([n_players_board,2,grid_size,grid_size])
    board[:,1] = e['Boards'][:,-1] # crowns
    for p in range(n_players_board):
        for c in range(8):
            for i in range(grid_size):
                for j in range(grid_size):
                    if i == (grid_size//2) and j == (grid_size//2):
                        board[p,0,i,j] = -2
                    if e['Boards'][p,c-1,i,j] == 1:
                        board[p,0,i,j] = c-2
    print(board)
    
    n_players = e['Current tiles'].shape[0] // 18
    cur_tile = torch.zeros([n_players,6])
    reshaped_current_tiles = e['Current tiles'].reshape(n_players,18)
    cur_tile[:,0] = torch.argmax(reshaped_current_tiles[:,:7],dim=1)
    cur_tile[:,1] = torch.argmax(reshaped_current_tiles[:,7:14],dim=1)
    cur_tile[:,2:] = reshaped_current_tiles[:,14:]
    
    n_players_prev = e['Previous tiles'].shape[0] // 17
    prev_tile = torch.zeros([n_players_prev,5])
    reshaped_previous_tiles = e['Previous tiles'].reshape(n_players_prev,17)
    prev_tile[:,0] = torch.argmax(reshaped_previous_tiles[:,:7],dim=1)
    prev_tile[:,1] = torch.argmax(reshaped_previous_tiles[:,7:14],dim=1)
    prev_tile[:,2:] = reshaped_previous_tiles[:,14:]
            
    return {'Boards':board, 'Current tiles':cur_tile, 'Previous tiles':prev_tile}

def draw_encoded_state(obs):
    if obs is None:
        print('None')
        return
    obs = encoded_to_original(obs)
    return draw_obs(obs)

# TODO: has to know the type of encoding to draw properly
def draw_obs(obs):
    grid_size = obs['Boards'].shape[-1]
    n_players = obs['Current tiles'].shape[0]
    img = Image.new("RGB", (img_size,img_size), "white")
    board_size = (img_size-(separation_size*(n_players+1))) // n_players

    for i,board in enumerate(obs['Boards']):
        draw_board(board[0], board[1], img, board_size, top_left=(
                        10 + i*(board_size+separation_size),10),
            grid_size=grid_size)
    for i,tile in enumerate(obs['Previous tiles']):
        draw_tile(tile, img,
                        top_left=(
                            10 + i*(board_size+separation_size),
                            20+board_size))
    for i,tile in enumerate(obs['Current tiles']):
        draw_tile(tile, img,
                        top_left=(
                            10 + i*75,
                            100+board_size))
        if tile[-1] != -1:
            draw_belong(img=img, top_left=(10+i*75,120+board_size))
            
    
    return img

def draw_belong(img, top_left):
    draw = ImageDraw.Draw(img)
    draw.rectangle((top_left[0],top_left[1],top_left[0]+10,top_left[1]+10), fill='black')

def draw_game(kingdomino):
    n_players = kingdomino.n_players
    img = Image.new("RGB", (img_size,img_size), "white")
    board_size = (img_size-(separation_size*(n_players+1))) // n_players
    
    for i,board in enumerate(kingdomino.boards):
        draw_board(board.board, board.crown, img, board_size, top_left=(
                        10 + i*(board_size+separation_size),10))
    for i,tile in zip(kingdomino.order, kingdomino.previous_tiles):
        draw_tile(tile, img,
                        top_left=(
                            10 + i*(board_size+separation_size),
                            20+board_size))
    return img


def draw_board(board, crown, img, board_size, top_left, grid_size):
    tile_size = board_size // grid_size
    draw = ImageDraw.Draw(img)
    pointer = top_left
    for y in range(grid_size):
        for x in range(grid_size):
            draw.rectangle((pointer[0], pointer[1],
                            pointer[0]+tile_size,pointer[1]+tile_size),
                           fill=type2color[int(board[x,y])+2])
            crown_pointer = pointer
            for c in range(int(crown[x,y])):
                draw.ellipse((
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
        type2color[int(tile[0])+2])
    pointer = top_left
    for i in range(int(tile[2])):
        draw.ellipse((
            pointer[0], pointer[1],
            pointer[0]+5, pointer[1]+5),
            'orange')
        pointer = (pointer[0]+5,pointer[1])

    draw.rectangle((
        top_left[0]+tile_size,top_left[1],
        top_left[0]+2*tile_size, top_left[1]+tile_size),
        type2color[int(tile[1])+2])
    pointer = (top_left[0]+tile_size, top_left[1])
    for i in range(int(tile[3])):
        draw.ellipse((
            pointer[0], pointer[1],
            pointer[0]+5, pointer[1]+5),
            'orange')
        pointer = (pointer[0]+5,pointer[1])
    
    
    
    
    
    
    