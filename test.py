from kingdomino import Kingdomino, TilePosition
from agent import HumanPlayer, RandomPlayer
import pygame

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

width = 500
height = 500

pygame.init()
screen = pygame.display.set_mode((400,500))
red = (255,0,0)
background_color = (255,255,255)

agents = [RandomPlayer(0), RandomPlayer(1), RandomPlayer(2), RandomPlayer(3)]
kingdomino = Kingdomino(len(agents), log=True)

kd_map = pygame.Rect(width//2, height//2, 100, 100)
current_tiles_rect = [pygame.Rect(i,25,50,50) for i in range(len(agents))]

done = False
first_turn = True
while not done:
    # Pygame logic
    for event in pygame.event.get():
        if event.type == QUIT:
            done = True
        if event.type in (QUIT, K_ESCAPE):
            done = True
    
    screen.fill(background_color)
    pygame.draw.rect(screen, red, (0,0,0,0))
    
    kingdomino.startTurn()
    
    # Kingdomino logic
    if kingdomino.last_turn:
        last_turn = True
    for player_id in kingdomino.order:
        player = agents[player_id]
        tile_id = None
        position = None
        if not kingdomino.last_turn:
            tile_id = player.chooseTile(kingdomino)
        if not first_turn:
            x1,y1,x2,y2 = player.placeTile(tile_id, kingdomino)
            position = TilePosition(x1,y1, x2,y2)
        kingdomino.play(player_id, tile_id, position)
    if kingdomino.last_turn and last_turn:
        scores = kingdomino.scores
        break
    first_turn = False
    
    pygame.display.flip()

pygame.quit()