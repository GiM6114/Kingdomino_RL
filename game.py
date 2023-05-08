import pygame

pygame.init()
screen = pygame.display.set_mode((400,500))

red = (255,0,0)
pygame.draw.rect(screen, red, pygame.Rect(30, 30, 60, 60),  2)

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    pygame.display.flip()
        
pygame.quit()