import pygame,sys
import pickle
from colors import Colors
from tetris_env_manual import TetrisEnvMan

pygame.init()

title_font = pygame.font.Font(None, 40)
score_surface = title_font.render("Score", True, Colors.white)
next_surface = title_font.render("Next", True, Colors.white)
game_over_surface = title_font.render("GAME OVER", True, Colors.white)

score_rect = pygame.Rect(320, 55, 170, 60)
next_rect = pygame.Rect(320, 215, 170, 180)

screen = pygame.display.set_mode((500, 620))
pygame.display.set_caption("Python Tetris")

clock = pygame.time.Clock()

env = TetrisEnvMan()
state = env.reset()

GAME_UPDATE = pygame.USEREVENT
pygame.time.set_timer(GAME_UPDATE, 1000)

last_time = pygame.time.get_ticks()

# record manual play for pre-training
recorded_data = []
action = 0
done = False

while not done:

	current_time = pygame.time.get_ticks()
	delta_time = current_time - last_time
	last_time = current_time

	if env.game.game_over == False:
		env.game.update(delta_time)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()
		if event.type == pygame.KEYDOWN:
			if env.game.game_over == True:
				pass
				#game.game_over = False
				#game.reset()
			if event.key == pygame.K_LEFT and env.game.game_over == False:
				action = 0 # Move left 
			if event.key == pygame.K_RIGHT and env.game.game_over == False:
				action = 1 # Move right
			if event.key == pygame.K_DOWN and env.game.game_over == False:
				action = 2 # Soft drop / down
			if event.key == pygame.K_UP and env.game.game_over == False:
				action = 3 # hard drop
			if event.key == pygame.K_PAGEUP and env.game.game_over == False:
				action = 5 # rotate clockwise
			if event.key == pygame.K_PAGEDOWN and env.game.game_over == False:
				action = 4 # rotate counterclockwise

			next_state, reward, done, info = env.step(action)
			recorded_data.append((state, action))
			state = next_state

		if event.type == GAME_UPDATE and env.game.game_over == False:
			env.game.move_down()

		

	#Drawing
	score_value_surface = title_font.render(str(env.game.score), True, Colors.white)

	screen.fill(Colors.dark_blue)
	screen.blit(score_surface, (365, 20, 50, 50))
	screen.blit(next_surface, (375, 180, 50, 50))

	if env.game.game_over == True:
		screen.blit(game_over_surface, (320, 450, 50, 50))

	pygame.draw.rect(screen, Colors.light_blue, score_rect, 0, 10)
	screen.blit(score_value_surface, score_value_surface.get_rect(centerx = score_rect.centerx, 
		centery = score_rect.centery))
	pygame.draw.rect(screen, Colors.light_blue, next_rect, 0, 10)
	env.game.draw(screen)

	pygame.display.update()
	clock.tick(60)

env.close()

with open("tetris_demonstrations.pkl", "wb") as f:
	pickle.dump(recorded_data, f)