# tetris_env.py
import os
# Suppress audio by using dummy SDL driver
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import numpy as np
import gym
import pygame
from gym import spaces
from colors import Colors

# Monkey-patch Pygame sound to avoid audio errors
class DummySound:
    def play(self): pass
    def stop(self): pass
pygame.mixer.Sound = lambda *args, **kwargs: DummySound()
pygame.mixer.music.load = lambda *args, **kwargs: None
pygame.mixer.music.play = lambda *args, **kwargs: None
pygame.mixer.music.stop = lambda *args, **kwargs: None

from game import Game

class TetrisEnv(gym.Env):
    """
    Custom Gym environment for Tetris with updated team controls:
      0: move left
      1: move right
      2: soft drop
      3: hard drop
      4: rotate clockwise
      5: rotate counterclockwise
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Game instance
        self.game = Game()
        # Grid dimensions
        self.grid_height, self.grid_width = len(self.game.grid.grid), len(self.game.grid.grid[0])

        # Action and observation spaces
        self.num_channels = 3
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.num_channels, self.grid_height, self.grid_width),
            dtype=np.int8
        )

        # For reward delta
        self.last_score = 0
        self._screen = None
        self.cleared = False
        self.stepCount = 0
        self.lastAction = 3
        self.lastGridScore = 0

        #visual setup stuff
        self.render = False


    def pyRender(self, render):
        self.render = render
        if self.render:
            # Initialize Pygame (dummy audio)
            pygame.init()
            try:
                pygame.mixer.pre_init()
                pygame.mixer.init()
            except pygame.error:
                pass

            self.title_font = pygame.font.Font(None, 40)
            self.score_surface = self.title_font.render("Score", True, Colors.white)
            self.next_surface = self.title_font.render("Next", True, Colors.white)
            self.game_over_surface = self.title_font.render("GAME OVER", True, Colors.white)

            self.score_rect = pygame.Rect(320, 55, 170, 60)
            self.next_rect = pygame.Rect(320, 215, 170, 180)

            self.screen = pygame.display.set_mode((500, 620))
            pygame.display.set_caption("Python Tetris")

    def reset(self):
        """Start a new episode."""
        self.game.reset()
        self.last_score = 0
        self._screen = None
        self.cleared = False
        self.stepCount = 0
        return self._get_observation_wide()

    def step(self, action):
        """Apply action, advance game, and return (obs, reward, done, info)."""

        rows_cleared = 0

        # Action mapping
        if action == 0:
            self.game.move_left()
        elif action == 1:
            self.game.move_right()
        elif action == 2:
            self.game.move_down()
            self.game.update_score(0, 1)
        elif action == 3 and hasattr(self.game, 'hard_drop'):
            #temp = self.game.score
            rows_cleared = self.game.hard_drop()
            #print(f"Hard drop: rows_cleared={rows_cleared}, reward={reward}")
            #self.game.score = temp # undo score gain of hard-drop for the sake of training
        elif action == 4:
            self.game.rotate_clockwise()
        elif action == 5:
            self.game.rotate_counterclockwise()

        
        # Gravity tick
        #self.game.move_down()

        if action == 3:
            #temp = self.game.update_bot()
            pass
        else:
            if self.stepCount % 5 == 0:
                self.game.move_down()
            rows_cleared = self.game.update_bot()


        # Check if Game Over
        done = self.game.game_over

        # Setup reward values
        grid_matrix = self.get_grid_matrix()
        holes = self.count_holes(grid_matrix)
        height_scare = self.get_height_scare_factor(grid_matrix) # height_var = self.get_height_variance(grid_matrix)
        bumpiness = self.get_bumpiness(grid_matrix)
        tetris_ready_bonus = self.tetris_ready(grid_matrix)

        current = self.game.score

        # Compute reward based on features
        reward = 0                               # Start with neutral reward
     
        if rows_cleared == 1:                    # Incentivise row clears
            reward += 400
        elif rows_cleared == 2:
            reward += 1000
        elif rows_cleared == 3:
            reward += 1500
        elif rows_cleared == 4:
            reward += 3000  # Tetris

        if tetris_ready_bonus:
            reward += 200

        reward -= holes * 5
        reward -= height_scare * 10
        reward -= bumpiness * 1

        reward += 5 # survival bonus 

        if action == 3 and self.lastAction == 3:
            reward -= 100                            # punish hard-drop spam

        if done: # death penalty
            reward -= 200                           #punish dying step

        reward = reward * 0.1                       #scale end reward      

        self.lastAction = action               

        info = {'score': current}
        self.stepCount = self.stepCount + 1

        #Drawing
        if self.render:
            score_value_surface = self.title_font.render(str(self.game.score), True, Colors.white)

            self.screen.fill(Colors.dark_blue)
            self.screen.blit(self.score_surface, (365, 20, 50, 50))
            self.screen.blit(self.next_surface, (375, 180, 50, 50))

            if self.game.game_over == True:
                self.screen.blit(self.game_over_surface, (320, 450, 50, 50))

            pygame.draw.rect(self.screen, Colors.light_blue, self.score_rect, 0, 10)
            self.screen.blit(score_value_surface, score_value_surface.get_rect(centerx = self.score_rect.centerx, 
                centery = self.score_rect.centery))
            pygame.draw.rect(self.screen, Colors.light_blue, self.next_rect, 0, 10)
            self.game.draw(self.screen)

            pygame.display.update()

        return self._get_observation_wide(), reward, done, info

    #reward functions

    def get_column_heights(self, grid):
        heights = [0] * self.game.grid.num_cols
        for col in range(self.game.grid.num_cols):
            for row in range(self.game.grid.num_rows):
                if grid[row][col] != 0:
                    heights[col] = self.game.grid.num_rows - row
                    break
        return heights
    
    def get_height_scare_factor(self, grid): #height scare instead - above threshold increase exponentially
        heights = self.get_column_heights(grid)
        well_column = 9 #disregard this for height calculations

        non_well_heights = [h for i, h in enumerate(heights) if i != well_column] # right side column - can remove from array
        mean_height = sum(non_well_heights) / len(non_well_heights)
        scare_height = 12
        excess_height = max(0, mean_height - scare_height)
        return excess_height

    # def get_height_variance(self, grid):
    #     heights = self.get_column_heights(grid)
    #     mean_height = sum(heights) / len(heights)
    #     imbalance_penalty = sum((h - mean_height) ** 2 for h in heights) / len(heights)  # variance
    #     base_sum = sum(heights)
    #     return base_sum + imbalance_penalty  # penalize imbalance
    
    def tetris_ready(self, grid, well_column=9):
        heights = self.get_column_heights(grid)
        well_height = heights[well_column]
        
        # Don't try Tetris if well is too high
        if well_height > self.game.grid.num_rows - 4:
            return False
        
        # Check multiple potential Tetris zones, not just bottom 4
        tetris_zones = 0
        for start_row in range(self.game.grid.num_rows - well_height - 3, -1, -1):
            zone_complete = True
            for row_offset in range(4):
                row_index = start_row + row_offset
                if row_index >= self.game.grid.num_rows:
                    zone_complete = False
                    break
                    
                for col in range(self.game.grid.num_cols):
                    if col == well_column:
                        if grid[row_index][col] != 0:  # Well should be empty
                            zone_complete = False
                            break
                    else:
                        if grid[row_index][col] == 0:  # Others should be filled
                            zone_complete = False
                            break
                if not zone_complete:
                    break
            
            if zone_complete:
                tetris_zones += 1
        
        return tetris_zones > 0  # Return True if any valid Tetris zone exists

    def count_holes(self, grid):
        holes = 0
        for col in range(self.game.grid.num_cols):
            block_found = False
            for row in range(self.game.grid.num_rows):
                if grid[row][col] != 0:
                    block_found = True
                elif block_found and grid[row][col] == 0:
                    holes += 1
        return holes

    # def get_bumpiness(self, grid):
    #     heights = self.get_column_heights(grid)
    #     bumpiness = 0
    #     for i in range(len(heights) - 1):
    #         bumpiness += abs(heights[i] - heights[i + 1])
    #     return bumpiness
    
    def get_bumpiness(self, grid): #introduce assymmetry to 
        heights = self.get_column_heights(grid)
        bumpiness = 0
        for i in range(len(heights) - 1):
            difference = abs(heights[i] - heights[i + 1])
            if (i < len(heights)) // 2:
                weight = 0.8
            else: 
                weight = 1.2
            bumpiness += difference * weight
        return bumpiness
    
    def get_grid_matrix(self):
        return [row[:] for row in self.game.grid.grid]

    def render(self, mode='human'):
        """Render the game via Pygame."""
        if self._screen is None:
            cell_size = 30
            w, h = self.game.grid_width * cell_size, self.game.grid_height * cell_size
            pygame.init()
            self._screen = pygame.display.set_mode((w, h))
        self._screen.fill((0, 0, 0))
        self.game.draw(self._screen)
        pygame.display.flip()

    def _get_observation(self):
        """Return the current grid as a {0,1} numpy array."""
        arr = np.array(self.game.grid.grid, dtype=np.int8)
        return (arr > 0).astype(np.int8)

    def _get_observation_wide(self):
        obs = np.zeros((self.num_channels, self.grid_height, self.grid_width), dtype=np.int8)

        # Channel 0: board grid
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                obs[0, i, j] = self.game.grid.grid[i][j]

        # Channel 1: current piece mask at its position
        for block in self.game.current_block.get_masked_blocks():  # You may need to implement this
            x, y = block
            if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                obs[1, y, x] = 1

        # Channel 2: encode next piece type
        next_piece_id = self.game.next_block.id  # assume index 0 to N-1
        obs[2, :, :] = next_piece_id / 7  # broadcast same value

        return obs

    def close(self):
        """Cleanup Pygame."""
        pygame.quit()

    def seed(self, seed=None):
        """Gym API: stubbed out (handled externally)."""
        return []
