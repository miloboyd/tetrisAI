# tetris_env_man.py
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

class TetrisEnvMan(gym.Env):
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

        #visual setup stuff
        self.render = False


    def reset(self):
        """Start a new episode."""
        self.game.reset()
        self.last_score = self.game.score
        return self._get_observation_wide()

    def step(self, action):
        """Apply action, advance game, and return (obs, reward, done, info)."""
        # Action mapping
        rows_cleared = 0
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
            #self.game.score = temp # undo score gain of hard-drop for the sake of training
        elif action == 4:
            self.game.rotate_clockwise()
        elif action == 5:
            self.game.rotate_counterclockwise()

        done = self.game.game_over

        # Setup reward values
        grid_matrix = self.get_grid_matrix()
        holes = self.count_holes(grid_matrix)
        height_var = self.get_height_variance(grid_matrix)
        bumpiness = self.get_bumpiness(grid_matrix)

        # Reward: change in score
        current = self.game.score

        # Compute reward based on features
        reward = 0                               
        #reward = current - self.last_score       # Incentivise gaining score
        if rows_cleared == 1:                    # Incentivise row clears
            reward += 5
        elif rows_cleared == 2:
            reward += 25
        elif rows_cleared == 3:
            reward += 50
        elif rows_cleared == 4:
            reward += 100  # Tetris

        reward -= holes * 0.01                    # Penalise holes
        reward -= height_var * 0.01                # Penalise height
        reward -= bumpiness * 0.01                # Penalise uneven surfaces
        reward += self.stepCount / 10000

        reward = reward / 10
        self.last_score = current

        if done:
            reward -= 0.5
        
        info = {'score': current}
        self.stepCount = self.stepCount + 1

        return self._get_observation_wide(), reward, done, info





    def get_column_heights(self, grid):
        heights = [0] * self.game.grid.num_cols
        for col in range(self.game.grid.num_cols):
            for row in range(self.game.grid.num_rows):
                if grid[row][col] != 0:
                    heights[col] = self.game.grid.num_rows - row
                    break
        return heights

    def get_height_variance(self, grid):
        heights = self.get_column_heights(grid)
        mean_height = sum(heights) / len(heights)
        imbalance_penalty = sum((h - mean_height) ** 2 for h in heights) / len(heights)  # variance
        base_sum = sum(heights)
        return base_sum + imbalance_penalty  # penalize imbalance

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

    def get_bumpiness(self, grid):
        heights = self.get_column_heights(grid)
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
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
