# tetris_env.py
import os
# Suppress audio by using dummy SDL driver
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import numpy as np
import gym
import pygame
from gym import spaces

# Initialize Pygame (dummy audio)
pygame.init()
try:
    pygame.mixer.pre_init()
    pygame.mixer.init()
except pygame.error:
    pass

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
        grid = self.game.grid.grid
        self.grid_height, self.grid_width = len(grid), len(grid[0])

        # Action and observation spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.grid_height, self.grid_width),
            dtype=np.int8
        )

        # For reward delta
        self.last_score = 0
        self._screen = None

    def reset(self):
        """Start a new episode."""
        self.game.reset()
        self.last_score = self.game.score
        return self._get_observation()

    def step(self, action):
        """Apply action, advance game, and return (obs, reward, done, info)."""
        # Action mapping
        if action == 0:
            self.game.move_left()
        elif action == 1:
            self.game.move_right()
        elif action == 2:
            self.game.move_down()
        elif action == 3 and hasattr(self.game, 'hard_drop'):
            self.game.hard_drop()
        elif action == 4:
            self.game.rotate_clockwise()
        elif action == 5:
            self.game.rotate_counterclockwise()

        # Gravity tick
        self.game.move_down()

        # Reward: change in score
        current = self.game.score
        reward = current - self.last_score
        self.last_score = current

        # Check termination
        done = self.game.game_over
        info = {'score': current}

        return self._get_observation(), reward, done, info

    def render(self, mode='human'):
        """Render the game via Pygame."""
        if self._screen is None:
            cell_size = 30
            w, h = self.grid_width * cell_size, self.grid_height * cell_size
            pygame.init()
            self._screen = pygame.display.set_mode((w, h))
        self._screen.fill((0, 0, 0))
        self.game.draw(self._screen)
        pygame.display.flip()

    def _get_observation(self):
        """Return the current grid as a {0,1} numpy array."""
        grid = self.game.grid.grid
        arr = np.array(grid, dtype=np.int8)
        return (arr > 0).astype(np.int8)

    def close(self):
        """Cleanup Pygame."""
        pygame.quit()

    def seed(self, seed=None):
        """Gym API: stubbed out (handled externally)."""
        return []
