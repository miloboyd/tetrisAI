# tetris_env_features_fixed.py
# FIXED: Reward system aligned with actual Tetris scoring

import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import numpy as np
import gym
import pygame
from gym import spaces
from colors import Colors

# Keep your existing pygame patches
class DummySound:
    def play(self): pass
    def stop(self): pass
pygame.mixer.Sound = lambda *args, **kwargs: DummySound()
pygame.mixer.music.load = lambda *args, **kwargs: None
pygame.mixer.music.play = lambda *args, **kwargs: None
pygame.mixer.music.stop = lambda *args, **kwargs: None

from game import Game

class TetrisEnvFeaturesFixed(gym.Env):
    """
    FIXED: Tetris environment with properly aligned reward system
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.game = Game()
        
        self.grid_height, self.grid_width = len(self.game.grid.grid), len(self.game.grid.grid[0])

        # Keep same 4-feature observation space
        self.observation_space = spaces.Box(
            low=0.0, high=200.0,
            shape=(4,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(6)

        # CRITICAL: Track score changes for proper reward calculation
        self.last_score = 0
        self.last_lines_cleared = 0
        self._screen = None
        self.cleared = False
        self.stepCount = 0
        self.lastAction = 3
        self.lastGridScore = 0
        self.render = False

    def pyRender(self, render):
        """Keep your existing render setup"""
        self.render = render
        if self.render:
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
            pygame.display.set_caption("Python Tetris - FIXED")

    def reset(self):
        """Start a new episode"""
        self.game.reset()
        self.last_score = 0  # CRITICAL: Reset score tracking
        self._screen = None
        self.cleared = False
        self.stepCount = 0
        self.last_lines_cleared = 0
        return self._get_board_props()

    def step(self, action):
        """Apply action with FIXED reward calculation"""
        rows_cleared = 0

        # Keep your existing action mapping
        if action == 0:
            self.game.move_left()
        elif action == 1:
            self.game.move_right()
        elif action == 2:
            self.game.move_down()
            self.game.update_score(0, 1)
        elif action == 3 and hasattr(self.game, 'hard_drop'):
            rows_cleared = self.game.hard_drop()
        elif action == 4:
            self.game.rotate_clockwise()
        elif action == 5:
            self.game.rotate_counterclockwise()

        # Keep your existing gravity logic
        if action == 3:
            pass
        else:
            if self.stepCount % 5 == 0:
                self.game.move_down()
            rows_cleared = self.game.update_bot()

        done = self.game.game_over

        # FIXED: Calculate reward based on SCORE IMPROVEMENT, not survival
        current_score = self.game.score
        score_improvement = current_score - self.last_score
        
        reward = self._calculate_fixed_reward(score_improvement, rows_cleared, done)
        
        # Update tracking
        self.last_score = current_score
        self.last_lines_cleared = rows_cleared
        
        info = {'score': self.game.score, 'score_improvement': score_improvement}
        self.stepCount = self.stepCount + 1

        # Keep your existing rendering logic
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

        return self._get_board_props(), reward, done, info

    def _calculate_fixed_reward(self, score_improvement, rows_cleared, done):
        """
        FIXED: Reward system that incentivizes score improvement, not just survival
        
        Primary goal: Maximize actual Tetris score
        Secondary goals: Clear lines efficiently, survive longer
        """
        # PRIMARY REWARD: Direct score improvement (this aligns reward with actual game score)
        reward = score_improvement * 0.1  # Scale down score to reasonable range
        
        # BONUS REWARDS: Additional incentives for good play
        if rows_cleared == 1:
            reward += 10      # Small bonus for clearing lines
        elif rows_cleared == 2:
            reward += 25      # Better bonus for double
        elif rows_cleared == 3:
            reward += 50      # Great bonus for triple  
        elif rows_cleared == 4:
            reward += 100     # Excellent bonus for Tetris!
        
        # SURVIVAL INCENTIVE: Very small positive reward for staying alive
        # But much smaller than score improvement potential
        if not done:
            reward += 0.1     # Tiny survival bonus (was 1.0 - the problem!)
        
        # DEATH PENALTY: Moderate penalty for dying
        if done:
            reward -= 50      # Death penalty
            
        return reward

    def _get_board_props(self):
        """Keep same feature extraction as before"""
        grid = self.get_grid_matrix()
        
        lines_cleared = self.last_lines_cleared
        holes = self._number_of_holes(grid)
        total_bumpiness = self._bumpiness(grid) 
        sum_height = self._height(grid)
        
        return np.array([
            lines_cleared,
            holes,
            total_bumpiness,
            sum_height
        ], dtype=np.float32)

    # Keep all your existing helper methods
    def _number_of_holes(self, board):
        """Count holes below blocks"""
        holes = 0
        for col in range(self.game.grid.num_cols):
            block_found = False
            for row in range(self.game.grid.num_rows):
                if board[row][col] != 0:
                    block_found = True
                elif block_found and board[row][col] == 0:
                    holes += 1
        return holes

    def _bumpiness(self, board):
        """Calculate surface bumpiness"""
        total_bumpiness = 0
        heights = []

        for col in range(self.game.grid.num_cols):
            height = 0
            for row in range(self.game.grid.num_rows):
                if board[row][col] != 0:
                    height = self.game.grid.num_rows - row
                    break
            heights.append(height)
        
        for i in range(len(heights) - 1):
            total_bumpiness += abs(heights[i] - heights[i+1])

        return total_bumpiness

    def _height(self, board):
        """Calculate sum of all column heights"""
        sum_height = 0
        for col in range(self.game.grid.num_cols):
            for row in range(self.game.grid.num_rows):
                if board[row][col] != 0:
                    height = self.game.grid.num_rows - row
                    sum_height += height
                    break
        return sum_height

    def get_grid_matrix(self):
        return [row[:] for row in self.game.grid.grid]

    def close(self):
        """Cleanup Pygame."""
        pygame.quit()

    def seed(self, seed=None):
        """Gym API: stubbed out (handled externally)."""
        return []