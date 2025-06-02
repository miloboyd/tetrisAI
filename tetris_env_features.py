# tetris_env_features.py
# Feature-based version of your existing tetris_env.py

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

class TetrisEnvFeatures(gym.Env):
    """
    Feature-based Tetris environment using your existing game logic
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Use your existing game instance
        self.game = Game()
        
        # Grid dimensions from your existing code
        self.grid_height, self.grid_width = len(self.game.grid.grid), len(self.game.grid.grid[0])

        # NEW: Exact 4-feature approach like online version
        # Features: [lines_cleared, holes, bumpiness, height_sum]
        self.observation_space = spaces.Box(
            low=0.0, high=200.0,  # Don't normalize - let network learn scale
            shape=(4,),  # Exact match to online version
            dtype=np.float32
        )
        
        # Keep your existing action space
        self.action_space = spaces.Discrete(6)

        # Keep existing tracking but add lines cleared tracking
        self.last_score = 0
        self._screen = None
        self.cleared = False
        self.stepCount = 0
        self.lastAction = 3
        self.lastGridScore = 0
        self.render = False
        self.last_lines_cleared = 0  # NEW: Track for features

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
            pygame.display.set_caption("Python Tetris")

    def reset(self):
        """Start a new episode - keep your existing reset logic"""
        self.game.reset()
        self.last_score = 0
        self._screen = None
        self.cleared = False
        self.stepCount = 0
        self.last_lines_cleared = 0  # Reset this too
        return self._get_board_props()

    def step(self, action):
        """Apply action - keep your existing step logic but return features"""
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

        # Track lines cleared for next observation
        self.last_lines_cleared = rows_cleared
        
        # Use exact online reward calculation
        reward = self._calculate_online_reward(rows_cleared)

        info = {'score': self.game.score}
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

        return self._get_board_props(), reward, done, info  # NEW: Use their function name

    def _get_board_props(self):
        """EXACT: Same as online version - 4 features, no normalization"""
        grid = self.get_grid_matrix()
        
        lines_cleared = self.last_lines_cleared  # From previous step
        holes = self._number_of_holes(grid)
        total_bumpiness = self._bumpiness(grid) 
        sum_height = self._height(grid)
        
        # Return raw values like online version (no normalization)
        return np.array([
            lines_cleared,
            holes,
            total_bumpiness,
            sum_height
        ], dtype=np.float32)

    def _calculate_online_reward(self, rows_cleared):
        """EXACT: Same simple reward as online version"""
        # Simple scoring: base + quadratic bonus for lines
        score = 1 + (rows_cleared ** 2) * self.game.grid.num_cols  # 10 for our grid
        
        if self.game.game_over:
            score -= 20  # Small death penalty only
            
        return score

    # EXACT: Online version's feature calculation methods
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

        # Get column heights
        for col in range(self.game.grid.num_cols):
            height = 0
            for row in range(self.game.grid.num_rows):
                if board[row][col] != 0:
                    height = self.game.grid.num_rows - row
                    break
            heights.append(height)
        
        # Sum absolute differences between adjacent columns
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

    # Keep all your existing helper methods but add the new ones needed for features
    def get_column_heights(self, grid):
        heights = [0] * self.game.grid.num_cols
        for col in range(self.game.grid.num_cols):
            for row in range(self.game.grid.num_rows):
                if grid[row][col] != 0:
                    heights[col] = self.game.grid.num_rows - row
                    break
        return heights

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

    def tetris_ready(self, grid, well_column=9):
        """Check if board is set up for a tetris (4-line clear)"""
        heights = self.get_column_heights(grid)
        well_height = heights[well_column]
        
        # Don't try Tetris if well is too high
        if well_height > self.game.grid.num_rows - 4:
            return False
        
        # Check if we have a valid tetris setup
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
                return True
        return False

    def get_grid_matrix(self):
        return [row[:] for row in self.game.grid.grid]

    def close(self):
        """Cleanup Pygame."""
        pygame.quit()

    def seed(self, seed=None):
        """Gym API: stubbed out (handled externally)."""
        return []