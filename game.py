from grid import Grid
from blocks import *
import random
import pygame

class Game:
	def __init__(self):
		self.grid = Grid()
		self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
		self.current_block = self.get_random_block()
		self.next_block = self.get_random_block()
		self.game_over = False
		self.score = 0
		self.lock_delay = 500 # Lock delay in milliseconds 
		self.lock_delay_timer = 0 # Timer to track current lock delay
		self.locking_phase = False # Flag to indicate if piece is in locking phase

	def update_score(self, lines_cleared, move_down_points):
		if lines_cleared == 1:
			self.score += 100
		elif lines_cleared == 2:
			self.score += 300
		elif lines_cleared == 3:
			self.score += 500
		elif lines_cleared == 4:
			self.score += 800

		self.score += move_down_points

	def get_random_block(self):
		if len(self.blocks) == 0:
			self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
		block = random.choice(self.blocks)
		self.blocks.remove(block)
		return block

	def move_left(self):
		self.current_block.move(0, -1)
		if self.block_inside() == False or self.block_fits() == False:
			self.current_block.move(0, 1)
		elif self.locking_phase:
			self.lock_delay_timer = self.lock_delay

	def move_right(self):
		self.current_block.move(0, 1)
		if self.block_inside() == False or self.block_fits() == False:
			self.current_block.move(0, -1)
		elif self.locking_phase: 
			self.lock_delay_timer = self.lock_delay

	def move_down(self):
		self.current_block.move(1, 0)
		if self.block_inside() == False or self.block_fits() == False:
			self.current_block.move(-1, 0)
			
			if not self.locking_phase:
				self.locking_phase = True
				self.lock_delay_timer = self.lock_delay
			else: 
				self.locking_phase = False
	
	def hard_drop(self):
		"""
		Drops the current block all the way down and locks it in place.
		"""
		# Keep moving down until collision
		drop_distance = 0
		while True:
			self.current_block.move(1, 0)
			if self.block_inside() == False or self.block_fits() == False:
				self.current_block.move(-1, 0)  # Move back up
				break
			drop_distance += 1
		
		# Add points for hard drop - typically 2 points per cell dropped
		self.update_score(0, drop_distance * 2)

		self.locking_phase = False
		
		# Lock the block in place
		tiles = self.current_block.get_cell_positions()
		for position in tiles:
			self.grid.grid[position.row][position.column] = self.current_block.id
		
		# Check and clear full rows
		rows_cleared = self.grid.clear_full_rows()
		if rows_cleared > 0:
			self.update_score(rows_cleared, 0)
		
		# Set up the next block
		self.current_block = self.next_block
		self.next_block = self.get_random_block()
		
		# Check if game is over
		if self.block_fits() == False:
			self.game_over = True

	def lock_block(self):
		tiles = self.current_block.get_cell_positions()
		for position in tiles:
			self.grid.grid[position.row][position.column] = self.current_block.id
		self.current_block = self.next_block
		self.next_block = self.get_random_block()
		rows_cleared = self.grid.clear_full_rows()
		if rows_cleared > 0:
			self.update_score(rows_cleared, 0)
		if self.block_fits() == False:
			self.game_over = True

	def reset(self):
		self.grid.reset()
		self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
		self.current_block = self.get_random_block()
		self.next_block = self.get_random_block()
		self.score = 0

	def block_fits(self):
		tiles = self.current_block.get_cell_positions()
		for tile in tiles:
			if self.grid.is_empty(tile.row, tile.column) == False:
				return False
		return True

	'''
	def rotate(self):
		self.current_block.rotate()
		if self.block_inside() == False or self.block_fits() == False:
			self.current_block.undo_rotation()
	'''

	def rotate_clockwise(self):
		self.current_block.rotate_clockwise()
		if self.block_inside() == False or self.block_fits() == False:
			self.current_block.rotate_counterclockwise()
		elif self.locking_phase:
			self.lock_delay_timer = self.lock_delay
	
	def rotate_counterclockwise(self):
		self.current_block.rotate_counterclockwise()
		if self.block_inside() == False or self.block_fits() == False:
			self.current_block.rotate_clockwise()
		elif self.locking_phase:
			self.lock_delay_timer = self.lock_delay

	def block_inside(self):
		tiles = self.current_block.get_cell_positions()
		for tile in tiles:
			if self.grid.is_inside(tile.row, tile.column) == False:
				return False
		return True

	def draw(self, screen):
		self.grid.draw(screen)
		self.current_block.draw(screen, 11, 11)

		if self.next_block.id == 3:
			self.next_block.draw(screen, 255, 290)
		elif self.next_block.id == 4:
			self.next_block.draw(screen, 255, 280)
		else:
			self.next_block.draw(screen, 270, 270)

	def update(self, delta_time):
		"""
		Update game state based on time elapsed.
		
		Args:
			delta_time: Time elapsed since last update in milliseconds
		"""
		# Only update lock delay timer if in locking phase
		if self.locking_phase:
			self.lock_delay_timer -= delta_time
			
			# If timer expired, lock the block
			if self.lock_delay_timer <= 0:
				self.lock_block()
				self.locking_phase = False