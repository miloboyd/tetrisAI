# TetrisAI

A Deep Q-Network (DQN) implementation that learns to play Tetris using reinforcement learning. This project includes both the original feature-based approach and a fixed version with properly aligned reward systems.

## Project Overview

This project implements a DQN agent that learns to play Tetris through trial and error. The AI uses a neural network to evaluate board states and make decisions about piece placement.

### Key Features

- **4-feature state representation** (lines cleared, holes, bumpiness, height sum)
- **Simple neural network architecture** (4→32→32→6)
- **Fixed reward system** that properly incentivizes score improvement
- **Comprehensive training monitoring** with plots and statistics
- **Resume training capability** for long training runs

## Setup

### Dependencies 

- Miniconda or Anaconda installed [See instructions](https://www.anaconda.com/docs/getting-started/miniconda/main)
- Python 3.8 or higher
- Torch, Gymnasium Pygame, Matplotlib, Numpy, Pybullet

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/miloboyd/tetrisAI.git
cd cd tetrisAI/


