# TetrisAI

A Deep Q-Network (DQN) implementation that learns to play Tetris using reinforcement learning with advanced board state analysis and sophisticated reward engineering.

## Project Overview

This project implements a DQN agent that learns to play Tetris through reinforcement learning, using a multi-channel observation space and carefully crafted reward functions. The AI analyzes board states through multiple features including holes, bumpiness, height factors, and Tetris setup detection to make strategic decisions about piece placement.

The implementation uses a 3-channel observation system that captures the current board state, active piece position, and next piece information, allowing the AI to plan ahead and make more informed decisions.

### Key Features

- **3-channel state representation** (board grid, current piece mask, next piece encoding)
- **Deep neural network architecture** (512→256→6 with ReLU activations)
- **Strategic reward system** with line clearing bonuses and Tetris setup detection
- **Advanced board analysis** (holes, height scare factor, asymmetric bumpiness, Tetris readiness)
- **Double DQN implementation** with experience replay and target networks
- **Resume training capability** with full checkpoint preservation

## Setup

### Dependencies 

- Miniconda or Anaconda installed [See instructions](https://www.anaconda.com/docs/getting-started/miniconda/main)
- Python 3.8 or higher
- Torch, Gymnasium Pygame, Matplotlib, Numpy, Pybullet

### Installation

1. **Install dependencies**

```bash
pip install torch gymnasium pygame matplotlib numpy pybullet
```

2. **Clone the repository:**
```bash
git clone https://github.com/miloboyd/tetrisAI.git
cd cd tetrisAI/
```

3. **Create new conda environment**

```bash
conda create -n ai4robotics python=3.8
conda activate ai4robotics
```

## Running the Code

### Without pre-training:

1. **Play manually**
```bash
python main.py
```

2. **Train the AI**
```bash
python train_dqn.py
```

3. **Run the trained model**

```bash
python run_dqn.py
```

### Manual pre-training:

1. **Play manually**
```bash
python main.py
```
When you die, and the program is killed, 'tetris_demonstrations.pkl' will be populated featuring state space and input data. 

2. **Save data to .pth file format**
```bash
python pretrainer.py
```

3. **Modify train_dqn.py**
Replace the hyperparameters within the file with the following:
- ```RESUME_TRAINING  = True```
- ```CHECKPOINT_PATH  = 'dqn_checkpoint.pth'```

4. **Train the AI with manual pre-training**
```bash
python train_dqn.py
```

5. **Run the trained model**

```bash
python run_dqn.py
```

