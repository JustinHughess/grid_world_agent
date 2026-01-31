# Grid World Q-Learning Agent

A reinforcement learning agent that uses Q-Learning to navigate a grid world, finding the optimal path from start to goal while avoiding obstacles.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)

## Overview

This project demonstrates a tabular Q-Learning implementation where an agent learns to navigate a 15x15 grid environment. The agent starts with no knowledge of the environment and gradually learns the optimal policy through trial and error, using epsilon-greedy exploration.

## Features

- **Q-Learning Algorithm**: Implements the classic temporal difference learning method
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation with decaying epsilon
- **Live Visualization**: Watch the agent learn in real-time with matplotlib
- **Configurable Environment**: Easily modify grid size, obstacles, start/goal positions
- **Training Progress**: Displays success rate and exploration metrics during training

## Project Structure

```
grid_world_agent/
├── environment.py   # GridWorld environment class
├── agent.py         # Q-Learning agent implementation
├── main.py          # Training loop and visualization
├── requirements.txt # Python dependencies
└── README.md
```

### Module Details

| File | Description |
|------|-------------|
| `environment.py` | Defines the `GridWorld` class with state transitions, rewards, and episode termination logic |
| `agent.py` | Implements `QLearningAgent` with Q-table, action selection, and value updates |
| `main.py` | Orchestrates training, handles visualization, and demonstrates the learned policy |

## Requirements

- Python 3.7+
- matplotlib
- numpy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/justinhughess/grid_world_agent.git
   cd grid_world_agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the training with live visualization:

```bash
python3 main.py
```

The visualization will show:
- **Yellow**: Agent's current position
- **Cyan**: Goal position
- **Red**: Obstacles

Training runs for 1000 episodes, visualizing every 100th episode. After training completes, the agent demonstrates the learned optimal path.

## How It Works

### Q-Learning Update Rule

The agent updates Q-values using the Bellman equation:

```
Q(s, a) ← Q(s, a) + α[r + γ·max Q(s', a') - Q(s, a)]
```

Where:
- `α` (alpha) = 0.1 — Learning rate
- `γ` (gamma) = 0.95 — Discount factor
- `ε` (epsilon) = 1.0 → 0.01 — Exploration rate (decays by 0.995 per episode)

### Reward Structure

| Event | Reward |
|-------|--------|
| Reach goal | +100 |
| Hit obstacle | -10 |
| Each step | -1 |

### Actions

| Action | Direction |
|--------|-----------|
| 0 | Up |
| 1 | Right |
| 2 | Down |
| 3 | Left |

## Customization

### Modify the Grid Layout

In `main.py`, adjust the obstacles, start position, and goal:

```python
obstacles = [(4, 0), (4, 1), ...]  # List of (row, col) tuples
env = GridWorld(
    grid_size=15,
    start_pos=(0, 0),
    goal_pos=(14, 14),
    obstacles=obstacles
)
```

### Adjust Learning Parameters

```python
agent = QLearningAgent(
    n_actions=4,
    learning_rate=0.1,      # How fast the agent learns
    discount_factor=0.95,   # Importance of future rewards
    epsilon=1.0,            # Initial exploration rate
    epsilon_decay=0.995,    # Exploration decay per episode
    epsilon_min=0.01        # Minimum exploration rate
)
```

### Change Training Duration

```python
n_trainings = 1000      # Number of episodes
max_steps = 200         # Max steps per episode
visualize_every = 100   # Visualize every N episodes
```

## License

MIT License
