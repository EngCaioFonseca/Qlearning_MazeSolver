# Maze Training with Deep Spiking Q-Learning

![image](https://github.com/user-attachments/assets/1dbc08fb-d6a8-4a9c-a54b-c362c77aa25b)

![image](https://github.com/user-attachments/assets/a5d1735a-7b8b-4c8a-b870-bba80f49eaef)

![image_2024_09_25T13_31_24_246Z](https://github.com/user-attachments/assets/491a2823-80c2-4156-a0b1-8bd81870d206)

This project implements a Deep Spiking Q-Learning approach to solve a maze problem. The agent uses a spiking neural network to learn the optimal path from the start to the goal in a given maze layout.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to train an agent to solve a maze using a combination of Deep Q-Learning and Spiking Neural Networks (SNN). The agent learns to navigate the maze by receiving rewards for reaching the goal and penalties for each step taken.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- numpy
- matplotlib

You can install the required libraries using pip:

pip install numpy matplotlib


## Usage

1. Clone the repository:

git clone https://github.com/yourusername/mazetraining.git
cd mazetraining

2. Run the `mazetraining.py` script:


The script will train the agent to solve the maze and generate various plots to visualize the training process and results.

## Code Overview

### Classes

- `Maze`: Represents the maze environment. It includes methods to reset the maze, take a step, and check if the goal is reached.
- `LIFNeuron`: Implements the Leaky Integrate-and-Fire (LIF) neuron model.
- `DeepSpikingQNetwork`: Represents the Q-network using spiking neurons. It includes methods for forward propagation and action selection.
- `DeepSpikingQLearning`: Implements the Deep Spiking Q-Learning algorithm. It includes methods for training, experience replay, and updating the target network.

### Functions

- `plot_benchmarks`: Plots the episode lengths, total rewards, loss, and success rate over time.
- `plot_raster`: Creates raster plots of neural activity.
- `visualize_maze_solution`: Visualizes the path taken by the agent to solve the maze.

### Main Execution

The main part of the script defines the maze layout, creates and trains the agent, and generates various plots to visualize the training process and results.

## Results

After running the script, you will see the following plots:

1. **Episode Lengths**: Shows the number of steps taken in each episode.
2. **Total Rewards**: Shows the total reward received in each episode.
3. **Loss**: Shows the loss over training iterations.
4. **Success Rate**: Shows the success rate over episodes.
5. **Initial Network Spike Activity**: Raster plot of the initial network's spike activity.
6. **Final Network Spike Activity**: Raster plot of the final network's spike activity.
7. **Maze Solution**: Visualization of the path taken by the trained agent to solve the maze.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

