import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st
import time
from PIL import Image

# Define the maze environment
class Maze:
    def __init__(self, maze):
        self.maze = maze
        self.n_rows, self.n_cols = maze.shape
        self.start = (0, 0)
        self.goal = (self.n_rows - 1, self.n_cols - 1)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state
        if action == 0:  # up
            new_row, new_col = max(0, row - 1), col
        elif action == 1:  # right
            new_row, new_col = row, min(self.n_cols - 1, col + 1)
        elif action == 2:  # down
            new_row, new_col = min(self.n_rows - 1, row + 1), col
        elif action == 3:  # left
            new_row, new_col = row, max(0, col - 1)

        if self.maze[new_row, new_col] == 1:  # If the new position is a barrier
            new_row, new_col = row, col  # Stay in the current position

        self.state = (new_row, new_col)
        reward = -1
        done = False

        if self.state == self.goal:
            reward = 100
            done = True

        return self.state, reward, done

    def get_possible_actions(self):
        return [0, 1, 2, 3]

# Q-learning algorithm with path recording
def q_learning(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros((env.n_rows, env.n_cols, 4))
    paths = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        path = [state]

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.get_possible_actions())
            else:
                action = np.argmax(q_table[state[0], state[1]])

            next_state, reward, done = env.step(action)
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1]])

            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state[0], state[1], action] = new_value

            state = next_state
            path.append(state)

        paths.append(path)

    return q_table, paths

# Function to create a random maze with a specified number of walls
def create_random_maze(n_rows, n_cols, n_walls):
    maze = np.zeros((n_rows, n_cols), dtype=int)
    walls = 0
    while walls < n_walls:
        row = random.randint(0, n_rows - 1)
        col = random.randint(0, n_cols - 1)
        if (row, col) != (0, 0) and (row, col) != (n_rows - 1, n_cols - 1) and maze[row, col] == 0:
            maze[row, col] = 1
            walls += 1
    return maze

# Function to simulate the agent's path
def get_agent_path(env, policy):
    path = [env.start]
    state = env.start
    while state != env.goal:
        action = policy[state]
        state, _, _ = env.step(action)
        path.append(state)
    return path

# Plot the maze, policy, and agent's path
def plot_maze_and_path(maze, policy_symbols, start, goal, path, current_pos=None):
    fig, ax = plt.subplots()
    ax.imshow(maze, cmap=plt.cm.binary)

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if (i, j) == start:
                ax.text(j, i, 'S', ha='center', va='center', color='green')
            elif (i, j) == goal:
                ax.text(j, i, 'G', ha='center', va='center', color='blue')
            elif maze[i, j] == 0:
                ax.text(j, i, policy_symbols[i, j], ha='center', va='center', color='red')
            elif maze[i, j] == 1:
                ax.text(j, i, 'X', ha='center', va='center', color='black')

    if current_pos:
        ax.plot(current_pos[1], current_pos[0], 'o', color='blue')

    return fig

# Streamlit app
st.title("Maze Training with Q-learning")

# Initial maze definition
n_rows, n_cols = 5, 5
maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])

env = Maze(maze)
q_table, paths = q_learning(env, num_episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1)

# Extract the best action for each state
policy = np.argmax(q_table, axis=2)

# Visualize the policy
action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
policy_symbols = np.vectorize(action_symbols.get)(policy)

st.write("### Maze Layout")
st.write(maze)

st.write("### Q-table")
st.write(q_table)

st.write("### Policy (symbols)")
st.write(policy_symbols)

st.write("### Maze with Policy and Agent's Path")

# Create a placeholder for the plot
plot_placeholder = st.empty()

# Animate the agent's learning process
for path in paths:
    for pos in path:
        fig = plot_maze_and_path(maze, policy_symbols, env.start, env.goal, path, current_pos=pos)
        plot_placeholder.pyplot(fig)
        time.sleep(0.1)  # Adjust the sleep time to control the speed of the animation

# Calculate the number of successful runs
successful_runs = 0
for _ in range(10):  # Simulate 10 runs
    final_path = get_agent_path(env, policy)
    if final_path[-1] == env.goal:
        successful_runs += 1

st.write(f"Number of successful runs: {successful_runs} out of 10")

# User input for the number of walls
n_walls = st.number_input("Enter the number of walls for the new maze:", min_value=0, max_value=n_rows * n_cols - 2, value=5)

# Button to generate a new maze
if st.button("Generate New Maze"):
    new_maze = create_random_maze(n_rows, n_cols, n_walls)
    env = Maze(new_maze)
    q_table, paths = q_learning(env, num_episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1)
    policy = np.argmax(q_table, axis=2)
    policy_symbols = np.vectorize(action_symbols.get)(policy)

    st.write("### New Maze Layout")
    st.write(new_maze)

    st.write("### New Q-table")
    st.write(q_table)

    st.write("### New Policy (symbols)")
    st.write(policy_symbols)

    st.write("### New Maze with Policy and Agent's Path")

    # Create a placeholder for the new plot
    new_plot_placeholder = st.empty()

    # Animate the agent's learning process in the new maze
    for path in paths:
        for pos in path:
            fig = plot_maze_and_path(new_maze, policy_symbols, env.start, env.goal, path, current_pos=pos)
            new_plot_placeholder.pyplot(fig)
            time.sleep(0.1)  # Adjust the sleep time to control the speed of the animation

    # Calculate the number of successful runs in the new maze
    successful_runs = 0
    for _ in range(10):  # Simulate 10 runs
        final_path = get_agent_path(env, policy)
        if final_path[-1] == env.goal:
            successful_runs += 1

    st.write(f"Number of successful runs in the new maze: {successful_runs} out of 10")
