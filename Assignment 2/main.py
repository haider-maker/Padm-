# Imports:
# --------
from padm_env import create_env
from Q_learning import train_q_learning, visualize_q_table
from constants import GOAL_COORDINATES, HURDLE_COORDINATES

goal_coordinates = GOAL_COORDINATES
hurdle_coordinates = HURDLE_COORDINATES

# User definitions:
# -----------------
train = False
test= True
visualize_results = True

"""
NOTE: Sometimes a fixed initializtion might push the agent to a local minimum.
In this case, it is better to use a random initialization.  
"""
random_initialization = True  # If True, the Q-table will be initialized randomly

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1_000  # Number of episodes

import numpy as np
import time
from padm_env import create_env
from constants import GOAL_COORDINATES, HURDLE_COORDINATES

def run_inference(
    q_table_path="q_table_20250704-004058.npy",
    random_initialization=True,
    render=True,
    delay=5.5
):
    """
    Run a single episode of the environment using a trained Q-table.

    Parameters
    ----------
    q_table_path : str
        Path to the saved Q-table .npy file.
    random_initialization : bool
        Whether to randomize the agent's start state (if supported).
    render : bool
        Whether to visually render the environment at each step.
    delay : float
        Time in seconds to pause between steps for visualization.
    """

    # Load the Q-table
    q_table = np.load(q_table_path)

    # Create the environment
    env = create_env(
        goal_coordinates=GOAL_COORDINATES,
        hurdle_coordinates=HURDLE_COORDINATES,
        random_initialization=random_initialization
    )

    # Reset the environment
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Select best action
        action = np.argmax(q_table[state[0], state[1], :])

        # Take action
        next_state, reward, done, info = env.step(action)

        if render:
            env.render()
            time.sleep(delay)

        print(
            f"Step: {steps} | State: {state} | Action: {action} "
            f"| Reward: {reward} | Total Reward: {total_reward}"
        )

        state = next_state
        total_reward += reward
        steps += 1

    env.close()

    print(f"\nFinished episode. Total reward: {total_reward}")

if test:
    run_inference(
    q_table_path="q_table.npy",
    random_initialization=True,
    render=True,
    delay=0.2)
# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env(goal_coordinates, hurdle_coordinates, random_initialization)

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
        # Visualize the Q-table:
        # ----------------------
        visualize_q_table(
            hurdle_coordinates=hurdle_coordinates,
            goal_coordinates=goal_coordinates, 
            q_values_path="q_table.npy")

