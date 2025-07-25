# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path=None,
                     max_steps=500
                     ):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(no_episodes):
        state, _ = env.reset()

        state = tuple(state)
        total_reward = 0

        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        for step in range(max_steps):
            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            #env.render()

            next_state = tuple(next_state)
            total_reward += reward

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma *
                 np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
            #! -------
            if done:
                break

        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")
    # Generate unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_filename = f"q_table_{timestamp}.npy"

    np.save(save_filename, q_table)
    print(f"Saved the Q-table to {save_filename}")

    #! Step 8: Save the trained Q-table
    #! -------
    # np.save(q_table_save_path, q_table)
    # print("Saved the Q-table.")
    # print("Final Q-table:")
    # for state, actions in q_table.items():
    #     print(f"State {state}: {actions}")
    # print(q_table)

# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(hurdle_coordinates=[(2, 1), (0, 4)],
                      goal_coordinates=(4, 4), 
                      grid_size=10,
                      actions=["Right", "Left", "Down", "Up"],
                      q_values_path="q_table.npy"):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
         # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

           # Create mask for visualization
            mask = np.zeros_like(heatmap_data, dtype=bool)

            # Mask the single goal cell
            for gx, gy in goal_coordinates:
                mask[gx, gy] = True

            # Mask hurdles
            for hx, hy in hurdle_coordinates:
                mask[hx, hy] = True

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                ax=ax,
                cbar=False,
                mask=mask,
                annot_kws={"size": 9})
            
            ax.invert_yaxis()

            for gx, gy in goal_coordinates:
                ax.text(gy + 0.5, gx + 0.5, 'G', color='green',
                        ha='center', va='center', weight='bold', fontsize=14)
            for hx, hy in hurdle_coordinates:
                ax.text(hy + 0.5, hx + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
