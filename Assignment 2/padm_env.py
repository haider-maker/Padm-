import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from constants import GOAL_COORDINATES, HURDLE_COORDINATES
import random

class HarryPotterEnv(gym.Env):
    def __init__(self, grid_size=10, goals=None, hurdles=None, random_initialization=False):
        super().__init__()
        self.grid_size = grid_size
        self.random_initialization = random_initialization
        self.visited_goal = False
        self.agent_state = np.array([1, 1])
        self.goal = np.array(goals[0]) if goals else None
        self.hurdles = [np.array(h) for h in hurdles] if hurdles is not None else []
        self.goal_img = mpimg.imread("dairy.png")
        self.hurdle_img = mpimg.imread("deatheater.png")
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

    def reset(self, seed=None, options=None):
        if self.random_initialization:
            self.agent_state = np.array([
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            ])
        else:
            self.agent_state = np.array([1, 1])

        self.visited_goal = False
        return tuple(self.agent_state), {}

    def step(self, action):

        # If already finished, return done immediately
        if self.visited_goal:
            info = {
                "goals_remaining": 0,
                "current_target": None
            }
            return tuple(self.agent_state), 0, True, info

        if action == 0 and self.agent_state[1] < self.grid_size - 1:
            self.agent_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:
            self.agent_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:
            self.agent_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size - 1:
            self.agent_state[0] += 1

        # Check for hurdle
        for hurdle in self.hurdles:
            if np.array_equal(self.agent_state, hurdle):
                print("Hit a hurdle!")
                reward = -5
                info = {
                    "goals_remaining": 1 if not self.visited_goal else 0,
                    "current_target": self.goal
                }
                return tuple(self.agent_state), reward, False, info

        # Check for goal
        if np.array_equal(self.agent_state, self.goal):
            if self.visited_goal:
                print(f"Revisited goal {tuple(self.goal)}. Penalizing agent.")
                reward = -5
            else:
                print(f"Reached new goal at {tuple(self.goal)}!")
                reward = 10
                self.visited_goal = True

            done = self.visited_goal
            info = {
                "goals_remaining": 0 if done else 1,
                "current_target": None if done else self.goal
            }
            return tuple(self.agent_state), reward, done, info

        # Otherwise normal move
        reward = 0
        done = False
        info = {
            "goals_remaining": 1 if not self.visited_goal else 0,
            "current_target": None if self.visited_goal else self.goal
        }
        return tuple(self.agent_state), reward, done, info

    def render(self):
        self.ax.clear()
        self.ax.set_facecolor("#000000")

        # Draw grid
        self.ax.set_xticks(np.arange(0, self.grid_size + 1))
        self.ax.set_yticks(np.arange(0, self.grid_size + 1))
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

        # Draw agent
        self.ax.plot(
            self.agent_state[0] + 0.5,
            self.agent_state[1] + 0.5,
            "ro",
            markersize=12,
        )

        # Draw goal
        if self.goal is not None:
            x, y = self.goal
            self.ax.imshow(
                self.goal_img,
                extent=[x, x + 1, y, y + 1],
                zorder=1,
            )

        # Draw hurdles
        for h in self.hurdles:
            x, y = h
            self.ax.imshow(
                self.hurdle_img,
                extent=[x, x + 1, y, y + 1],
                zorder=1,
            )

        self.ax.set_aspect("equal")
        plt.pause(0.1)

    def close(self):
        plt.close()

def create_env(goal_coordinates,
               hurdle_coordinates,
               random_initialization):
    env = HarryPotterEnv(
        grid_size=10,
        goals=goal_coordinates,
        hurdles=hurdle_coordinates,
        random_initialization=random_initialization
    )
    return env
