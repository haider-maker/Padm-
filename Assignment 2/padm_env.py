import gymnasium as gym #is openai lib for the environment 
import numpy as np #is the lib for maths 
import matplotlib.pyplot as plt #the lib for the grding and plots 
import matplotlib.image as mpimg
from constants import GOAL_COORDINATES, HURDLE_COORDINATES
import random 

class HarryPotterEnv(gym.Env):
    def __init__(self, grid_size=10, goals=None, hurdles=None):
        super().__init__() 
        self.grid_size = grid_size 
        self.agent_state = np.array([1,1])
        self.goals = [np.array(g) for g in goals] if goals is not None else []
        self.current_goal_idx = 0
        self.hurdles = [np.array(h) for h in hurdles] if hurdles is not None else []
        self.goal_imgs = [
            mpimg.imread("dairy.png"),
            mpimg.imread("nagini.png"),
            mpimg.imread("cup.jpeg"),
            mpimg.imread("diadem.png"),
            mpimg.imread("locket.png"),
            mpimg.imread("ring.jpg"),
        ]
        self.hurdle_img = mpimg.imread("deatheater.png") 
        self.action_space = gym.spaces.Discrete(4) 
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32) 
        self.fig, self.ax = plt.subplots() 
        plt.show(block=False) 

    def reset(self, seed=None, options=None):
        self.agent_state = np.array([1,1])
        #self.current_goal_idx = 0
        return tuple(self.agent_state), {}

    def step(self, action):

        # If already finished, return done immediately
        if self.current_goal_idx >= len(self.goals):
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

        for hurdle in self.hurdles:
            if np.array_equal(self.agent_state, hurdle):
                print("Hit a hurdle!")
                reward = -5
                info = {
                    "goals_remaining": len(self.goals) - self.current_goal_idx,
                    "current_target": self.goals[self.current_goal_idx]
                }
                return tuple(self.agent_state), reward, False, info

        reward = 0
        done = False
        if np.array_equal(self.agent_state, self.goals[self.current_goal_idx]):
             print(f"Reached goal {self.current_goal_idx + 1}!")
        reward = 10
        self.current_goal_idx += 1

        if self.current_goal_idx >= len(self.goals):
            done = True
            print("🎯 All goals reached!")
        else:
            # teleport randomly
            while True:
                rand_x = random.randint(1, 9)
                rand_y = random.randint(1, 9)
                rand_pos = np.array([rand_x, rand_y])
                if not np.array_equal(rand_pos, self.goals[self.current_goal_idx]):
                    break
            self.agent_state = rand_pos
            print(f"Teleported to {tuple(self.agent_state)} for next goal.")
            done = False

        info = {
            "goals_remaining": len(self.goals) - self.current_goal_idx,
            "current_target": None if self.current_goal_idx >= len(self.goals) else self.goals[self.current_goal_idx]
        }
        return tuple(self.agent_state), reward, done, info



    def render(self):
        self.ax.clear()
        self.ax.set_facecolor("#000000")  
        
        # Draw grid
        self.ax.set_xticks(np.arange(0, self.grid_size+1))
        self.ax.set_yticks(np.arange(0, self.grid_size+1))
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        #self.ax.invert_yaxis()
        
        # Draw agent
        self.ax.plot(
            self.agent_state[0] + 0.5,
            self.agent_state[1] + 0.5,
            "ro",
            markersize=12,
        )
        
        # Draw all goals
        for idx, goal in enumerate(self.goals):
            x, y = goal
            img = self.goal_imgs[idx]
            self.ax.imshow(
                img,
                extent=[x, x+1, y, y+1],
                zorder=1,
            )
        
        # Draw hurdles
        for h in self.hurdles:
            x, y = h
            self.ax.imshow(
                self.hurdle_img,
                extent=[x, x+1, y, y+1],
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
        hurdles=hurdle_coordinates
    )
    return env



# if __name__ == "__main__":
#     env = create_env((9, 9), [(2, 2), (3, 4), (5, 5), (1, 7), (7, 1)], random_initialization=False)
#     state, info = env.reset()
#     for _ in range(500):
#         action = env.action_space.sample()
#         state, reward, done, info = env.step(action)
#         env.render()
#         print(f"state:{state}, Reward:{reward}, Action:{action}, Done:{done}, Info:{info}")
#         if done:
#             print("Bruhhhhhhhhhhhhhhhh!!!!!!! I reached the destination ")
#             break
#     env.close()
