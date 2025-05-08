import  gymnasium as gym #is openai lib for the environment 
import numpy as np #is the lib for maths 
import matplotlib.pyplot as plt #the lib for the grding and plots 
import matplotlib.image as mpimg

class HarryPotterEnv(gym.Env):
    def __init__(self, grid_size=10):
        super().__init__() #default func and gives power to the class 
        self.grid_size = grid_size #self is like this in our language it's pointing toward something 
        self.agent_state = np.array([1,1])#using npies array object 
        self.goals = [np.array([2, 5]), np.array([5, 4]),np.array([5, 8]),np.array([7, 3]),np.array([1, 8]),np.array([9, 9])]
        self.current_goal_idx = 0
        self.goal_state = np.array([9,9]) #default pos where the goal is placed
        self.hurdles = [np.array([2, 2]), np.array([3, 4]), np.array([5, 5]), np.array([1, 7]), np.array([7, 1])]  # 5 hurdles
        self.goal_img = mpimg.imread("nagini.png")  # Load your image once
        self.action_space = gym.spaces.Discrete(4) #consit of fnitely manly element in our case 25 elements. 4 possible actions up,dwm.....
        self.observationobservation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(2,)) #??? both line 17 and 18 are setting up the 5x5 grid starting lowest point as 0 and going up till 4 coz in python the the upper bound is one less then the written 
        self.fig, self.ax = plt.subplots() #plot in a plot 
        plt.show(block=False) #completely unecessary line can be commented 

    def reset(self):
        self.agent_state = np.array([1,1]) #whereever the agent is right now place it back to the starting point wich is 1,1
        return self.agent_state 
    
    def step(self, action):
        if action == 0 and self.agent_state[1] < self.grid_size: # up, 1 means that the x coordinate is same and we only accessing the y coordinate 
            self.agent_state[1] += 1
        
        elif action == 1 and self.agent_state[1] > 0: # down, 0 coz its lower bound it it goes down 0 then it'll be out of grid 
            self.agent_state[1] -= 1

        elif action == 2 and self.agent_state[0] > 0: # left using the x coordinate it can't be lesser then 0 in left right case 
            self.agent_state[0] -= 1

        elif action == 3 and self.agent_state[0] < self.grid_size: # right
            self.agent_state[0] += 1
        
        # Check if hit a hurdle
        for hurdle in self.hurdles:
            if np.array_equal(self.agent_state, hurdle):
                print("Hit a hurdle! Resetting environment.")
                obs = self.reset()
                return obs, -5, False, {"reason": "hurdle"}
        # Check current goal
        reward = 0
        done = False
        if np.array_equal(self.agent_state, self.goals[self.current_goal_idx]):
            print(f"Reached goal {self.current_goal_idx + 1}!")
            reward = 10
            self.current_goal_idx += 1
            if self.current_goal_idx >= len(self.goals):
                done = True
                print("🎯 All goals reached!")  

        info = {"goals_remaining": len(self.goals) - self.current_goal_idx,
            "current_target": self.goals[self.current_goal_idx - 1] if self.current_goal_idx > 0 else self.goals[0]}
        
        return self.agent_state, reward, done, info 
    
    def render(self):
        self.ax.clear()#buildin clearing the screen 
        self.ax.plot(self.agent_state[0],self.agent_state[1],"ro") # writing the x axis and y axis separately , ro is the buildin means red dot 
        
        # Draw all goals as image
        for idx, goal in enumerate(self.goals):
            x, y = goal
            # Scale down image to fit grid cell
            self.ax.imshow(self.goal_img, extent=[x - 0.4, x + 0.4, y - 0.4, y + 0.4], zorder=1)

        # draw hurdles
        for h in self.hurdles:
            self.ax.plot(h[0], h[1], "rx")  # hurdle (red X)
        self.ax.set_xlim(-1,self.grid_size) #for out own ease we are making it from -1 till the grid size
        self.ax.set_ylim(-1,self.grid_size)
        self.ax.set_aspect('equal') #equally distribute the aspect ratio 
        plt.pause(0.1) #taking the next step with pause of 0.1 sec for our ease 

    def close(self):
        plt.close()

if __name__=="__main__": # double underscore is the important syntax mtlab this is patthar par lakeer or yai hokr hi rhay ga always true, its needed coz we want to run the code in this block at any cost 
    env = HarryPotterEnv()
    state = env.reset() 
    for _ in range(50000):
        action = env.action_space.sample() #randomly taking the descion up, down, left, right and action spce is the descrete(4) already discuused in the code randomly take any action from the 4
        state,reward,done,info = env.step(action) #
        env.render()
        print(f"state:{state},Reward{reward},Action{action},Done{done},Info{info}")
        if done:
            print("Bruhhhhhhhhhhhhhhhh!!!!!!! I reached the destination ")
            break
    env.close()
