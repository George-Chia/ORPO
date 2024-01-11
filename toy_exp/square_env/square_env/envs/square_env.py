import gym
from gym import spaces
import numpy as np
import random

def in_high_value(pos):
    x = pos[0]
    y = pos[1]
    if x <= 3 and y <= 3 and (x - 3) ** 2 + (y - 3) ** 2 <= 1:
        return True
    else:
        return False
    


def generate_toy_state():
    width = 0.25
    while True:
        x = random.uniform(-3, 3)  
        y = random.uniform(-3, 3)  
        if -x - width < y < -x + width: 
            return (x, y)  

class SquareEnvironment(gym.Env):
    def __init__(self):
        super(SquareEnvironment, self).__init__()

        self.observation_space = spaces.Box(low=-3, high=3, shape=(2,))

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        self.agent_pos = np.zeros(2)

    def step(self, action):
        current_pos = self.agent_pos

        new_pos = current_pos + action

        new_pos = np.clip(new_pos, -3, 3)


        reward = self._calculate_reward(new_pos)

        self.agent_pos = new_pos

        done = False 

        return new_pos, reward, done, {}

    def reset(self):
        init_state = generate_toy_state()
        self.agent_pos = np.array(init_state)
        return self.agent_pos

    def _calculate_reward(self, pos):
        line_distance = abs(pos[1] + pos[0]) / np.sqrt(2)

        reward = line_distance if pos[1] > -pos[0] else -line_distance
        
        return reward
