import gym
import numpy as np
import math

class HalfcheetahvelRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, info = self.env.step(action)
        info["reward_run"] = np.clip(info["reward_run"],-3,3)
        reward = info["reward_run"] + info["reward_ctrl"]
        info['modified_reward'] = reward + 15 * obs[0]
        # self.render()
        return obs, reward, terminated, info

class HalfcheetahvelSafeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, info = self.env.step(action)
        info["reward_run"] = np.clip(info["reward_run"],-3,3)
        reward = info["reward_run"] + info["reward_ctrl"]
        info['modified_reward'] = reward + 15 * obs[0]
        if obs[1] > 1:
            info['modified_reward'] = -3
        # self.render()
        return obs, reward, terminated, info



class AntRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, info = self.env.step(action)
        # info["reward_run"] = np.clip(info["reward_run"],-3,3)
        reward = info['reward_forward'] + info['reward_ctrl'] + info['reward_contact'] + info['reward_survive']
        self.render()
        info['modified_reward'] = info['x_velocity'] * math.cos(math.pi / 6) + info['y_velocity'] * math.sin(math.pi / 6)\
                                  + info['reward_ctrl'] + info['reward_contact'] #+ info['reward_survive']
        # print(reward, info['modified_reward'])
        return obs, reward, terminated, info

