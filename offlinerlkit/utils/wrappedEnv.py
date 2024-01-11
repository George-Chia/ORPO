import gym
import numpy as np
import math

class HalfcheetahjumpRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, info = self.env.step(action)
        info["reward_run"] = np.clip(info["reward_run"],-3,3)
        reward = info["reward_run"] + info["reward_ctrl"]
        new_reward = reward + 15 * obs[0]
        return obs, new_reward, terminated, info

class HalfcheetahSafejumpRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, info = self.env.step(action)
        info["reward_run"] = np.clip(info["reward_run"],-3,3)
        reward = info["reward_run"] + info["reward_ctrl"]
        new_reward = reward + 15 * obs[0]
        if obs[1] > 1:
            new_reward = -3
        return obs, new_reward, terminated, info

class AntangleRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, info = self.env.step(action)
        # info["reward_run"] = np.clip(info["reward_run"],-3,3)
        # reward = info['reward_forward'] + info['reward_ctrl'] + info['reward_contact'] + info['reward_survive']
        new_reward = info['x_velocity'] * math.cos(math.pi / 6) + info['y_velocity'] * math.sin(math.pi / 6) \
          + info['reward_ctrl'] + info['reward_contact'] #+ info['reward_survive']
        return obs, new_reward, terminated, info
