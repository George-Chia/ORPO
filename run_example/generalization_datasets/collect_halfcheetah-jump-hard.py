import numpy as np

import d3rlpy
import gym
import d4rl
from gym.wrappers import RescaleAction
import argparse

from d3rlpy.preprocessing import MinMaxActionScaler
from d3rlpy.wrappers.wrappedEnv import HalfcheetahvelSafeRewardWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--n_critics', type=int, default=2)
args = parser.parse_args()

# env
env = gym.make('halfcheetah-random-v2')
env = RescaleAction(env, -3, 3)
env = HalfcheetahvelSafeRewardWrapper(env)
action_scaler = MinMaxActionScaler(minimum=-3, maximum=3)

# setup algorithm
# sac = d3rlpy.algos.SAC()
random_policy = d3rlpy.algos.RandomPolicyRecord2rewards(use_gpu=args.gpu, action_scaler=action_scaler)

# prepare experience replay buffer
buffer = d3rlpy.online.buffers.ReplayBufferRecord2rewards(maxlen=1000000, env=env)


# start data collection
random_policy.collect(env, buffer, n_steps=1000000)

# export as MDPDataset
dataset = buffer.to_mdp_dataset()
dataset_modified_reward = buffer.to_mdp_dataset_modified_reward()

# save MDPDataset
# dataset.dump("halfcheetah-jump-hard-original-rewards.h5")
dataset_modified_reward.dump("halfcheetah-jump-hard.h5")






