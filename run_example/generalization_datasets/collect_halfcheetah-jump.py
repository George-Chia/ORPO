import numpy as np

import d3rlpy
import gym
import d4rl
from gym.wrappers import RescaleAction
import argparse

from d3rlpy.preprocessing import MinMaxActionScaler
from d3rlpy.wrappers.wrappedEnv import HalfcheetahvelRewardWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_critics', type=int, default=2)
args = parser.parse_args()

# env
env = gym.make('halfcheetah-random-v2')
env = RescaleAction(env, -3, 3)
env = HalfcheetahvelRewardWrapper(env)
action_scaler = MinMaxActionScaler(minimum=-3, maximum=3)

# setup algorithm
# sac = d3rlpy.algos.SAC()
sac = d3rlpy.algos.SACRecord2rewards(actor_learning_rate=1e-5,
                       critic_learning_rate=1e-5,
                       temp_learning_rate=1e-5,
                       batch_size=256,
                       use_gpu=args.gpu,
                       n_critics=args.n_critics,
                       action_scaler=action_scaler)

# prepare experience replay buffer
buffer = d3rlpy.online.buffers.ReplayBufferRecord2rewards(maxlen=1000000, env=env)

# prepare exploration strategy if necessary
explorer = d3rlpy.online.explorers.ConstantEpsilonGreedy(0.3)

# start data collection
sac.fit_online(env, buffer, n_steps=1000000)

# export as MDPDataset
dataset = buffer.to_mdp_dataset()
dataset_modified_reward = buffer.to_mdp_dataset_modified_reward()

# save MDPDataset
# dataset.dump("halfcheetah-jump-original-rewards.h5")
dataset_modified_reward.dump("halfcheetah-jump.h5")






