from gym.envs.registration import register
from .envs.square_env import SquareEnvironment

register(
    id='square_env/SquareEnv-v0',
    entry_point='square_env.envs:SquareEnvironment',
    max_episode_steps=10,
)