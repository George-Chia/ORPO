import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict
from offlinerlkit.buffer.buffer import ReplayBuffer


class ReplayBufferPlus(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        super(ReplayBufferPlus, self).__init__(buffer_size=buffer_size,
        obs_shape=obs_shape,
        obs_dtype=obs_dtype,
        action_dim=action_dim,
        action_dtype=action_dtype,
        device=device)
        self.bonus_rewards = np.zeros((self._max_size, 1), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminal: np.ndarray,
            bonus_reward: np.ndarray,
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()
        self.bonus_rewards[self._ptr] = np.array(reward).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def add_batch(
            self,
            obss: np.ndarray,
            next_obss: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminals: np.ndarray,
            bonus_rewards: np.ndarray,
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()
        self.bonus_rewards[indexes] = np.array(bonus_rewards).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)

        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        },batch_indexes

    def sample_bonus_reward(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.bonus_rewards[batch_indexes]).to(self.device)
        }



    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy(),
            "bonus_rewards": self.bonus_rewards[:self._size].copy(),
        }