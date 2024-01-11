import numpy as np
import torch
from offlinerlkit.dynamics import BaseDynamics
from typing import Dict, Union, Tuple
from collections import defaultdict
import torch.nn as nn
from offlinerlkit.utils.scaler import StandardScaler
import torch.nn.functional as F
import os


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750).to(device)
        self.d2 = nn.Linear(750, 750).to(device)
        self.d3 = nn.Linear(750, action_dim).to(device)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "fvae-policy.pth"), map_location=self.device))


class VaeRolloutPolicy():
    def __init__(
            self,
            dynamics: BaseDynamics,
            action_space: np.ndarray,
            state_dim:int,
            action_dim:int,
            max_action:float,
            scaler: StandardScaler = None,
            device: str = "cpu",
            load_vae_path: str = None

    ) -> None:
        self.action_space = action_space
        self.dynamics = dynamics
        self.device = device
        self.fvae = VAE(state_dim, action_dim, action_dim*2, max_action, device)
        self.fvae.load(load_vae_path)
        self.scaler = scaler
        self.device = device

    def rollout(
            self,
            init_obss: np.ndarray,
            rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            # unnormalized obs for vae
            with torch.no_grad():
                if self.scaler is not None:
                    observations = self.scaler.inverse_transform(observations)
                observations = torch.from_numpy(observations).to(self.device)
                actions = self.fvae.decode(observations)
            observations = observations.cpu().numpy()
            actions = actions.cpu().numpy()
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)
            rollout_transitions["bonus_rewards"].append(info["bouns_reward"])

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
               {"num_transitions_rollout_policy": num_transitions, "reward_mean_rollout_policy": rewards_arr.mean()}

    def select_action_using_normalized_obs_if_needed(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # if self.scaler is not None:
        #     obs = self.scaler.transform(obs)
        obs = obs.astype(np.float32)
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(self.device)
            action = self.fvae.decode(obs)
        return action.cpu().numpy()

