import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Callable
from collections import defaultdict
from offlinerlkit.policy import TD3BCPolicy
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler

from offlinerlkit.modules.dist_module import TanhNormalWrapper

class OOMITD3BCPolicy(TD3BCPolicy):

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        rollout_policy,
        tau: float = 0.005,
        gamma: float = 0.99,
        max_action: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        alpha: float = 2.5,
        scaler: StandardScaler = None,
        # uniform_rollout: bool = False,
        device="cpu",
        action_scale: float = 1.0,
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            max_action=max_action,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq,
            alpha=alpha,
            scaler=scaler,
            action_scale=action_scale
        )

        self.dynamics = dynamics
        # self._uniform_rollout = uniform_rollout
        self.rollout_policy = rollout_policy
        self.device = device


    # this is for reproducing TD3BC version combo using oomi-TD3BC
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
            actions = self.select_action_using_normalized_obss(observations)
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


    def actforward_with_dist(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
        # NoNeedTODO: calculate dists to deterministic actions. Use L2 norm instead
        # batch_size = obs.shape[0]
        # num_repeat_actions = 10
        # tmp_obss = obs.unsqueeze(1) \
        #     .repeat(1, num_repeat_actions, 1) \
        #     .view(batch_size * num_repeat_actions, obs.shape[-1])
        #
        # action = self.actor(tmp_obss)
        # action = action.view(batch_size, num_repeat_actions, -1)
        # mean = action.mean(dim=1)
        # std = action.std(dim=1)
        # return TanhNormalWrapper(mean, std)

    def select_action_using_normalized_obss(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action*self.action_scale



    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, final_policy_fake_batch, rollout_policy_fake_batch = batch["real"], batch["final_policy_fake"], batch["rollout_policy_fake"]
        mix_batch = {k: torch.cat([real_batch[k], final_policy_fake_batch[k], rollout_policy_fake_batch[k]], 0) for k in real_batch.keys()}

        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
            mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        batch_size = obss.shape[0]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            q = self.critic1(obss, a)
            lmbda = self._alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + ((a - actions).pow(2)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._sync_weight()

        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

