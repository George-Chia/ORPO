import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import SACPolicy
from offlinerlkit.dynamics import BaseDynamics

from torch.distributions.kl import kl_divergence
from offlinerlkit.policy import OOMICQLPolicy,OOMITD3BCPolicy
from offlinerlkit.utils.scaler import StandardScaler

class OptimisticRolloutPolicy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    """

    def __init__(
            self,
            dynamics: BaseDynamics,
            actor: nn.Module,
            critic1: nn.Module,
            critic2: nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic1_optim: torch.optim.Optimizer,
            critic2_optim: torch.optim.Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
            distance_coefficient: float = 0,
            scaler: StandardScaler = None,
            device: str = "cpu",
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
            alpha=alpha,
            action_scale=action_scale
        )

        self.dynamics = dynamics
        self.distance_coefficient = distance_coefficient
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
            actions = self.select_action(observations)
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

    def learn(self, batch: Dict, final_policy) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        batch = mix_batch
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs, dist = self.actforward_with_dist(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()

        # if final_policy is not None:
        #     if isinstance(final_policy, OOMICQLPolicy):
        #         with torch.no_grad():
        #             _, _, reference_dist = final_policy.actforward_with_dist(obss)
        #         distance_loss = kl_divergence(reference_dist,dist).mean()
        #         actor_loss += self.distance_coefficient * distance_loss
        #     elif isinstance(final_policy, OOMITD3BCPolicy):
        #         reference_a = final_policy.select_action_using_normalized_obss(obss,deterministic=True)
        #         distance_loss = ((torch.from_numpy(reference_a).to(self.device) - a).pow(2)).mean()
        #         actor_loss += self.distance_coefficient * distance_loss
        #     else:
        #         raise NotImplementedError

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()
        if final_policy is not None:
            if isinstance(final_policy, OOMICQLPolicy):
                result = {
                    "loss/rollout_actor": actor_loss.item(),
                    # "loss/kl_divergence": distance_loss.item(),
                    "loss/rollout_critic1": critic1_loss.item(),
                    "loss/rollout_critic2": critic2_loss.item(),
                }
            elif isinstance(final_policy, OOMITD3BCPolicy):
                result = {
                    "loss/rollout_actor": actor_loss.item(),
                    # "loss/distance_a": distance_loss.item(),
                    "loss/rollout_critic1": critic1_loss.item(),
                    "loss/rollout_critic2": critic2_loss.item(),
                }
            else:
                raise NotImplementedError
        else:
            result = {
                "loss/rollout_actor": actor_loss.item(),
                "loss/rollout_critic1": critic1_loss.item(),
                "loss/rollout_critic2": critic2_loss.item(),
            }

        if self._is_auto_alpha:
            result["loss/rollout_alpha"] = alpha_loss.item()
            result["rollout_alpha"] = self._alpha.item()

        return result

    def actforward_with_dist(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action*self.action_scale, log_prob, dist

    def select_action_using_normalized_obs_if_needed(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        with torch.no_grad():
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()