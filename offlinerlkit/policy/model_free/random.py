import numpy as np
from offlinerlkit.dynamics import BaseDynamics
from typing import Dict, Union, Tuple
from collections import defaultdict

class RandomRolloutPolicy():
    def __init__(
            self,
            dynamics: BaseDynamics,
            action_space: np.ndarray
    ) -> None:
        self.action_space = action_space
        self.dynamics = dynamics

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
            actions = np.random.uniform(
                self.action_space.low[0],
                self.action_space.high[0],
                size=(len(observations), self.action_space.shape[0])
            )
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
        actions = np.random.uniform(
            self.action_space.low[0],
            self.action_space.high[0],
            size=(len(obs), self.action_space.shape[0])
        )
        return actions


