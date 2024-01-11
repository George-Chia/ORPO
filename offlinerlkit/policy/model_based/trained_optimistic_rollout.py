import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import OptimisticRolloutPolicy
from offlinerlkit.dynamics import BaseDynamics

from torch.distributions.kl import kl_divergence
from offlinerlkit.policy import OOMICQLPolicy, OOMITD3BCPolicy
from offlinerlkit.utils.scaler import StandardScaler
import os

class TrainedOptimisticRolloutPolicy(OptimisticRolloutPolicy):
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
            load_optimistic_policy_path=None
    ) -> None:
        super().__init__(
            dynamics,
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            distance_coefficient=distance_coefficient,
            scaler=scaler,
            device=device,
            action_scale=action_scale
        )
        self.load(load_optimistic_policy_path)

    def learn(self, batch: Dict, final_policy) -> None:
        result = None
        return result

    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "trained-optimistic-policy.pth"), map_location=self.device))
