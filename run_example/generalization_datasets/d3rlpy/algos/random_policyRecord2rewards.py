from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from ..argument_utility import ActionScalerArg
from ..constants import ActionSpace
from .base import AlgoBase
from .random_policy import RandomPolicy

from typing import Any, Callable, Dict, List, Optional, Union
import gym
import numpy as np
from tqdm.auto import trange
from typing_extensions import Protocol

from ..dataset import TransitionMiniBatch
from ..logger import LOG, D3RLPyLogger
from ..metrics.scorer import evaluate_on_environment
from ..preprocessing import ActionScaler, Scaler
from ..preprocessing.stack import StackedObservation
from d3rlpy.online.buffers import Buffer,ReplayBufferRecord2rewards
from d3rlpy.online.explorers import Explorer
from d3rlpy.online.iterators import _setup_algo
from .base import _assert_action_space
from d3rlpy.online.iterators import collectRecord2rewards


class RandomPolicyRecord2rewards(RandomPolicy):
    r"""Random Policy for continuous control algorithm.

    This is designed for data collection and lightweight interaction tests.
    ``fit`` and ``fit_online`` methods will raise exceptions.

    Args:
        distribution (str): random distribution. The available options are
            ``['uniform', 'normal']``.
        normal_std (float): standard deviation of the normal distribution. This
            is only used when ``distribution='normal'``.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.

    """

    _distribution: str
    _normal_std: float
    _action_size: int

    def __init__(
        self,
        *,
        distribution: str = "uniform",
        normal_std: float = 1.0,
        action_scaler: ActionScalerArg = None,
        **kwargs: Any,
    ):
        super().__init__(
            distribution=distribution,
            normal_std=normal_std,
            action_scaler=action_scaler,
            kwargs=kwargs,
        )

    def collect(
        self,
        env: gym.Env,
        buffer: Optional[Buffer] = None,
        explorer: Optional[Explorer] = None,
        deterministic: bool = False,
        n_steps: int = 1000000,
        show_progress: bool = True,
        timelimit_aware: bool = True,
    ) -> Buffer:
        """Collects data via interaction with environment.

        If ``buffer`` is not given, ``ReplayBuffer`` will be internally created.

        Args:
            env: gym-like environment.
            buffer : replay buffer.
            explorer: action explorer.
            deterministic: flag to collect data with the greedy policy.
            n_steps: the number of total steps to train.
            show_progress: flag to show progress bar for iterations.
            timelimit_aware: flag to turn ``terminal`` flag ``False`` when
                ``TimeLimit.truncated`` flag is ``True``, which is designed to
                incorporate with ``gym.wrappers.TimeLimit``.

        Returns:
            replay buffer with the collected data.

        """
        # create default replay buffer
        if buffer is None:
            buffer = ReplayBufferRecord2rewards(1000000, env=env)

        # check action-space
        _assert_action_space(self, env)

        collectRecord2rewards(
            algo=self,
            env=env,
            buffer=buffer,
            explorer=explorer,
            deterministic=deterministic,
            n_steps=n_steps,
            show_progress=show_progress,
            timelimit_aware=timelimit_aware,
        )

        return buffer
