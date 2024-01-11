from typing import Any, Dict, Optional, Sequence

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.sac_impl import DiscreteSACImpl, SACImpl


from d3rlpy.algos import SAC
import gym
from ..online.buffers import Buffer, ReplayBuffer
from ..online.explorers import Explorer
from typing import Callable
from ..online.iterators import AlgoProtocol
from .base import _assert_action_space

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

from d3rlpy.online.iterators import collectRecord2rewards

def train_single_env_Record2rewards(
    algo: AlgoProtocol,
    env: gym.Env,
    buffer: ReplayBufferRecord2rewards,
    explorer: Optional[Explorer] = None,
    n_steps: int = 1000000,
    n_steps_per_epoch: int = 10000,
    update_interval: int = 1,
    update_start_step: int = 0,
    random_steps: int = 0,
    eval_env: Optional[gym.Env] = None,
    eval_epsilon: float = 0.0,
    save_metrics: bool = True,
    save_interval: int = 1,
    experiment_name: Optional[str] = None,
    with_timestamp: bool = True,
    logdir: str = "d3rlpy_logs",
    verbose: bool = True,
    show_progress: bool = True,
    tensorboard_dir: Optional[str] = None,
    timelimit_aware: bool = True,
    callback: Optional[Callable[[AlgoProtocol, int, int], None]] = None,
) -> None:
    """Start training loop of online deep reinforcement learning.

    Args:
        algo: algorithm object.
        env: gym-like environment.
        buffer : replay buffer.
        explorer: action explorer.
        n_steps: the number of total steps to train.
        n_steps_per_epoch: the number of steps per epoch.
        update_interval: the number of steps per update.
        update_start_step: the steps before starting updates.
        random_steps: the steps for the initial random explortion.
        eval_env: gym-like environment. If None, evaluation is skipped.
        eval_epsilon: :math:`\\epsilon`-greedy factor during evaluation.
        save_metrics: flag to record metrics. If False, the log
            directory is not created and the model parameters are not saved.
        save_interval: the number of epochs before saving models.
        experiment_name: experiment name for logging. If not passed,
            the directory name will be ``{class name}_online_{timestamp}``.
        with_timestamp: flag to add timestamp string to the last of
            directory name.
        logdir: root directory name to save logs.
        verbose: flag to show logged information on stdout.
        show_progress: flag to show progress bar for iterations.
        tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
        timelimit_aware: flag to turn ``terminal`` flag ``False`` when
            ``TimeLimit.truncated`` flag is ``True``, which is designed to
            incorporate with ``gym.wrappers.TimeLimit``.
        callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called at the end of epochs.

    """
    # setup logger
    if experiment_name is None:
        experiment_name = algo.__class__.__name__ + "_online"

    logger = D3RLPyLogger(
        experiment_name,
        save_metrics=save_metrics,
        root_dir=logdir,
        verbose=verbose,
        tensorboard_dir=tensorboard_dir,
        with_timestamp=with_timestamp,
    )
    algo.set_active_logger(logger)

    # initialize algorithm parameters
    _setup_algo(algo, env)

    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    # prepare stacked observation
    if is_image:
        stacked_frame = StackedObservation(observation_shape, algo.n_frames)

    # save hyperparameters
    algo.save_params(logger)

    # switch based on show_progress flag
    xrange = trange if show_progress else range

    # setup evaluation scorer
    eval_scorer: Optional[Callable[..., float]]
    if eval_env:
        eval_scorer = evaluate_on_environment(eval_env, epsilon=eval_epsilon)
    else:
        eval_scorer = None

    # start training loop
    observation = env.reset()
    rollout_return = 0.0
    for total_step in xrange(1, n_steps + 1):
        with logger.measure_time("step"):
            # stack observation if necessary
            if is_image:
                stacked_frame.append(observation)
                fed_observation = stacked_frame.eval()
            else:
                observation = observation.astype("f4")
                fed_observation = observation

            # sample exploration action
            with logger.measure_time("inference"):
                if total_step < random_steps:
                    action = env.action_space.sample()
                elif explorer:
                    x = fed_observation.reshape((1,) + fed_observation.shape)
                    action = explorer.sample(algo, x, total_step)[0]
                else:
                    action = algo.sample_action([fed_observation])[0]

            # step environment
            with logger.measure_time("environment_step"):
                next_observation, reward, terminal, info = env.step(action)
                rollout_return += reward

            # special case for TimeLimit wrapper
            if timelimit_aware and "TimeLimit.truncated" in info:
                clip_episode = True
                terminal = False
            else:
                clip_episode = terminal

            # store observation
            buffer.append_with_modified_reward(
                observation=observation,
                action=action,
                reward=reward,
                modified_reward=info['modified_reward'],
                terminal=terminal,
                clip_episode=clip_episode,
            )

            # reset if terminated
            if clip_episode:
                observation = env.reset()
                logger.add_metric("rollout_return", rollout_return)
                rollout_return = 0.0
                # for image observation
                if is_image:
                    stacked_frame.clear()
            else:
                observation = next_observation

            # psuedo epoch count
            epoch = total_step // n_steps_per_epoch

            if total_step > update_start_step and len(buffer) > algo.batch_size:
                if total_step % update_interval == 0:
                    # sample mini-batch
                    with logger.measure_time("sample_batch"):
                        batch = buffer.sample(
                            batch_size=algo.batch_size,
                            n_frames=algo.n_frames,
                            n_steps=algo.n_steps,
                            gamma=algo.gamma,
                        )

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = algo.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)

            # call callback if given
            if callback:
                callback(algo, epoch, total_step)

        if epoch > 0 and total_step % n_steps_per_epoch == 0:
            # evaluation
            if eval_scorer:
                logger.add_metric("evaluation", eval_scorer(algo))

            if epoch % save_interval == 0:
                logger.save_model(total_step, algo)

            # save metrics
            logger.commit(epoch, total_step)

    # clip the last episode
    buffer.clip_episode()

    # close logger
    logger.close()


class SACRecord2rewards(SAC):
    '''
    For sampling original reward and modified reward for tasks requiring generalization.
    '''


    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        initial_temperature: float = 1.0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[SACImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            actor_encoder_factor=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            initial_temperature=initial_temperature,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            impl=impl,
            kwargs=kwargs,
        )

    def fit_online(
        self,
        env: gym.Env,
        buffer: Optional[Buffer] = None,
        explorer: Optional[Explorer] = None,
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        update_interval: int = 1,
        update_start_step: int = 0,
        random_steps: int = 0,
        eval_env: Optional[gym.Env] = None,
        eval_epsilon: float = 0.0,
        save_metrics: bool = True,
        save_interval: int = 1,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        timelimit_aware: bool = True,
        callback: Optional[Callable[[AlgoProtocol, int, int], None]] = None,
    ) -> None:
        """Start training loop of online deep reinforcement learning.

        Args:
            env: gym-like environment.
            buffer : replay buffer.
            explorer: action explorer.
            n_steps: the number of total steps to train.
            n_steps_per_epoch: the number of steps per epoch.
            update_interval: the number of steps per update.
            update_start_step: the steps before starting updates.
            random_steps: the steps for the initial random explortion.
            eval_env: gym-like environment. If None, evaluation is skipped.
            eval_epsilon: :math:`\\epsilon`-greedy factor during evaluation.
            save_metrics: flag to record metrics. If False, the log
                directory is not created and the model parameters are not saved.
            save_interval: the number of epochs before saving models.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be ``{class name}_online_{timestamp}``.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            timelimit_aware: flag to turn ``terminal`` flag ``False`` when
                ``TimeLimit.truncated`` flag is ``True``, which is designed to
                incorporate with ``gym.wrappers.TimeLimit``.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called at the end of epochs.

        """

        # create default replay buffer
        if buffer is None:
            buffer = ReplayBufferRecord2rewards(1000000, env=env)

        # check action-space
        _assert_action_space(self, env)

        train_single_env_Record2rewards(
            algo=self,
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            update_interval=update_interval,
            update_start_step=update_start_step,
            random_steps=random_steps,
            eval_env=eval_env,
            eval_epsilon=eval_epsilon,
            save_metrics=save_metrics,
            save_interval=save_interval,
            experiment_name=experiment_name,
            with_timestamp=with_timestamp,
            logdir=logdir,
            verbose=verbose,
            show_progress=show_progress,
            tensorboard_dir=tensorboard_dir,
            timelimit_aware=timelimit_aware,
            callback=callback,
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


class DiscreteSAC(AlgoBase):
    r"""Soft Actor-Critic algorithm for discrete action-space.

    This discrete version of SAC is built based on continuous version of SAC
    with additional modifications.

    The target state-value is calculated as expectation of all action-values.

    .. math::

        V(s_t) = \pi_\phi (s_t)^T [Q_\theta(s_t) - \alpha \log (\pi_\phi (s_t))]

    Similarly, the objective function for the temperature parameter is as
    follows.

    .. math::

        J(\alpha) = \pi_\phi (s_t)^T [-\alpha (\log(\pi_\phi (s_t)) + H)]

    Finally, the objective function for the policy function is as follows.

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D}
            [\pi_\phi(s_t)^T [\alpha \log(\pi_\phi(s_t)) - Q_\theta(s_t)]]

    References:
        * `Christodoulou, Soft Actor-Critic for Discrete Action Settings.
          <https://arxiv.org/abs/1910.07207>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        initial_temperature (float): initial temperature value.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.sac_impl.DiscreteSACImpl):
            algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _temp_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _n_critics: int
    _initial_temperature: float
    _target_update_interval: int
    _use_gpu: Optional[Device]
    _impl: Optional[DiscreteSACImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(eps=1e-4),
        critic_optim_factory: OptimizerFactory = AdamFactory(eps=1e-4),
        temp_optim_factory: OptimizerFactory = AdamFactory(eps=1e-4),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 64,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_critics: int = 2,
        initial_temperature: float = 1.0,
        target_update_interval: int = 8000,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[DiscreteSACImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=None,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._n_critics = n_critics
        self._initial_temperature = initial_temperature
        self._target_update_interval = target_update_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DiscreteSACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            initial_temperature=self._initial_temperature,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        # lagrangian parameter update for SAC temeprature
        if self._temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(batch)
            metrics.update({"temp_loss": temp_loss, "temp": temp})

        critic_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss})

        actor_loss = self._impl.update_actor(batch)
        metrics.update({"actor_loss": actor_loss})

        if self._grad_step % self._target_update_interval == 0:
            self._impl.update_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE
