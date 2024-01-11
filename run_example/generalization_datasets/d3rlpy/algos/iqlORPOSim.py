from d3rlpy.algos.iql import *
import copy
import json
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import gym
import numpy as np
from tqdm.auto import tqdm

from ..argument_utility import (
    ActionScalerArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_action_scaler,
    check_reward_scaler,
    check_scaler,
)
from ..constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from ..context import disable_parallel
from ..dataset import Episode, MDPDataset, Transition, TransitionMiniBatch
from ..decorators import pretty_repr
from ..gpu import Device
from ..iterators import RandomIterator, RoundIterator, TransitionIterator
from ..logger import LOG, D3RLPyLogger
from ..models.encoders import EncoderFactory, create_encoder_factory
from ..models.optimizers import OptimizerFactory
from ..models.q_functions import QFunctionFactory, create_q_func_factory
from ..online.utility import get_action_size_from_env
from ..preprocessing import (
    ActionScaler,
    RewardScaler,
    Scaler,
    create_action_scaler,
    create_reward_scaler,
    create_scaler,
)

from .baseORPOSim import AlgoBaseORPOSim


class IQLORPOSim(IQL,AlgoBaseORPOSim):
    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        value_encoder_factory: EncoderArg = "default",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        expectile: float = 0.7,
        weight_temp: float = 3.0,
        max_weight: float = 100.0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[IQLImpl] = None,

        # ORPO
        rollout_horizon: int = 5,
        generated_maxlen: int = 50000 * 5 * 5,
        real_ratio: float = 0.05,  # MOPO
        # ORPO default
        rollout_batch_size: int = 50000,
        rollout_interval: int = 1000,

        **kwargs: Any,
    ):
        super().__init__(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            value_encoder_factory=value_encoder_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            expectile=expectile,
            weight_temp=weight_temp,
            max_weight=max_weight,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            impl=impl,
        )

        # ORPO
        self._rollout_batch_size = rollout_batch_size
        self._rollout_horizon = rollout_horizon
        self._rollout_interval = rollout_interval
        self._generated_maxlen=generated_maxlen
        self._real_ratio = real_ratio

    '''
    def fit(
        self,
        dataset: Union[List[Episode], List[Transition], MDPDataset],
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        george_tensorboard=None,
        george_num=None,
        update_rololut_policy=None,
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Trains with the given dataset.

        .. code-block:: python

            algo.fit(episodes, n_steps=1000000)

        Args:
            dataset: list of episodes to train.
            n_epochs: the number of epochs to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            shuffle: flag to shuffle transitions on each epoch.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.

        Returns:
            list of result tuples (epoch, metrics) per epoch.

        """
        results = list(
            self.fitter(
                dataset,
                n_epochs,
                n_steps,
                n_steps_per_epoch,
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                show_progress,
                tensorboard_dir,
                eval_episodes,
                save_interval,
                scorers,
                shuffle,
                callback,
                george_tensorboard,
                george_num,
                update_rololut_policy=update_rololut_policy
            )
        )
        return results


    def fitter(
        self,
        dataset: Union[List[Episode], List[Transition], MDPDataset],
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        george_tensorboard=None,
        george_num=None,
        update_rololut_policy=None,
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
             iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: offline dataset to train.
            n_epochs: the number of epochs to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            shuffle: flag to shuffle transitions on each epoch.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.

        Returns:
            iterator yielding current epoch and metrics dict.

        """

        transitions = []      # memory increasing...........(George)
        if isinstance(dataset, MDPDataset):
            for episode in dataset.episodes:
                transitions += episode.transitions
        elif not dataset:
            raise ValueError("empty dataset is not supported.")
        elif isinstance(dataset[0], Episode):
            for episode in cast(List[Episode], dataset):
                transitions += episode.transitions
        elif isinstance(dataset[0], Transition):
            transitions = list(cast(List[Transition], dataset))
        else:
            raise ValueError(f"invalid dataset type: {type(dataset)}")

        # how to delete it????????
        # for transition in transitions:
        #     del transition
        # import gc
        # gc.collect()


        # check action space
        if self.get_action_type() == ActionSpace.BOTH:
            pass
        elif transitions[0].is_discrete:
            assert (
                self.get_action_type() == ActionSpace.DISCRETE
            ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
        else:
            assert (
                self.get_action_type() == ActionSpace.CONTINUOUS
            ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR

        iterator: TransitionIterator
        if n_epochs is None and n_steps is not None:
            assert n_steps >= n_steps_per_epoch
            n_epochs = n_steps // n_steps_per_epoch
            iterator = RandomIterator(
                transitions,
                n_steps_per_epoch,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
            )
            LOG.debug("RandomIterator is selected.")
        elif n_epochs is not None and n_steps is None:
            iterator = RoundIterator(
                transitions,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
                shuffle=shuffle,
            )
            LOG.debug("RoundIterator is selected.")
        else:
            raise ValueError("Either of n_epochs or n_steps must be given.")

        # setup logger
        logger = self._prepare_logger(
            save_metrics,
            experiment_name,
            with_timestamp,
            logdir,
            verbose,
            tensorboard_dir,
        )

        # add reference to active logger to algo class during fit
        self._active_logger = logger

        # initialize scaler
        if self._scaler:
            LOG.debug("Fitting scaler...", scaler=self._scaler.get_type())
            self._scaler.fit(transitions)

        # initialize action scaler
        if self._action_scaler:
            LOG.debug(
                "Fitting action scaler...",
                action_scaler=self._action_scaler.get_type(),
            )
            self._action_scaler.fit(transitions)

        # initialize reward scaler
        if self._reward_scaler:
            LOG.debug(
                "Fitting reward scaler...",
                reward_scaler=self._reward_scaler.get_type(),
            )
            self._reward_scaler.fit(transitions)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            transition = iterator.transitions[0]
            action_size = transition.get_action_size()
            observation_shape = tuple(transition.get_observation_shape())
            self.create_impl(
                self._process_observation_shape(observation_shape), action_size
            )
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # save hyperparameters
        self.save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)

        # training loop
        total_step = 0
        for epoch in range(1, n_epochs + 1):
            dynamics,rollout_policy = update_rololut_policy(epoch)

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(iterator)),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            iterator.reset()

            for itr in range_gen:

                # generate new transitions with dynamics models
                new_transitions = self.generate_new_data(
                    transitions=iterator.transitions,
                    dynamics=dynamics,
                    rollout_policy=rollout_policy
                )
                if new_transitions:
                    iterator.add_generated_transitions(new_transitions)
                    LOG.debug(
                        f"{len(new_transitions)} transitions are generated.",
                        real_transitions=len(iterator.transitions),
                        fake_transitions=len(iterator.generated_transitions),
                    )

                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = next(iterator)

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # save loss to loss history dict
            self._loss_history["epoch"].append(epoch)
            self._loss_history["step"].append(total_step)
            for name, vals in epoch_loss.items():
                if vals:
                    self._loss_history[name].append(np.mean(vals))

            if scorers and eval_episodes:
                self._evaluate(eval_episodes, scorers, logger)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            if george_tensorboard is not None:
                for key in metrics.keys():
                    george_tensorboard.add_scalar(key, metrics[key], george_num)


            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        # drop reference to active logger since out of fit there is no active
        # logger
        self._active_logger.close()
        self._active_logger = None


    def generate_new_data(
        self, transitions: List[Transition],dynamics, rollout_policy
    ) -> Optional[List[Transition]]:
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR

        if not self._is_generating_new_data():
            return None

        init_transitions = self._sample_initial_transitions(transitions)

        rets: List[Transition] = []

        # rollout
        batch = TransitionMiniBatch(init_transitions)
        observations = batch.observations
        actions = rollout_policy.sample_action(observations)   # George
        prev_transitions: List[Transition] = []
        for _ in range(self._get_rollout_horizon()):
            # predict next state
            pred = dynamics.step(observations, actions)
            # pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], pred)
            next_observations, penalized_rewards, terminals, infos, encouraged_rewards = pred

            # sample policy action
            next_actions = rollout_policy.sample_action(next_observations)

            # append new transitions
            new_transitions = []
            for i in range(len(init_transitions)):
                transition = Transition(
                    observation_shape=self._impl.observation_shape,
                    action_size=self._impl.action_size,
                    observation=observations[i],
                    action=actions[i],
                    reward=float(penalized_rewards[i][0]),
                    next_observation=next_observations[i],
                    terminal=terminals[i],
                )

                if prev_transitions:
                    prev_transitions[i].next_transition = transition
                    transition.prev_transition = prev_transitions[i]

                new_transitions.append(transition)

            prev_transitions = new_transitions
            rets += new_transitions
            observations = next_observations.copy()
            actions = next_actions.copy()

        return rets

    def _sample_initial_transitions(
        self, transitions: List[Transition]
    ) -> List[Transition]:
        # uniformly sample transitions
        n_transitions = self._rollout_batch_size
        indices = np.random.randint(len(transitions), size=n_transitions)
        return [transitions[i] for i in indices]

    def _get_rollout_horizon(self) -> int:
        return self._rollout_horizon

    def _is_generating_new_data(self) -> bool:
        return self._grad_step % self._rollout_interval == 0
    '''
