from d3rlpy.algos.sac import *

from typing import (
    Any,
    Optional,

)



from ..argument_utility import (
    ActionScalerArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
)

from ..models.optimizers import OptimizerFactory


from .baseORPOSim import AlgoBaseORPOSim


class SACORPOSim(SAC,AlgoBaseORPOSim):
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
            temp_learning_rate=temp_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
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
        )
        # ORPO
        self._rollout_batch_size = rollout_batch_size
        self._rollout_horizon = rollout_horizon
        self._rollout_interval = rollout_interval
        self._generated_maxlen = generated_maxlen
        self._real_ratio = real_ratio
