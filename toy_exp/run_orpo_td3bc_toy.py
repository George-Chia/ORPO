import argparse
import os
import sys
import random

import gym
import d4rl

import numpy as np
import torch

from offlinerlkit.nets import MLP
from offlinerlkit.modules import Actor, ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset,get_dataset
from offlinerlkit.buffer import ReplayBuffer, ReplayBufferPlus
from offlinerlkit.utils.logger import Logger, make_log_dirs
# from offlinerlkit.utils.recompute_reward_fns import recompute_reward_fn_halfcheetahjump,recompute_reward_fn_antangle

from offlinerlkit.policy import OptimisticRolloutPolicy, OOMITD3BCPolicy, RandomRolloutPolicy, VaeRolloutPolicy, TrainedOptimisticRolloutPolicy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.policy_trainer import MBPolicyOOMITrainer
import pickle
import square_env



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="orpo-td3bc")
    parser.add_argument("--task", type=str, default="toy")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # for final policy
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--penalty-coef", type=float, default=0)
    parser.add_argument("--real-ratio-final", type=float, default=0.5)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--final-policy-rollout-ratio-final", type=float, default=0.0)
    # parser.add_argument("--rho-s", type=str, default="mix", choices=["model", "mix"])
    parser.add_argument("--evaluation-only", default=None, action='store_true')
    parser.add_argument("--load-policy-path", type=str, default=None)

    # for rollout policy SAC
    parser.add_argument("--rollout_actor-lr", type=float, default=1e-4)
    parser.add_argument("--rollout_critic-lr", type=float, default=3e-4)
    parser.add_argument("--rollout_hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--rollout_gamma", type=float, default=0.99)
    parser.add_argument("--rollout_tau", type=float, default=0.005)
    parser.add_argument("--rollout_alpha", type=float, default=0.2)
    parser.add_argument("--rollout_auto-alpha", default=True)
    parser.add_argument("--rollout_target-entropy", type=int, default=None)
    parser.add_argument("--rollout_alpha-lr", type=float, default=1e-4)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length-rollout-policy", type=int, default=5)
    parser.add_argument("--distance-coef", type=float, default=0)
    parser.add_argument("--bonus-coef", type=float, default=0.015)
    parser.add_argument("--real-ratio-rollout", type=float, default=0.05)
    # parser.add_argument("--uniform-rollout", type=bool, default=False)
    # parser.add_argument("--final-policy-rollout", type=bool, default=False)  # TD3BC version combo
    parser.add_argument("--rollout-policy-type",type=str, default="optimistic",help="optimistic, random, vae or trained_optimistic")


    # for dynamics model
    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--uncertainty_mode", type=str, default='ensemble_std')
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)


    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make('square_env/SquareEnv-v0')
    is_generalization_env = False
    args.action_scale = 1.0
    args.original_max_action = env.action_space.high[0] / args.action_scale

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    with open("./random_dataset_0.25width_10000.pkl", "rb") as file:
        dataset = pickle.load(file)



    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    actor = Actor(actor_backbone, args.action_dim, device=args.device)

    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.rollout_policy_type == "optimistic" or args.rollout_policy_type == "trained_optimistic":
        # create rollout policy model
        rollout_actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.rollout_hidden_dims)
        rollout_critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.rollout_hidden_dims)
        rollout_critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.rollout_hidden_dims)
        rollout_dist = TanhDiagGaussian(
            latent_dim=getattr(rollout_actor_backbone, "output_dim"),
            output_dim=args.action_dim,
            unbounded=True,
            conditioned_sigma=True
        )
        rollout_actor = ActorProb(rollout_actor_backbone, rollout_dist, args.device)
        rollout_critic1 = Critic(rollout_critic1_backbone, args.device)
        rollout_critic2 = Critic(rollout_critic2_backbone, args.device)
        rollout_actor_optim = torch.optim.Adam(rollout_actor.parameters(), lr=args.rollout_actor_lr)
        rollout_critic1_optim = torch.optim.Adam(rollout_critic1.parameters(), lr=args.rollout_critic_lr)
        rollout_critic2_optim = torch.optim.Adam(rollout_critic2.parameters(), lr=args.rollout_critic_lr)
        rollout_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rollout_actor_optim, args.epoch)

        if args.rollout_auto_alpha:
            rollout_target_entropy = args.rollout_target_entropy if args.rollout_target_entropy \
                else -np.prod(env.action_space.shape)
            args.target_entropy = rollout_target_entropy
            rollout_log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            rollout_alpha_optim = torch.optim.Adam([rollout_log_alpha], lr=args.rollout_alpha_lr)
            rollout_alpha = (rollout_target_entropy, rollout_log_alpha, rollout_alpha_optim)
        else:
            rollout_alpha = args.rollout_alpha
    else:
        rollout_lr_scheduler = None

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        penalty_coef=args.penalty_coef,
        bonus_coef=args.bonus_coef
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    fake_buffer = ReplayBufferPlus(
        buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    fake_buffer_rollout = ReplayBufferPlus(
        buffer_size=args.rollout_batch_size * args.rollout_length_rollout_policy * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    obs_scaler = None

    if args.rollout_policy_type == "optimistic":
        # create rollout policy
        rollout_policy = OptimisticRolloutPolicy(
            dynamics,
            rollout_actor,
            rollout_critic1,
            rollout_critic2,
            rollout_actor_optim,
            rollout_critic1_optim,
            rollout_critic2_optim,
            tau=args.rollout_tau,
            gamma=args.rollout_gamma,
            alpha=rollout_alpha,
            distance_coefficient=args.distance_coef,
            scaler=obs_scaler,
            device=args.device,
            action_scale=args.action_scale  # for halfcheetahvel
        )
    elif args.rollout_policy_type == "random":
        rollout_policy = RandomRolloutPolicy(dynamics,action_space=env.action_space)
    elif args.rollout_policy_type == "trained_optimistic":
        load_optimistic_policy_path = ''
        rollout_policy = TrainedOptimisticRolloutPolicy(
            dynamics,
            rollout_actor,
            rollout_critic1,
            rollout_critic2,
            rollout_actor_optim,
            rollout_critic1_optim,
            rollout_critic2_optim,
            tau=args.rollout_tau,
            gamma=args.rollout_gamma,
            alpha=rollout_alpha,
            distance_coefficient=args.distance_coef,
            scaler=obs_scaler,
            device=args.device,
            action_scale=args.action_scale,  # for halfcheetahvel
            load_optimistic_policy_path=load_optimistic_policy_path
        )




    # create policy
    policy = OOMITD3BCPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        max_action=args.original_max_action,
        rollout_policy=rollout_policy,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        update_actor_freq=args.update_actor_freq,
        alpha=args.alpha,
        scaler=obs_scaler,
        # uniform_rollout=args.uniform_rollout,
        device=args.device,
        action_scale=args.action_scale
    )
    if args.load_policy_path:
        policy.load(args.load_policy_path)





    # log
    # assert (args.uniform_rollout and args.final_policy_rollout) == False
    if args.rollout_policy_type == "optimistic":
        log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args),
                                 record_params=["penalty_coef", "bonus_coef", "uncertainty_mode",
                                                "rollout_length", "distance_coef", "real_ratio_rollout",
                                                "real_ratio_final", "final_policy_rollout_ratio_final",
                                                "rollout_length_rollout_policy"])
        if args.evaluation_only:
            log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args),
                                     record_params=["penalty_coef", "bonus_coef", "uncertainty_mode",
                                                    "rollout_length", "distance_coef", "real_ratio_rollout",
                                                    "real_ratio_final", "final_policy_rollout_ratio_final",
                                                    "rollout_length_rollout_policy", "evaluation_only"])
    else:
        log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args),
                                 record_params=["penalty_coef", "bonus_coef", "uncertainty_mode",
                                                "rollout_length", "distance_coef",
                                                # "real_ratio_rollout","real_ratio_final", "final_policy_rollout_ratio_final",
                                                "rollout_length_rollout_policy",
                                                "rollout_policy_type"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MBPolicyOOMITrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        fake_buffer_rollout=fake_buffer_rollout,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio_rollout=args.real_ratio_rollout,
        real_ratio_final=args.real_ratio_final,
        eval_episodes=args.eval_episodes,
        lr_scheduler=None,
        rollout_lr_scheduler=rollout_lr_scheduler,
        rollout_length_rollout_policy=args.rollout_length_rollout_policy,
        final_policy_rollout_ratio_final=args.final_policy_rollout_ratio_final,
        is_generalization_env=is_generalization_env,
        evaluation_only=args.evaluation_only
        # final_policy_rollout=args.final_policy_rollout,
    )

    # train
    if not load_dynamics_model:
        dynamics.train(real_buffer.sample_all(), logger, max_epochs_since_update=5)

    policy_trainer.train()


if __name__ == "__main__":
    train()