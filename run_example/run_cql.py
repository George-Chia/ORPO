import argparse
import random

import gym
import sys
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset,get_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import CQLPolicy


"""
suggested hypers
cql-weight=5.0, temperature=1.0 for all D4RL-Gym tasks
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="cql")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    if args.task.split('-')[0] == "halfcheetahveljump":
        original_task = "halfcheetah" + args.task.lstrip(args.task.split('-')[0])
        env = gym.make("halfcheetah-random-v2")
        from gym.wrappers import RescaleAction
        from offlinerlkit.utils.wrappedEnv import HalfcheetahjumpRewardWrapper
        env = RescaleAction(env, -3, 3)
        env = HalfcheetahjumpRewardWrapper(env)
        loaded_dataset = get_dataset(args.dataset)
        dataset = qlearning_dataset(env, dataset=loaded_dataset)
        # dataset = qlearning_dataset(env)
        # dataset['rewards'] = recompute_reward_fn_halfcheetahjump(dataset['observations'],dataset['actions'],
        #                                                         dataset['next_observations'],dataset['rewards'])
        is_generalization_env = True
        args.action_scale = 3.0
        args.original_max_action = env.action_space.high[0] / args.action_scale
    elif args.task.split('-')[0] == "antangle":
        env = gym.make('Ant-v3', exclude_current_positions_from_observation=False)
        from offlinerlkit.utils.wrappedEnv import AntangleRewardWrapper
        env = AntangleRewardWrapper(env)
        loaded_dataset = get_dataset(args.dataset)
        dataset = qlearning_dataset(env, dataset=loaded_dataset)
        # dataset['rewards'] = recompute_reward_fn_antangle(dataset['observations'], dataset['actions'],
        #                                                          dataset['next_observations'], dataset['rewards'])
        is_generalization_env = True
        args.action_scale = 1.0
        args.original_max_action = env.action_space.high[0] / args.action_scale
    elif args.task.split('-')[0] == "halfcheetahvelSafejump":
        original_task = "halfcheetah" + args.task.lstrip(args.task.split('-')[0])
        env = gym.make("halfcheetah-random-v2")
        from gym.wrappers import RescaleAction
        from offlinerlkit.utils.wrappedEnv import HalfcheetahSafejumpRewardWrapper
        env = RescaleAction(env, -3, 3)
        env = HalfcheetahSafejumpRewardWrapper(env)
        loaded_dataset = get_dataset(args.dataset)
        dataset = qlearning_dataset(env,dataset=loaded_dataset)
        # dataset = qlearning_dataset(env)
        # dataset['rewards'] = recompute_reward_fn_halfcheetahjump(dataset['observations'],dataset['actions'],
        #                                                         dataset['next_observations'],dataset['rewards'])
        is_generalization_env = True
        args.action_scale = 3.0
        args.original_max_action = env.action_space.high[0] / args.action_scale
    else:
        env = gym.make(args.task)
        if 'antmaze' in args.task:
            dataset = qlearning_dataset(env, terminate_on_end=True)
        else:
            dataset = qlearning_dataset(env)
        is_generalization_env = False
        args.action_scale = 1.0
        args.original_max_action = env.action_space.high[0] / args.action_scale

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

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
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        action_scale=args.action_scale
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        is_generalization_env=is_generalization_env
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()