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
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="orpo")
    parser.add_argument("--task", type=str, default="toy")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")

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


     # Attention: normalize_obs() will change the buffer!!!!!!';;;normalize_obs is beneficial except for the hopper tasks
     # if is_normalize_obs:
     #     obs_mean, obs_std = real_buffer.normalize_obs()
     #     obs_scaler = StandardScaler(mu=obs_mean, std=obs_std)
     # else:
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
     elif args.rollout_policy_type == "vae":
          load_vae_path = ''
          rollout_policy = VaeRolloutPolicy(dynamics,action_space=env.action_space,scaler=obs_scaler,device=args.device,
                                             state_dim=args.obs_shape[0],action_dim=args.action_dim,max_action=args.original_max_action*args.action_scale,load_vae_path=load_vae_path)
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


     args.load_policy_path = './policy#10.pth'
     if args.load_policy_path:
          policy.load(args.load_policy_path)
          # state_dict = torch.load(args.load_policy_path)
          # policy.load_state_dict(state_dict)
     state = env.reset()
     trajectories = []
     for i in range(5000):
          trajectories.append(state)
          action = policy.select_action(state)
          state, reward, done, _ = env.step(action)
          if done:
               # break
               state = env.reset()
     
     # 创建散点图
     fig, ax = plt.subplots(figsize=(8, 8))
     # 添加点到散点图中
     x = [point[0] for point in trajectories]
     y = [point[1] for point in trajectories]
     ax.scatter(x, y, s=50,alpha=0.6,marker='x')
     
     # # 绘制圆
     # circle = patches.Circle((3, 3), 1, linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.6)
     # ax.add_patch(circle)
     
     
     ax.tick_params(axis='x', labelsize=18)
     ax.tick_params(axis='y', labelsize=18)  
     # x_line = np.linspace(-3, 3, 100)
     # y_line1 = -x_line + 0.25
     # y_line2 = -x_line - 0.25
     # ax.plot(x_line, y_line1, color='green', linewidth=1.5)
     # ax.plot(x_line, y_line2, color='green', linewidth=1.5)
     
     # 绘制正方形
     square = patches.Rectangle((-3, -3), 6, 6, linewidth=2, edgecolor='black', facecolor='none')
     ax.add_patch(square)
     # 绘制直线
     x_line = np.linspace(-3, 3, 100)
     y_line1 = -x_line + 0.25
     y_line2 = -x_line - 0.25
     ax.plot(x_line, y_line1, color='darkblue',linewidth=1.5)
     ax.plot(x_line, y_line2, color='darkblue', linewidth=1.5)
     
     
     # 设置坐标轴范围
     ax.set_xlim(-3, 3)
     ax.set_ylim(-3, 3)
     # 显示散点图
     plt.savefig('./trajectories_orpo_fig2-d.png')


if __name__ == "__main__":
    train()