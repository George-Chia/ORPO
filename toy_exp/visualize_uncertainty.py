import argparse
import os
import sys
import random

import gym

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
# from offlinerlkit.utils.recompute_reward_fns import recompute_reward_fn_halfcheetahjump,recompute_reward_fn_antangle
import pickle
import square_env
import random
import matplotlib.pyplot as plt



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mopo")
    parser.add_argument("--task", type=str, default="toy")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--penalty-coef", type=float, default=2.5)
    parser.add_argument("--uncertainty_mode", type=str, default='ensemble_std')
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--load-dynamics-path", type=str, default=None)
    parser.add_argument("--load-policy-path", type=str, default=None)
    parser.add_argument("--evaluation-only", default=None, action='store_true')

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:5" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
     # create env and dataset
     env = gym.make('square_env/SquareEnv-v0')
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
     # if is_generalization_env:
     #     dist = ScaledTanhDiagGaussian(
     #         latent_dim=getattr(actor_backbone, "output_dim"),
     #         output_dim=args.action_dim,
     #         unbounded=True,
     #         conditioned_sigma=True,
     #         scale_times=3.0,
     #     )
     # else:
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

     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

     if args.auto_alpha:
          target_entropy = args.target_entropy if args.target_entropy \
               else -np.prod(env.action_space.shape)

          args.target_entropy = target_entropy

          log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
          alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
          alpha = (target_entropy, log_alpha, alpha_optim)
     else:
          alpha = args.alpha

     # create dynamics
     load_dynamics_model = True if args.load_dynamics_path else False
     dynamics_model = EnsembleDynamicsModel(
          obs_dim=np.prod(args.obs_shape),
          action_dim=args.action_dim,
          hidden_dims=args.dynamics_hidden_dims,
          num_ensemble=args.n_ensemble,
          num_elites=args.n_elites,
          weight_decays=args.dynamics_weight_decay,
          device=args.device,
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
          uncertainty_mode=args.uncertainty_mode
     )

     if args.load_dynamics_path:
          dynamics.load(args.load_dynamics_path)
     
     # obs = np.random.uniform(-3, 3, (3, 2))
     # action = np.random.uniform(-1, 1, (10, 2))
     # print(dynamics.step(obs, action))
     
     plot_dict = {}
     
     # 方形边长
     side_length = 6
     # 步长
     step_length = 0.1
     # 计算方形的顶点坐标
     vertices = np.array([[-side_length/2, -side_length/2],
                         [-side_length/2, side_length/2],
                         [side_length/2, side_length/2],
                         [side_length/2, -side_length/2]])
     states = []
     # 遍历所有点
     # for x in np.arange(-side_length/2, side_length/2+step_length, step_length):
     #      for y in np.arange(-side_length/2, side_length/2+step_length, step_length):
     #           # 判断点是否在方形内部
     #           if np.all(vertices[0] <= [x, y]) and np.all(vertices[2] >= [x, y]):
     #                print(f"({x}, {y})")
     #                states.append([x,y])
                    # plot_dict[(x, y)] = dynamics.step(np.array([x, y]), np.array([0, 0]))
                    
     # x_vals = np.linspace(-side_length/2, side_length/2, int(side_length/step_length) + 1)
     # y_vals = np.linspace(-side_length/2, side_length/2, int(side_length/step_length) + 1)
     # states = np.array(states)
     # heatmap = np.zeros_like(np.array(states))
     # for i, x in enumerate(x_vals):
     #      for j, y in enumerate(y_vals):
     #           heatmap[i, j] = plot_dict[(x, y)]
               
     x_vals = np.linspace(-side_length/2, side_length/2, int(side_length/step_length) + 1)
     y_vals = np.linspace(-side_length/2, side_length/2, int(side_length/step_length) + 1)
     heatmap = np.zeros((len(x_vals), len(y_vals)))
     for i, x in enumerate(x_vals):
          for j, y in enumerate(y_vals):
               states = np.array([[x,y]]*10)
               actions = np.random.uniform(-1, 1, size=states.shape)
               heatmap[i, j] = dynamics.step(states,actions)[3]['uncertainty'].mean().item()
     
     plt.imshow(heatmap, cmap='hot', origin='lower', extent=[-side_length/2, side_length/2, -side_length/2, side_length/2])
     plt.colorbar()
     plt.title("Heatmap of Points Inside Square")
     # plt.xlabel("x")
     # plt.ylabel("y")
     # plt.xlabel("x", fontsize=18)
     # plt.ylabel("y", fontsize=18)
     plt.savefig('./uncertainty_fig2-b.png')


if __name__ == "__main__":
    train()   

