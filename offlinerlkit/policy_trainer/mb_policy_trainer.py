import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer,ReplayBufferPlus
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy


# model-based policy trainer
class MBPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: 'ReplayBuffer|ReplayBufferPlus',
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0,
        is_generalization_env: bool = False,
        evaluation_only: bool = False,
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger

        self._rollout_freq, self._rollout_batch_size, \
            self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self._is_generalization_env = is_generalization_env
        self.evaluation_only = evaluation_only

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        last_10_performance_Dynamics = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):
            if not self.evaluation_only:
                self.policy.train()

                pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
                for it in pbar:
                    if num_timesteps % self._rollout_freq == 0:
                        init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                        rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                        self.fake_buffer.add_batch(**rollout_transitions)
                        self.logger.log(
                            "num rollout transitions: {}, reward mean: {:.4f}".\
                                format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                        )
                        for _key, _value in rollout_info.items():
                            self.logger.logkv_mean("rollout_info/"+_key, _value)

                    real_sample_size = int(self._batch_size * self._real_ratio)
                    fake_sample_size = self._batch_size - real_sample_size
                    real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                    fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                    batch = {"real": real_batch, "fake": fake_batch}
                    loss = self.policy.learn(batch)
                    pbar.set_postfix(**loss)

                    for k, v in loss.items():
                        self.logger.logkv_mean(k, v)

                    # update the dynamics if necessary
                    if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                        dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                        for k, v in dynamics_update_info.items():
                            self.logger.logkv_mean(k, v)

                    num_timesteps += 1

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            if not self._is_generalization_env:
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                last_10_performance.append(norm_ep_rew_mean)
                self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            else:
                ep_rew_mean = ep_reward_mean
                ep_rew_std = ep_reward_std
                last_10_performance.append(ep_rew_mean)
                self.logger.logkv("eval/episode_reward", ep_rew_mean)
                self.logger.logkv("eval/episode_reward_std", ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.logkv("eval/last_10_performance", np.mean(last_10_performance))
            self.logger.logkv("eval/last_10_performance_std", np.std(last_10_performance))

            '''
            # evaluate current policy using dynamics
            eval_info_Dynamics = self._evaluate_Dynamics()
            ep_reward_mean_Dynamics, ep_reward_std_Dynamics = np.mean(eval_info_Dynamics["eval_Dynamics/episode_reward"]), np.std(eval_info_Dynamics["eval_Dynamics/episode_reward"])
            ep_length_mean_Dynamics, ep_length_std_Dynamics = np.mean(eval_info_Dynamics["eval_Dynamics/episode_length"]), np.std(eval_info_Dynamics["eval_Dynamics/episode_length"])
            ep_uncertainty_mean_Dynamics, ep_uncertainty_std_Dynamics = np.mean(eval_info_Dynamics["eval_Dynamics/episode_uncertainty"]), np.std(eval_info_Dynamics["eval_Dynamics/episode_uncertainty"])
            if not self._is_generalization_env:
                norm_ep_rew_mean_Dynamics = self.eval_env.get_normalized_score(ep_reward_mean_Dynamics) * 100
                norm_ep_rew_std_Dynamics = self.eval_env.get_normalized_score(ep_reward_std_Dynamics) * 100
                last_10_performance_Dynamics.append(norm_ep_rew_mean_Dynamics)
                self.logger.logkv("eval_Dynamics/normalized_episode_reward", norm_ep_rew_mean_Dynamics)
                self.logger.logkv("eval_Dynamics/normalized_episode_reward_std", norm_ep_rew_std_Dynamics)
            else:
                ep_rew_mean_Dynamics = ep_reward_mean_Dynamics
                ep_rew_std_Dynamics = ep_reward_std_Dynamics
                last_10_performance_Dynamics.append(ep_rew_mean_Dynamics)
                self.logger.logkv("eval_Dynamics/episode_reward", ep_rew_mean_Dynamics)
                self.logger.logkv("eval_Dynamics/episode_reward_std", ep_rew_std_Dynamics)
            self.logger.logkv("eval_Dynamics/episode_length", ep_length_mean_Dynamics)
            self.logger.logkv("eval_Dynamics/episode_length_std", ep_length_std_Dynamics)
            self.logger.logkv("eval_Dynamics/episode_uncertainty", ep_uncertainty_mean_Dynamics)
            self.logger.logkv("eval_Dynamics/episode_uncertainty_std", ep_uncertainty_std_Dynamics)
            self.logger.logkv("eval/last_10_performance_Dynamics", np.mean(last_10_performance_Dynamics))
            self.logger.logkv("eval/last_10_performance_Dynamics_std", np.std(last_10_performance_Dynamics))
            '''
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            if e % 100 == 0:
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, f"policy#{e}.pth"))


        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance),"last_10_performance_Dynamics":np.mean(last_10_performance_Dynamics)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            # self.eval_env.render()
            # if self._is_generalization_env:
            #     reward = recompute_reward_fn_halfcheetahjump(obs, action.flatten(), next_obs, reward)
            episode_reward += reward
            episode_length += 1

            obs = next_obs
            # print(num_episodes, ' - ', episode_length, ' - ', episode_reward)
            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()


            # time.sleep(0.01)
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    def _evaluate_Dynamics(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        obs = np.expand_dims(obs,axis=0)
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        episode_uncertainty = 0

        while num_episodes < self._eval_episodes:
            try:
                action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
            except Exception as e:
                print(e)
                print(obs)
            next_obs, penalized_reward, terminal, info = self.policy.dynamics.step(obs, action)
            # if self._is_generalization_env:
            #     reward = recompute_reward_fn_halfcheetahvel(obs, action, next_obs, reward)
            episode_reward += info["raw_reward"]
            episode_length += 1
            episode_uncertainty += info['uncertainty']

            obs = next_obs

            if terminal or episode_length>1000 or np.isnan(obs).sum()>0:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length, "episode_uncertainty":episode_uncertainty}
                )
                num_episodes += 1
                episode_reward, episode_length = 0, 0
                episode_uncertainty = 0
                obs = self.eval_env.reset()
                obs = np.expand_dims(obs, axis=0)

        return {
            "eval_Dynamics/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval_Dynamics/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval_Dynamics/episode_uncertainty": [ep_info["episode_uncertainty"] for ep_info in eval_ep_info_buffer],
        }