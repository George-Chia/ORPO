import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer, ReplayBufferPlus
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy_trainer.mb_policy_trainer import MBPolicyTrainer


from offlinerlkit.policy import OptimisticRolloutPolicy, OOMITD3BCPolicy, RandomRolloutPolicy


# model-based policy trainer
class MBPolicyOOMITrainer(MBPolicyTrainer):
    def __init__(
            self,
            policy,
            eval_env: gym.Env,
            real_buffer: ReplayBuffer,
            fake_buffer: ReplayBufferPlus,
            fake_buffer_rollout: ReplayBufferPlus,
            # rollout_policy_fake_buffer: ReplayBuffer,
            logger: Logger,
            rollout_setting: Tuple[int, int, int],
            epoch: int = 1000,
            step_per_epoch: int = 1000,
            batch_size: int = 256,
            real_ratio_rollout: float = 0.05,
            real_ratio_final: float = 0.5,
            eval_episodes: int = 10,
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            dynamics_update_freq: int = 0,
            rollout_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            rollout_length_rollout_policy: int = 1,
            final_policy_rollout_ratio_final: float = 0.0,
            # final_policy_rollout: bool = False,
            is_generalization_env: bool = False,
            evaluation_only: bool = False,
    ) -> None:
        super(MBPolicyOOMITrainer, self).__init__(
            policy = policy,
            eval_env = eval_env,
            real_buffer = real_buffer,
            fake_buffer = fake_buffer,
            logger = logger,
            rollout_setting =rollout_setting,
            dynamics_update_freq = dynamics_update_freq,
            epoch = epoch,
            step_per_epoch = step_per_epoch,
            batch_size = batch_size,
            real_ratio = real_ratio_final,
            eval_episodes = eval_episodes,
            lr_scheduler = lr_scheduler,
            is_generalization_env=is_generalization_env,
            evaluation_only=evaluation_only
        )
        # self.rollout_policy_fake_buffer = rollout_policy_fake_buffer
        self.rollout_lr_scheduler = rollout_lr_scheduler
        self._real_ratio_rollout = real_ratio_rollout
        # self._final_policy_rollout = final_policy_rollout
        self.fake_buffer_rollout = fake_buffer_rollout
        self._rollout_length_rollout_policy = rollout_length_rollout_policy
        self._final_policy_rollout_ratio_final = final_policy_rollout_ratio_final

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        last_10_performance_Dynamics = deque(maxlen=10)
        last_10_performance_rollout_policy = deque(maxlen=10)
        last_10_performance_rollout_policy_Dynamics = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):
            if not self.evaluation_only:
                self.policy.train()
                if isinstance(self.policy.rollout_policy,OptimisticRolloutPolicy):
                    self.policy.rollout_policy.train()

                pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
                for it in pbar:
                    if num_timesteps % self._rollout_freq == 0:
                        init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                        # if self._final_policy_rollout:
                        final_policy_rollout_transitions, final_policy_rollout_info = self.policy.rollout(init_obss, self._rollout_length) # this is for use oomi to implement combo
                        rollout_transitions, rollout_info = self.policy.rollout_policy.rollout(init_obss, self._rollout_length_rollout_policy)
                        # self.fake_buffer.add_batch(**rollout_transitions)
                        self.fake_buffer_rollout.add_batch(**rollout_transitions)
                        self.fake_buffer.add_batch(**final_policy_rollout_transitions)

                        for _key, _value in rollout_info.items():
                            self.logger.logkv_mean("rollout_info/" + _key, _value)

                        for _key, _value in final_policy_rollout_info.items():
                            self.logger.logkv_mean("rollout_info_final_policy/" + _key, _value)

                    real_sample_size_final = int(self._batch_size * self._real_ratio)
                    final_policy_fake_sample_size_final = int(self._batch_size * self._final_policy_rollout_ratio_final)
                    rollout_policy_fake_sample_size_final = self._batch_size - real_sample_size_final - final_policy_fake_sample_size_final
                    real_batch_final = self.real_buffer.sample(batch_size=real_sample_size_final)
                    final_policy_fake_batch_final,final_policy_batch_indexes_final = self.fake_buffer.sample(
                        batch_size=final_policy_fake_sample_size_final)
                    rollout_policy_fake_batch_final, rollout_policy_batch_indexes_final = self.fake_buffer_rollout.sample(
                        batch_size=rollout_policy_fake_sample_size_final)
                    mix_batch_final = {"real": real_batch_final, "final_policy_fake": final_policy_fake_batch_final, "rollout_policy_fake":rollout_policy_fake_batch_final}
                    loss = self.policy.learn(mix_batch_final)

                    # TODO: delete comment
                    '''
                    if isinstance(self.policy.rollout_policy, OptimisticRolloutPolicy):
                        real_sample_size_rollout = int(self._batch_size * self._real_ratio_rollout)
                        fake_sample_size_rollout = self._batch_size - real_sample_size_rollout
                        real_batch_rollout = self.real_buffer.sample(batch_size=real_sample_size_rollout)
                        fake_batch_rollout = self.fake_buffer_rollout.sample_bonus_reward(batch_size=fake_sample_size_rollout)
                        mix_batch_rollout = {"real": real_batch_rollout, "fake": fake_batch_rollout}
                        rollout_policy_loss = self.policy.rollout_policy.learn(mix_batch_rollout, final_policy=self.policy)

                    # bonus_fake_batch = self.fake_buffer.sample_bonus_reward(batch_size=fake_sample_size_rollout,batch_indexes=batch_indexes)
                    # bonus_mix_batch =  {"real": real_batch_rollout, "fake": bonus_fake_batch}
                    # rollout_policy_loss = self.policy.rollout_policy.learn(bonus_mix_batch, final_policy=self.policy)
                    pbar.set_postfix(**loss)
                    # pbar.set_postfix(**rollout_policy_loss)

                    for k, v in loss.items():
                        self.logger.logkv_mean(k, v)
                    if isinstance(self.policy.rollout_policy, OptimisticRolloutPolicy) and rollout_policy_loss is not None:
                        for k, v in rollout_policy_loss.items():
                            self.logger.logkv_mean(k, v)
                    '''

                    # update the dynamics if necessary
                    if 0 < self._dynamics_update_freq and (num_timesteps + 1) % self._dynamics_update_freq == 0:
                        dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                        for k, v in dynamics_update_info.items():
                            self.logger.logkv_mean(k, v)

                    num_timesteps += 1



                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # TODO: delete comment
                # if isinstance(self.policy.rollout_policy, OptimisticRolloutPolicy):
                #     if self.rollout_lr_scheduler is not None:
                #         self.rollout_lr_scheduler.step()

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
                self.logger.logkv("eval/last_10_performance", np.mean(last_10_performance))
                self.logger.logkv("eval/last_10_performance_std", np.std(last_10_performance))
            else:
                ep_rew_mean = ep_reward_mean
                ep_rew_std = ep_reward_std
                last_10_performance.append(ep_rew_mean)
                self.logger.logkv("eval/episode_reward", ep_rew_mean)
                self.logger.logkv("eval/episode_reward_std", ep_rew_std)
                self.logger.logkv("eval/last_10_performance", np.mean(last_10_performance))
                self.logger.logkv("eval/last_10_performance_std", np.std(last_10_performance))
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)

            if not self.evaluation_only and not self._is_generalization_env:
                # if not self._is_generalization_env:
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
            # evaluate rollout policy
            eval_rollout_policy_info = self._evaluate_rollout_policy()
            ep_rollout_policy_reward_mean, ep_rollout_policy_reward_std = np.mean(eval_rollout_policy_info["eval_rollout_policy/episode_reward"]), np.std(
                eval_rollout_policy_info["eval_rollout_policy/episode_reward"])
            ep_rollout_policy_length_mean, ep_rollout_policy_length_std = np.mean(eval_rollout_policy_info["eval_rollout_policy/episode_length"]), np.std(
                eval_rollout_policy_info["eval_rollout_policy/episode_length"])
            if not self._is_generalization_env:
                norm_ep_rollout_policy_rew_mean = self.eval_env.get_normalized_score(ep_rollout_policy_reward_mean) * 100
                norm_ep_rollout_policy_rew_std = self.eval_env.get_normalized_score(ep_rollout_policy_reward_std) * 100
                last_10_performance_rollout_policy.append(norm_ep_rollout_policy_rew_mean)
                self.logger.logkv("eval_rollout_policy/normalized_episode_reward", norm_ep_rollout_policy_rew_mean)
                self.logger.logkv("eval_rollout_policy/normalized_episode_reward_std", norm_ep_rollout_policy_rew_std)
            else:
                ep_rollout_policy_rew_mean = ep_rollout_policy_reward_mean
                ep_rollout_policy_rew_std = ep_rollout_policy_reward_std
                last_10_performance_rollout_policy.append(ep_rollout_policy_rew_mean)
                self.logger.logkv("eval_rollout_policy/episode_reward", ep_rollout_policy_rew_mean)
                self.logger.logkv("eval_rollout_policy/episode_reward_std", ep_rollout_policy_rew_std)
            self.logger.logkv("eval_rollout_policy/episode_length", ep_rollout_policy_length_mean)
            self.logger.logkv("eval_rollout_policy/episode_length_std", ep_rollout_policy_length_std)
            self.logger.logkv("eval/last_10_performance_rollout_policy", np.mean(last_10_performance_rollout_policy))
            self.logger.logkv("eval/last_10_performance_rollout_policy_std", np.std(last_10_performance_rollout_policy))

            if not self._is_generalization_env:
            # evaluate rollout policy using dynamics
                eval_rollout_policy_info_Dynamics = self._evaluate_rollout_policy_Dynamics()
                ep_rollout_policy_reward_mean_Dynamics, ep_rollout_policy_reward_std_Dynamics = np.mean(eval_rollout_policy_info_Dynamics["eval_rollout_policy_Dynamics/episode_reward"]), np.std(
                    eval_rollout_policy_info_Dynamics["eval_rollout_policy_Dynamics/episode_reward"])
                ep_rollout_policy_length_mean_Dynamics, ep_rollout_policy_length_std_Dynamics = np.mean(eval_rollout_policy_info_Dynamics["eval_rollout_policy_Dynamics/episode_length"]), np.std(
                    eval_rollout_policy_info_Dynamics["eval_rollout_policy_Dynamics/episode_length"])
                ep_rollout_policy_uncertainty_mean_Dynamics, ep_rollout_policy_uncertainty_std_Dynamics = np.mean(
                    eval_rollout_policy_info_Dynamics["eval_rollout_policy_Dynamics/episode_uncertainty"]), np.std(
                    eval_rollout_policy_info_Dynamics["eval_rollout_policy_Dynamics/episode_uncertainty"])
                if not self._is_generalization_env:
                    norm_ep_rollout_policy_rew_mean_Dynamics = self.eval_env.get_normalized_score(ep_rollout_policy_reward_mean_Dynamics) * 100
                    norm_ep_rollout_policy_rew_std_Dynamics = self.eval_env.get_normalized_score(ep_rollout_policy_reward_std_Dynamics) * 100
                    last_10_performance_rollout_policy.append(norm_ep_rollout_policy_rew_mean_Dynamics)
                    self.logger.logkv("eval_rollout_policy_Dynamics/normalized_episode_reward", norm_ep_rollout_policy_rew_mean_Dynamics)
                    self.logger.logkv("eval_rollout_policy_Dynamics/normalized_episode_reward_std", norm_ep_rollout_policy_rew_std_Dynamics)
                else:
                    ep_rollout_policy_rew_mean_Dynamics = ep_rollout_policy_reward_mean_Dynamics
                    ep_rollout_policy_rew_std_Dynamics = ep_rollout_policy_reward_std_Dynamics
                    last_10_performance_rollout_policy_Dynamics.append(ep_rollout_policy_rew_mean_Dynamics)
                    self.logger.logkv("eval_rollout_policy_Dynamics/episode_reward",
                                      ep_rollout_policy_rew_mean_Dynamics)
                    self.logger.logkv("eval_rollout_policy_Dynamics/episode_reward_std",
                                      ep_rollout_policy_rew_std_Dynamics)
                self.logger.logkv("eval_rollout_policy_Dynamics/episode_length", ep_rollout_policy_length_mean_Dynamics)
                self.logger.logkv("eval_rollout_policy_Dynamics/episode_length_std", ep_rollout_policy_length_std_Dynamics)
                self.logger.logkv("eval_rollout_policy_Dynamics/episode_uncertainty", ep_rollout_policy_uncertainty_mean_Dynamics)
                self.logger.logkv("eval_rollout_policy_Dynamics/episode_uncertainty_std", ep_rollout_policy_uncertainty_std_Dynamics)
                self.logger.logkv("eval/last_10_performance_rollout_policy_Dynamics", np.mean(last_10_performance_rollout_policy_Dynamics))
                self.logger.logkv("eval/last_10_performance_rollout_policy_Dynamics_std", np.std(last_10_performance_rollout_policy_Dynamics))
            '''

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])

            # save checkpoint
            if e % 100 == 0:
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, f"policy#{e}.pth"))
                torch.save(self.policy.rollout_policy.state_dict(),
                           os.path.join(self.logger.checkpoint_dir, f"rollout_policy#{e}.pth"))



        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate_rollout_policy(self) -> Dict[str, List[float]]:
        if isinstance(self.policy.rollout_policy, OptimisticRolloutPolicy):
            self.policy.rollout_policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.rollout_policy.select_action_using_normalized_obs_if_needed(obs.reshape(1, -1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            # if self._is_generalization_env:
            #     reward = recompute_reward_fn_halfcheetahjump(obs, action.flatten(), next_obs, reward)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes += 1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

        return {
            "eval_rollout_policy/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval_rollout_policy/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    def _evaluate_rollout_policy_Dynamics(self) -> Dict[str, List[float]]:
        if isinstance(self.policy.rollout_policy, OptimisticRolloutPolicy):
            self.policy.rollout_policy.eval()
        obs = self.eval_env.reset()
        obs = np.expand_dims(obs, axis=0)
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        episode_uncertainty = 0

        while num_episodes < self._eval_episodes:
            try:
                action = self.policy.rollout_policy.select_action_using_normalized_obs_if_needed(obs.reshape(1, -1),deterministic=True)
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

            if terminal or episode_length>1000:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length, "episode_uncertainty":episode_uncertainty}
                )
                num_episodes += 1
                episode_reward, episode_length = 0, 0
                episode_uncertainty = 0
                obs = self.eval_env.reset()
                obs = np.expand_dims(obs, axis=0)

        return {
            "eval_rollout_policy_Dynamics/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval_rollout_policy_Dynamics/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval_rollout_policy_Dynamics/episode_uncertainty": [ep_info["episode_uncertainty"] for ep_info in eval_ep_info_buffer]
        }