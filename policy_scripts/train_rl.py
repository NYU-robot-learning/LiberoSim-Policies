#!/usr/bin/env python3

import warnings
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import hydra
import torch
import numpy as np
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader
from video import TrainVideoRecorder, VideoRecorder

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
	obs_shape = {}
	for key in cfg.suite.pixel_keys:
		obs_shape[key] = obs_spec[key].shape
	if cfg.use_proprio:
		obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
	obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
	cfg.agent.obs_shape = obs_shape
	cfg.agent.action_shape = action_spec.shape
	return hydra.utils.instantiate(cfg.agent)

class WorkspaceIL:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		
		# load data
		dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
		self.expert_replay_loader = make_expert_replay_loader(
			dataset_iterable, self.cfg.batch_size)
		self.expert_replay_iter = iter(self.expert_replay_loader)
		
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, mode='rl')
		# create envs
		self.cfg.suite.task_make_fn.max_episode_len = self.expert_replay_loader.dataset._max_episode_len * self.expert_replay_loader.dataset.subsample
		self.cfg.suite.task_make_fn.max_state_dim = self.expert_replay_loader.dataset._max_state_dim
		self.train_env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)
		self.eval_env, _ = hydra.utils.call(self.cfg.suite.task_make_fn)
		
		self.setup()

		# create agent
		self.agent = make_agent(self.train_env[0].observation_spec(),
								self.train_env[0].action_spec(), cfg)
		
		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0

		if self.cfg.irl:
			self.expert_demo = []
			for env_idx in range(len(self.train_env)):
				episodes = self.expert_replay_loader.dataset._episodes[env_idx]
				self.expert_demo.append([])
				for episode in episodes:
					self.expert_demo[env_idx].append(np.transpose(episode['observation'], (0,3,1,2)))
			
	def setup(self):
		# create obs spec
		obs_spec = []
		env_obs_spec = self.train_env[0].observation_spec()
		self.obs_keys = self.cfg.suite.pixel_keys if self.cfg.obs_type == 'pixels' else [self.cfg.suite.feature_key]
		if self.cfg.obs_type == 'pixels' and self.cfg.use_proprio:
			self.obs_keys.append(self.cfg.suite.proprio_key)
		for key in self.obs_keys:
			obs_spec.append(specs.Array(
								env_obs_spec[key].shape, 
								env_obs_spec[key].dtype, 
								key))

		# create replay buffer		
		data_specs = [
			*obs_spec,
			self.train_env[0].action_spec(),
			specs.Array((1,), np.float32, 'reward'),
			specs.Array((1,), np.float32, 'discount')
		]

		self.replay_storage = ReplayBufferStorage(data_specs, self.obs_keys,
												  self.work_dir / 'buffer')

		self.replay_loader = make_replay_loader(
			self.work_dir / 'buffer', self.cfg.replay_buffer_size, self.cfg.batch_size, 
			self.cfg.replay_buffer_num_workers, self.cfg.suite.save_snapshot, 
			self.cfg.nstep, self.cfg.suite.discount, self.obs_keys, self.cfg.suite.history,
			self.cfg.suite.history_len, self.cfg.temporal_agg, self.cfg.num_queries,
			self.cfg.suite.task_make_fn.max_episode_len, self.cfg.suite.task_make_fn.max_state_dim)

		self._replay_iter = None

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(
			self.work_dir if self.cfg.save_train_video else None)

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.suite.action_repeat
	
	@property
	def replay_iter(self):
		if self._replay_iter is None:
			self._replay_iter = iter(self.replay_loader)
		return self._replay_iter

	def eval(self):
		episode_rewards = []
		for env_idx in range(len(self.eval_env)):
			episode, total_reward = 0, 0
			eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

			while eval_until_episode(episode):
				time_step = self.eval_env[env_idx].reset()
				self.agent.buffer_reset()
				step = 0

				if episode == 0:
					self.video_recorder.init(self.eval_env[env_idx], enabled=True)
				
				# plot obs with cv2
				while not time_step.last():
					with torch.no_grad(), utils.eval_mode(self.agent):
						action = self.agent.act(time_step.observation,
												step,
												self.global_step,
												eval_mode=True)
					time_step = self.eval_env[env_idx].step(action)
					self.video_recorder.record(self.eval_env[env_idx])
					total_reward += time_step.reward
					step += 1

				episode += 1
			self.video_recorder.save(f'{self.global_frame}_env{env_idx}.mp4')
			episode_rewards.append(total_reward / episode)
			
		with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
			for env_idx, reward in enumerate(episode_rewards):
				log(f'episode_reward_env{env_idx}', reward)
			log('episode_reward', np.mean(episode_rewards))
			log('episode_length', step * self.cfg.suite.action_repeat / episode)
			log('episode', self.global_episode)
			log('step', self.global_step)

	def train(self):
		# predicates
		train_until_step = utils.Until(self.cfg.suite.num_train_steps,
									   self.cfg.suite.action_repeat)
		seed_until_step = utils.Until(self.cfg.suite.num_seed_steps,
									  self.cfg.suite.action_repeat)
		eval_every_step = utils.Every(self.cfg.suite.eval_every_steps,
									  self.cfg.suite.action_repeat)

		episode_step, episode_reward = 0, 0
		obs_key = self.cfg.suite.pixel_keys[0] if self.cfg.obs_type == 'pixels' else self.cfg.suite.feature_key # only one modality
		render_key = self.cfg.suite.pixel_keys[0]

		time_steps = list()
		observations = list()
		actions = list()

		env_idx = np.random.randint(0, len(self.train_env))
		time_step = self.train_env[env_idx].reset()
		self.agent.buffer_reset()
		time_steps.append(time_step)
		observations.append(time_step.observation[obs_key])
		actions.append(time_step.action)

		if self.cfg.irl:
			if self.agent.auto_rew_scale:
				self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

		self.train_video_recorder.init(time_step.observation[render_key])
		metrics = None
		while train_until_step(self.global_step):
			if time_step.last():
				self._global_episode += 1
				if self._global_episode % 10 == 0:
					self.train_video_recorder.save(f'{self.global_frame}_env{env_idx}.mp4')
				
				observations = np.stack(observations, 0)
				actions = np.stack(actions, 0)
				if self.cfg.irl:
					new_rewards = self.agent.ot_rewarder(
						observations, self.expert_demo[env_idx], self.global_step)
					new_rewards_sum = np.sum(new_rewards)

					if self.agent.auto_rew_scale:
						if self._global_episode == 1:
							self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(
								np.abs(new_rewards_sum))
							new_rewards = self.agent.ot_rewarder(
								observations, self.expert_demo[env_idx], self.global_step)
							new_rewards_sum = np.sum(new_rewards)
					
				for i, elt in enumerate(time_steps):
					obs = {}
					for key in self.obs_keys:
						obs[key] = elt.observation[key]
					elt = elt._replace(observation=obs)
					if self.cfg.irl:
						elt = elt._replace(reward=new_rewards[i])
					self.replay_storage.add(elt)
				
				if metrics is not None:
					# log stats
					elapsed_time, total_time = self.timer.reset()
					episode_frame = episode_step * self.cfg.suite.action_repeat
					with self.logger.log_and_dump_ctx(self.global_frame,
													  ty='train') as log:
						log('fps', episode_frame / elapsed_time)
						log('total_time', total_time)
						log('episode_reward', episode_reward)
						log('episode_length', episode_frame)
						log('episode', self.global_episode)
						log('buffer_size', len(self.replay_storage))
						log('step', self.global_step)
						if self.cfg.irl:
							log('imitation_reward', new_rewards_sum)
				
				# try to save snapshot
				if self.cfg.suite.save_snapshot:
					self.save_snapshot()

				# reset env
				time_steps = list()
				observations = list()
				actions = list()

				env_idx = np.random.randint(0, len(self.train_env))
				time_step = self.train_env[env_idx].reset()
				self.agent.buffer_reset()
				time_steps.append(time_step)
				observations.append(time_step.observation[obs_key])
				actions.append(time_step.action)

				self.train_video_recorder.init(time_step.observation[render_key])
				episode_step, episode_reward = 0, 0

			# try to evaluate
			if eval_every_step(self.global_step):
				self.logger.log('eval_total_time', self.timer.total_time(),
								self.global_frame)
				self.eval()
			
			# sample action
			with torch.no_grad(), utils.eval_mode(self.agent):
				action = self.agent.act(time_step.observation,
										episode_step,
										self.global_step,
										eval_mode=False)
			
				
			# try to update agent
			if not seed_until_step(self.global_step):
				# Update
				metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, 
											self.global_step, self.cfg.bc_regularize)
				self.logger.log_metrics(metrics, self.global_frame, ty='train')

			# take env step
			time_step = self.train_env[env_idx].step(action)
			episode_reward += time_step.reward

			time_steps.append(time_step)
			observations.append(time_step.observation[obs_key])
			actions.append(time_step.action)

			self.train_video_recorder.record(time_step.observation[render_key])
			episode_step += 1
			self._global_step += 1

	def save_snapshot(self):
		snapshot = self.work_dir / 'snapshot.pt'
		self.agent.clear_buffers()
		keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		# payload.update(self.agent.save_snapshot())
		with snapshot.open('wb') as f:
			torch.save(payload, f)

	def load_snapshot(self, snapshot):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		# agent_payload = {}
		# for k, v in payload.items():
		# 	if k not in self.__dict__:
		# 		agent_payload[k] = v
		# self.agent.load_snapshot(agent_payload)
		# self.agent.load_snapshot_eval(agent_payload)
		self.agent.load_snapshot(payload['agent'])

@hydra.main(config_path='cfgs', config_name='config_rl')
def main(cfg):
	from train_rl import WorkspaceIL as W
	root_dir = Path.cwd()
	workspace = W(cfg)
	
	# Load weights
	if cfg.load_bc:
		snapshot = Path(cfg.bc_weight)
		if snapshot.exists():
			print(f'resuming bc: {snapshot}')
			workspace.load_snapshot(snapshot)
	
	workspace.train()


if __name__ == '__main__':
	main()
