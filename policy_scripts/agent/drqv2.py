import einops
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

import utils
from agent.networks.rgb_modules import *
from agent.networks.policy_head import *
from agent.networks.gpt import GPT, GPTConfig
from agent.networks.mlp import MLP
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance
import time
import copy

class Actor(nn.Module):
	def __init__(self, repr_dim, act_dim, hidden_dim, policy_type='gpt', 
			  	 policy_head='deterministic', num_feat_per_step=1):
		super().__init__()

		self._policy_type = policy_type
		hidden_dim = 256 if policy_type=='gpt' else hidden_dim
		self._act_dim = act_dim
		self._num_feat_per_step = num_feat_per_step
		
		# GPT model
		if policy_type == 'gpt':
			self._policy = GPT(
				GPTConfig(
					block_size=300, #30,
					input_dim=repr_dim,
					output_dim=hidden_dim,
					# n_layer=6,
					# n_head=6,
					# n_embd=120,
					n_layer=8,
					n_head=4,
					n_embd=hidden_dim,
				)
			)
		elif policy_type == 'mlp':
			self._policy = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
										nn.ReLU(inplace=True),
										nn.Linear(hidden_dim, hidden_dim),
										nn.ReLU(inplace=True))	
		
		head = DeterministicHead if policy_head=='deterministic' else GMMHead
		self._action_head = head(hidden_dim, self._act_dim, num_layers=0)
		
		self.apply(utils.weight_init)

	def forward(self, obs, stddev):
		
		# process observation
		features = self._policy(obs)
		
		# Filter actions - take 1 action for all tokens per time step 
		# and predict extra actions of action chunking
		features = self._policy(obs)[:, ::self._num_feat_per_step]
		# action head
		pred_action = self._action_head(features, stddev)

		return pred_action


class Critic(nn.Module):
	def __init__(self, repr_dim, action_shape, hidden_dim):
		super().__init__()

		self.action_repeat = repr_dim // action_shape[0]

		self.Q1 = nn.Sequential(
			nn.Linear(repr_dim + action_shape[0] * self.action_repeat, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.Q2 = nn.Sequential(
			nn.Linear(repr_dim + action_shape[0] * self.action_repeat, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.apply(utils.weight_init)

	def forward(self, obs, action):
		action = action.repeat(1, self.action_repeat)
		h_action = torch.cat([obs, action], dim=-1)
		q1 = self.Q1(h_action)
		q2 = self.Q2(h_action)

		return q1, q2


class DrQv2Agent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, critic_target_tau, 
			  	 num_expl_steps, update_every_steps, stddev_schedule, stddev_clip, use_tb, 
				 augment, obs_type, encoder_type, policy_type, policy_head, pixel_keys, 
				 proprio_key, feature_key, use_proprio, bc_weight_type, bc_weight_schedule, 
				 rewards, sinkhorn_rew_scale, update_target_every, auto_rew_scale, auto_rew_scale_factor, 
				 norm, history, history_len, eval_history_len, separate_encoders, temporal_agg,
				 max_episode_len, num_queries):
		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.update_every_steps = update_every_steps
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.augment = augment
		self.obs_type = obs_type
		self.encoder_type = encoder_type
		self.policy_head = policy_head
		self.use_proprio = use_proprio
		self.norm = norm
		self.bc_weight_type = bc_weight_type
		self.bc_weight_schedule = bc_weight_schedule
		self.history_len = history_len if history else 1
		self.eval_history_len = eval_history_len if history else 1
		self.separate_encoders = separate_encoders

		assert policy_head == 'deterministic', "Only deterministic policy head is supported for now"

		# actor_parameters
		self._act_dim = action_shape[0]

		# For irl
		self.rewards = rewards
		self.sinkhorn_rew_scale = sinkhorn_rew_scale
		self.update_target_every = update_target_every
		self.auto_rew_scale = auto_rew_scale
		self.auto_rew_scale_factor = auto_rew_scale_factor

		# keys
		if obs_type == 'pixels':
			self.pixel_keys = pixel_keys
			self.proprio_key = proprio_key
		else:
			self.feature_key = feature_key
		
		# action chunking params
		self.temporal_agg = temporal_agg
		self.max_episode_len = max_episode_len
		self.num_queries = num_queries

		# number of inputs per time step
		if policy_type == 'mlp' or obs_type == 'features':
			num_feat_per_step = 1
		elif obs_type == 'pixels':
			num_feat_per_step = len(self.pixel_keys)
			if use_proprio:
				num_feat_per_step += 1
	
		# observation params
		if obs_type == 'pixels':
			if use_proprio:
				proprio_shape = obs_shape[self.proprio_key]
			obs_shape = obs_shape[self.pixel_keys[0]]
		else:
			obs_shape = obs_shape[self.feature_key]

		# models
		if obs_type == 'pixels':
			if self.separate_encoders:
				self.encoder = {}
			if self.encoder_type == 'base':
				if self.separate_encoders:
					for key in self.pixel_keys:
						self.encoder[key] = BaseEncoder(obs_shape).to(device)
						self.repr_dim = self.encoder[key].repr_dim
				else:
					self.encoder = BaseEncoder(obs_shape).to(device)
					self.repr_dim = self.encoder.repr_dim
			elif self.encoder_type == 'resnet':
				if self.separate_encoders:
					for key in self.pixel_keys:
						self.encoder[key] = ResnetEncoder(obs_shape, 512, language_fusion="none").to(device)
				else:
					self.encoder = ResnetEncoder(obs_shape, 512, language_fusion="none").to(device)
				self.repr_dim = 512
				# self.norm = True
			elif self.encoder_type == 'patch':
				pass
		else:
			self.encoder = MLP(obs_shape[0], hidden_channels=[512, 512]).to(device)
			self.repr_dim = 512
		self.encoder_target = copy.deepcopy(self.encoder)

		# projector for proprioceptive features
		if obs_type == 'pixels' and use_proprio:
			self.proprio_projector = MLP(proprio_shape[0], hidden_channels=[self.repr_dim, self.repr_dim]).to(device)
			self.proprio_projector.apply(utils.weight_init)

		if obs_type == 'pixels' and policy_type == 'mlp':
			new_repr_dim = self.repr_dim * len(self.pixel_keys)
			if use_proprio:
				new_repr_dim += self.repr_dim
			self.repr_dim = new_repr_dim

		# actor
		action_dim = self._act_dim * self.num_queries if self.temporal_agg else self._act_dim
		self.actor = Actor(self.repr_dim, action_dim, hidden_dim, policy_type, 
					 	   policy_head, num_feat_per_step).to(device)
		
		self.critic = Critic(self.repr_dim, action_shape, hidden_dim).to(device)
		self.critic_target = Critic(self.repr_dim, action_shape, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		# encoder
		if self.separate_encoders:
			params = []
			for key in self.pixel_keys:
				params += list(self.encoder[key].parameters())
		else:
			params = list(self.encoder.parameters())
		self.encoder_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
		# self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_opt, T_max=5e5)
		# proprio
		if obs_type == 'pixels' and self.use_proprio:
			self.proprio_opt = torch.optim.AdamW(self.proprio_projector.parameters(), lr=lr, weight_decay=1e-4)
			# self.proprio_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.proprio_opt, T_max=5e5)
		# actor
		params = list(self.actor.parameters())
		self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
		# self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=5e5)
		# critic
		self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)
		# self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_opt, T_max=5e5)

		# augmentations
		if obs_type == 'pixels' and self.norm:
			if self.encoder_type=='small':
				MEAN = torch.tensor([0.0, 0.0, 0.0])
				STD = torch.tensor([1.0, 1.0, 1.0])
			elif self.encoder_type=='resnet' or self.norm:
				MEAN = torch.tensor([0.485, 0.456, 0.406])
				STD = torch.tensor([0.229, 0.224, 0.225])
			self.customAug = T.Compose([
								T.Normalize(
									mean=MEAN,
									std=STD)])
			# normalize = T.Normalize(mean=MEAN, std=STD)

		# data augmentation
		if obs_type == 'pixels' and self.augment:
			# self.aug = utils.RandomShiftsAug(pad=4)
			self.test_aug = T.Compose([
				T.ToPILImage(),
				T.ToTensor()
			])

		self.train()
		self.critic_target.train()

	def __repr__(self):
		return "drqv2"
	
	def train(self, training=True):
		self.training = training
		# encoder
		if self.obs_type == "pixels" and self.separate_encoders:
			for key in self.pixel_keys:
				self.encoder[key].train(training)
		else:
			self.encoder.train(training)
		# proprio
		if self.obs_type == "pixels" and self.use_proprio:
			self.proprio_projector.train(training)

		self.actor.train(training)
		self.critic.train(training)

	def buffer_reset(self):
		if self.obs_type == 'pixels':
			self.observation_buffer = {}
			for key in self.pixel_keys:
				self.observation_buffer[key] = deque(maxlen=self.eval_history_len)
			if self.use_proprio:
				self.proprio_buffer = deque(maxlen=self.eval_history_len)
		else:
			self.observation_buffer = deque(maxlen=self.eval_history_len)

		# temporal aggregation
		if self.temporal_agg:
			self.all_time_actions = torch.zeros([self.max_episode_len, self.max_episode_len+self.num_queries, self._act_dim]).to(self.device)

	def clear_buffers(self):
		del self.observation_buffer
		if self.obs_type == 'pixels' and self.use_proprio:
			del self.proprio_buffer
		if self.temporal_agg:
			del self.all_time_actions

	def act(self, obs, step, global_step, eval_mode=False):
		if self.obs_type == 'pixels':
			# add to buffer
			features = []
			for key in self.pixel_keys:
				self.observation_buffer[key].append(self.test_aug(obs[key].transpose(1,2,0)).numpy())
				pixels = torch.as_tensor(np.array(self.observation_buffer[key]), device=self.device).float()
				pixels = self.customAug(pixels / 255.0) if self.norm else pixels
				#encoder
				pixels = self.encoder[key](pixels) if self.separate_encoders else self.encoder(pixels)
				features.append(pixels)
			if self.use_proprio:
				self.proprio_buffer.append(obs[self.proprio_key])
				proprio = torch.as_tensor(np.array(self.proprio_buffer), device=self.device).float()
				proprio = self.proprio_projector(proprio)
				features.append(proprio)
			features = torch.cat(features, dim=-1).view(-1, self.repr_dim)
		else:
			self.observation_buffer.append(obs[self.feature_key])
			features = torch.as_tensor(np.array(self.observation_buffer), device=self.device).float()
			features = self.encoder(features)
			
		stddev = utils.schedule(self.stddev_schedule, global_step)
		action = self.actor(features.unsqueeze(0), stddev)
		if eval_mode:
			action = action.mean
		else:
			action = action.sample(clip=None)
			if global_step < self.num_expl_steps:
				action.uniform_(-1.0, 1.0)
		
		if self.temporal_agg:
			action = action.view(-1, self.num_queries, self._act_dim)
			self.all_time_actions[[step], step : step + self.num_queries] = action[-1:]
			actions_for_curr_step = self.all_time_actions[:, step]
			actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
			actions_for_curr_step = actions_for_curr_step[actions_populated]
			k = 0.01
			exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
			exp_weights = exp_weights / exp_weights.sum()
			exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
			action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
			return action[0].cpu().float().numpy()
		else:
			return action[0, -1].cpu().float().numpy()

	def update_critic(self, features, action, reward, discount, next_features, step):
		metrics = dict()

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			dist = self.actor(next_features, stddev)
			next_action = dist.sample(clip=self.stddev_clip)
			target_Q1, target_Q2 = self.critic_target(next_features[:, -1], next_action[:, -1])
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		Q1, Q2 = self.critic(features[:, -1], action[:, -1])

		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		# optimize encoder and critic
		if self.obs_type == "pixels" and self.use_proprio:
			self.proprio_opt.zero_grad(set_to_none=True)
		self.encoder_opt.zero_grad(set_to_none=True)
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
		self.encoder_opt.step()
		if self.obs_type == "pixels" and self.use_proprio:
			self.proprio_opt.step()

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()
			
		return metrics

	def update_actor(self, features, features_bc, features_qfilter, action_bc, bc_regularize, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(features, stddev)
		action = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		Q1, Q2 = self.critic(features[:, -1], action[:, -1])
		Q = torch.min(Q1, Q2)

		# Compute bc weight
		if not bc_regularize:
			bc_weight = 0.0
		elif self.bc_weight_type == "linear":
			bc_weight = utils.schedule(self.bc_weight_schedule, step)
		elif self.bc_weight_type == "qfilter":
			"""
			Soft Q-filtering inspired from 			
			Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
			learning with demonstrations." 2018 IEEE international 
			conference on robotics and automation (ICRA). IEEE, 2018.
			"""
			with torch.no_grad():
				stddev = 0.1
				dist_qf = self.actor_bc(features_qfilter, stddev)
				action_qf = dist_qf.mean
				Q1_qf, Q2_qf = self.critic(features_qfilter.clone()[:, -1], action_qf[:, -1])
				Q_qf = torch.min(Q1_qf, Q2_qf)
				bc_weight = (Q_qf>Q).float().mean().detach()

		actor_loss = -Q.mean() * (1-bc_weight)

		if bc_regularize:
			stddev = 0.1
			dist_bc = self.actor(features_bc, stddev)
			log_prob_bc = dist_bc.log_prob(action_bc).sum(-1, keepdim=True)
			actor_loss += - log_prob_bc.mean()*bc_weight*0.03

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
			metrics['rl_loss'] = -Q.mean().item()* (1-bc_weight)
			if bc_regularize:
				metrics['bc_weight'] = bc_weight
				metrics['bc_loss'] = - log_prob_bc.mean().item()*bc_weight*0.03

		return metrics

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		batch = next(replay_iter)
		data = utils.to_torch(batch, self.device)
		action = data['action'].float()
		reward = data['reward'].float()
		discount = data['discount'].float()
		
		# get features
		if self.obs_type == 'pixels':			
			features = []
			next_features = []
			obs_qfilter = {}
			for key in self.pixel_keys:
				pixel = data[key].float()
				next_pixel = data['next_' + key].float()
				shape = pixel.shape
				# rearrange
				pixel = einops.rearrange(pixel, 'b t c h w -> (b t) c h w')
				next_pixel = einops.rearrange(next_pixel, 'b t c h w -> (b t) c h w')
				# augment
				pixel = self.customAug(pixel / 255.0) if self.norm else pixel
				next_pixel = self.customAug(next_pixel / 255.0) if self.norm else next_pixel
				# store for qfilter
				obs_qfilter[key] = pixel.clone()
				# encode
				pixel = self.encoder[key](pixel) if self.separate_encoders else self.encoder(pixel)
				with torch.no_grad():
					next_pixel = self.encoder[key](next_pixel) if self.separate_encoders else self.encoder(next_pixel)
				# bring back time dimension
				pixel = einops.rearrange(pixel, '(b t) d -> b t d', t=shape[1])
				next_pixel = einops.rearrange(next_pixel, '(b t) d -> b t d', t=shape[1])
				# store
				features.append(pixel)
				next_features.append(next_pixel)
			if self.use_proprio:
				proprio = data[self.proprio_key].float()
				next_proprio = data['next_' + self.proprio_key].float()
				# save for qfilter
				obs_qfilter[self.proprio_key] = proprio.clone()
				# project
				proprio = self.proprio_projector(proprio)
				with torch.no_grad():
					next_proprio = self.proprio_projector(next_proprio)
				# store
				features.append(proprio)
				next_features.append(next_proprio)
			features = torch.cat(features, dim=-1).view(action.shape[0], -1, self.repr_dim) # (B, T * num_feat_per_step, D)
			next_features = torch.cat(next_features, dim=-1).view(action.shape[0], -1, self.repr_dim) # (B, T * num_feat_per_step, D)
		else:
			features = data[self.feature_key].float()
			next_features = data['next_' + self.feature_key].float()
			features = self.encoder(features)
			with torch.no_grad():
				next_features = self.encoder(next_features)

		if bc_regularize:
			batch = next(expert_replay_iter)
			data = utils.to_torch(batch, self.device)
			action_bc = data['actions'].float()
			if self.obs_type == 'pixels':
				features_bc = []
				for key in self.pixel_keys:
					pixel = data[key].float()
					shape = pixel.shape
					# rearrange
					pixel = einops.rearrange(pixel, 'b t c h w -> (b t) c h w')
					# augment
					pixel = self.customAug(pixel / 255.0) if self.norm else pixel
					# encode
					pixel = self.encoder[key](pixel) if self.separate_encoders else self.encoder(pixel)
					# bring back time dimension
					pixel = einops.rearrange(pixel, '(b t) d -> b t d', t=shape[1])
					# store
					features_bc.append(pixel)
				if self.use_proprio:
					proprio = data[self.proprio_key].float()
					# project
					proprio = self.proprio_projector_bc(proprio)
					# store
					features_bc.append(proprio)
				features_bc = torch.cat(features_bc, dim=-1).view(action_bc.shape[0], -1, self.repr_dim)
			else:
				features_bc = data[self.feature_key].float()
				features_bc = self.encoder(features_bc)
			# detach grads
			features_bc = features_bc.detach()

			# qfilter obs
			if self.bc_weight_type=="qfilter":
				if self.obs_type == 'pixels':
					features_qfilter = []
					for key in self.pixel_keys:
						pixel = obs_qfilter[key]
						# encode
						pixel = self.encoder_bc[key](pixel) if self.separate_encoders else self.encoder_bc(pixel)
						# bring back time dimension
						pixel = einops.rearrange(pixel, '(b t) d -> b t d', t=shape[1])
						# store
						features_qfilter.append(pixel)
					if self.use_proprio:
						proprio = obs_qfilter[self.proprio_key]
						# project
						proprio = self.proprio_projector_bc(proprio)
						# store
						features_qfilter.append(proprio)
					features_qfilter = torch.cat(features_qfilter, dim=-1).view(action_bc.shape[0], -1, self.repr_dim)
				else:
					features_qfilter = obs_qfilter[self.feature_key].float()
					features_qfilter = self.encoder_bc(features_qfilter)
				# detach grads
				features_qfilter = features_qfilter.detach()
			else:
				features_qfilter = None
			

			# obs_bc, action_bc = utils.to_torch(batch, self.device)
			# import ipdb; ipdb.set_trace()
			# # augment
			# if self.obs_type=='pixels' and self.augment:
			# 	obs_bc = self.aug(obs_bc.float())
			# else:
			# 	obs_bc = obs_bc.float()
			# # encode
			# if bc_regularize and self.bc_weight_type=="qfilter":
			# 	obs_qfilter = self.encoder_bc(obs_qfilter)
			# 	obs_qfilter = obs_qfilter.detach()
			# else:
			# 	obs_qfilter = None
			# obs_bc = self.encoder(obs_bc)
			# # Detach grads
			# obs_bc = obs_bc.detach()
		else:
			features_qfilter = None
			features_bc = None 
			action_bc = None

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(features, action, reward, discount, next_features, step))

		# update actor
		metrics.update(self.update_actor(features.detach(), features_bc, features_qfilter, action_bc, bc_regularize, step))
			
		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def ot_rewarder(self, observations, demos, step):

		if step % self.update_target_every == 0:
			self.encoder_target.load_state_dict(self.encoder.state_dict())
			
		scores_list = list()
		ot_rewards_list = list()
		for demo in demos:
			obs = torch.tensor(observations).to(self.device).float()
			obs = self.encoder_target(obs)
			exp = torch.tensor(demo).to(self.device).float()
			exp = self.encoder_target(exp)
			obs = obs.detach()
			exp = exp.detach()
			
			if self.rewards == 'sinkhorn_cosine':
				cost_matrix = cosine_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'sinkhorn_euclidean':
				cost_matrix = euclidean_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'cosine':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(1. - F.cosine_similarity(obs, exp))
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			elif self.rewards == 'euclidean':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(obs - exp).norm(dim=1)
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			else:
				raise NotImplementedError()

			scores_list.append(np.sum(ot_rewards))
			ot_rewards_list.append(ot_rewards)

		closest_demo_index = np.argmax(scores_list)
		return ot_rewards_list[closest_demo_index]

	def save_snapshot(self):
		keys_to_save = ['encoder', 'actor', 'critic']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		# load
		# encoder
		if self.obs_type == 'pixels' and self.separate_encoders:
			for key in self.pixel_keys:
				self.encoder[key].load_state_dict(payload.encoder[key].state_dict())
		else:
			self.encoder.load_state_dict(payload.encoder.state_dict())
		# proprio
		if self.obs_type == 'pixels' and self.use_proprio:
			self.proprio_projector.load_state_dict(payload.proprio_projector.state_dict())
		# actor
		self.actor.load_state_dict(payload.actor.state_dict())

		# target networks
		self.critic_target.load_state_dict(self.critic.state_dict())
		if self.obs_type == 'pixels' and self.separate_encoders:
			for key in self.pixel_keys:
				self.encoder_target[key].load_state_dict(self.encoder[key].state_dict())
		else:
			self.encoder_target.load_state_dict(self.encoder.state_dict())
		
		if self.bc_weight_type == "qfilter":
			# Store a copy of the BC policy with frozen weights
			# encoder
			if self.obs_type == 'pixels' and self.separate_encoders:
				self.encoder_bc = {}
				for key in self.pixel_keys:
					self.encoder_bc[key] = copy.deepcopy(self.encoder[key])
					self.encoder_bc[key].eval()
					for param in self.encoder_bc[key].parameters():
						param.requires_grad = False
			else:
				self.encoder_bc = copy.deepcopy(self.encoder)
				self.encoder_bc.eval()
				for param in self.encoder_bc.parameters():
					param.requires_grad = False
			# proprio
			if self.obs_type == 'pixels' and self.use_proprio:
				self.proprio_projector_bc = copy.deepcopy(self.proprio_projector)
				self.proprio_projector_bc.eval()
				for param in self.proprio_projector_bc.parameters():
					param.requires_grad = False
			# actor
			self.actor_bc = copy.deepcopy(self.actor)
			self.actor_bc.eval()
			for param in self.actor_bc.parameters():
				param.required_grad = False

		# Update optimizers
		# encoder
		if self.obs_type == "pixels" and self.separate_encoders:
			params = []
			for key in self.pixel_keys:
				params += list(self.encoder[key].parameters())
		else:
			params = list(self.encoder.parameters())
		self.encoder_opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
		# self.encoder_opt = torch.optim.Adam(params, lr=self.lr)
		# proprio
		if self.obs_type == "pixels" and self.use_proprio:
			self.proprio_opt = torch.optim.AdamW(self.proprio_projector.parameters(), lr=self.lr, weight_decay=1e-4)
			# self.proprio_opt = torch.optim.Adam(self.proprio_projector.parameters(), lr=self.lr)
		# actor
		self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
		# self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		# critic
		self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=1e-4)
		# self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)


