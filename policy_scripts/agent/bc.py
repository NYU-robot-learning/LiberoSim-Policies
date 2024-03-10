import einops
import numpy as np
from pathlib import Path
from collections import deque

import torch
from torch import nn

from torchvision import transforms as T
from torch.nn import functional as F
import torch.distributions as D

import utils
from agent.networks.rgb_modules import *
from agent.networks.policy_head import *
from agent.networks.gpt import GPT, GPTConfig
from agent.networks.mlp import MLP
	
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

class BCAgent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, stddev_schedule, 
	      		 stddev_clip, use_tb, augment, obs_type, encoder_type, policy_type, policy_head,
				 pixel_keys, proprio_key, feature_key, use_proprio, train_encoder, norm, 
				 history, history_len, eval_history_len, separate_encoders, temporal_agg,
				 max_episode_len, num_queries):
		self.device = device
		self.lr = lr
		self.hidden_dim = hidden_dim
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.augment = augment
		self.obs_type = obs_type
		self.encoder_type = encoder_type
		self.policy_head = policy_head
		self.use_proprio = use_proprio
		self.norm = norm
		self.train_encoder = train_encoder
		self.history_len = history_len if history else 1
		self.eval_history_len = eval_history_len if history else 1
		self.separate_encoders = separate_encoders

		# actor parameters
		self._act_dim = action_shape[0]

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
		
		# encoder
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
		
		# optimizers
		# encoder
		if self.train_encoder:
			if self.separate_encoders:
				params = []
				for key in self.pixel_keys:
					params += list(self.encoder[key].parameters())
			else:
				params = list(self.encoder.parameters())
			self.encoder_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
			self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_opt, T_max=5e5)
		# proprio
		if obs_type == 'pixels' and self.use_proprio:
			self.proprio_opt = torch.optim.AdamW(self.proprio_projector.parameters(), lr=lr, weight_decay=1e-4)
			self.proprio_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.proprio_opt, T_max=5e5)
		# actor
		params = list(self.actor.parameters())
		self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
		self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=5e5)
		# self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_opt, step_size=5000, gamma=0.1)

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
		self.buffer_reset()

	def __repr__(self):
		return "bc"
	
	def train(self, training=True):
		self.training = training
		if training:
			if self.separate_encoders:
				for key in self.pixel_keys:
					if self.train_encoder:
						self.encoder[key].train(training)
					else:
						self.encoder[key].eval()
			else:
				if self.train_encoder:
					self.encoder.train(training)
				else:
					self.encoder.eval()
			if self.obs_type == 'pixels' and self.use_proprio:
				self.proprio_projector.train(training)
			self.actor.train(training)
		else:
			if self.separate_encoders:
				for key in self.pixel_keys:
					self.encoder[key].eval()
			else:
				self.encoder.eval()
			if self.obs_type == 'pixels' and self.use_proprio:
				self.proprio_projector.eval()
			self.actor.eval()

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
			action = action.sample()
		
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

	def update(self, expert_replay_iter, step):
		metrics = dict()

		batch = next(expert_replay_iter)
		data = utils.to_torch(batch, self.device)
		action = data['actions'].float()

		# get features
		if self.obs_type == 'pixels':			
			features = []
			for key in self.pixel_keys:
				pixel = data[key].float()
				shape = pixel.shape
				# rearrange
				pixel = einops.rearrange(pixel, 'b t c h w -> (b t) c h w')
				# augment
				# pixel = self.aug(pixel) if self.augment else pixel
				pixel = self.customAug(pixel / 255.0) if self.norm else pixel
				# encode
				if self.train_encoder:
					pixel = self.encoder[key](pixel) if self.separate_encoders else self.encoder(pixel)
				else:
					with torch.no_grad():
						pixel = self.encoder[key](pixel) if self.separate_encoders else self.encoder(pixel)
				pixel = einops.rearrange(pixel, '(b t) d -> b t d', t=shape[1])
				features.append(pixel)
			if self.use_proprio:
				proprio = data[self.proprio_key].float()
				proprio = self.proprio_projector(proprio)
				features.append(proprio)
			features = torch.cat(features, dim=-1).view(action.shape[0], -1, self.repr_dim) # (B, T * num_feat_per_step, D)
		else:
			features = data[self.feature_key].float()
			if self.train_encoder:
				features = self.encoder(features)
			else:
				with torch.no_grad():
					features = self.encoder(features)
		
		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(features, stddev)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		
		# loss
		if self.temporal_agg:
			action = einops.rearrange(action, 'b t1 t2 d -> b t1 (t2 d)')
		actor_loss = -log_prob.mean()

		if self.train_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		if self.obs_type == 'pixels' and self.use_proprio:
			self.proprio_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		if self.train_encoder:
			self.encoder_opt.step()
			self.encoder_scheduler.step()
		if self.obs_type == 'pixels' and self.use_proprio:
			self.proprio_opt.step()
			self.proprio_scheduler.step()
		self.actor_opt.step()
		self.actor_scheduler.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()

		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor', 'actor_opt', 'encoder']
		if self.train_encoder:
			keys_to_save += ['encoder_opt']
		if self.obs_type == 'pixels' and self.use_proprio:
			keys_to_save += ['proprio_projector', 'proprio_opt']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		self.encoder = payload['model'].base_encoder.to(self.device)
		self.encoder.eval()
		self.actor = Actor(256, (self._act_dim,), self.hidden_dim).to(self.device)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

	def load_snapshot_eval(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v
		
		# set networks to eval mode
		self.train(training=False)