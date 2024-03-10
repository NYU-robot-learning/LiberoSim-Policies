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
		self._action_head = head(hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=0)
		
		self.apply(utils.weight_init)

	def forward(self, obs, num_prompt_feats, stddev):
		
		# process observation
		features = self._policy(obs)

		# remove action prediction from prompt
		features = features[:, num_prompt_feats:]
		
		# Filter actions - take 1 action for all tokens per time step 
		# and predict extra actions of action chunking
		features = features[:, ::self._num_feat_per_step]

		# action head
		pred_action = self._action_head(features, stddev)

		return pred_action

class BCAgent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, stddev_schedule, 
		  		 stddev_clip, use_tb, augment, obs_type, encoder_type, policy_type, policy_head,
				 pixel_keys, proprio_key, feature_key, use_proprio, train_encoder, norm, 
				 history, history_len, eval_history_len, separate_encoders, temporal_agg,
				 max_episode_len, num_queries, prompt, use_language, use_actions):
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
		self.use_proprio = use_proprio if obs_type == 'pixels' else False
		self.norm = norm
		self.train_encoder = train_encoder
		self.history_len = history_len if history else 1
		self.eval_history_len = eval_history_len if history else 1
		self.separate_encoders = separate_encoders
		self.use_language = use_language
		self.language_proj_type = 'mlp' # mlp or identity
		self.prompt = prompt
		self.use_actions = use_actions # only for the prompt
		
		# language 
		self.language_fusion = 'none' if not self.use_language else 'film'
		self.language_dim = 384
		self.lang_repr_dim = 512

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
		# ##############################################################################
		# self.max_episode_len = 380
		# ##############################################################################
		self.num_queries = num_queries

		# number of inputs per time step
		if policy_type == 'mlp' or obs_type == 'features':
			num_feat_per_step = 1
		elif obs_type == 'pixels':
			num_feat_per_step = len(self.pixel_keys)
			if use_proprio:
				num_feat_per_step += 1
		if policy_type == 'gpt' and self.use_language:
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
						self.encoder[key] = ResnetEncoder(obs_shape, 512, language_dim=self.lang_repr_dim, language_fusion=self.language_fusion).to(device)
				else:
					self.encoder = ResnetEncoder(obs_shape, 512, language_dim=self.lang_repr_dim, language_fusion=self.language_fusion).to(device)
				self.repr_dim = 512
				# self.norm = True
			elif self.encoder_type == 'patch':
				pass
		else:
			self.encoder = MLP(obs_shape[0], hidden_channels=[512, 512]).to(device)
			self.repr_dim = 512
			if self.use_language:
				self.repr_dim += self.lang_repr_dim

		# language encoder
		if self.use_language:
			# projector
			if self.language_proj_type == 'mlp':
				self.language_projector = MLP(self.language_dim, hidden_channels=[self.lang_repr_dim, self.lang_repr_dim]).to(device)
			else:
				self.language_projector = nn.Identity()
			self.language_projector.apply(utils.weight_init)


		# projector for proprioceptive features
		if use_proprio:
			self.proprio_projector = MLP(proprio_shape[0], hidden_channels=[self.repr_dim, self.repr_dim]).to(device)
			self.proprio_projector.apply(utils.weight_init)

		# projector for actions
		if self.use_actions:
			self.action_projector = MLP(self._act_dim, hidden_channels=[self.repr_dim, self.repr_dim]).to(device)
			self.action_projector.apply(utils.weight_init)

		if obs_type == 'pixels' and policy_type == 'mlp':
			new_repr_dim = self.repr_dim * len(self.pixel_keys)
			if use_proprio:
				new_repr_dim += self.repr_dim
			if self.use_language:
				new_repr_dim += self.lang_repr_dim
			if self.use_actions:
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
		if self.use_proprio:
			self.proprio_opt = torch.optim.AdamW(self.proprio_projector.parameters(), lr=lr, weight_decay=1e-4)
			self.proprio_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.proprio_opt, T_max=5e5)
		# language
		if self.use_language:
			self.language_opt = torch.optim.AdamW(self.language_projector.parameters(), lr=lr, weight_decay=1e-4)
			self.language_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.language_opt, T_max=5e5)
		# action projector
		if self.use_actions:
			self.action_opt = torch.optim.AdamW(self.action_projector.parameters(), lr=lr, weight_decay=1e-4)
			self.action_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.action_opt, T_max=5e5)
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
			if self.use_language:
				self.language_projector.train(training)
			if self.obs_type == 'pixels' and self.use_proprio:
				self.proprio_projector.train(training)
			if self.use_actions:
				self.action_projector.train(training)
			self.actor.train(training)
		else:
			if self.separate_encoders:
				for key in self.pixel_keys:
					self.encoder[key].eval()
			else:
				self.encoder.eval()
			if self.use_language:
				self.language_projector.eval()
			if self.obs_type == 'pixels' and self.use_proprio:
				self.proprio_projector.eval()
			if self.use_actions:
				self.action_projector.eval()
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

	def act(self, obs, prompt, step, global_step, eval_mode=False):
		# # save cv2 images
		# if step == 0:
		# 	import cv2
		# 	pix = obs['pixels'].transpose(1,2,0).astype(np.uint8)
		# 	pixego = obs['pixels_egocentric'].transpose(1,2,0).astype(np.uint8)
		# 	cv2.imwrite("env_pix.png", pix)
		# 	cv2.imwrite("env_pixego.png", pixego)

		# lang projection
		if self.use_language:
			key = self.pixel_keys[0] if self.obs_type == 'pixels' else self.feature_key
			repeat_len = min(len(self.observation_buffer[key]) + 1, self.eval_history_len) if self.obs_type == 'pixels' else len(self.observation_buffer)
			lang_features = torch.as_tensor(obs['task_emb'], device=self.device).float()[None].repeat(repeat_len, 1)
			lang_features = self.language_projector(lang_features)
			# lang_features = einops.rearrange(lang_features, 'b t d -> (b t) d')
		else:
			lang_features = None
		
		if self.obs_type == 'pixels':
			# add to buffer
			features = []
			for key in self.pixel_keys:
				self.observation_buffer[key].append(self.test_aug(obs[key].transpose(1,2,0)).numpy())
				pixels = torch.as_tensor(np.array(self.observation_buffer[key]), device=self.device).float()
				pixels = self.customAug(pixels / 255.0) if self.norm else pixels
				#encoder
				pixels = self.encoder[key](pixels, lang=lang_features) if self.separate_encoders else self.encoder(pixels, lang=lang_features)
				features.append(pixels)
			if self.use_proprio:
				self.proprio_buffer.append(obs[self.proprio_key])
				proprio = torch.as_tensor(np.array(self.proprio_buffer), device=self.device).float()
				proprio = self.proprio_projector(proprio)
				features.append(proprio)
			if self.use_language:
				features.append(lang_features)
			features = torch.cat(features, dim=-1).view(-1, self.repr_dim)
		else:
			self.observation_buffer.append(obs[self.feature_key])
			features = torch.as_tensor(np.array(self.observation_buffer), device=self.device).float()
			features = self.encoder(features)
			if self.use_language:
				features = torch.cat([features, lang_features], dim=-1)
		
		# prompt
		if self.prompt not in [None, 'text']:
			if self.use_language:
				prompt_lang_features = lang_features[-1:]
				reshape_lang = True

			if self.obs_type == 'pixels':
				prompt_features = []
				for key in self.pixel_keys:
					pixel = torch.as_tensor(prompt[f"prompt_{key}"], device=self.device).float()
					shape = pixel.shape
					# reshape lang features
					if self.use_language and reshape_lang:
						prompt_lang_features = prompt_lang_features.repeat(shape[0], 1)
						# prompt_lang_features = einops.rearrange(prompt_lang_features, 'b t d -> (b t) d')
						reshape_lang = False
					# augment
					pixel = self.customAug(pixel / 255.0) if self.norm else pixel
					# encode
					pixel = self.encoder[key](pixel, lang=prompt_lang_features) if self.separate_encoders else self.encoder(pixel, lang=prompt_lang_features)
					prompt_features.append(pixel)
				if self.use_proprio:
					proprio = torch.as_tensor(prompt[f"prompt_{self.proprio_key}"], device=self.device).float()
					proprio = self.proprio_projector(proprio)
					prompt_features.append(proprio)
				if self.use_actions:
					action = torch.as_tensor(prompt[f"prompt_actions"], device=self.device).float()
					action = self.action_projector(action)
					prompt_features.append(action)
				prompt_features = torch.cat(prompt_features, dim=-1).view(-1, self.repr_dim)
			else:
				prompt_features = torch.as_tensor(prompt[f"prompt_{self.feature_key}"], device=self.device).float()
				prompt_features = self.encoder(prompt_features)
			features = torch.cat([prompt_features, features], dim=0)
			num_prompt_feats = prompt_features.shape[0]
		else:
			num_prompt_feats = 0
			
		stddev = utils.schedule(self.stddev_schedule, global_step)
		action = self.actor(features.unsqueeze(0), num_prompt_feats, stddev)
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

		# # save cv2 images
		# import cv2
		# pix = data['pixels'][0,0].cpu().numpy().transpose(1,2,0)
		# pixego = data['pixels_egocentric'][0,0].cpu().numpy().transpose(1,2,0)
		# pix = (pix * 255).astype(np.uint8)
		# pixego = (pixego * 255).astype(np.uint8)
		# cv2.imwrite("read_pix.png", pix)
		# cv2.imwrite("read_pixego.png", pixego)
		
		# lang projection
		if self.use_language:
			lang_features = data['task_emb'].float()[:, None].repeat(1, self.history_len, 1)
			lang_features = self.language_projector(lang_features)
			lang_features = einops.rearrange(lang_features, 'b t d -> (b t) d')
		else:
			lang_features = None
		
		# features
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
					pixel = self.encoder[key](pixel, lang=lang_features) if self.separate_encoders else self.encoder(pixel, lang=lang_features)
				else:
					with torch.no_grad():
						pixel = self.encoder[key](pixel, lang=lang_features) if self.separate_encoders else self.encoder(pixel, lang=lang_features)
				pixel = einops.rearrange(pixel, '(b t) d -> b t d', t=shape[1])
				features.append(pixel)
			if self.use_proprio:
				proprio = data[self.proprio_key].float()
				proprio = self.proprio_projector(proprio)
				features.append(proprio)
			if self.use_language:
				lang_features = einops.rearrange(lang_features, '(b t) d -> b t d', t=shape[1])
				features.append(lang_features)
			# for feat in features:
			# 	print(feat.shape, len(self.pixel_keys), self.use_proprio, self.use_proprio, self.use_language)
			# print("######################################")
			features = torch.cat(features, dim=-1).view(action.shape[0], -1, self.repr_dim) # (B, T * num_feat_per_step, D)
		else:
			features = data[self.feature_key].float()
			if self.train_encoder:
				features = self.encoder(features)
			else:
				with torch.no_grad():
					features = self.encoder(features)
			if self.use_language:
				features = torch.cat([features, lang_features], dim=-1)
		
		# prompt
		if self.prompt not in [None, 'text']:
			if self.use_language:
				prompt_lang_features = lang_features[:, -1:]
				reshape_lang = True

			if self.obs_type == 'pixels':
				prompt_features = []
				for key in self.pixel_keys:
					pixel = data[f'prompt_{key}'].float()
					shape = pixel.shape
					# reshape lang features
					if self.use_language and reshape_lang:
						prompt_lang_features = prompt_lang_features.repeat(1, shape[1], 1)
						prompt_lang_features = einops.rearrange(prompt_lang_features, 'b t d -> (b t) d')
						reshape_lang = False
					# rearrange
					pixel = einops.rearrange(pixel, 'b t c h w -> (b t) c h w')
					# augment
					# pixel = self.aug(pixel) if self.augment else pixel
					pixel = self.customAug(pixel / 255.0) if self.norm else pixel
					# encode
					if self.train_encoder:
						pixel = self.encoder[key](pixel, lang=prompt_lang_features) if self.separate_encoders else self.encoder(pixel, lang=prompt_lang_features)
					else:
						with torch.no_grad():
							pixel = self.encoder[key](pixel, lang=prompt_lang_features) if self.separate_encoders else self.encoder(pixel, lang=prompt_lang_features)
					pixel = einops.rearrange(pixel, '(b t) d -> b t d', t=shape[1])
					prompt_features.append(pixel)
				if self.use_proprio:
					proprio = data[f"prompt_{self.proprio_key}"].float()
					proprio = self.proprio_projector(proprio)
					prompt_features.append(proprio)
				if self.use_actions:
					prompt_action = data[f"prompt_actions"].float()
					prompt_action = self.action_projector(prompt_action)
					prompt_features.append(prompt_action)
				prompt_features = torch.cat(prompt_features, dim=-1).view(action.shape[0], -1, self.repr_dim) # (B, T * num_feat_per_step, D)
			else:
				prompt_features = data[f"prompt_{self.feature_key}"].float()
				if self.train_encoder:
					prompt_features = self.encoder(prompt_features)
				else:
					with torch.no_grad():
						prompt_features = self.encoder(prompt_features)
		
			# prepend prompt features
			features = torch.cat([prompt_features, features], dim=1)
			num_prompt_feats = prompt_features.shape[1]
		else:
			num_prompt_feats = 0
		
		# action
		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(features, num_prompt_feats, stddev)
		
		# loss
		if self.temporal_agg:
			# action = einops.rearrange(action, 'b t1 t2 d -> b t2 (t1 d)')
			action = einops.rearrange(action, 'b t1 t2 d -> b t1 (t2 d)')
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_loss = -log_prob.mean()

		if self.train_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		if self.obs_type == 'pixels' and self.use_proprio:
			self.proprio_opt.zero_grad(set_to_none=True)
		if self.use_language:
			self.language_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		if self.train_encoder:
			self.encoder_opt.step()
			self.encoder_scheduler.step()
		if self.obs_type == 'pixels' and self.use_proprio:
			self.proprio_opt.step()
			self.proprio_scheduler.step()
		if self.use_language:
			self.language_opt.step()
			self.language_scheduler.step()
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
		if self.use_language:
			keys_to_save += ['language_projector', 'language_opt']
		if self.use_actions:
			keys_to_save += ['action_projector', 'action_opt']
		keys_to_save += ['use_proprio', 'use_language', 'use_actions', 'max_episode_len']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload, encoder_only=False):
		if encoder_only:
			self.encoder = payload.model.base_encoder.to(self.device)
			if self.train_encoder:
				if self.separate_encoders:
					params = []
					for key in self.pixel_keys:
						self.encoder[key].train(True)
						params += list(self.encoder[key].parameters())
				else:
					self.encoder.train(True)
					params = list(self.encoder.parameters())
				self.encoder_opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
				self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_opt, T_max=5e5)
			else:
				if self.separate_encoders:
					for key in self.pixel_keys:
						self.encoder[key].eval()
				else:
					self.encoder.eval()
		else:
			self.encoder = payload['model'].base_encoder.to(self.device)
			self.encoder.eval()
			self.actor = Actor(256, (self._act_dim,), self.hidden_dim).to(self.device)
			self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

	def load_snapshot_eval(self, payload):
		# self.encoder = payload.encoder.to(self.device)
		# self.encoder.load_state_dict(payload.encoder.state_dict())
		self.encoder.load_state_dict(payload['encoder'].state_dict())
		for p in self.encoder.parameters():
			p.requires_grad = False
		# self.encoder.eval()

		# self.actor = payload.actor.to(self.device)
		# self.actor.load_state_dict(payload.actor.state_dict())
		self.actor.load_state_dict(payload['actor'].state_dict())
		for p in self.actor.parameters():
			p.requires_grad = False
		# self.actor.eval()

		# self.use_proprio = payload.use_proprio
		# self.use_language = payload.use_language
		# if self.use_actions:
		# 	self.use_actions = payload.use_actions
		self.use_proprio = payload['use_proprio']
		self.use_language = payload['use_language']
		if self.use_actions:
			self.use_actions = payload['use_actions']

		if self.use_proprio:
			# self.proprio_projector = payload.proprio_projector.to(self.device)
			# self.proprio_projector.load_state_dict(payload.proprio_projector.state_dict())
			self.proprio_projector.load_state_dict(payload['proprio_projector'].state_dict())
			for p in self.proprio_projector.parameters():
				p.requires_grad = False
			# self.proprio_projector.eval()
		if self.use_language:
			# self.language_projector = payload.language_projector.to(self.device)
			# self.language_projector.load_state_dict(payload.language_projector.state_dict())
			self.language_projector.load_state_dict(payload['language_projector'].state_dict())
			for p in self.language_projector.parameters():
				p.requires_grad = False
			# self.language_projector.eval()
		if self.use_actions:
			# self.action_projector = payload.action_projector.to(self.device)
			# self.action_projector.load_state_dict(payload.action_projector.state_dict())
			self.action_projector.load_state_dict(payload['action_projector'].state_dict())
			for p in self.action_projector.parameters():
				p.requires_grad = False
			# self.action_projector.eval()
				
		


