import cv2
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset

class LiberoDataset(IterableDataset):
    def __init__(self, path, suites, task_name, obs_type, history, history_len, 
                 prompt, temporal_agg, num_queries, img_size=128, subsample=10):

        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.subsample = subsample if self._prompt == 'episode' else 1
        self.img_size = img_size

        # temporal_aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries
        
        # get data paths
        self._paths = []
        for suite in suites:
            self._paths.extend(list((Path(path) / suite).glob('*')))
        
        if task_name is not None:
            paths = {}
            idx2name = {}
            for path in self._paths:
                task = str(path).split('.')[0].split('/')[-1]
                if task in task_name:
                    # get idx of task in task_name
                    idx = task_name.index(task)
                    paths[idx] = path
                    idx2name[idx] = task
            del self._paths
            self._paths = paths

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), 'rb'))
            observations = data['observations'] if self._obs_type == 'pixels' else data['states']
            actions = data['actions']
            # store
            self._episodes[_path_idx] = []
            for i in range(len(observations)):
                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(self._max_episode_len, len(observations[i]) if not isinstance(observations[i], dict) else len(observations[i]['pixels']))
                # if obs_type == 'features':
                self._max_state_dim = max(self._max_state_dim, data['states'][i].shape[-1])
                self._num_samples += len(observations[i]) if self._obs_type == 'features' else len(observations[i]['pixels'])
        
        # subsample
        if self._prompt == 'episode':
            self._max_episode_len = self._max_episode_len // self.subsample + 1

        # # augmentation
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop(self.img_size, scale=(0.9, 1.)),
            transforms.ToTensor(),
            # normalize
        ])
        
    def _sample_episode(self, env_idx=None):
        idx = random.choice(list(self._episodes.keys())) if env_idx is None else env_idx
        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode
    
    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes['observation']
        actions = episodes['action']

        if self._obs_type == 'pixels':
            # Sample obs, action
            sample_idx = np.random.randint(0, len(observations['pixels'])-self._history_len)
            if hasattr(self, 'aug'):
                sampled_pixel = observations['pixels'][sample_idx:sample_idx+self._history_len]
                sampled_pixel_egocentric = observations['pixels_egocentric'][sample_idx:sample_idx+self._history_len]
                sampled_pixel = torch.stack([self.aug(sampled_pixel[i]) for i in range(len(sampled_pixel))])
                sampled_pixel_egocentric = torch.stack([self.aug(sampled_pixel_egocentric[i]) for i in range(len(sampled_pixel_egocentric))])
            else:
                sampled_pixel = np.transpose(observations['pixels'][sample_idx:sample_idx+self._history_len], (0,3,1,2))
                sampled_pixel_egocentric = np.transpose(observations['pixels_egocentric'][sample_idx:sample_idx+self._history_len], (0,3,1,2))
            sampled_proprioceptive_state = np.concatenate(
                [observations['joint_states'][sample_idx:sample_idx+self._history_len],
                 observations['gripper_states'][sample_idx:sample_idx+self._history_len]], axis=-1
            )
            if self._temporal_agg:
                # arrange sampled action to be of shape (history_len, num_queries, action_dim)
                sampled_action = np.zeros((self._history_len, self._num_queries, actions.shape[-1]))
                num_actions = self._history_len + self._num_queries - 1 # -1 since its num_queries including the last action of the history
                act = np.zeros((num_actions, actions.shape[-1]))
                act[:min(len(actions), sample_idx+num_actions) - sample_idx] = actions[sample_idx:sample_idx+num_actions]
                sampled_action = np.lib.stride_tricks.sliding_window_view(act, (self._history_len, actions.shape[-1]))
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[sample_idx:sample_idx+self._history_len]

            if self._prompt == None:
                return {
                    'pixels': sampled_pixel,
                    'pixels_egocentric': sampled_pixel_egocentric,
                    'proprioceptive': sampled_proprioceptive_state,
                    'actions': sampled_action,
                    # 'mask': mask if self._temporal_agg else None,
                }
            elif self._prompt == 'text':
                pass
            elif self._prompt == 'goal':
                prompt_episode = self._sample_episode(env_idx)
                prompt_observations = prompt_episode['observation']
                if hasattr(self, 'aug'):
                    prompt_pixel = self.aug(prompt_observations['pixels'][-1])[None]
                    prompt_pixel_egocentric = self.aug(prompt_observations['pixels_egocentric'][-1])[None]
                else:
                    prompt_pixel = np.transpose(prompt_observations['pixels'][-1:], (0,3,1,2))
                    prompt_pixel_egocentric = np.transpose(prompt_observations['pixels_egocentric'][-1:], (0,3,1,2))
                prompt_proprioceptive_state = np.concatenate(
                    [prompt_observations['joint_states'][-1:],
                     prompt_observations['gripper_states'][-1:]], axis=-1
                )
                prompt_action = prompt_episode['action'][-1:]
                return {
                    'pixels': sampled_pixel,
                    'pixels_egocentric': sampled_pixel_egocentric,
                    'proprioceptive': sampled_proprioceptive_state,
                    'actions': sampled_action,
                    'prompt_pixels': prompt_pixel,
                    'prompt_pixels_egocentric': prompt_pixel_egocentric,
                    'prompt_proprioceptive': prompt_proprioceptive_state,
                    'prompt_actions': prompt_action,
                }
            elif self._prompt == 'episode':
                prompt_pixel = torch.zeros((self._max_episode_len, *sampled_pixel.shape[1:]))
                prompt_pixel_egocentric = torch.zeros((self._max_episode_len, *sampled_pixel_egocentric.shape[1:]))
                prompt_proprioceptive_state = np.zeros((self._max_episode_len, sampled_proprioceptive_state.shape[-1]))
                prompt_action = np.zeros((self._max_episode_len, sampled_action.shape[-1]))

                prompt_episode = self._sample_episode(env_idx)
                prompt_observations = prompt_episode['observation']
                prompt_actions = prompt_episode['action']
                # store
                if hasattr(self, 'aug'):
                    pixel = prompt_observations['pixels'][::self.subsample]
                    pixel_egocentric = prompt_observations['pixels_egocentric'][::self.subsample]
                    pixel = torch.stack([self.aug(pixel[i]) for i in range(len(pixel))])
                    pixel_egocentric = torch.stack([self.aug(pixel_egocentric[i]) for i in range(len(pixel_egocentric))])
                    prompt_pixel[-len(prompt_observations['pixels'][::self.subsample]):] = pixel
                    prompt_pixel_egocentric[-len(prompt_observations['pixels_egocentric'][::self.subsample]):] = pixel_egocentric
                else:
                    prompt_pixel[-len(prompt_observations['pixels'][::self.subsample]):] = np.transpose(np.array(prompt_observations['pixels'][::self.subsample]), (0,3,1,2))
                    prompt_pixel_egocentric[-len(prompt_observations['pixels_egocentric'][::self.subsample]):] = np.transpose(np.array(prompt_observations['pixels_egocentric'][::self.subsample]), (0,3,1,2))
                prompt_proprioceptive_state[-len(prompt_observations['joint_states'][::self.subsample]):] = np.concatenate(
                    [prompt_observations['joint_states'][::self.subsample],
                    prompt_observations['gripper_states'][::self.subsample]], axis=-1
                )
                prompt_action[-len(prompt_actions[::self.subsample]):] = prompt_actions[::self.subsample]
                return {
                    'pixels': sampled_pixel,
                    'pixels_egocentric': sampled_pixel_egocentric,
                    'proprioceptive': sampled_proprioceptive_state,
                    'actions': sampled_action,
                    'prompt_pixels': prompt_pixel,
                    'prompt_pixels_egocentric': prompt_pixel_egocentric,
                    'prompt_proprioceptive': prompt_proprioceptive_state,
                    'prompt_actions': prompt_action,
                }

        elif self._obs_type == 'features':
            # Sample obs, action
            sample_idx = np.random.randint(0, len(observations) - self._history_len)
            sampled_obs = np.array(observations[sample_idx:sample_idx+self._history_len])
            sampled_action = actions[sample_idx:sample_idx+self._history_len]
            # pad obs to match self._max_state_dim
            obs = np.zeros((self._history_len, self._max_state_dim))
            state_dim = sampled_obs.shape[-1]
            obs[:, :state_dim] = sampled_obs
            sampled_obs = obs

            # prompt obs, action
            if self._prompt == None:
                return {
                    'features': sampled_obs,
                    'actions': sampled_action,
                }
            elif self._prompt == 'text':
                pass
            elif self._prompt == 'goal':
                prompt_episode = self._sample_episode(env_idx)
                prompt_obs = np.array(prompt_episode['observation'][-1:])
                prompt_action = prompt_episode['action'][-1:]
                return {
                    'features': sampled_obs,
                    'actions': sampled_action,
                    'prompt_obs': prompt_obs,
                    'prompt_actions': prompt_action,
                }
            elif self._prompt == 'episode':
                prompt_obs = np.zeros((self._max_episode_len, *obs.shape[1:]))
                prompt_action = np.zeros((self._max_episode_len, *sampled_action.shape[1:]))
                mask = np.zeros(self._max_episode_len)
                # Entire episode is the prompt
                prompt_episode = self._sample_episode(env_idx)
                prompt_observations = prompt_episode['observation']
                prompt_obs[-len(prompt_observations[::self.subsample]):, :state_dim] = prompt_observations[::self.subsample]
                prompt_action[-len(prompt_episode['action'][::self.subsample]):] = prompt_episode['action'][::self.subsample]
                prompt_obs = np.array(prompt_obs)
                mask[-len(prompt_observations[::self.subsample]):] = 1
                return {
                    'features': sampled_obs,
                    'actions': sampled_action,
                    'prompt_features': prompt_obs,
                    'prompt_actions': prompt_action,
                    'mask': mask,
                }
    
    def sample_test(self, env_idx):
        episode = self._sample_episode(env_idx)
        observations = episode['observation']
        actions = episode['action']
        
        if self._obs_type == 'pixels':
            pixels_shape = observations['pixels'].shape
            
            # observation
            if self._prompt == None:
                prompt_pixel = None
                prompt_pixel_egocentric = None
                prompt_proprioceptive_state = None
            elif self._prompt == 'text':
                pass
            elif self._prompt == 'goal':
                prompt_pixel = np.transpose(observations['pixels'][-1:], (0,3,1,2))
                prompt_pixel_egocentric = np.transpose(observations['pixels_egocentric'][-1:], (0,3,1,2))
                prompt_proprioceptive_state = np.concatenate(
                    [observations['joint_states'][-1:],
                    observations['gripper_states'][-1:]], axis=-1
                )
            elif self._prompt == 'episode':        
                prompt_pixel = np.zeros((self._max_episode_len, pixels_shape[3], pixels_shape[1], pixels_shape[2]))
                prompt_pixel_egocentric = np.zeros((self._max_episode_len, pixels_shape[3], pixels_shape[1], pixels_shape[2]))
                prompt_proprioceptive_state = np.zeros((self._max_episode_len, observations['joint_states'].shape[-1] + observations['gripper_states'].shape[-1]))
                # Entire episode is the prompt
                prompt_pixel[-len(observations['pixels'][::self.subsample]):] = np.transpose(np.array(observations['pixels'][::self.subsample]), (0,3,1,2))
                prompt_pixel_egocentric[-len(observations['pixels_egocentric'][::self.subsample]):] = np.transpose(np.array(observations['pixels_egocentric'][::self.subsample]), (0,3,1,2))
                # prompt_proprioceptive_state[-len(observations['proprioceptive'][::self.subsample]):] = np.array(observations['proprioceptive'][::self.subsample])
                prompt_proprioceptive_state[-len(observations['joint_states'][::self.subsample]):] = np.concatenate(
                    [observations['joint_states'][::self.subsample],
                    observations['gripper_states'][::self.subsample]], axis=-1
                )

            # actions
            prompt_action = np.zeros((self._max_episode_len, actions.shape[-1]))
            prompt_action[-len(actions[::self.subsample]):] = actions[::self.subsample]

            return {
                'prompt_pixels': prompt_pixel,
                'prompt_pixels_egocentric': prompt_pixel_egocentric,
                'prompt_proprioceptive': prompt_proprioceptive_state,
                'prompt_actions': prompt_action,
            }

        elif self._obs_type == 'features':
            # observation
            if self._prompt == None:
                prompt_obs = None
            elif self._prompt == 'text':
                pass
            elif self._prompt == 'goal':
                prompt_obs = np.array(observations[-1:])
            elif self._prompt == 'episode':
                prompt_obs = np.zeros((self._max_episode_len, self._max_state_dim))
                # Entire episode is the prompt
                prompt_obs[-len(observations[::self.subsample]):, :observations.shape[-1]] = observations[::self.subsample]

            # actions
            prompt_action = np.zeros((self._max_episode_len, actions.shape[-1]))
            prompt_action[-len(actions[::self.subsample]):] = actions[::self.subsample]

            return {
                'prompt_features': prompt_obs,
                'prompt_actions': prompt_action
            }

    def __iter__(self):
        while True:
            yield self._sample()
    
    def __len__(self):
        return self._num_samples
