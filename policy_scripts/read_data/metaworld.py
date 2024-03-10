import cv2
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
import pickle

class MetaworldDataset(IterableDataset):
    def __init__(self, dataset_path, task_name, num_demos, obs_type, history, history_len,
                 prompt, temporal_agg, num_queries, img_size=84, subsample=1):

        self._obs_type = obs_type
        self._history = history
        self._history_len = history_len if history else 1
        self._prompt = prompt
        self._img_size = img_size
        self.subsample = subsample if self._prompt == 'episode' else 1

        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # get data paths
        self._paths = {}
        for idx, name in enumerate(task_name):
            self._paths[idx] = Path(dataset_path) / f'{name}.pkl'
        
        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), 'rb'))
            observations = data['observations'] if obs_type == 'pixels' else data['states']
            actions = data['actions']
            # store
            self._episodes[_path_idx] = []
            for i in range(num_demos):
                episode = dict(observation=observations[i], action=actions[i])
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(self._max_episode_len, len(observations[i]))
                self._max_state_dim = max(self._max_state_dim, data['states'][i].shape[-1])
                self._num_samples += len(observations[i])
        
        # subsample
        if self._prompt == 'episode':
            self._max_episode_len = self._max_episode_len // self.subsample + 1
        
        # # augmentation
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop(self._img_size, scale=(0.95, 1.), ratio=(1., 1.)),
            transforms.RandomCrop(self._img_size, padding=(4,4,4,4), padding_mode='edge'),
            transforms.ToTensor(),
            # normalize
        ])

    def _sample_episode(self, env_idx=None):
        idx = random.choice(list(self._episodes.keys())) if env_idx is None else env_idx
        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episode, env_idx = self._sample_episode()
        observations = episode['observation']
        actions = episode['action']
        
        sample_idx = np.random.randint(0, len(episode['observation']) - self._history_len)
        # obs
        sampled_obs = observations[sample_idx:sample_idx+self._history_len]
        if self._obs_type == 'pixels':
            if hasattr(self, 'aug'):
                sampled_obs = np.stack([self.aug(obs) for obs in sampled_obs])
            else:
                sampled_obs = np.transpose(sampled_obs, (0, 3, 1, 2))
        
        # action
        if self._temporal_agg:
            # arrange sampled action to be of shape (history_len, num_queries, action_dim)
            sampled_action = np.zeros((self._history_len, self._num_queries, actions.shape[-1]))
            num_actions = self._history_len + self._num_queries - 1 # -1 since its num_queries including the last action of the history
            act = np.zeros((num_actions, actions.shape[-1]))
            act[:min(len(actions), sample_idx+num_actions) - sample_idx] = actions[sample_idx:sample_idx+num_actions]
            sampled_action = np.lib.stride_tricks.sliding_window_view(act, (self._history_len, actions.shape[-1]))
            sampled_action = sampled_action[:, 0]
        else:
            sampled_action = episode['action'][sample_idx:sample_idx+self._history_len]

        obs_key = "pixels" if self._obs_type == 'pixels' else "features"
        return {
            obs_key: sampled_obs,
            'actions': sampled_action,
        }

    def __iter__(self):
        while True:
            yield self._sample()
