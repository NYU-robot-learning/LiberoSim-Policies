import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torchvision import transforms
import pickle

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

class ReplayBufferStorage:
    def __init__(self, data_specs, obs_keys, replay_dir):
        self._data_specs = data_specs
        self._obs_keys = obs_keys
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step.observation[spec.name] if spec.name in self._obs_keys \
                                                     else time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, obs_keys, history, history_len,
                 temporal_agg, num_queries, max_episode_len, max_state_dim):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._obs_keys = obs_keys
        self._history_len = history_len if history else 1
        self._max_episode_len = max_episode_len
        self._max_state_dim = max_state_dim
        self._img_size = 84
        
        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # augmentation
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomAffine(0, translate=(0.1, 0.1)),
            # transforms.RandomResizedCrop(self._img_size, scale=(0.95, 1.), ratio=(1., 1.)),
            transforms.RandomCrop(self._img_size, padding=(4,4,4,4), padding_mode='edge'),
            transforms.ToTensor(),
            # normalize
        ])

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _augment(self, obs):
        return self.aug(obs.transpose(1, 2, 0))
    
    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1 - self._history_len) + 1
        obs = {}
        for key in self._obs_keys:
            obs[key] = episode[key][idx - 1:idx + self._history_len - 1]
            for i in range(self._history_len):
                obs[key][i] = self._augment(obs[key][i])
        # obs = episode['observation'][idx - 1]
        action = episode['action'][idx:idx + self._history_len]
        next_obs = {}
        for key in self._obs_keys:
            next_obs["next_" + key] = episode[key][idx + self._nstep - 1:idx + self._nstep + self._history_len - 1]
            for i in range(self._history_len):
                next_obs["next_" + key][i] = self._augment(next_obs["next_" + key][i])
        # next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx + self._history_len])
        discount = np.ones_like(episode['discount'][idx + self._history_len])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i + self._history_len]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i + self._history_len] * self._discount

        return {
            **obs,
            'action': action,
            'reward': reward,
            'discount': discount,
            **next_obs,
        }
        # return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

def make_replay_loader(replay_dir, max_size, batch_size, num_workers, save_snapshot,
                       nstep, discount, obs_keys, history, history_len, temporal_agg, 
                       num_queries, max_episode_len, max_state_dim):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            obs_keys=obs_keys,
                            history=history,
                            history_len=history_len,
                            temporal_agg=temporal_agg,
                            num_queries=num_queries,
                            max_episode_len=max_episode_len,
                            max_state_dim=max_state_dim)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader


def make_expert_replay_loader(iterable, batch_size):
    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=2,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
