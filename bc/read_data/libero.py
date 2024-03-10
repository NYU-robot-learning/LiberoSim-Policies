import cv2
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset

class BCDataset(IterableDataset):
    def __init__(self, path, suites, task_name, num_demos_per_task, obs_type, history, history_len, 
                 prompt, temporal_agg, num_queries, img_size, subsample):

        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.subsample = subsample
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
            task_emb = data['task_emb']
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(self._max_episode_len, len(observations[i]) if not isinstance(observations[i], dict) else len(observations[i]['pixels']))
                # if obs_type == 'features':
                self._max_state_dim = 123 #123 #77 #47 # max(self._max_state_dim, data['states'][i].shape[-1])
                self._num_samples += len(observations[i]) if self._obs_type == 'features' else len(observations[i]['pixels'])
        
        # # subsample
        # self._max_episode_len = self._max_episode_len // self.subsample + 1
        # ############################################################################
        # self._max_episode_len = 380
        # ############################################################################

        # augmentation
        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop(self.img_size, scale=(0.9, 1.)),
            transforms.ToTensor(),
        ])
        
    def _sample_episode(self, env_idx=None):
        idx = random.choice(list(self._episodes.keys())) if env_idx is None else env_idx
        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes['observation']
        actions = episodes['action']
        task_emb = episodes['task_emb']

        if self._obs_type == 'pixels':
            # Sample obs, action
            sample_idx = np.random.randint(0, len(observations['pixels'])-self._history_len)
            sampled_pixel = observations['pixels'][sample_idx:sample_idx+self._history_len]
            # sampled_pixel_egocentric = observations['pixels_egocentric'][sample_idx:sample_idx+self._history_len]
            sampled_pixel = torch.stack([self.aug(sampled_pixel[i]) for i in range(len(sampled_pixel))])
            # sampled_pixel_egocentric = torch.stack([self.aug(sampled_pixel_egocentric[i]) for i in range(len(sampled_pixel_egocentric))])
            # sampled_proprioceptive_state = np.concatenate(
            #     [observations['eef_states'][sample_idx:sample_idx+self._history_len],
            #      observations['gripper_states'][sample_idx:sample_idx+self._history_len]], axis=-1
            # )
            # TODO: Test after correct data collected
            sampled_proprioceptive_state = observations['proprio'][sample_idx:sample_idx+self._history_len] 
            # sampled_proprioceptive_state = np.zeros((self._history_len, 9)) #observations['proprio'][sample_idx:sample_idx+self._history_len] 
            if self._temporal_agg:
                # arrange sampled action to be of shape (history_len, num_queries, action_dim)
                sampled_action = np.zeros((self._history_len, self._num_queries, actions.shape[-1]))
                num_actions = self._history_len + self._num_queries - 1 # -1 since its num_queries including the last action of the history
                act = np.zeros((num_actions, actions.shape[-1]))
                act[:min(len(actions), sample_idx+num_actions) - sample_idx] = actions[sample_idx:sample_idx+num_actions]
                # sampled_action = np.lib.stride_tricks.sliding_window_view(act, (self._history_len, actions.shape[-1]))
                sampled_action = np.lib.stride_tricks.sliding_window_view(act, (self._num_queries, actions.shape[-1]))
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[sample_idx:sample_idx+self._history_len]

            if self._prompt == None or self._prompt == 'text':
                # print(sampled_pixel.shape, sampled_pixel_egocentric.shape, sampled_proprioceptive_state.shape, sampled_action.shape)
                return {
                    'pixels': sampled_pixel,
                    # 'pixels_egocentric': sampled_pixel_egocentric,
                    'proprioceptive': sampled_proprioceptive_state,
                    'actions': sampled_action,
                    # 'mask': mask if self._temporal_agg else None,
                    'task_emb': task_emb,
                }
            elif self._prompt == 'goal':
                prompt_episode = self._sample_episode(env_idx)
                prompt_observations = prompt_episode['observation']
                prompt_pixel = self.aug(prompt_observations['pixels'][-1])[None]
                # prompt_pixel_egocentric = self.aug(prompt_observations['pixels_egocentric'][-1])[None]
                # prompt_proprioceptive_state = np.concatenate(
                #     [prompt_observations['eef_states'][-1:],
                #      prompt_observations['gripper_states'][-1:]], axis=-1
                # )
                prompt_proprioceptive_state = prompt_observations['proprio'][-1:]
                prompt_action = prompt_episode['action'][-1:]
                return {
                    'pixels': sampled_pixel,
                    'pixels_egocentric': sampled_pixel_egocentric,
                    'proprioceptive': sampled_proprioceptive_state,
                    'actions': sampled_action,
                    'prompt_pixels': prompt_pixel,
                    # 'prompt_pixels_egocentric': prompt_pixel_egocentric,
                    'prompt_proprioceptive': prompt_proprioceptive_state,
                    'prompt_actions': prompt_action,
                    'task_emb': task_emb,
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
            if self._prompt == None or self._prompt == 'text':
                return {
                    'features': sampled_obs,
                    'actions': sampled_action,
                    'task_emb': task_emb,
                }
            elif self._prompt == 'goal':
                prompt_episode = self._sample_episode(env_idx)
                prompt_obs = np.array(prompt_episode['observation'][-1:])
                prompt_action = prompt_episode['action'][-1:]
                return {
                    'features': sampled_obs,
                    'actions': sampled_action,
                    'prompt_obs': prompt_obs,
                    'prompt_actions': prompt_action,
                    'task_emb': task_emb,
                }
    
    def sample_test(self, env_idx):
        episode = self._sample_episode(env_idx)
        observations = episode['observation']
        actions = episode['action']
        task_emb = episode['task_emb']
        
        if self._obs_type == 'pixels':
            pixels_shape = observations['pixels'].shape
            
            # observation
            if self._prompt == None or self._prompt == 'text':
                prompt_pixel = None
                # prompt_pixel_egocentric = None
                prompt_proprioceptive_state = None
                prompt_action = None
            elif self._prompt == 'goal':
                prompt_pixel = np.transpose(observations['pixels'][-1:], (0,3,1,2))
                # prompt_pixel_egocentric = np.transpose(observations['pixels_egocentric'][-1:], (0,3,1,2))
                # prompt_proprioceptive_state = np.concatenate(
                #     [observations['eef_states'][-1:],
                #     observations['gripper_states'][-1:]], axis=-1
                # )
                prompt_proprioceptive_state = observations['proprio'][-1:]
                prompt_action = None
            return {
                'prompt_pixels': prompt_pixel,
                # 'prompt_pixels_egocentric': prompt_pixel_egocentric,
                'prompt_proprioceptive': prompt_proprioceptive_state,
                'prompt_actions': prompt_action,
                'task_emb': task_emb,
            }

        elif self._obs_type == 'features':
            # observation
            if self._prompt == None or self._prompt == 'text':
                prompt_obs, prompt_action = None, None
            elif self._prompt == 'goal':
                prompt_obs = np.array(observations[-1:])
                prompt_action = None

            return {
                'prompt_features': prompt_obs,
                'prompt_actions': prompt_action,
                'task_emb': task_emb,
            }

    def __iter__(self):
        while True:
            yield self._sample()
    
    def __len__(self):
        return self._num_samples

class MocoDataset(IterableDataset):
    def __init__(self, obs_type, path, suites, task_name, num_demos_per_task, img_size=128):

        self._obs_type = obs_type
        self.img_size = img_size

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
            task_emb = data['task_emb']
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(self._max_episode_len, len(observations[i]) if not isinstance(observations[i], dict) else len(observations[i]['pixels']))
                # if obs_type == 'features':
                self._max_state_dim = max(self._max_state_dim, data['states'][i].shape[-1])
                self._num_samples += len(observations[i]) if self._obs_type == 'features' else len(observations[i]['pixels'])
        
        # split train/eval
        eval_perc = 0.1
        self._train_idx = np.random.choice(list(self._paths.keys()), int(len(self._paths.keys())*(1-eval_perc)), replace=False)
        self._eval_idx = [idx for idx in list(self._paths.keys()) if idx not in self._train_idx]
        # save train and eval names
        with open('train_names.txt', 'w') as f:
            for idx in self._train_idx:
                f.write(f"{idx}:{self._paths[idx]}\n")
        with open('eval_names.txt', 'w') as f:
            for idx in self._eval_idx:
                f.write(f"{idx}:{self._paths[idx]}\n")


        # augmentation
        import agent.ssl.utils.moco.loader as loader
        crop_min = 0.6 #0.08
        # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        augmentation1 = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(self.img_size, scale=(crop_min, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=1.0),
            transforms.ToTensor(),
            normalize
        ]
        augmentation2 = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(self.img_size, scale=(crop_min, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomSolarize(0.5, p=0.2),
            transforms.ToTensor(),
            normalize
        ]
        self.aug = loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                            transforms.Compose(augmentation2))
        
    def _sample_episode(self, env_idx=None):
        idx = random.choice(list(self._episodes.keys())) if env_idx is None else env_idx
        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode
    
    def _sample(self, train=True):
        env_idx = random.choice(self._train_idx) if train else random.choice(self._eval_idx)
        episode = self._sample_episode(env_idx)
        observations = episode['observation']
        actions = episode['action']
        task_emb = episode['task_emb']

        if self._obs_type == 'pixels':
            sample_idx = np.random.randint(0, len(observations['pixels']))
            # augment
            # reverse channel order to match bc training
            sampled_pixel = self.aug(observations['pixels'][sample_idx])
            sampled_pixel_egocentric = self.aug(observations['pixels_egocentric'][sample_idx])
            sampled_pixel = torch.stack([sampled_pixel[0], sampled_pixel[1]], dim=0).numpy()
            sampled_pixel_egocentric = torch.stack([sampled_pixel_egocentric[0], sampled_pixel_egocentric[1]], dim=0).numpy()
            sampled_proprioceptive_state = np.concatenate(
                [observations['eef_states'][sample_idx],
                 observations['gripper_states'][sample_idx]], axis=-1
            )
            sampled_action = actions[sample_idx]
            return {
                'pixels': sampled_pixel,
                'pixels_egocentric': sampled_pixel_egocentric,
                'proprioceptive': sampled_proprioceptive_state,
                'actions': sampled_action,
                'task_emb': task_emb,
            }
        elif self._obs_type == 'features':
            sample_idx = np.random.randint(0, len(observations))
            features = observations[sample_idx]
            actions = actions[sample_idx]
            return {
                'features': features,
                'actions': actions,
                'task_emb': task_emb,
            }

    def __iter__(self):
        while True:
            yield self._sample()
    
    def __len__(self):
        return self._num_samples
    
class VIPDataset(IterableDataset):
    def __init__(self, obs_type, path, suites, task_name, num_demos_per_task, img_size, doaug):

        self._obs_type = obs_type
        self.img_size = img_size
        self.doaug = doaug

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
            task_emb = data['task_emb']
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(self._max_episode_len, len(observations[i]) if not isinstance(observations[i], dict) else len(observations[i]['pixels']))
                # if obs_type == 'features':
                self._max_state_dim = max(self._max_state_dim, data['states'][i].shape[-1])
                self._num_samples += len(observations[i]) if self._obs_type == 'features' else len(observations[i]['pixels'])
        
        # split train/eval
        eval_perc = 0.1
        self._train_idx = np.random.choice(list(self._paths.keys()), int(len(self._paths.keys())*(1-eval_perc)), replace=False)
        self._eval_idx = [idx for idx in list(self._paths.keys()) if idx not in self._train_idx]
        # save train and eval names
        with open('train_names.txt', 'w') as f:
            for idx in self._train_idx:
                f.write(f"{idx}:{self._paths[idx]}\n")
        with open('eval_names.txt', 'w') as f:
            for idx in self._eval_idx:
                f.write(f"{idx}:{self._paths[idx]}\n")


        # augmentation
        self.preprocess = torch.nn.Sequential(
                        transforms.Resize(int(self.img_size * 1.142)), #256),
                        transforms.CenterCrop(self.img_size), #224)
                )
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                # transforms.ToPILImage(),
                # transforms.RandomResizedCrop(224, scale = (0.2, 1.0)),
                transforms.RandomResizedCrop(self.img_size, scale = (0.2, 1.0)),
                # transforms.ToTensor(),
            )
        else:
            # self.aug = torch.nn.Sequential(
            #     transforms.ToPILImage(),
            #     transforms.ToTensor(),
            # )
            self.aug = lambda x: x
        
    def _sample_episode(self, env_idx=None):
        idx = random.choice(list(self._episodes.keys())) if env_idx is None else env_idx
        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode
    
    def _sample(self, train=True):
        env_idx = random.choice(self._train_idx) if train else random.choice(self._eval_idx)
        episode = self._sample_episode(env_idx)
        observations = episode['observation']
        task_emb = episode['task_emb']

        # indices 
        vidlen = len(observations['pixels']) if self._obs_type == 'pixels' else len(observations)
        start_idx = np.random.randint(0, vidlen-2)
        end_idx = np.random.randint(start_idx+1, vidlen)
        s0_idx_vip = np.random.randint(start_idx, end_idx)
        s1_idx_vip = min(s0_idx_vip+1, end_idx)

        # Self-supervised reward (this is always -1)
        reward = float(s0_idx_vip == end_idx) - 1

        if self._obs_type == 'pixels':
            # agentview cam
            im0 = torch.tensor(observations['pixels'][start_idx] / 255.0).transpose(0,2).transpose(1,2)
            img = torch.tensor(observations['pixels'][end_idx] / 255.0).transpose(0,2).transpose(1,2)
            imts0_vip = torch.tensor(observations['pixels'][s0_idx_vip] / 255.0).transpose(0,2).transpose(1,2)
            imts1_vip = torch.tensor(observations['pixels'][s1_idx_vip] / 255.0).transpose(0,2).transpose(1,2)

            # egocentric cam
            ego_im0 = torch.tensor(observations['pixels_egocentric'][start_idx] / 255.0).transpose(0,2).transpose(1,2)
            ego_img = torch.tensor(observations['pixels_egocentric'][end_idx] / 255.0).transpose(0,2).transpose(1,2)
            ego_imts0_vip = torch.tensor(observations['pixels_egocentric'][s0_idx_vip] / 255.0).transpose(0,2).transpose(1,2)
            ego_imts1_vip = torch.tensor(observations['pixels_egocentric'][s1_idx_vip] / 255.0).transpose(0,2).transpose(1,2)

            if self.doaug == "rctraj":
                ### Encode each image in the video at once the same way
                allims = torch.stack([im0, img, imts0_vip, imts1_vip], 0)
                allims_aug = self.aug(allims)

                im0 = allims_aug[0]
                img = allims_aug[1]
                imts0_vip = allims_aug[2]
                imts1_vip = allims_aug[3]

                allims = torch.stack([ego_im0, ego_img, ego_imts0_vip, ego_imts1_vip], 0)
                allims_aug = self.aug(allims)

                ego_im0 = allims_aug[0]
                ego_img = allims_aug[1]
                ego_imts0_vip = allims_aug[2]
                ego_imts1_vip = allims_aug[3]
            
            else:
                ### Encode each image individually
                im0 = self.aug(im0)
                img = self.aug(img)
                imts0_vip = self.aug(imts0_vip)
                imts1_vip = self.aug(imts1_vip)

                ego_im0 = self.aug(ego_im0)
                ego_img = self.aug(ego_img)
                ego_imts0_vip = self.aug(ego_imts0_vip)
                ego_imts1_vip = self.aug(ego_imts1_vip)
            
            im = torch.stack([im0, img, imts0_vip, imts1_vip])
            ego_im = torch.stack([ego_im0, ego_img, ego_imts0_vip, ego_imts1_vip])

            # proprocess
            im = self.preprocess(im)
            ego_im = self.preprocess(ego_im)

            return {
                'pixels': im,
                'pixels_egocentric': ego_im,
                'task_emb': task_emb,
                'reward': reward,
            }

        elif self._obs_type == 'features':
            f0 = observations[start_idx]
            fg = observations[end_idx]
            fts0_vip = observations[s0_idx_vip]
            fts1_vip = observations[s1_idx_vip]
            features = torch.stack([f0, fg, fts0_vip, fts1_vip])
            return {
                'features': features,
                'task_emb': task_emb,
                'reward': reward,
            }

    def __iter__(self):
        while True:
            yield self._sample()
    
    def __len__(self):
        return self._num_samples