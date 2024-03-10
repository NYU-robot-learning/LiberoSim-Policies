import metaworld
import random
import  metaworld.policies as policies
import cv2
import numpy as np

import pickle
from pathlib import Path
from collections import deque

from video import VideoRecorder

env_names = ["hammer-v2", "drawer-close-v2", "drawer-open-v2", "door-open-v2", "bin-picking-v2", "button-press-topdown-v2", "door-unlock-v2"]
num_demos = 1
frame_stack = 1
SAVE_DATA_PATH = Path('../../expert_demos/metaworld')

# create save directory
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

POLICY = {
	'hammer-v2': policies.SawyerHammerV2Policy,
	'drawer-close-v2': policies.SawyerDrawerCloseV2Policy,
	'drawer-open-v2': policies.SawyerDrawerOpenV2Policy,
	'door-open-v2': policies.SawyerDoorOpenV2Policy,
	'bin-picking-v2': policies.SawyerBinPickingV2Policy,
	'button-press-topdown-v2': policies.SawyerButtonPressTopdownV2Policy,
	'door-unlock-v2': policies.SawyerDoorUnlockV2Policy
}

CAMERA = {
	'hammer-v2': 'corner3',
	'drawer-close-v2': 'corner',
	'drawer-open-v2': 'corner',
	'door-open-v2': 'corner3',
	'bin-picking-v2': 'corner',
	'button-press-topdown-v2': 'corner',
	'door-unlock-v2': 'corner'
}

NUM_STEPS = {
	'hammer-v2': 125,		
	'drawer-close-v2': 125,
	'drawer-open-v2': 125,
	'door-open-v2': 125,
	'bin-picking-v2': 175,
	'button-press-topdown-v2': 125,
	'door-unlock-v2': 125
}


for env_name in env_names:
	print(f"Generating demo for: {env_name}")
	# Initialize policy
	policy = POLICY[env_name]()

	# Initialize env
	ml1 = metaworld.ML1(env_name) # Construct the benchmark, sampling tasks
	env = ml1.train_classes[env_name]()  # Create an environment with task `pick_place`

	# # Initialize save dir
	# save_dir = SAVE_DATA_PATH / env_name
	# save_dir.mkdir(parents=True, exist_ok=True)

	# Initialize video recorder
	video_recorder = VideoRecorder(SAVE_DATA_PATH, camera_name=CAMERA[env_name])

	images_list = list()
	states_list = list()
	actions_list = list()
	rewards_list = list()

	episode = 0
	video_recorder.init(env)
	while episode < num_demos:
		print(f"Episode {episode}")
		images = list()
		states = list()
		actions = list()
		rewards = list()
		image_stack = deque([], maxlen=frame_stack)
		goal_achieved = 0

		# Set random goal
		task = ml1.train_tasks[episode] #random.choice(ml1.train_tasks)
		env.set_task(task)  # Set task

		# Reset env
		state = env.reset()  # Reset environment
		video_recorder.record(env)
		num_steps = NUM_STEPS[env_name]
		for step in range(num_steps):
			# Get state
			states.append(state)
			# Get frames
			frame = env.render(offscreen=True, camera_name=CAMERA[env_name])
			frame = cv2.resize(frame, (84,84))
			# frame = np.transpose(frame, (2,0,1))
			image_stack.append(frame)
			while(len(image_stack)<frame_stack):
				image_stack.append(frame)
			images.append(np.concatenate(image_stack, axis=0))
			# Get action
			action = policy.get_action(state)
			action = np.clip(action, -1.0, 1.0)
			actions.append(action)
			# Act in the environment
			state, reward, done, info = env.step(action)
			rewards.append(reward)
			video_recorder.record(env)
			goal_achieved += info['success'] 

		# Store trajectory
		episode = episode + 1
		images_list.append(np.array(images))
		states_list.append(np.array(states))
		actions_list.append(np.array(actions))
		rewards_list.append(np.array(rewards))

	video_recorder.save(f'{env_name}.mp4')

	file_path = SAVE_DATA_PATH / f'{env_name}.pkl'
	payload = {
		'observations': images_list,
		'states': states_list,
		'actions': actions_list,
		'rewards': rewards_list
	}


	with open(str(file_path), 'wb') as f:
		pickle.dump(payload, f)

