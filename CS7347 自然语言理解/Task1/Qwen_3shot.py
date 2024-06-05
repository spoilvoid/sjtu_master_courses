from gradio_client import Client
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import alfworld
import json
import os
import logging
import random
import numpy as np


def seed_all(seed):
	random.seed(seed)
	np.random.seed(seed)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


DEFAULT_ROLE_PALY_PROMPT = "Suppose you are a person in a virtual world tasked with completing a specific mission. You will be provided with background information about the game, details of your task and a list of admissible actions. In order to solve the task, you need to break the task down into a series of admissible actions. Every turn, you should think or output an admissible action with '> ', e.g. '> go to armchair 1' and continue until the task is completed. If your answer begins with 'think', You will not take any action and receive an 'OK.\n"
DEFAULT_QUERY_PROMPT = "What should you do next?"


def play_alfworld_game(model_client, environment, system_prompt, init_obs, init_info, logger, max_steps=50, store_file="2shot_history_action.txt"):
	obs = init_obs
	info = init_info
	# get environment task prompt
	env_task_prompt = '\n'.join(obs[0].split('\n\n')[1:]) + '\n'
	logger.info(f"env_task_prompt: {env_task_prompt}")

	# initialization of variables
	action_response_pairs = "History actions and responses:\n"
	result_str = ""
	for step in range(1, max_steps+1):
		logger.info(f"step: {step}")
		# collect admissible actions to str
		admissible_actions = "Admissible action list:\n"
		for action in info['admissible_commands'][0]:
			admissible_actions += f'> {action}\n'
		# setup query
		query = env_task_prompt + action_response_pairs + admissible_actions + DEFAULT_QUERY_PROMPT

		# LLM predict next action
		_, history, _ = model_client.predict(
			query=query,
			history=[],
			system=system_prompt,
			api_name="/model_chat"
		)
		action = history[-1][1]
		logger.info(f"LLM output: {action}")
		record_flag = False
		action_flag = False
		symbol_flag = False
		if '> think: ' in action:
			thought = action.split('> think: ')[1].split('\n')[0]
			action = action.replace('> think: '+ thought, "")
			logger.info(f"thought: {thought}")
			action_response_pairs += f"> think: {thought}\nOK.\n"
			result_str = "OK."
			record_flag = True
		elif 'think: ' in action:
			thought = action.split('think: ')[1].split('\n')[0]
			action = action.replace('think: '+ thought, "")
			action_response_pairs += f"> think: {thought}\nOK.\n"
			result_str = "OK."
			record_flag = True
		# if action contains "> sth.". It indicates LLM's action attempt
		if '> ' in action:
			symbol_flag = True
			# extract action from response
			action = action.split('> ')[1].split('\n')[0]
			print(f'first part action: {action}')
			# check if there is an admissible action
			max_match_len = 0
			max_match_action = ""
			for admissible_action in info['admissible_commands'][0]:
				if admissible_action in action and len(admissible_action)> max_match_len:
					max_match_action = admissible_action
					max_match_len = len(admissible_action)
					action_flag = True
			if max_match_len > 0:
				action = max_match_action
				logger.info(f"action: {action}")
			else:
				action_flag = False
			print(f'matched action: {max_match_action}')
			admissible_actions = info['admissible_commands'][0]
			print(f'admissable actions: {admissible_actions}')
		if action_flag:
			# take action in environment
			obs, reward, done, info = environment.step([action])
			# check if task is done
			if done[0]:
				logger.info("Task is finished")
				# write messages to a file
				with open(store_file, 'a') as f:
					f.write(str(history) + '\n')
				return reward
			# process new observation into string and remove loc information
			result_str = obs[0]
			if result_str.startswith('You arrive at loc '):
				result_str = result_str[result_str.find('. ')+2:]
			action_response_pairs += f"> {action}\n{result_str}\n"
		if not record_flag and not action_flag:
			action = action.replace('\n', ' ')
			if not symbol_flag:
				obs, reward, done, info = environment.step([action])
				result_str = obs[0]
				if result_str.startswith('You arrive at loc '):
					result_str = result_str[result_str.find('. ')+2:]
				if "Nothing happens" in result_str:
					result_str = "Incorrect action. Take action should start with '>'."
				else:
					action_response_pairs += f"> {action}\n{result_str}\n"
			else:
				result_str = "Nothing happens"
			action_response_pairs += f"> {action}\n{result_str}\n"
		# add new action to action_response_prompt
		logger.info(f"Result: {result_str}")

	# write messages to a file
	with open(store_file, 'a') as f:
		f.write(str(history) + '\n')
	return (0,)


if __name__=="__main__":
	seed_all(42)
	# load config
	config = generic.load_config()
	env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

	# setup model client
	client = Client("Qwen/Qwen1.5-110B-Chat-demo")

	# setup environment
	split = "eval_out_of_distribution"
	env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
	env = env.init_env(batch_size=1)

	# set up logger
	log_dir = "3shot"
	store_file = '3shot.txt'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	logger = get_logger('log', log_dir)

	PREFIXES = {
		'pick_and_place': 'put',
		'pick_clean_then_place': 'clean',
		'pick_heat_then_place': 'heat',
		'pick_cool_then_place': 'cool',
		'look_at_obj': 'examine',
		'pick_two_obj': 'puttwo'
	}
	count_list = [0, 0, 0, 0, 0, 0]
	# reward 0 means failure, 1 means success
	reward_list = [0, 0, 0, 0, 0, 0]
	for _ in range(1, 135):
		obs, info = env.reset()
		logger.info(f"Task {_} Start")
		# setup system prompt
		game_detail = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
		logger.info(f"Game Detail: {game_detail}")
		with open('prompts/alfworld_3prompts.json', 'r') as f:
			example_database = json.load(f)
		for i, (k, v) in enumerate(PREFIXES.items()):
			if game_detail.startswith(k):
				system_prompt = DEFAULT_ROLE_PALY_PROMPT + 'Here are three examples.\n' + example_database[f'react_{v}_2'] + example_database[f'react_{v}_1'] + example_database[f'react_{v}_0'] + '\n'
				# system_prompt = DEFAULT_ROLE_PALY_PROMPT + 'Here are two examples.\n' + example_database[f'react_{v}_1'] + example_database[f'react_{v}_0'] + '\n'
				# system_prompt = DEFAULT_ROLE_PALY_PROMPT + 'Here is an example.\n' + example_database[f'react_{v}_0'] + '\n'
				#system_prompt = DEFAULT_ROLE_PALY_PROMPT
				logger.info(f"System Prompt: {system_prompt}")

				# play the game with LLM
				reward = play_alfworld_game(client, env, system_prompt, obs, info, logger, store_file=os.path.join(log_dir, store_file))
				reward_list[i] += reward[0]
				count_list[i] += 1

				logger.info(f"Task {_} End")
				logger.info(f"Reward: {reward[0]}\n\n")

	logger.info(f"Count List: {count_list}")
	logger.info(f"Reward List: {reward_list}")

