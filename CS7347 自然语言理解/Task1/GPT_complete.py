from openai import OpenAI
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import alfworld
import json
import os
import logging


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


DEFAULT_ROLE_PALY_PROMPT = "Suppose you are a person in a virtual world tasked with completing a specific mission. You will be provided with background information about the game, details of your task and a list of admissible actions. In order to solve the task, you need to break the task down into a series of admissible actions. ***In every turn, you should think or output an admissible action from given choices, e.g. 'go to armchair 1' and continue until the task is completed.*** If your answer begins with 'think', You will not take any action and receive an 'OK.\n"
DEFAULT_QUERY_PROMPT = "What should you do next?"


def play_alfworld_game(model_client, environment, system_prompt, init_obs, init_info, logger, max_steps=50, message_record="2shot_history_action.txt"):
	obs = init_obs
	info = init_info
	# get environment task prompt
	env_task_prompt = '\n'.join(obs[0].split('\n\n')[1:]) + '\n'
	logger.info(f"task: {env_task_prompt}")

	# initialization of variables
	result_str = ""
	messages = [{"role": "system", "content": system_prompt}]
	messages.append({"role": "user", "content": env_task_prompt})
	messages.append({"role": "assistant", "content": "OK. I will try my best to complete the task."})
	for step in range(1, max_steps):
		logger.info(f"step: {step}")
		print(f"step: {step}")
		# collect admissible actions to str
		admissible_actions = "Action list:\n"
		for i, action in enumerate(info['admissible_commands'][0]):
			admissible_actions += f'{i+1}: {action}\n'
			# if i != 0 and i != len(info['admissible_commands'][0])-1:
			# 	admissible_actions += ", "
			# if i == len(info['admissible_commands'][0])-1:
			# 	admissible_actions += " and " + action + '.\n'
			# else:
			# 	admissible_actions += action
		logger.info(admissible_actions)
		# setup query with environment response action&response history, admissible actions and question
		query = result_str + admissible_actions + DEFAULT_QUERY_PROMPT
		messages.append({"role": "user", "content": query})

		# LLM predict next action
		response = model_client.chat.completions.create(
			model="gpt-3.5-turbo",
			messages=messages,
			temperature=0,
			max_tokens=100,
			top_p=1,
			frequency_penalty=0.0,
			presence_penalty=0.0
		)
		action = response.choices[0].message.content
		messages.append({"role": "assistant", "content": action})
		logger.info(f"Action: {action}")
		# if action is think, do nothing and response OK.
		if action.startswith('think'):
			result_str = 'OK.'
		else:
			# check if response contains admissible action, if not input whole response
			for admissible_action in info['admissible_commands'][0]:
				if admissible_action in action:
					action = admissible_action
					break
			# take action in environment
			obs, reward, done, info = environment.step([action])
			# check if task is done
			if done[0]:
				logger.info("Task is finished")
				# write messages to a file
				with open(message_record, 'a') as f:
					f.write(str(messages) + '\n')
				return reward
			# process new observation into string and remove loc information
			result_str = obs[0]
			if result_str.startswith('You arrive at loc '):
				result_str = result_str[result_str.find('. ')+2:]
			result_str += '\n'

		# add new action to action_response_prompt
		logger.info(f"Result: {result_str}")

	# write messages to a file
	with open(message_record, 'a') as f:
		f.write(str(messages) + '\n')
	return 0


if __name__=="__main__":
	# load config
	config = generic.load_config()
	env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

	# setup model client
	client = OpenAI(api_key="tokens", base_url="connection_url")


	# setup environment
	split = "eval_out_of_distribution"
	env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
	env = env.init_env(batch_size=1)

	# set up logger
	log_dir = "2shot_actionlist"
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
	reward_list = [[], [], [], [], [], []]
	for _ in range(134):
		logger.info(f"Task {_} Start")
		obs, info = env.reset()

		# setup system prompt
		game_detail = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
		logger.info(f"Game Detail: {game_detail}")
		with open('prompts/alfworld_3prompts.json', 'r') as f:
			example_database = json.load(f)
		for i, (k, v) in enumerate(PREFIXES.items()):
			if game_detail.startswith(k):
				system_prompt = DEFAULT_ROLE_PALY_PROMPT + 'Here are two examples.\n' + example_database[f'react_{v}_1'] + example_database[f'react_{v}_0'] + '\n'
				# system_prompt = DEFAULT_ROLE_PALY_PROMPT
				logger.info(f"System Prompt: {system_prompt}")

				# play the game with LLM
				reward = play_alfworld_game(client, env, system_prompt, obs, info, logger)
				reward_list[i].append(reward)
				count_list[i] += 1

				logger.info(f"Task {_} End")
				logger.info(f"Reward: {reward}")
		break

	logger.info(f"Count List: {count_list}")
	logger.info(f"Reward List: {reward_list}")

