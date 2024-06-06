import os
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm
import collections

import torch
import matplotlib.pyplot as plt

import gymnasium as gym

from utils.agent import DDPGAgent


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v4', choices=['Hopper-v4', 'Humanoid-v4', 'HalfCheetah-v4', 'Ant-v4'])
    parser.add_argument('--episodes', type=int, default=10000, help="training episodes")
    parser.add_argument('--buffer_feed', type=int, default=100, help="when to learn from buffer")
    parser.add_argument('--batch_size', type=int, default=64, help="replay sample size")
    parser.add_argument('--actor_lr', type=float, default=3e-4, help="actor learning rate")
    parser.add_argument('--critic_lr', type=float, default=3e-3, help="critic learning rate")
    parser.add_argument('--gpu_order', type=int, default=0, help="which gpu to load model")

    args = parser.parse_args()

    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    num_episodes = args.episodes
    actor_hidden_dim = [64]
    critic_hidden_dim = [64, 32]
    gamma = 0.9
    tau = 0.005  
    buffer_size = 10000
    minimal_size = args.buffer_feed
    batch_size = args.batch_size
    sigma = 0.01  
    device = torch.device(f"cuda:{args.gpu_order}") if torch.cuda.is_available() else torch.device("cpu")
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0] 

    agent = DDPGAgent(state_dim, actor_hidden_dim, critic_hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, buffer_size, batch_size, minimal_size, args.env_name, device)

    return_list = agent.train_off_policy_agent(env, num_episodes)
    episodes_list = list(range(len(return_list)))


    # draw figure
    plt.plot(episodes_list, return_list)
    plt.ylabel('Episode Reward')
    plt.xlabel('Episodes')
    plt.title('DDPG  ' + args.env_name)
    fig_name = f'DDPG_{args.env_name}'
    plt.savefig(fig_name)
    plt.clf()
    plt.show()