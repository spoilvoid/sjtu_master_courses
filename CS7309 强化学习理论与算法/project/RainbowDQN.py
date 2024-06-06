import os
import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt

import torch

from utils.agent import RainbowDQNAgent


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='VideoPinball-ramNoFrameskip-v4', choices=['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'])
    parser.add_argument('--steps', type=int, default=50000, help="max steps per episode")
    parser.add_argument('--episodes', type=int, default=10000, help="training episodes")
    parser.add_argument('--buffer_feed', type=int, default=1000, help="when to learn from buffer")
    parser.add_argument('--copy_freq', type=int, default=500, help="frequency to copy to target net")
    parser.add_argument('--n_step', type=int, default=2, help="step n before learning")
    parser.add_argument('--noisy', action='store_true', help="whether choose to noisy layer")
    parser.add_argument('--batch_size', type=int, default=128, help="replay sample size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--gpu_order', type=int, default=0, help="which gpu to load model")
    
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_order}") if torch.cuda.is_available() else torch.device("cpu")
    agent = RainbowDQNAgent(
        args.env_name,
        device,
        V_min=-10.0,
        V_max=10.0,
        batch_size=args.batch_size,
        copy_every=args.copy_freq,
        play_before_learn=args.buffer_feed,    
        n_step=args.n_step,
        lr=args.lr,
        num_atoms=51,
        noisy=args.noisy,
        image=False
    )
    results, losses = agent.train(args.steps, args.episodes)

    episodes_list = range(len(results))
    plt.plot(episodes_list, results)
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.suptitle('RainbowDQN  ' + args.env_name)
    fig_name = f'RainbowDQN_{args.env_name}'
    plt.savefig(fig_name)
    plt.clf()
    plt.show()
