import os
import sys
import numpy as np
import pandas as pd
import random
import collections
from tqdm import tqdm
import argparse

import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import matplotlib.pyplot as plt


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# class DQNnet(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         '''Args:
#             state_dim  int: state data's last dimension
#             hidden_dim  List[int]: hidden dimension of every hidden layer
#             action_dim  int:output action data's last dimension  
#         '''
#         super(DQNnet, self).__init__()
#         self.num_layers = len(hidden_dim) + 1
#         self.layers = nn.ModuleList(nn.Linear(in_channels, out_channels) for in_channels, out_channels in zip([state_dim] + hidden_dim, hidden_dim + [action_dim]))
#         self.__init_parameters__()

#     def forward(self, x):
#         for idx, layer in enumerate(self.layers):
#             if idx < self.num_layers - 1:
#                 x = F.leaky_relu(layer(x))
#             else:
#                 x = layer(x)
#         return x
    
#     def __init_parameters__(self):
#         # initialize the parameters of the modules
#         for p in self.layers.parameters():
#             if p.dim() > 1:
#                 init.xavier_uniform_(p)


class DuelDQNnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        '''Args:
            state_dim  int: state data's last dimension
            hidden_dim  List[int]: hidden dimension of every hidden layer
            action_dim  int:output action data's last dimension  
        '''
        super(DuelDQNnet, self).__init__()
        self.action_dim = action_dim
        self.pre_layers = nn.ModuleList(nn.Linear(in_channels, out_channels) for in_channels, out_channels in zip([state_dim] + hidden_dim[:-1], hidden_dim))
        self.A_head = torch.nn.Linear(hidden_dim[-1], action_dim)
        self.V_head = torch.nn.Linear(hidden_dim[-1], 1)
        self.__init_parameters__()

    def forward(self, x):
        for layer in self.pre_layers:
            x = F.leaky_relu(layer(x))
        A_value = self.A_head(x)
        V_value = self.V_head(x)
        Q_value = torch.Tensor.repeat(V_value,[1, self.action_dim]) + A_value - torch.mean(A_value, dim=1).view(-1, 1) # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q_value
        
    def __init_parameters__(self):
        # initialize the parameters of the modules
        for p in self.pre_layers.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)        
        for p in self.A_head.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)
        for p in self.V_head.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)


class ReplayBuffer:
    ''' experience buffer for DQN learning '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def clear(self):
        self.buffer.clear()

    def sample(self, sample_size):
        transitions = random.sample(self.buffer, sample_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
    
    
class DQN_agent:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update_frequency, device, env_name='naive'):
        # if mode == 'dueling':
        #     self.target_DQN_net = DuelDQNnet(state_dim, hidden_dim, action_dim).to(device)
        #     self.DQN_net = DuelDQNnet(state_dim, hidden_dim, action_dim).to(device)
        # elif mode == 'naive':
        #     self.target_DQN_net = DQNnet(state_dim, hidden_dim, action_dim).to(device)
        #     self.DQN_net = DQNnet(state_dim, hidden_dim, action_dim).to(device)  
        # else:
        #     raise ValueError("Unknown DQN algorithm mode")
        self.target_DQN_net = DuelDQNnet(state_dim, hidden_dim, action_dim).to(device)
        self.DQN_net = DuelDQNnet(state_dim, hidden_dim, action_dim).to(device)

        self.target_DQN_net.to(device)
        self.DQN_net.to(device)
        
        self.action_dim = action_dim
        self.env_name = env_name
        self.optimizer = torch.optim.Adam(self.DQN_net.parameters(), lr=lr)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon-greedy rate
        self.target_update_frequency = target_update_frequency
        self.count = 0  # counter for step num
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
            action = torch.argmax(self.DQN_net(state)).item()
        return action

    def update(self, transition_dict):
        # get split record data
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        max_next_q_values = self.target_DQN_net(next_states).max(dim=1)[0].view(-1, 1)
        # here become a single value tensor
        q_values = self.DQN_net(states).gather(1, actions)  # Q(s,a)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        self.optimizer.zero_grad()
        dqn_loss = F.mse_loss(q_values, q_targets)
        dqn_loss.backward() 
        self.optimizer.step()

        if self.count % self.target_update_frequency == 0:
            self.target_DQN_net.load_state_dict(self.DQN_net.state_dict())
        self.count += 1

    def save_DQN_net(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{self.env_name}_DQN.pth")
        torch.save(self.DQN_net, output_path)


def train(agent: DQN_agent, buffer: ReplayBuffer, env: gym.Env, epochs: int, sample_size: int, minimal_size: int = 0, save_dir = "model"):
    return_list = []
    num_bar = 10
    update_frequency = 10
    best_return = -sys.maxsize - 1
    for bar_idx in range(num_bar):
        with tqdm(total=int(epochs / num_bar), desc=f'Iteration {bar_idx+1}') as pbar:
            for episode in range(int(epochs / num_bar)):
                episode_return = 0
                state, _ = env.reset()
                # env.render()
                done = False
                # do actions to get an episode train
                count = 0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # for stability, only when buffer size > minimal_size, agent will update parameters
                    if buffer.size() > minimal_size:
                        states, actions, rewards, next_states, dones = buffer.sample(sample_size)
                        transition_dict = {
                            'states': states,
                            'actions': actions,
                            'rewards': rewards,
                            'next_states': next_states,
                            'dones': dones
                        }
                        agent.update(transition_dict)
                if best_return < episode_return:
                    agent.save_DQN_net(save_dir)
                    best_return = episode_return
                # update reward
                pbar.set_postfix({'episode': f'{epochs / num_bar * bar_idx + episode + 1}', 'return': f'{episode_return}'})
                pbar.update(1)    
                print('\n', episode_return)
                return_list.append(episode_return)
    return return_list


def evaluate(agent: DQN_agent, env: gym.Env, test_num: int):
    total_reward = 0
    for epoch in range(test_num):
        print(epoch)
        epoch_reward = 0
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ ,_ = env.step(action)
            state = next_state
            epoch_reward += reward
        total_reward += epoch_reward
    env.close()
    return total_reward / test_num


if __name__=="__main__":
    '''
    # 2 kicker with down when take down or none action
    action space:
    0:NOOP do nothing,  1:FIRE fire all ,2:UP let 2 kicker and right bar up a bit
    3:RIGHT up right kicker, 4:LEFT up left kicker, 5:DOWN hold right bar and 2 kicker down a bit
    6:UPFIRE release right bar and kick center, 7:RIGHTFIRE release right bar and kick right, 8:LEFTFIRE release right bar and kick left
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='VideoPinball-ramNoFrameskip-v4', choices=['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'])
    args = parser.parse_args()

    # parameter setting
    env = gym.make(args.env_name, obs_type="ram", full_action_space=False)

    lr = 1e-2
    epochs = 1000
    state_dim = env.observation_space.shape[0]
    hidden_dim = [64, 64, 16, 4]
    action_dim = env.action_space.n

    gamma = 0.9
    epsilon = 0.01
    target_update_frequency = 10
    buffer_size = 10000
    minimal_size = 200
    sample_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test_num = 10
    save_dir = args.env_name

    seed = 42
    seed_all(seed)

    # train the agent
    print(f"start training {args.env_name} with DQN")
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DQN_agent(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update_frequency, device, args.env_name)
    returns_list = train(agent, replay_buffer, env, epochs, sample_size, minimal_size, save_dir)
    print(f"finish training {args.env_name} with DQN")

    # evaluate the agent
    print(f"start evaluating {args.env_name} with DQN")
    eval_reward_list = evaluate(agent, env, test_num)
    print("test average rewards:", eval_reward_list)
    print(f"finish evaluating {args.env_name} with DQN")

    # draw figure of rewards curve
    episodes_list = range(len(returns_list))
    plt.plot(episodes_list, returns_list)
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.title('Episode——DQN Reward Curve')
    fig_name = f'{args.env_name}_DQN'
    plt.savefig(fig_name)
    plt.clf()
    plt.show()
