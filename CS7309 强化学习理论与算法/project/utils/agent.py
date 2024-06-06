import gymnasium as gym
import numpy as np
import time
import os
import sys


from tqdm import tqdm
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.network import RainbowDQN, ActorNet, CriticNet
from utils.ExpReplay import PriorityReplayBuffer, NormalReplayBuffer


class RainbowDQNAgent:
    def __init__(
            self,
            gym_env: str,
            device,
            V_min: float = -10.0,
            V_max: float = 10.0,
            play_before_learn: int = 3000,
            num_atoms: int = 51,
            gamma: float = 0.99,
            batch_size: int = 100,
            copy_every: int = 500,
            n_step: int = 2,
            lr: float = 1e-3,
            noisy: bool = True,
            image: bool = False
    ):
        self.env_name = gym_env
        self.env = gym.make(gym_env, obs_type='ram')

        self.V_min = V_min
        self.V_max = V_max
        self.num_atoms = num_atoms
        self.support = torch.linspace(V_min, V_max, num_atoms)
        self.dz = (self.V_max - self.V_min) / (self.num_atoms - 1)

        self.replay_buffer = PriorityReplayBuffer(self.env.observation_space.shape)
        self.pbl = play_before_learn
        self.noisy = noisy
        self.image = image
        if image:
            self.obs_n = 1000
        else:
            self.obs_n = sum(self.env.observation_space.shape)
        self.act_n = self.env.action_space.n
        self.n_step = n_step

        epsilon_start = 1.0
        epsilon_final = 0.01
        self.epsilon_decay = lambda: 500
        self.epsilon = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) \
                                         * np.exp(-1. * frame_idx / self.epsilon_decay())
        self.frame_idx = 0

        self.gamma = gamma
        self.copy_every = copy_every
        self.batch_size = batch_size

        self.writer = SummaryWriter()
        self.device = device
        self.q_model = RainbowDQN(
            inp_dim=self.obs_n, out_dim=self.act_n, lr=lr, V_min=V_min, V_max=V_max, num_atoms=num_atoms, noisy=self.noisy, image=self.image
        ).to(device)
        self.target_model = RainbowDQN(
            inp_dim=self.obs_n, out_dim=self.act_n, V_min=V_min, V_max=V_max, num_atoms=num_atoms, noisy=self.noisy,
            image=self.image
        ).to(device)

        self.__copy2target()

    def __del__(self):
        self.writer.close()

    def __copy2target(self):
        self.target_model.load_state_dict(self.q_model.state_dict())

    def act(self, obs: torch.Tensor, train: bool = True):
        if not self.noisy and train and np.random.randn() < self.epsilon(self.frame_idx):
            return np.random.randint(0, self.act_n)
        with torch.no_grad():
            dist = self.q_model.predict(obs.float().to(self.device)).cpu()
            dist = dist * self.support
            action = dist.sum(dim=2).argmax(dim=1).item()
            return action

    def replay(self):
        data = self.replay_buffer.sample(batch_size=self.batch_size)
        indices, states, actions, rewards, next_states, dones, weights = data

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8)).float()

        probs = self.q_model.predict(states.to(self.device)).cpu()
        probs_a = probs[range(self.batch_size), actions]
        probs_a = probs_a + torch.tensor(0.0001)  # zero from softmax, wtf

        with torch.no_grad():
            next_probs = self.q_model.predict(next_states.to(self.device)).cpu()
            dist = self.support.expand_as(next_probs) * next_probs
            ns_a = dist.sum(dim=2).argmax(dim=1)
            if self.noisy: self.target_model.reset_noise()
            next_probs = self.target_model.predict(next_states.to(self.device)).cpu()  # double q learning
            next_probs_a = next_probs[range(self.batch_size), ns_a]

            # compute Tz
            Tz = rewards.view(-1, 1) + (torch.tensor(1) - dones).view(-1, 1) * (self.gamma ** self.n_step) * self.support.view(1, -1)
            Tz = Tz.clamp(min=self.V_min, max=self.V_max)

            b = (Tz - self.V_min) / self.dz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # some magic here
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            m = states.new_zeros([self.batch_size, self.num_atoms])
            offset = torch.linspace(0, ((self.batch_size - 1) * self.num_atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.num_atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (next_probs_a * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (next_probs_a * (b - l.float())).view(-1))

        loss = - torch.sum(m * probs_a.log(), dim=1)
        self.q_model.zero_grad()
        weights = torch.from_numpy(np.array(weights)).float()
        (weights * loss).mean().backward()
        self.q_model.opt.step()

        loss = loss.detach().numpy()
        self.replay_buffer.update(indices=indices, tderrors=loss)
        return np.mean(loss)

    def preprocess_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
        transform_resize = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        pil_image = transform_resize(pil_image)

        # 转换为NumPy数组，并将值缩放到[0, 1]
        image_array = np.array(image) / 255.0

        # 标准化
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform_normalize(image_array)
        input_batch = input_tensor.unsqueeze(0)

        return input_batch

    def train(self, steps: int, episodes: int = 500):
        # 不进行学习的部分不放入训练步数中
        steps += self.pbl

        self.epsilon_decay = lambda: steps // 4

        ep_losses = []
        ep_results = []
        best_ep_sum = -sys.maxsize - 1
        for episode in range(episodes):
            with tqdm(total=steps, desc=f'Iteration {episode+1}') as pbar:
                obs, _ = self.env.reset()
                ep_sum = .0
                losses = .0
                for i in range(steps):
                    # n-step
                    if self.image:
                        action = self.act(self.preprocess_image(obs))
                    else:
                        action = self.act(torch.from_numpy(obs).unsqueeze(0))
                    next_obs, rew, done, _, _ = self.env.step(action)
                    x = 1
                    cur_rew = rew
                    # 走多少步后将经验放入ReplayBuffer中
                    while x < self.n_step and not done:
                        cur_obs = next_obs
                        if self.image:
                            action = self.act(self.preprocess_image(obs))
                        else:
                            action = self.act(torch.from_numpy(cur_obs).unsqueeze(0))
                        next_obs, rew, done, _, _ = self.env.step(action)
                        cur_rew += rew
                        x += 1
                    ep_sum += rew
                    self.replay_buffer.append((obs, action, rew, next_obs, done))
                    obs = next_obs

                    # 超过阈值后开始学习经验
                    if i > self.pbl:
                        self.frame_idx += 1
                        loss = self.replay()
                        self.writer.add_scalar("Rainbow/loss", loss, i)
                        losses += loss

                    # 更新参数至target network
                    if i % self.copy_every == 0:
                        self.__copy2target()

                    pbar.update(1)  
                    # 如果结束了或达到步数上限，则进入下一个episode
                    if done or i == steps-1:
                        ep_results.append(ep_sum)
                        ep_losses.append(losses)
                        self.writer.add_scalar("Rainbow/reward", ep_sum, i)
                        break
                    
                if ep_sum > best_ep_sum:
                    best_ep_sum = ep_sum
                    self.save()
                
                pbar.set_postfix({'episode': f'{episode+1}', 'return': f'{ep_sum}'})
                pbar.update(1)    

        return ep_results, ep_losses

    def show(self):
        self.q_model.train(False)
        obs, _ = self.env.reset()
        while True:
            if self.image:
                act = self.act(self.preprocess_image(obs))
            else:
                act = self.act(torch.from_numpy(obs).unsqueeze(0))
            obs, _, done, _, _ = self.env.step(act)
            obs = obs if not done else self.env.reset()[0]
            time.sleep(0.04)
            self.env.render()

    def save(self):
        output_dir = 'models'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, self.env_name + "_RainbowDQN.pth")
        torch.save(self.q_model, output_path)


class DDPGAgent:
    def __init__(self, state_dim, actor_hidden_dim, critic_hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, buffer_size, batch_size, minimal_size, env_name, device):
        self.actor = ActorNet(state_dim, actor_hidden_dim, action_dim, action_bound).to(device)
        self.critic = CriticNet(state_dim, critic_hidden_dim, action_dim).to(device)
        self.target_actor = ActorNet(state_dim, actor_hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = CriticNet(state_dim, critic_hidden_dim, action_dim).to(device)
        
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = NormalReplayBuffer(buffer_size)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.minimal_size = minimal_size
        self.env_name = env_name
        self.device = device
        

    def take_action(self, state):
        state = np.array(state, dtype=np.float32).flatten()
        state = torch.tensor(state, dtype=torch.float).to(self.device).view(-1,self.state_dim)
        action = self.actor(state).cpu().detach().numpy().reshape(-1)
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device).view(-1, self.state_dim)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        next_q_values = self.target_critic(next_states, self.target_actor(next_states).view(-1, self.action_dim))
        
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        states = states.view(-1, self.state_dim)
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update Target ActorNet and CriticNet
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def save(self):
        output_dir = 'models'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        actor_output_path = os.path.join(output_dir, self.env_name + "_DDPG_Actor.pth")
        critic_output_path = os.path.join(output_dir, self.env_name + "_DDPG_Critic.pth")
        torch.save(self.actor, actor_output_path)
        torch.save(self.critic, critic_output_path)

    def train_off_policy_agent(self, env, num_episodes):
        return_list = []
        best_ep_sum = -sys.maxsize - 1
        for i in range(10):
            with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes/10)):
                    episode_return = 0
                    state,_ = env.reset()
                    done = False
                    cnt = 0
                    while not done and cnt < 200:
                        cnt += 1
                        action = self.take_action(state)
                        next_state, reward, done, _, _ = env.step(action)
                        #print("take a step:",cnt)
                        self.replay_buffer.add(state.flatten(), action, reward, next_state.flatten(), done)
                        state = next_state
                        episode_return += reward
                        if self.replay_buffer.size() > self.minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                            self.update(transition_dict)
                    if episode_return > best_ep_sum:
                        self.save()
                        best_ep_sum = episode_return
                    return_list.append(episode_return)
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % episode_return})
                    pbar.update(1)
        return return_list
