import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from utils.NoiseLayer import NoisyDense

import typing


class RainbowDQN(nn.Module):
    def __call__(self, *inp, **kwargs) -> typing.Any:
        return super().__call__(*inp, **kwargs)

    def __init__(
            self, inp_dim: int,
            out_dim: int,
            V_min: float = -10.0,
            V_max: float = 10.0,
            lr: float = 1e-3,
            num_atoms: int = 51,
            noisy: bool = True,
            image: bool = False):
        """Input: (1, inp_dim)"""
        super().__init__()

        self.V_min = V_min
        self.V_max = V_max

        self.num_atoms = num_atoms
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.noisy = noisy
        if noisy:
            wL = NoisyDense
        else:
            wL = nn.Linear
        
        self.image = image
        if image:
            self.feature_extractor = models.resnet50(pretrained=True)
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor = None

        self.fea1 = nn.Linear(inp_dim, 256)
        self.fea2 = nn.Linear(256, 128)

        self.adv1 = wL(128, 128)
        self.adv2 = wL(128, out_dim * num_atoms)

        self.val1 = wL(128, 128)
        self.val2 = wL(128, num_atoms)

        self.lr = lr
        self.opt = optim.Adam(self.parameters(), lr=self.lr)

    def predict(self, state: torch.Tensor):
        if self.image:
            state = self.feature_extractor(state).view(-1)
        state = F.relu(self.fea2(F.relu(self.fea1(state))))
        advantage = self.adv2(F.relu(self.adv1(state)))
        value = self.val2(F.relu(self.val1(state)))

        value = value.view(-1, 1, self.num_atoms)
        advantage = advantage.view(-1, self.out_dim, self.num_atoms)
        q = advantage + value - advantage.mean(dim=1, keepdim=True)
        q = F.softmax(q.view(-1, self.num_atoms), dim=-1).view(-1, self.out_dim, self.num_atoms)
        return q

    def reset_noise(self):
        if self.noisy:
            self.adv1.reset_noise()
            self.adv2.reset_noise()
            self.val1.reset_noise()
            self.val2.reset_noise()


class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(ActorNet, self).__init__()
        self.num_layers = len(hidden_dim) + 1
        self.layers = nn.ModuleList(nn.Linear(in_channels, out_channels) for in_channels, out_channels in zip([state_dim] + hidden_dim, hidden_dim + [action_dim]))
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            if idx < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = F.tanh(layer(x))
        return x * self.action_bound


class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CriticNet, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_layers = len(hidden_dim) + 1
        self.layers = nn.ModuleList(nn.Linear(in_channels, out_channels) for in_channels, out_channels in zip([state_dim + action_dim] + hidden_dim, hidden_dim + [1]))

    def forward(self, x, a):
        x = x.view(-1,self.state_dim)
        a = a.view(-1,self.action_dim)
        cat = torch.cat([x, a], dim=1)
        for idx, layer in enumerate(self.layers):
            if idx < self.num_layers - 1:
                cat = F.relu(layer(cat))
            else:
                cat = layer(cat)
        return cat


