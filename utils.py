from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

class NoisyLinear(nn.Module):

    def __init__(
            self, 
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon, 
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))
        return x.sign().mul(x.abs().sqrt())
    

class DenseNet(nn.Module):

    def __init__(
            self, 
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor,
            hidden_size: int,
            no_dueling=False,
            no_noise=False
    ):
        
        super(DenseNet, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        self.hidden_size = hidden_size

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.ReLU()
        )

        self.advantage_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.advantage_layer = NoisyLinear(self.hidden_size, out_dim * atom_size)

        self.value_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.value_layer = NoisyLinear(self.hidden_size, atom_size)

        self.no_dueling = no_dueling
        self.no_noise = no_noise

        if no_noise:

            self.advantage_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.advantage_layer = nn.Linear(self.hidden_size, out_dim * atom_size)
            self.value_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.value_layer = nn.Linear(self.hidden_size, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )

        if not self.not_dueling:
            value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        else:
            q_atoms = advantage

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist
    
    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

class ConvNet(nn.Module):

    def __init__(
            self,
            in_dim: List[int],
            out_dim: int,
            atom_size: int,
            support: torch.Tensor,
            no_dueling=False,
            no_noise=False
    ):
        super(ConvNet, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.conv1 = nn.Conv2d(in_dim[2], 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, strider=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=8, stride=4):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_dim[1], 8, 4), 4, 2), 3, 1)
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_dim[0], 8, 4), 3, 1))
        self.hidden_size = conv_w * conv_h * 32

        self.advantage_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.advantage_layer = NoisyLiear(self.hidden_size, out_dim * atom_size)

        self.value_hidden_layer = NoisyLinear(self.hidden_size, self.hidden_size)
        self.value_layer = NoisyLinear(self.hidden_size, atom_size)

        self.no_dueling = no_dueling
        self.no_noise = no_noise

        if no_noise:

            self.advantage_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.advantage_layer = nn.Linear(self.hidden_size, out_dim * atom_size)
            self.value_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.value_layer = nn.Linear(self.hidden_size, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:

        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.convv2(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        feature = x.reshape(x.shape[0], -1)


        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        if not self.no_dueling:
            value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
            q_atoms = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_atoms = advantage

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist
    
    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


if __name__ == '__main__':
    dims = [1, 100, 100, 3]
    s1 = torch.FloatTensor(np.random.uniform(size=dims))
    atom_size = 51
    support = torch.linspace(0, 200, atom_size)
    m = ConvNet(dims[1:], 4, atom_size, support)
    o = m(s1)
    print(o)

    dims = 4
    s1 = torch.FloatTensor(np.random.uniform(size=dims))
    atom_size = 51
    support = torch.linspace(0, 200, atom_size)
    m = DenseNet(4, 4, atom_size, support, 100)
    o = m(s1.unsqueeze(0))
    print(o)