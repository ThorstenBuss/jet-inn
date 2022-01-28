import os

import numpy as np
from scipy.stats import special_ortho_group

import torch
from torch import nn
import torch.nn.functional as F

class Subnet(nn.Module):

    def __init__(self, dims_in, dims_out, hidden, dropout=0.0):
        super(Subnet, self).__init__()

        self.dims_in = dims_in
        self.dims_out = dims_out

        self.fc1 = nn.Linear(dims_in, hidden)
        self.fc2 = nn.Linear(hidden,  dims_out)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2.weight.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class INN(nn.Module):

    def __init__(self, data_dim, num_layers, hidden, dropout=0.0):
        super(INN, self).__init__()

        self.data_dim = data_dim
        self.num_layers = num_layers

        self.splits = (data_dim - (data_dim // 2), data_dim // 2)

        self.subnets = nn.ModuleList()
        self.transformations = nn.ParameterList()

        cache_file = f'cache/orthos_{data_dim:d}_dim_{num_layers:d}_layer.npy'

        if data_dim > 100:
            if not os.path.isfile(cache_file):
                os.makedirs('cache', exist_ok=True)
                np.save(
                    cache_file,
                    special_ortho_group.rvs(data_dim, size=num_layers)
                )

            orthos = np.load(cache_file)
        else:
            orthos = special_ortho_group.rvs(data_dim, size=num_layers)

        for i in range(self.num_layers):
            self.subnets.append(Subnet(self.splits[0], 2 * self.splits[1], hidden, dropout))
            ortho = nn.Parameter(torch.FloatTensor(orthos[i]), requires_grad=False)
            self.transformations.append(ortho)

    def forward(self, x, rev=False):
        log_jac_det = 0

        for i in range(self.num_layers):
            if rev:
                i = self.num_layers - i - 1
                x = F.linear(x, torch.transpose(self.transformations[i],0,1))
            x1, x2 = torch.split(x, self.splits, dim=1)
            a1 = 0.1*self.subnets[i](x1)

            log_jac = 2 * torch.tanh(a1[:, :self.splits[1]])
            if not rev:
                x2 = x2 * torch.exp(log_jac) + a1[:, self.splits[1]:]
            else:
                x2 = (x2 - a1[:, self.splits[1]:]) / torch.exp(log_jac)

            x = torch.cat((x1, x2), 1)
            log_jac_det += torch.sum(log_jac, dim=1)

            if not rev:
                x = F.linear(x, self.transformations[i])

        if not rev:
            return x, log_jac_det
        else:
            return x, -log_jac_det
