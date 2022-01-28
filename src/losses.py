import torch
import torch.nn as nn

class INN_loss(nn.Module):

    def __init__(self, num_dim):
        super(INN_loss, self).__init__()
        self.num_dim = num_dim

    def forward(self, Z, log_jac_det):
        losses = 0.5*torch.sum(Z**2, 1) - log_jac_det
        loss = losses.mean() / self.num_dim
        return loss
