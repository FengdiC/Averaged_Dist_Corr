import torch
from torch import nn
import numpy as np

class AvgDiscount(nn.Module):
    def __init__(self,o_dim,hidden):
        super(AvgDiscount,self).__init__()
        self.body = nn.Sequential(nn.Linear(o_dim,hidden),nn.ReLU()
                              ,nn.Linear(hidden,hidden),nn.ReLU())
        self.weight = nn.Sequential(nn.Linear(hidden,1))

    def forward(self,obs):
        obs = obs.float()
        body = self.body(obs)
        weight = self.weight(body)
        weight = torch.sigmoid(weight)
        return torch.squeeze(weight)