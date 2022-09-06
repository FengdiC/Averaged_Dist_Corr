import torch
from torch import nn
import numpy as np

class AvgDiscount(nn.Module):
    def __init__(self,o_dim,hidden,scale=1.0):
        super(AvgDiscount,self).__init__()
        self.body = nn.Sequential(nn.Linear(o_dim,hidden),nn.ReLU()
                              ,nn.Linear(hidden,hidden),nn.ReLU())
        self.weight = nn.Sequential(nn.Linear(hidden,1))
        self.scale = scale

    def forward(self,obs):
        obs = obs.float()
        body = self.body(obs)
        weight = self.weight(body)
        weight = torch.sigmoid(weight)
        return torch.squeeze(weight)/self.scale

class AvgDiscount_sigmoid(nn.Module):
    def __init__(self,o_dim,hidden,scale=1.0):
        super(AvgDiscount_sigmoid,self).__init__()
        self.body = nn.Sequential(nn.Linear(o_dim,hidden),nn.ReLU()
                              ,nn.Linear(hidden,hidden),nn.ReLU())
        self.weight = nn.Sequential(nn.Linear(hidden,1))
        self.scale = scale

    def forward(self,obs):
        obs = obs.float()
        body = self.body(obs)
        weight = self.weight(body)
        weight = torch.sigmoid(weight)
        return torch.squeeze(weight)/self.scale

class AvgDiscount_ReLU(nn.Module):
    def __init__(self,o_dim,hidden,scale=10.0):
        super(AvgDiscount_sigmoid,self).__init__()
        self.body = nn.Sequential(nn.Linear(o_dim,hidden),nn.ReLU()
                              ,nn.Linear(hidden,hidden),nn.ReLU())
        self.weight = nn.Sequential(nn.Linear(hidden,1),nn.ReLU)
        self.scale = scale

    def forward(self,obs):
        obs = obs.float()
        body = self.body(obs)
        weight = self.weight(body)
        return torch.squeeze(weight)/self.scale

class AvgDiscount_tanh(nn.Module):
    def __init__(self,o_dim,hidden,scale=1.0):
        super(AvgDiscount_sigmoid,self).__init__()
        self.body = nn.Sequential(nn.Linear(o_dim,hidden),nn.ReLU()
                              ,nn.Linear(hidden,hidden),nn.ReLU())
        self.weight = nn.Sequential(nn.Linear(hidden,1))
        self.scale = scale

    def forward(self,obs):
        obs = obs.float()
        body = self.body(obs)
        weight = self.weight(body)
        weight = ( torch.tanh(weight)+1 )/2.0
        return torch.squeeze(weight)/self.scale