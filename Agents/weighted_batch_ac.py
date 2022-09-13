import torch

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from Components.utils import A
from Components.buffer import Buffer
from Networks.weight import AvgDiscount, AvgDiscount_sigmoid, AvgDiscount_ReLU, AvgDiscount_tanh
from Networks.actor_critic import MLPCategoricalActor, NNCategoricalActor
from Networks.actor_critic import NNGammaCritic, NNGaussianActor

class WeightedBatchActorCritic(A):
    # the current code works for shared networks with categorical actions only
    def __init__(self,lr,gamma,BS,o_dim,n_actions,hidden,args,device=None,shared=False):
        super(WeightedBatchActorCritic,self).__init__(lr=lr,gamma=gamma,BS=BS,o_dim=o_dim,n_actions=n_actions,
                                              hidden=hidden,args=args,device=device,shared=shared)
        self.network = MLPCategoricalActor(o_dim,n_actions,hidden,shared)
        if args.weight_activation == 'sigmoid':
            self.weight_network = AvgDiscount_sigmoid(o_dim,hidden,args.scale_weight)
        elif args.weight_activation == 'ReLU':
            self.weight_network = AvgDiscount_ReLU(o_dim,hidden,args.scale_weight)
        else:
            self.weight_network = AvgDiscount_tanh(o_dim, hidden, args.scale_weight)
        self.network.to(device)
        self.weight_network.to(device)
        self.opt = torch.optim.Adam(self.network.parameters(),lr=lr)  #decay schedule?
        self.weight_opt = torch.optim.Adam(self.weight_network.parameters(), lr=args.lr_weight)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10000, gamma=0.9)

    def update(self,closs_weight):
        # input: data  Job: finish one round of gradient update
        _, self.next_values,_ = self.network.forward(torch.from_numpy(self.next_frames).to(self.device),
                                                     torch.from_numpy(self.actions).to(self.device))
        self.new_lprobs, self.values,_ = self.network.forward(torch.from_numpy(self.frames).to(self.device),
                                                              torch.from_numpy(self.actions).to(self.device))

        self.dones = torch.from_numpy(self.dones).to(self.device)
        # returns =  torch.from_numpy(self.rewards) + [(1-self.dones)* self.args.gamma+self.dones*self.args.gamma**2]*self.next_values.detach()
        returns = torch.from_numpy(self.rewards).to(self.device) + (1 - self.dones) * self.gamma * self.next_values.detach()
        self.closs = closs_weight*torch.mean((returns-self.values)**2)

        self.weights = self.weight_network.forward(torch.from_numpy(self.frames).to(self.device))
        pobj = self.new_lprobs * (returns - self.values).detach() *self.weights.detach() *self.buffer_size * (1-self.gamma)
        self.ploss = -torch.mean(pobj)
        self.opt.zero_grad()
        self.ploss.backward()
        self.closs.backward()
        self.opt.step()

    def update_weight(self,scale=1.0):
        # input: data  Job: finish one round of gradient update
        self.weights = self.weight_network.forward(torch.from_numpy(self.frames).to(self.device))
        self.labels = self.gamma**self.times

        self.wloss = torch.mean((torch.from_numpy(self.labels).to(self.device)*scale-self.weights)**2)
        self.weight_opt.zero_grad()
        self.wloss.backward()
        self.weight_opt.step()

    def create_buffer(self,env):
        # Create the buffer
        self.buffer_size=self.args.buffer
        o_dim = env.observation_space.shape[0]
        self.buffer = Buffer(self.args.gamma,self.args.lam, o_dim, 0, self.args.buffer)

    def act(self,op):
        a, lprob = self.network.act(torch.from_numpy(op).to(self.device))
        return a, lprob.detach()

    def store(self,op,r,done,a,lprob,time):
        self.buffer.add(op, r, done, a, lprob.item(), time)

    def learn(self,count,obs):
        # Update
        if count == self.buffer_size:
            self.buffer.add_last(obs)
            for epoch in range(self.args.epoch_weight):
                self.buffer.shuffle()
                for turn in range(self.buffer_size // self.BS):  # buffer_size//self.BS
                    # value functions may not be well learnt
                    self.frames, self.rewards, self.dones, self.actions, self.old_lprobs, self.times, self.next_frames \
                        = self.buffer.sample(self.BS, turn)
                    self.update_weight(self.args.scale_weight)
                    # self.scheduler.step()
            # update policy and action values
            for epoch in range(1):
                self.buffer.shuffle()
                for turn in range(1):  # buffer_size//self.BS
                    # value functions may not be well learnt
                    self.frames, self.rewards, self.dones, self.actions, self.old_probs, self.times, self.next_frames \
                        = self.buffer.sample(self.BS, turn)
                    self.update(self.args.LAMBDA_2)
                    # self.scheduler.step()
            self.buffer.empty()

            # print("ploss is: ", self.ploss.detach().numpy(), ":::", self.closs.detach().numpy())
            loss = float(self.ploss.detach().cpu().numpy() + self.closs.detach().cpu().numpy())
            count = 0
            return loss,count
        else:
            return 0,count


class SharedWeightedCriticBatchAC(A):
    def __init__(self, lr, gamma, BS, o_dim, n_actions, hidden, args, device=None, shared=False, continuous=False):
        super(SharedWeightedCriticBatchAC, self).__init__(lr=lr, gamma=gamma, BS=BS, o_dim=o_dim, n_actions=n_actions,
                                hidden=hidden, args=args, device=device, shared=shared)
        self.args = args
        self.continuous = continuous

        if continuous:
            self.network = NNGaussianActor(o_dim, n_actions, hidden,device)
        else:
            self.network = NNCategoricalActor(o_dim, n_actions, hidden,shared)

        self.weight_critic = NNGammaCritic(o_dim, hidden, args.scale_weight)
        self.network.to(device)
        self.weight_critic.to(device)
        self.opt = torch.optim.Adam(self.network.parameters(), lr=lr)  #decay schedule?
        self.weight_critic_opt = torch.optim.Adam(self.weight_critic.parameters(), lr=args.lr_weight)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10000, gamma=0.9)

    def update(self, closs_weight, scale=1.0):
        # input: data  Job: finish one round of gradient update
        self.next_values, self.weights = self.weight_critic(torch.from_numpy(self.next_frames).to(self.device))
        self.values, self.weights = self.weight_critic(torch.from_numpy(self.frames).to(self.device))
        self.new_lprobs, _ = self.network(torch.from_numpy(self.frames).to(self.device),
                                          torch.from_numpy(self.actions).to(self.device))

        self.dones = torch.from_numpy(self.dones).to(self.device)
        returns = torch.from_numpy(self.rewards).to(self.device) +(1 - self.dones) * self.gamma * self.next_values.detach()
                
        # Actor loss
        pobj = self.new_lprobs * (returns - self.values).detach() *self.weights.detach() *self.buffer_size * (1-self.gamma)
        self.ploss = -torch.mean(pobj)
        self.opt.zero_grad()
        self.ploss.backward()        
        self.opt.step()

        # Weight & Critic loss
        ## Critic
        self.closs = closs_weight*torch.mean((returns-self.values)**2)
        ## Weight
        self.labels = self.gamma**self.times
        self.wloss = torch.mean((torch.from_numpy(self.labels).to(self.device)  *scale-self.weights)**2)
        self.weight_critic_opt.zero_grad()
        (self.closs + self.wloss).backward()
        self.weight_critic_opt.step()

    def create_buffer(self,env):
        # Create the buffer
        self.buffer_size=self.args.buffer
        o_dim = env.observation_space.shape[0]
        n_actions = 0
        if self.continuous:
            n_actions = env.action_space.shape[0]
        self.buffer = Buffer(self.args.gamma,self.args.lam, o_dim, n_actions, self.args.buffer)
        
    def act(self,op):
        a, lprob = self.network.act(torch.from_numpy(op).to(self.device)  )
        return a, lprob.detach()

    def store(self,op,r,done,a,lprob,time):
        self.buffer.add(op, r, done, a, lprob.item(), time)

    def learn(self,count,obs):
        # Update
        if count == self.buffer_size:
            self.buffer.add_last(obs)            
            for epoch in range(1):
                self.buffer.shuffle()
                for turn in range(1):  # buffer_size//self.BS
                    # value functions may not be well learnt
                    self.frames, self.rewards, self.dones, self.actions, self.old_probs, self.times, self.next_frames \
                        = self.buffer.sample(self.BS, turn)
                    self.update(self.args.LAMBDA_2,self.args.scale_weight)
                    # self.scheduler.step()
            self.buffer.empty()

            # print("ploss is: ", self.ploss.detach().numpy(), ":::", self.closs.detach().numpy())
            loss = float(self.ploss.detach().cpu().numpy() + self.closs.detach().cpu().numpy())
            count = 0
            return loss,count
        else:
            return 0,count