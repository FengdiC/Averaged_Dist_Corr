import torch
from torch import nn
import numpy as np
from Networks.actor_critic import MLPCategoricalActor
import gym
from Components.utils import meanstdnormalizaer, A
from Components.buffer import Buffer
import matplotlib.pyplot as plt

class BatchActorCritic(A):
    # the current code works for shared networks with categorical actions only
    def __init__(self,lr,gamma,BS,o_dim,n_actions,hidden,shared=False):
        super(BatchActorCritic,self).__init__(lr=lr,gamma=gamma,BS=BS,o_dim=o_dim,n_actions=n_actions,
                                              hidden=hidden,shared=shared)
        self.network = MLPCategoricalActor(o_dim,n_actions,hidden,shared)
        self.opt = torch.optim.Adam(self.network.parameters(),lr=lr)  #decay schedule?
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10000, gamma=0.9)

    def update(self,closs_weight,naive=True):
        # input: data  Job: finish one round of gradient update
        _, self.next_values,_ = self.network.forward(torch.from_numpy(self.next_frames),torch.from_numpy(self.actions))
        self.new_lprobs, self.values,_ = self.network.forward(torch.from_numpy(self.frames),torch.from_numpy(self.actions))

        self.dones = torch.from_numpy(self.dones)
        # returns =  torch.from_numpy(self.rewards) + ((1-self.dones)* self.gamma+self.dones*self.gamma**2)*self.next_values.detach()
        returns = torch.from_numpy(self.rewards) + (1 - self.dones) * self.gamma* self.next_values.detach()
        self.closs = closs_weight*torch.mean((returns-self.values)**2)
        if naive:
            pobj = self.gamma**torch.from_numpy(self.times) * self.new_lprobs * (returns - self.values).detach()
        else:
            pobj = self.new_lprobs * (returns - self.values).detach()
        self.ploss = -torch.mean(pobj)
        self.opt.zero_grad()
        self.ploss.backward()
        self.closs.backward()
        self.opt.step()

    def create_buffer(self,env,args,buffer_size):
        # Create the buffer
        self.buffer_size=buffer_size
        self.args=args
        o_dim = env.observation_space.shape[0]
        self.buffer = Buffer(args, o_dim, 0, buffer_size)

    def act(self,op):
        a, lprob = self.network.act(torch.from_numpy(op))
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
                    self.frames, self.rewards, self.dones, self.actions, self.old_lprobs, self.times, self.next_frames \
                        = self.buffer.sample(self.BS, turn)
                    self.update(self.args.LAMBDA_2,self.args.naive)
                    # self.scheduler.step()
            self.buffer.empty()

            # print("ploss is: ", self.ploss.detach().numpy(), ":::", self.closs.detach().numpy())
            loss = float(self.ploss.detach().numpy() + self.closs.detach().numpy())
            count = 0
            return loss,count
        else:
            return 0,count



