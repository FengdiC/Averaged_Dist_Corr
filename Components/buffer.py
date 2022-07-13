import torch
from torch import nn
import numpy as np
from Components.utils import A

class Buffer(A):
    def __init__(self,args,o_dim,n_actions=0,size=2000):
        super(Buffer,self).__init__(args=args,o_dim=o_dim,n_actions=n_actions,size=size)
        self.frames= np.zeros((size+1,o_dim))

        if self.n_actions>0:
            self.actions= np.zeros((size,n_actions),np.float32)
        else:
            self.actions = np.zeros(size)
        self.rewards = np.zeros(size)
        self.dones=np.zeros(size)
        self.current_size=0
        self.idx = np.arange(self.size)
        self.old_probs =np.zeros(size)
        self.times = np.zeros(size)

    def empty(self):
        size=self.size
        self.frames = np.zeros((size+1, self.o_dim))

        if self.n_actions > 0:
            self.actions = np.zeros((size, self.n_actions), np.float32)
        else:
            self.actions = np.zeros(size)
        self.rewards = np.zeros(size)
        self.dones = np.zeros(size)
        self.current_size = 0
        self.idx = np.arange(self.size)
        self.old_probs = np.zeros(size)
        self.times = np.zeros(size)

    def add(self,op, reward, done, action, lprob, time):
        # reward = np.sign(reward)*np.log(1+reward)
        self.frames[self.current_size] = op
        if self.n_actions==0:
            self.actions[self.current_size] = int(action)
        else:
            self.actions[self.current_size,:] = action
        self.rewards[self.current_size] = reward
        self.dones[self.current_size] = float(done)
        self.old_probs[self.current_size] = lprob
        self.times[self.current_size] = time
        self.current_size +=1

    def add_last(self,op):
        self.frames[self.current_size] = op

    def sample(self,BS=100,turn=0):
        index= self.idx[turn*BS:(turn+1)*BS]
        self.index = index
        return np.copy(self.frames[index]),np.copy(self.rewards[index]),np.copy(self.dones[index]), \
               np.copy(self.actions[index]),np.copy(self.old_probs[index]),np.copy(self.times[index]),\
               np.copy(self.frames[index+1])

    def shuffle(self):
        self.idx = np.arange(self.size)
        np.random.shuffle(self.idx)

    def all_frames(self):
        return self.frames[:self.size]

    def compute_gae(self,values):
        self.values=values

        # This function is only called when the buffer is full.
        self.returns = np.zeros(self.size)
        self.returns[-1] = self.values[-1]
        for i in reversed(range(self.size-1)):
            # self.returns[i] = self.rewards[i] + \
            #                   ((1-self.dones[i])* self.args.gamma+self.dones[i]*self.args.gamma**2)*self.returns[i+1]
            self.returns[i] = self.rewards[i] + (1 - self.dones[i]) * self.args.gamma * self.returns[i + 1]

        self.advantages = np.zeros(self.size)
        self.advantages[-1] = self.returns[-1] - self.values[-1]
        for i in reversed(range(self.size - 1)):
            # self.advantages[i] = self.rewards[i] + \
            #                      ((1-self.dones[i])* self.args.gamma+self.dones[i]*self.args.gamma**2) * self.values[i + 1] - self.values[i] \
            #                     + ((1-self.dones[i])* self.args.gamma*self.args.lam+self.dones[i]*(self.args.gamma*self.args.lam)**2) * self.advantages[i + 1]
            self.advantages[i] = self.rewards[i] + (1 - self.dones[i]) * self.args.gamma * self.values[i + 1] - self.values[i] \
                                 + (1 - self.dones[i]) * self.args.gamma * self.args.lam * self.advantages[i + 1]
        # self.returns = (self.returns - np.mean(self.returns)) / (np.std(self.returns) + eps)
        self.advantages = (self.advantages - np.mean(self.advantages)) / (np.std(self.advantages) + np.finfo(float).eps)
        return np.copy(self.returns[self.index]), np.copy(self.advantages[self.index])