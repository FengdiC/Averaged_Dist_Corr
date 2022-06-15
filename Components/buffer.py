import torch
from torch import nn
import numpy as np
from Components.utils import A

class Buffer(A):
    def __init__(self,args,o_dim,n_actions=0,size=2000):
        super(Buffer,self).__init__(args=args,o_dim=o_dim,n_actions=n_actions,size=size)
        self.frames= np.zeros((size+1,o_dim))
        self.actions = np.zeros(size)
        if self.n_actions>0:
            self.action= np.zeros((size,n_actions),np.float32)
        self.rewards = np.zeros(size)
        self.dones=np.zeros(size)
        self.current_size=0
        self.idx = np.arange(self.size)
        self.old_probs =np.zeros(size)
        self.times = np.zeros(size)

    def empty(self):
        size=self.size
        self.frames = np.zeros((size+1, self.o_dim))
        self.actions = np.zeros(size)
        if self.n_actions > 0:
            self.action = np.zeros((size, self.n_actions), np.float32)
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
            self.actions[self.current_size,:] = np.array(action)
        self.rewards[self.current_size] = reward
        self.dones[self.current_size] = float(done)
        self.old_probs[self.current_size] = lprob
        self.times[self.current_size] = time
        self.current_size +=1

    def add_last(self,op):
        self.frames[self.current_size] = op

    def sample(self,BS=100,turn=0):
        index= self.idx[turn*BS:(turn+1)*BS]
        return np.copy(self.frames[index]),np.copy(self.rewards[index]),np.copy(self.dones[index]), \
               np.copy(self.actions[index]),np.copy(self.old_probs[index]),np.copy(self.times[index]),\
               np.copy(self.frames[index+1])

    def shuffle(self):
        self.idx = np.arange(self.size)
        np.random.shuffle(self.idx)