import torch
from torch import nn
import numpy as np
from Networks.actor_critic import MLPCategoricalActor,MLPGaussianActor
import gym
from Components.utils import meanstdnormalizaer, A
from Components.buffer import Buffer
import matplotlib.pyplot as plt

# The current version of PPO does not have entropy loss !!!!!

class PPO(A):
    # the current code works for categorical actions only
    def __init__(self,lr,gamma,BS,o_dim,n_actions,hidden,args,device,shared=False,continuous=False):
        super(PPO,self).__init__(lr=lr,gamma=gamma,BS=BS,o_dim=o_dim,n_actions=n_actions,
                                              hidden=hidden,args=args,device=device,shared=shared,
                                              continuous=continuous)
        if continuous:
            self.network = MLPGaussianActor(o_dim,n_actions,hidden,shared,device)
        else:
            self.network = MLPCategoricalActor(o_dim,n_actions,hidden,shared)
        self.network.to(device)
        self.opt = torch.optim.Adam(self.network.parameters(),lr=lr)  #decay schedule?
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=100000, gamma=0.9)

    def update(self,closs_weight,entropy_weight,naive=True):
        # input: data  Job: finish one round of gradient update
        _, self.next_values,_ = self.network.forward(torch.from_numpy(self.next_frames).to(self.device),
                                                     torch.from_numpy(self.actions).to(self.device))
        self.new_lprobs, self.values, entropy = self.network.forward(torch.from_numpy(self.frames).to(self.device),
                                                                     torch.from_numpy(self.actions).to(self.device))

        # Compute clipped gradients
        ratio = torch.exp(self.new_lprobs - torch.from_numpy(self.old_lprobs).to(self.device))
        original = ratio * torch.from_numpy(self.advantages).to(self.device)
        clip = torch.clip(ratio, 1 - 0.2, 1 + 0.2)
        if naive:
            self.ploss = -torch.mean(self.gamma**torch.from_numpy(self.times).to(self.device) *
                                     torch.minimum(original, clip * torch.from_numpy(self.advantages).to(self.device))) + \
                         entropy_weight * torch.mean(entropy)

        else:
            self.ploss = -torch.mean(torch.minimum(original, clip * torch.from_numpy(self.advantages).to(self.device))) +\
                         entropy_weight * torch.mean(entropy)

        # minimize TD squared error or mse between returns and values???
        # self.dones = torch.from_numpy(self.dones)
        # self.returns =  torch.from_numpy(self.rewards) + self.gamma*(1-self.dones)*self.next_values.detach()
        self.closs = closs_weight*torch.mean((torch.from_numpy(self.returns).to(self.device)-self.values)**2)

        self.opt.zero_grad()
        self.ploss.backward()
        self.closs.backward()
        self.opt.step()

    def create_buffer(self,env):
        # Create the buffer
        self.buffer_size=self.args.buffer
        o_dim = env.observation_space.shape[0]
        if self.continuous:
            self.buffer = Buffer(self.args.gamma,self.args.lam, o_dim, self.n_actions, self.args.buffer)
        else:
            self.buffer = Buffer(self.args.gamma,self.args.lam,o_dim, 0, self.args.buffer)

    def act(self,op):
        a, lprob = self.network.act(torch.from_numpy(op).to(self.device))
        return a, lprob.detach()

    def store(self,op,r,done,a,lprob,time):
        self.buffer.add(op, r, done, a, lprob.item(), time)

    def learn(self,count,obs):
        # Update
        if count == self.buffer_size:
            self.buffer.add_last(obs)
            self.all_frames = self.buffer.all_frames()
            if self.continuous:
                _, self.values, _ = self.network.forward(torch.from_numpy(self.all_frames).to(self.device),
                                                            torch.from_numpy(np.zeros((self.buffer_size,self.n_actions))).to(self.device))
            else:
                _,self.values,_ = self.network.forward(torch.from_numpy(self.all_frames).to(self.device),
                                                    torch.from_numpy(np.zeros(self.buffer_size)).to(self.device))
            self.buffer.compute_gae(self.values)
            for epoch in range(self.args.epoch):
                self.buffer.shuffle()
                for turn in range(self.buffer_size//self.BS):  # buffer_size//self.BS
                    # value functions may not be well learnt
                    self.frames, self.rewards, self.dones, self.actions, self.old_lprobs, self.times, self.next_frames \
                        = self.buffer.sample(self.BS, turn)
                    self.returns,self.advantages = self.buffer.sample_adv()
                    self.update(self.args.LAMBDA_2,self.args.LAMBDA_1,self.args.naive)
                    # self.scheduler.step()
                # print("ploss is: ", self.ploss.detach().numpy(), ":::", self.closs.detach().numpy())
            self.buffer.empty()

            loss = float(self.ploss.detach().cpu().numpy() + self.closs.detach().cpu().numpy())
            count = 0
            return loss,count
        else:
            return 0,count



