import random

import torch
import numpy as np
from Components.utils import argsparser
from config import agents_dict
import matplotlib.pyplot as plt
from Envs.counterexample_cal import pi, TwoStates, batch_weighted_grad,batch_biased_grad,batch_naive_grad, Actor, plot_grad
from Components import logger
import itertools
from Components.utils import meanstdnormalizaer, A
from torch import nn


class NNGammaCritic(torch.nn.Module):
    def __init__(self, o_dim, hidden=16, scale=1.):
        super(NNGammaCritic, self).__init__()
        self.body = nn.Sequential(nn.Linear(o_dim, hidden), nn.ReLU())
        # self.critic = nn.Linear(hidden, 1)
        self.weight = nn.Sequential(nn.Linear(hidden, 1), nn.ReLU())
        self.scale = scale

    def forward(self, obs):
        obs = obs.float()
        body = self.body(obs)
        # value = self.critic(body)
        weight = self.weight(body)
        # weight = torch.sigmoid(weight)
        # weight = (torch.tanh(weight) +1)/2.0
        # return torch.squeeze(value), torch.squeeze(weight)
        return torch.squeeze(weight)

class NaiveAgent(A):
    # the current code works for shared networks with categorical actions only
    def __init__(self,lr,gamma,BS,o_dim,n_actions,hidden,args,device=None,agent='weighted',shared=True,continuous=False):
        super(NaiveAgent,self).__init__(lr=lr,gamma=gamma,BS=BS,o_dim=o_dim,n_actions=n_actions,
                                              hidden=hidden,args=args,device=device,agent=agent,shared=shared,
                                              continuous=continuous)
        self.actor = Actor(device)
        self.weight = NNGammaCritic(o_dim, hidden=16, scale=1)
        self.actor.to(device)
        self.weight.to(device)
        self.opt = torch.optim.Adam(self.weight.parameters(),lr=lr)
        self.states = []
        self.actions = []
        self.times = []
        self.idx = []
        self.buffer_size= self.args.buffer

    def update(self,naive=False):
        # input: data  Job: finish one round of gradient update
        self.correction = self.weight.forward(torch.from_numpy(self.frames).to(self.device))
        self.new_lprobs = self.actor.forward(torch.from_numpy(self.actions).to(self.device))

        if self.agent=='our correction':
            grad = batch_weighted_grad(self.actor.theta.detach().numpy(),self.new_lprobs,self.correction,self.idx,
                              self.actions,self.times,self.args.gamma)
        elif self.agent=='biased':
            grad = batch_biased_grad(self.actor.theta.detach().numpy(),self.new_lprobs,self.correction,self.idx,
                              self.actions,self.times,self.args.gamma)
        elif self.agent=='existing correction':
            grad = batch_naive_grad(self.actor.theta.detach().numpy(),self.new_lprobs,self.correction,self.idx,
                              self.actions,self.times,self.args.gamma)
        else:
            print("The agent is not coded.")
        self.actor.update(grad,self.lr)
        self.wloss = torch.mean((self.correction - torch.from_numpy(self.args.gamma**self.times))**2)
        self.opt.zero_grad()
        self.wloss.backward()
        self.opt.step()

    def act(self):
        a = self.actor.act()
        return a

    def store(self,op,idx,a,time):
        self.states.append(op)
        self.idx.append(idx)
        self.actions.append(a)
        self.times.append(time)

    def learn(self,count,obs):
        # Update
        if count == self.buffer_size:
            for epoch in range(1):
                index =  np.arange(self.buffer_size)
                np.random.shuffle(index)
                for turn in range(1):  # buffer_size//self.BS
                    # value functions may not be well learnt
                    self.frames = np.array(self.states)[index]
                    self.idx = np.array(self.idx)[index]
                    self.actions = np.array(self.actions)[index]
                    self.times = np.array(self.times)[index]
                    self.update()
            self.states = []
            self.actions = []
            self.times = []
            self.idx = []
            count = 0
            return None,count
        else:
            return 0,count

def train(args,agent='our correction'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    seed = args.seed

    # Create Env
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed((seed))
    env = TwoStates()
    o_dim = env.observation_space.n
    a_dim = env.action_space.n

    # Set the agent
    agent = NaiveAgent(args.lr,args.gamma,args.buffer,o_dim,a_dim,8,args,device,agent)

    ret = step = 0
    rets = []
    policies = []
    ep_lens = []
    avgrets = []
    losses = []
    avglos = []
    avgpi = []
    op, idx = env.reset()

    num_steps = 100000
    checkpoint = 20
    num_episode = 0
    count = 0
    time = 0
    old_theta = agent.actor.theta.detach().numpy()
    for steps in range(num_steps):
        # does torch need expand_dims
        a = agent.act()
        agent.store(op, idx, a, time)
        obs, idx, r, done, infos = env.step(a)

        # Observe
        op = obs
        time += 1
        count += 1

        loss, count = agent.learn(count, obs)
        losses.append(loss)

        # End of Episode
        ret += r * args.gamma**time
        # For convience
        # ret += r
        step += 1
        if done:
            num_episode += 1
            rets.append(ret)
            policies.append(pi(agent.actor.theta.detach().numpy()[0]))
            ep_lens.append(step)
            # print("Episode {} ended with return {:.2f} in {} steps. Total steps: {}".format(num_episode, ret, step, steps))

            ret = 0
            step = 0
            time = 0
            op,idx = env.reset()

        if (steps + 1) % checkpoint == 0:
            # plt.rcParams["figure.figsize"]
            # gymdisplay(env,MAIN)
            # avgrets.append(np.mean(rets))
            avgpi.append(np.mean(policies))
            # avglos.append(np.mean(losses))
            rets = []
            policies = []
            losses = []
            # plt.clf()
            # plt.subplot(211)
            # plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            # plt.subplot(212)
            # theta = agent.actor.theta.detach().numpy()
            # plt.arrow(old_theta[0],0,theta[0]-old_theta[0],0,width=.08)
            # old_theta = theta
            # plt.pause(0.005)
    return avgpi

agents = ['our correction','existing correction','biased']
num_steps = 10000
checkpoint = 20
plt.figure()
for agent in agents:
    if agent == 'our correction':
        color = 'orangered'
    elif agent =='existing correction':
        color = 'dodgerblue'
    else:
        color = 'blueviolet'
    args = argsparser()
    rets = train(args,agent)
    plt.plot(range(len(rets)),rets,label=agent,color=color)
plt.legend()
plt.xlabel("training steps")
plt.ylabel("policy probability of the top action")
plt.title("Performance on the counterexample")
plt.show()
# plot_grad()