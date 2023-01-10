import random

import torch
import numpy as np

import os,sys, inspect, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from Components.utils import argsparser
from config import agents_dict
import matplotlib.pyplot as plt
from Envs.counterexample_cal import pi, TwoStates, batch_weighted_grad,batch_biased_grad,batch_naive_grad, Actor, plot_grad

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
        self.opt = torch.optim.Adam(self.weight.parameters(),lr=args.lr_weight)
        self.states = []
        self.actions = []
        self.times = []
        self.idx = []
        self.buffer_size= self.args.buffer

    def update(self,naive=False):
        # input: data  Job: finish one round of gradient update
        self.correction = self.weight.forward(torch.from_numpy(self.frames).to(self.device))
        self.wloss = torch.mean((self.correction - torch.from_numpy(self.args.gamma ** self.times)) ** 2)
        self.opt.zero_grad()
        self.wloss.backward()
        self.opt.step()

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

    def act(self,seed):
        a = self.actor.act(seed)
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

def train(args,agent='our correction',seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
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
        a = agent.act(seed)
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


def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 17.0
    plt.rcParams['ytick.labelsize'] = 17.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['ytick.minor.pad'] = 50.0


def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.axes.set_ylim(0,None)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
                   labelsize=16, pad=8)
    ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
                   labelsize=16, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())

def tune(agent):
    lr = [0.95,0.9,0.8,0.5,0.1,0.05]
    lr_weight = [0.1,0.05,0.01,0.008,0.005]
    if agent!='our correction':
        lr_weight=[0]
    buffer = [1,8,20,40]
    rets = []
    for gamma in [0.3, 0.5, 0.7, 0.9]:
        args = argsparser()
        args.gamma = gamma
        max_ret = 0
        for values in list(itertools.product(lr, lr_weight,buffer)):
            args.lr=values[0]
            args.lr_weight = values[1]
            args.buffer = values[2]
            values = [str(s) for s in values]
            name = '-'.join(values)
            for seed in range(5):
                ret = train(args, agent,seed)
                rets += ret
            if np.mean(np.array(ret))> max_ret:
                max_ret = np.mean(np.array(ret))
                print(agent,"gamma:",gamma,":::",name)
    #     plt.plot(range(len(ret)), ret, label=name)
    # plt.legend()
    # plt.xlabel("training steps")
    # plt.ylabel("policy probability of the top action")
    # plt.title("Performance on the counterexample")
    # plt.show()


def compare():
    plt.figure(figsize=(11,6.8),dpi=60)
    setsizes()
    setaxes()

    agents = ['our correction','existing correction','biased']
    num_steps = 100000
    checkpoint = 20
    for agent in agents:
        if agent == 'our correction':
            color = 'tab:orange'
        elif agent =='existing correction':
            color = 'tab:blue'
        else:
            color = 'tab:green'
        args = argsparser()
        rets = []
        for seed in range(2):
            ret = train(args,agent,seed)
            rets.append(ret)
        rets=np.array(rets)
        mean = np.mean(rets, axis=0)
        std = np.std(rets, axis=0) / np.sqrt(30)
        print(rets.shape)
        plt.plot(range(1,rets.shape[1]*20,20), mean, color=color, label=agent)
        plt.fill_between(range(1,rets.shape[1]*20,20), mean - std, mean + std, color=color, alpha=0.2)
    plt.ylim(0, None)
    plt.legend()
    plt.legend(prop={"size": 17})
    plt.xlabel("training steps",fontsize=19)
    plt.ylabel("probability of the optimal action",fontsize=19)
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=17, rotation=45)
    plt.title("Performance on the counterexample",fontsize=19)
    plt.show()
    # plot_grad()

# tune('existing correction') #lr=0.8, buffer=20, gamma=0.99
# tune('our correction') #lr=0.8, buffer=20 lr_weight=0.01, gamma=0.99
tune('biased')
# compare()

"""existing correction gamma: 0.3::: 0.95 - 0 - 1
gamma: 0.5::: 0.95 - 0 - 1
gamma: 0.7::: 0.95 - 0 - 1
gamma: 0.9::: 0.95 - 0 - 8
our correction gamma: 0.3::: 0.95 - 0.008 - 1
our correction gamma: 0.5::: 0.95 - 0.01 - 1
our correction gamma: 0.7::: 0.95 - 0.005 - 1
our correction gamma: 0.9::: 0.5 - 0.005 - 1
"""