import random

import torch
import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from config import agents_dict
import matplotlib.pyplot as plt
from Envs.reacher import DotReacher, DotReacherRepeat
from Components import logger, utils
import itertools
from reacher_config import repeated, episodic

def train(args,env,stepsize=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    seed = args.seed

    # Create Env
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed((seed))
    # env = DotReacherRepeat(stepsize=stepsize)
    o_dim = env.observation_space.shape[0]
    if args.continuous:
        a_dim= env.action_space.shape[0]
    else:
        a_dim = env.action_space.n

    # Set the agent
    network = agents_dict[args.agent]
    if args.continuous:
        agent = network(args.lr, args.gamma, args.batch_size, o_dim, a_dim, args.hidden,args,device,continuous=True)
    else:
        agent = network(args.lr,args.gamma,args.batch_size,o_dim,a_dim,args.hidden,args,device)

    # Experiment block starts
    # Create the buffer
    agent.create_buffer(env)

    ret = 0
    rets = []
    errs= []
    errs_buffer=[]
    err_ratios= []
    avgrets = []
    losses = []
    avglos = []
    avgerr = []
    avgerr_buffer = []
    avgerr_ratio = []
    op = env.reset()

    num_steps = 10000
    checkpoint = 1000
    num_episode = 0
    count = 0
    time = 0
    for steps in range(num_steps):
        # does torch need expand_dims
        a, lprob = agent.act(op)
        obs, r, done, infos = env.step(a)
        agent.store(op,r,done,a,lprob,time)

        # Observe
        op = obs
        time += 1
        count += 1

        loss,count=agent.learn(count,obs)
        losses.append(loss)


        # End of Episode
        # ret += r * args.gamma**time
        # For convience
        ret += r
        if done:
            num_episode += 1
            rets.append(ret)
            ret = 0
            time = 0
            op = env.reset()

        if (steps + 1) % checkpoint == 0:
            rets = test(agent,env)
            avgrets.append(np.mean(rets))
            rets = []
    return avgrets

def test(agent,env):
    rets= 0
    op = env.reset()
    r=-1
    done=False
    time =0
    while r!=0 and not done and time<100:
        a, lprob = agent.act(op)
        obs, r, done, infos = env.step(a)
        rets+= r
        time+=1
    return rets


param = {'agent': ['batch_ac_shared_gc', 'batch_ac',"weighted_batch_ac"], 'naive': [True, False]}
args = utils.argsparser()

for values in list(itertools.product(param['agent'], param['naive'])):
    args.agent = values[0]
    args.naive = values[1]
    returns = []
    if args.agent=='batch_ac' and args.naive==True:
        name = 'naive'
    elif args.agent=='batch_ac' and args.naive==False:
        name = 'biased'
    else:
        name = args.agent
    if args.agent != 'batch_ac' and args.naive == True:
        continue

    hyperparam = repeated[name]
    args.buffer = hyperparam['buffer']
    args.batch_size = hyperparam['buffer']
    args.lr = hyperparam['lr']
    args.lr_weight = hyperparam['lr_weight']
    args.scale_weight = hyperparam['scale']
    args.LAMBDA_2 = hyperparam['gamma_coef']
    args.gamma = hyperparam['gamma']
    args.hidden = hyperparam['hid']
    args.hidden_weight = hyperparam['critic_hid']
    env = DotReacherRepeat(stepsize=0.2)
    returns = []

    for run_seed in range(10):
        args.seed = run_seed
        rets = train(args,env)
        returns.append(rets)
    returns = np.array(returns)
    plt.plot(range(returns.shape[1]),np.mean(returns,axis=0),label='repeated-'+name)

    env = DotReacher(stepsize=0.2)
    hyperparam = episodic[name]
    args.buffer = hyperparam['buffer']
    args.batch_size = hyperparam['buffer']
    args.lr = hyperparam['lr']
    args.lr_weight = hyperparam['lr_weight']
    args.scale_weight = hyperparam['scale']
    args.LAMBDA_2 = hyperparam['gamma_coef']
    args.gamma = hyperparam['gamma']
    args.hidden = hyperparam['hid']
    args.hidden_weight = hyperparam['critic_hid']
    env = DotReacher(stepsize=0.2)
    returns = []

    for run_seed in range(10):
        args.seed = run_seed
        print(run_seed)
        rets = train(args,env)
        returns.append(rets)
    returns = np.array(returns)
    plt.plot(range(returns.shape[1]), np.mean(returns, axis=0), label='episodic-' + name)

plt.legend()
plt.show()