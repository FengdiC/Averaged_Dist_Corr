import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
granddir = os.path.dirname(parentdir)
sys.path.insert(0, granddir)
from Components.random_search import random_search_cartpole, set_one_thread
from train import train
import numpy as np
import torch, random
from Components import utils, logger
import gym
from config import agents_dict
import itertools
import matplotlib.pyplot as plt

param = {'agent': ['batch_ac_shared_gc', 'batch_ac'], 'naive': [True, False],
         'env': ['MountainCarContinuous-v0','Acrobot-v1', 'CartPole-v1']}
# 'MountainCarContinuous-v0'   'Acrobot-v1'

args = utils.argsparser()
# env, gamma, continuous are decided through args input

logger.configure(args.log_dir, ['csv'], log_suffix='classic-control-discounted-993')
set_one_thread()

for values in list(itertools.product(param['agent'], param['naive'], param['env'])):
    args.agent = values[0]
    args.naive = values[1]
    args.env = values[2]
    args.continuous=False
    seeds = range(10)
    returns = []

    if args.agent=='batch_ac':
        num=32
    else:
        num=153

    hyperparam = random_search_cartpole(num)
    args.buffer = hyperparam['buffer']
    args.batch_size = hyperparam['buffer']
    args.lr = hyperparam['lr']
    args.lr_weight = hyperparam['lr_weight']
    args.scale_weight = hyperparam['scale']
    args.LAMBDA_2 = hyperparam['gamma_coef']
    # args.gamma = hyperparam['gamma']
    args.gamma = 0.993
    args.hidden = hyperparam['hid']
    args.hidden_weight = hyperparam['critic_hid']

    if args.agent == 'batch_ac_shared_gc' and args.naive == True:
        continue
    if args.env in ['MountainCarContinuous-v0', 'Pendulum-v1']:
        args.continuous = True

    for seed in seeds:
        args.seed = seed

        checkpoint = 10000
        result = train(args,repeated=False)

        ret = np.array(result)
        print(ret.shape)
        returns.append(ret)
        name = [str(k) for k in values]
        name.append('993')
        name.append(str(seed))
        print("hyperparam", '-'.join(name))
        logger.logkv("hyperparam", '-'.join(name))
        for n in range(ret.shape[0]):
            logger.logkv(str((n + 1) * checkpoint), ret[n])
        logger.dumpkvs()
    returns = np.array(returns)
    # plt.plot(range(returns.shape[1]),np.mean(returns,axis=0),label = name)
    # plt.show()

