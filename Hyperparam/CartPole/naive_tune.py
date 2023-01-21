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

seeds = range(30)

args = utils.argsparser()
# env, gamma, continuous are decided through args input

# Torch Shenanigans fix
set_one_thread()
logger.configure(args.log_dir, ['csv'], log_suffix='cartpole_tune_naive-'+str(args.seed))
hyperparam = random_search_cartpole(args.seed)
args.agent='batch_ac'
args.naive=True

args.buffer = hyperparam['buffer']
args.batch_size = hyperparam['buffer']
args.lr = hyperparam['lr']
args.lr_weight = hyperparam['lr_weight']
args.scale_weight = hyperparam['scale']
args.LAMBDA_2 = hyperparam['gamma_coef']
args.gamma =hyperparam['gamma']
args.hidden = hyperparam['hid']
args.hidden_weight = hyperparam['critic_hid']

for seed in seeds:
    args.seed = seed

    checkpoint = 10000
    result = train(args)

    name = list(hyperparam.values())
    name = [str(s) for s in name]
    name.append(str(seed))
    logger.logkv("hyperparam", '-'.join(name))

    ret = np.array(result)
    print(ret.shape)
    print("hyperparam", '-'.join(name))
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n + 1) * checkpoint), ret[n])
    logger.dumpkvs()

