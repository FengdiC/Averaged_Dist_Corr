import numpy as np
import itertools
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from Components import utils, logger
from train import train

# param = {'lr_weight':[0.0001,0.0003,0.003,0.01],'weight_activation':['sigmoid','ReLU','tanh'],
#          'scale_weight':[1.0,10.0,100.0]}

param = {'agent': ['batch_ac_shared_gc', 'batch_ac'], 'naive': [True, False],
         'env': ['Acrobot-v1','MountainCarContinuous-v0']}

args = utils.argsparser()
# env, gamma, continuous are decided through args input

args.buffer = 64
args.batch_size = 64
args.lr = 0.0003
args.scale_weight = 10.0
args.LAMBDA_2 = 10.0
args.lr_weight = 0.003
args.gamma = 0.99

logger.configure(args.log_dir, ['csv'], log_suffix='classic-control-repeated')

for values in list(itertools.product(param['agent'], param['naive'], param['env'])):
    args.agent = values[0]
    args.naive = values[1]
    args.env = values[2]
    args.continuous=False
    seeds = range(30)
    returns = []

    if args.agent=='batch_ac' and args.naive==False:
        args.buffer= 32
        args.batch_size = 32
    if args.agent=='batch_ac' and args.naive==True:
        args.buffer= 16
        args.batch_size = 16
    if args.agent == 'batch_ac_shared_gc' and args.naive == True:
        continue
    if args.env in ['MountainCarContinuous-v0', 'Pendulum-v1']:
        args.continuous = True

    for seed in seeds:
        args.seed = seed

        checkpoint = 10000
        result = train(args)

        ret = np.array(result)
        print(ret.shape)
        returns.append(ret)
        name = [str(k) for k in values]
        name.append(str(seed))
        print("hyperparam", '-'.join(name))
        logger.logkv("hyperparam", '-'.join(name))
        for n in range(ret.shape[0]):
            logger.logkv(str((n + 1) * checkpoint), ret[n])
        logger.dumpkvs()

    ret = np.array(returns)
    print(ret.shape)
    ret = np.mean(ret, axis=0)
    name = [str(k) for k in values]
    name.append('mean')
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n + 1) * checkpoint), ret[n])
    logger.dumpkvs()
