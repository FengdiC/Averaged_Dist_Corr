import numpy as np
import sys
import itertools
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from Components import utils, logger
from train import train
import mujoco

param = {'lr': [0.0003,0.0001],'buffer':[1024,2048,3072],'LAMBDA_2':[1,10,20]}

args = utils.argsparser()
# env, gamma, continuous are decided through args input

args.env='Hopper-v4'
args.agent='ppo'
args.naive=False
args.continuous=True
args.buffer = 2048
args.batch_size = 64
args.lr = 0.0003
args.scale_weight = 1.0
args.LAMBDA_2 = 10.0
args.lr_weight = 0.003
args.gamma = 0.99
args.continuous=True

logger.configure(args.log_dir, ['csv'], log_suffix='Hopper-ppo-tune2')

for values in list(itertools.product(param['lr'], param['buffer'], param['LAMBDA_2'])):
    args.lr= values[0]
    args.buffer = values[1]
    args.LAMBDA_2 = values[2]

    if args.agent == 'weighted_ppo' and args.LAMBDA_2!=10:
        continue
    seeds = range(5)
    returns = []

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
