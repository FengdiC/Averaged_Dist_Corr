import numpy as np
import utils, logger
import itertools
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from train import train


# param = {'batch_size':[100,200,500,1000],'buffer':[100,200,500,1000,3000],'lr':[0.0003],
#          'LAMBDA_2':[10,40],'epoch':[1]}

param = {'agent':['weighted_ppo','ppo'],'naive':[True, False]}

args = utils.argsparser()
args.env='Hopper-v2'
args.batch_size = 128
args.buffer = 4000
args.lr = 0.0003
args.LAMBDA_2 = 10
args.gamma=0.99
args.continuous = True

logger.configure(args.log_dir,['csv'], log_suffix='cartpole_ppo_naive-hyperparam-tune')

for values in list(itertools.product(param['agent'],param['naive'])):
    args.agent = values[0]
    args.naive = bool(values[1])
    seeds = range(10)
    returns = []
	
    # if args.agent=='batch_ac' and args.epoch>1:
    #     continue
    if args.agent == 'weighted_ppo' and args.naive==True:
        continue

    for seed in seeds:
        args.seed= seed

        checkpoint = 10000
        result =train(args)

        ret = np.array(result)
        print(ret.shape)
        returns.append(ret)
        name = [str(k) for k in values]
        name.append(str(seed))
        print("hyperparam",'-'.join(name))
        logger.logkv("hyperparam",'-'.join(name))
        for n in range(ret.shape[0]):
            logger.logkv(str((n+1)*checkpoint),ret[n])
        logger.dumpkvs()

    ret = np.array(returns)
    print(ret.shape)
    ret = np.mean(ret,axis=0)
    name = [str(k) for k in values]
    name.append('mean')
    logger.logkv("hyperparam",'-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n+1)*checkpoint),ret[n])
    logger.dumpkvs()
