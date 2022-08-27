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

param = {'buffer':[1024,2048,4096],'lr':[0.000125,0.0003,0.000625],'epoch_weight':[5,10,15,20]}

# param = {'agent':['weighted_ppo','ppo'],'naive':[True, False]}

args = utils.argsparser()
# env, gamma, continuous are decided through args input

args.batch_size = 64
# args.buffer = 2048
# args.lr = 0.0003
args.LAMBDA_2 = 10
args.agent='weighted_ppo'

logger.configure(args.log_dir,['csv'], log_suffix=str(args.env)+'-ppo-avg-param')

for values in list(itertools.product(param['buffer'],param['lr'],param['epoch_weight'])):
    args.buffer = values[0]
    args.lr = values[1]
    args.epoch_weight = values[2]
    seeds = range(10)
    returns = []
	
    # if args.agent=='batch_ac' and args.epoch>1:
    #     continue
    # if args.agent == 'weighted_ppo' and args.naive==True:
    #     continue

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
