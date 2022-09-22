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
param = {'gamma':[0.9,0.93,0.95,0.97,0.99,0.995]}
args = utils.argsparser()
args.agent= 'batch_ac_shared_gc'
args.batch_size = 64
args.buffer = 64
args.lr = 0.0003
args.lr_weight=0.003
args.LAMBDA_2 = 1

logger.configure(args.log_dir,['csv'], log_suffix='CartPole-weighted-batch-ac-gamma')

for values in list(itertools.product(param['gamma'])):
    args.gamma = values[0]
    seeds = range(5)
    result = []
	
    # if args.agent=='batch_ac' and args.epoch>1:
    #     continue
    # if args.agent == 'weighted_batch_ac' and args.epoch==1:
    #     continue

    for seed in seeds:
        args.seed= seed
        checkpoint = 10000
        result =train(args)

        ret = np.array(result)
        print(ret.shape)
        # ret = np.mean(ret,axis=0)
        name = [str(k) for k in values]
        name.append(str(seed))
        print("hyperparam",'-'.join(name))
        logger.logkv("hyperparam",'-'.join(name))
        for n in range(ret.shape[0]):
            logger.logkv(str((n+1)*checkpoint),ret[n])
        logger.dumpkvs()

    # ret = np.array(result)
    # print(ret.shape)
    # ret = np.mean(ret,axis=0)
    # name = [str(k) for k in values]
    # logger.logkv("hyperparam",'-'.join(name))
    # for n in range(ret.shape[0]):
    #     logger.logkv(str((n+1)*checkpoint),ret[n])
    # logger.dumpkvs()
