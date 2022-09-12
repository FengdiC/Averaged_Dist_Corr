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

# param = {'lr_weight':[0.0001,0.0003,0.003,0.01],'weight_activation':['sigmoid','ReLU','tanh'],
#          'scale_weight':[1.0,10.0,100.0]}

param = {'agent':['weighted_batch_ac','batch_ac_shared_gc'],'lr_weight':[0.0001,0.0003,0.003,0.01],
         'closs_weight':[1,10,20]}

args = utils.argsparser()
# env, gamma, continuous are decided through args input

args.buffer=64
args.batch_size = 64
args.lr = 0.0003
args.scale_weight = 10
args.weight_activation = 'ReLU'
args.gamma = 0.99
# args.LAMBDA_2 = 10
# args.agent='weighted_batch_ac'

logger.configure(args.log_dir,['csv'], log_suffix=str(args.env)+'-weighted-batch-ac-shared-network')

for values in list(itertools.product(param['agent'],param['lr_weight'],param['closs_weight'])):
    args.agent = values[0]
    args.lr_weight = values[1]
    args.LAMBDA_2 = values[2]
    seeds = range(5)
    returns = []
	
    # if args.agent=='batch_ac' and args.epoch>1:
    #     continue
    # if args.agent == 'weighted_ppo' and args.naive==True:
    #     continue
    # if args.scale_weight>1 and args.weight_activation!='ReLU':
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
