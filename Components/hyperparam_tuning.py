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


param = {'batch_size':[100,200,500,1000],'buffer':[100,200,500,1000,3000],'lr':[0.0003],
         'LAMBDA_2':[10,40],'epoch':[1]}
args = utils.argsparser()

logger.configure(args.log_dir,['csv'], log_suffix='batchAC-hyperparam-tune')

for values in list(itertools.product(param['batch_size'],param['buffer'],param['lr'],param['LAMBDA_2'],param['epoch'])):
    args.batch_size = values[0]
    args.buffer = values[1]
    args.lr = values[2]
    args.LAMBDA_2 = values[3]
    args.epoch = values[4]
    seeds = [111,345,3]
    result = []
	
    if args.buffer<args.batch_size:
        continue

    for seed in seeds:
        args.seed= seed

        num_steps = 1000000
        checkpoint = 10000
        result.append(train(args))
        print(result)

    ret = np.array(result)
    print(ret.shape)
    ret = np.mean(ret,axis=0)
    name = [str(k) for k in values]
    logger.logkv("hyperparam",'-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n+1)*checkpoint),ret[n])
    logger.dumpkvs()
