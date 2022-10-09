import itertools
import os
import sys
import inspect

import numpy as np

from Components import utils, logger

def grid_search_args(args):
    """ TODO: Naive implementation. Make it efficient """
    
    count = 0
    for batch_size in [32, 128, 512, 1024]:
        for lr in [0.3, 0.03, 0.003]:
            for scale_weight in [1, 5, 10]:
                for value_loss_coef in [0.25, 1]:
                    for entropy_coef in [0, 0.01]:
                        for lr_weight in [0.003, 0.0003]:
                            count += 1
                            if count == args.hyp_seed:
                                args.batch_size = batch_size
                                args.lr = lr
                                args.scale_weight = scale_weight
                                args.value_loss_coef = value_loss_coef
                                args.entropy_coef = entropy_coef
                                args.lr_weight= lr_weight
                                args.gamma = 0.99
                                return args
    
    raise Exception("Hyp seed not found")
   

def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        print(e)
    return dir_path

def main():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from train import train

    args = utils.argsparser()
    args = grid_search_args(args)
    
    base_dir = '{}_{}_{}'.format(args.env, args.agent, args.naive)
    work_dir = os.path.join(currentdir, "results", base_dir, str(args.hyp_seed))
    make_dir(work_dir)        

    seeds = range(30)

    if args.env in ['MountainCarContinuous-v0', 'Pendulum-v1', 'Reacher-v2']:
        args.continuous = True

    for seed in seeds:
        args.seed = seed
        result = train(args)
        savepath = os.path.join(work_dir, "returns_{}.txt".format(seed))
        np.savetxt(savepath, result)

if __name__ == "__main__":
    main()