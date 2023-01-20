import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from Components.random_search import random_search_Reacher, set_one_thread
import numpy as np
import torch, random
from Components import utils, logger
from Components.utils import argsparser
import gym
from config import agents_dict
from Envs.reacher import DotReacher, DotReacherRepeat
import itertools

def train(args,stepsize=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    seed = args.seed

    # Create Env
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed((seed))
    # env = DotReacherRepeat(stepsize=stepsize)
    env = DotReacher(stepsize=stepsize)
    o_dim = env.observation_space.shape[0]
    if args.continuous:
        a_dim= env.action_space.shape[0]
    else:
        a_dim = env.action_space.n

    # Set the agent
    network = agents_dict[args.agent]
    if args.continuous:
        agent = network(args.lr, args.gamma, args.batch_size, o_dim, a_dim, args.hidden,args,device,continuous=True)
    else:
        agent = network(args.lr,args.gamma,args.batch_size,o_dim,a_dim,args.hidden,args,device)

    # Experiment block starts
    # Create the buffer
    agent.create_buffer(env)

    ret = 0
    rets = []
    avgrets = []
    losses = []
    avglos = []
    op = env.reset()

    num_steps = 20000
    checkpoint = 200
    num_episode = 0
    count = 0
    time = 0
    for steps in range(num_steps):
        # does torch need expand_dims
        a, lprob = agent.act(op)
        obs, r, done, infos = env.step(a)
        agent.store(op,r,done,a,lprob,time)

        # Observe
        op = obs
        time += 1
        count += 1

        loss,count=agent.learn(count,obs)
        losses.append(loss)
        # End of Episode
        # ret += r * args.gamma**time
        # For convience
        ret += r
        if done:
            num_episode += 1
            rets.append(ret)
            ret = 0
            time = 0
            op = env.reset()

        if (steps + 1) % checkpoint == 0:
            # plt.rcParams["figure.figsize"]
            # gymdisplay(env,MAIN)
            avgrets.append(np.mean(rets))
            avglos.append(np.mean(losses))
            rets = []
            losses = []
            # plt.clf()
            # plt.subplot(311)
            # plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            # plt.subplot(312)
            # plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avglos)
            # plt.subplot(313)
            # plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgerr)
            # # plt.savefig('Hopper_hyper_graph/hopper_ppo_lr_' + floatToString(args.lr) + "_seed_" + str(
            # #     args.seed) + "_agent_" + str(args.agent)  + "_var_" + floatToString(args.var))
            # plt.pause(0.001)
    return avgrets

# param = {'lr_weight':[0.0001,0.0003,0.003,0.01],'weight_activation':['sigmoid','ReLU','tanh'],
#          'scale_weight':[1.0,10.0,100.0]}

param = {'agent': ['batch_ac_shared_gc', 'batch_ac',"weighted_batch_ac"], 'naive': [True, False]}

args = utils.argsparser()
# env, gamma, continuous are decided through args input

# Torch Shenanigans fix
set_one_thread()
logger.configure(args.log_dir, ['csv'], log_suffix='Reacher_tune_no_repeat-'+str(args.seed))
hyperparam = random_search_Reacher(args.seed)
args.buffer = hyperparam['buffer']
args.batch_size = hyperparam['buffer']
args.lr = hyperparam['lr']
args.lr_weight = hyperparam['lr_weight']
args.scale_weight = hyperparam['scale']
args.LAMBDA_2 = hyperparam['gamma_coef']
args.gamma =hyperparam['gamma']
args.hidden = hyperparam['hid']
args.hidden_weight = hyperparam['critic_hid']

for values in list(itertools.product(param['agent'], param['naive'])):
    args.agent = values[0]
    args.naive = values[1]
    returns = []
    # if args.agent=='batch_ac' and args.epoch>1:
    #     continue
    if args.agent != 'batch_ac' and args.naive == True:
        continue

    for run_seed in range(30):
        args.seed = run_seed
        checkpoint = 200
        result = train(args)
        ret = np.array(result)
        print(ret.shape)
        returns.append(ret)
        name = list(hyperparam.values())
        name = [str(s) for s in name]
        name.append(str(run_seed))
        logger.logkv("hyperparam", '-'.join(name))
        agent = [str(k) for k in values]
        agent.append(str(run_seed))
        print("agent", '-'.join(agent))
        logger.logkv("agent", '-'.join(agent))
        for n in range(ret.shape[0]):
            logger.logkv(str((n + 1) * checkpoint), ret[n])
        logger.dumpkvs()

    ret = np.array(returns)
    print(ret.shape)
    ret = np.mean(ret, axis=0)
    agent = [str(k) for k in values]
    logger.logkv("agent", '-'.join(agent))
    name = list(hyperparam.values())
    name = [str(s) for s in name]
    name.append('mean')
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n + 1) * checkpoint), ret[n])
    logger.dumpkvs()