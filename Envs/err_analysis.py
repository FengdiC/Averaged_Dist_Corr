import random

import torch
import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from Components.utils import argsparser
import gym
from config import agents_dict
import matplotlib.pyplot as plt
from Envs.reacher import DotReacher, DotReacherRepeat
from Components import logger
import itertools
import pandas as pd
import seaborn as sns
from reacher_bias_variance import setaxes, setsizes
from reacher_config import repeated, episodic

def train(args,env=None,stepsize=0.4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    seed = args.seed

    # Create Env
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed((seed))
    if env == None:
        env = DotReacherRepeat(stepsize=stepsize)
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
    errs= []
    errs_buffer=[]
    err_ratios= []
    avgrets = []
    losses = []
    avglos = []
    avgerr = []
    avgerr_buffer = []
    avgerr_ratio = []
    op = env.reset()

    num_steps = 40000
    checkpoint = 1000
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

        # when we compare biases from the missing discount factor and our correction approximation, should it be
        # on all states or just states shown up in the last buffer. But anyhow, these states should be weighted
        # according to the stationary distribution.
        if count == agent.buffer_size:
            all_frames = agent.buffer.all_frames()

        loss,count=agent.learn(count,obs)
        losses.append(loss)

        if count == 0:
            correction, d_pi = plot_correction(env, agent, args.gamma, device)
            # print("check", np.sum(correction*d_pi))
            est = plot_est_corr(env, agent, device, correction,args)
            err = np.matmul(d_pi, np.abs(correction - est))
            err_ratio, err_buffer = bias_compare(env, all_frames, d_pi, correction, est)
            errs.append(err)
            errs_buffer.append(err_buffer)
            err_ratios.append(err_ratio)


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

            # print(np.matmul(d_pi,correction)) #should be close to 1
            avgerr.append(np.mean(errs))
            avgerr_ratio.append(np.mean(err_ratios))
            avgerr_buffer.append(np.mean(errs_buffer))
            avgrets.append(np.mean(rets))
            avglos.append(np.mean(losses))
            rets = []
            losses = []
            errs= []
            err_ratios = []
            errs_buffer = []
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
    return avgrets,avgerr,avgerr_buffer,avgerr_ratio

def test(agent,env):
    rets= 0
    op = env.reset()
    r=-1
    while r!=0:
        a, lprob = agent.act(op)
        obs, r, done, infos = env.step(a)
        rets+= r
    return rets

def plot_correction(env,agent,gamma,device,policy=np.array([None,None])):
    # get the policy
    states = env.get_states()
    if (policy==None).all():
        policy = agent.network.get_policy(torch.from_numpy(states).to(device))
    # policy = np.ones((25,8))/8.0

    # get transition matrix P
    P = env.transition_matrix(policy)
    n = env.num_pt
    # # check if the matrix is a transition matrix
    # print(np.sum(P,axis=1))
    power = 1
    err = np.matmul(np.ones(n**2),np.linalg.matrix_power(P,power+1))-\
          np.matmul(np.ones(n**2), np.linalg.matrix_power(P, power))
    err = np.sum(np.abs(err))
    while err > 1.2 and power<10:
        power+=1
        err = np.matmul(np.ones(n**2), np.linalg.matrix_power(P,  power + 1)) - \
              np.matmul(np.ones(n**2), np.linalg.matrix_power(P, power))
        err = np.sum(np.abs(err))
    # print(np.sum(np.linalg.matrix_power(P, 3),axis=1))
    d_pi = np.matmul(np.ones(n**2)/float(n**2), np.linalg.matrix_power(P, power + 1))
    # print("stationary distribution",d_pi,np.sum(d_pi))

    if np.sum(d_pi - np.matmul(np.transpose(d_pi),P))>0.001:
        print("not the stationary distribution")

    # compute the special transition function M
    M = np.matmul(np.diag(d_pi) , P)
    M = np.matmul(M, np.diag(1/d_pi))

    correction = np.matmul(np.linalg.inv(np.eye(n**2)-gamma* np.transpose(M)) , (1-gamma) * 1/(d_pi*n**2))
    # print(correction.shape)

    # # plot heatmap
    # data = pd.DataFrame(data={'x': states[:,0], 'y': states[:,1], 'z': np.squeeze(correction)})
    # data = data.pivot(index='x', columns='y', values='z')
    # sns.heatmap(data)
    # plt.show()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.tricontourf(states[:,0], states[:,1], np.squeeze(correction), 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('correction')
    # plt.show()
    return correction,d_pi


def plot_est_corr(env,agent,device,correction,args):
    # get the weights
    states = env.get_states()
    if args.agent == 'weighted_batch_ac':
        weights = agent.weight_network.forward(torch.from_numpy(states).to(device)).detach().cpu().numpy()/args.scale_weight
    else:
        _, weights = agent.weight_critic.forward(torch.from_numpy(states).to(device))
        weights = weights.detach().cpu().numpy()/args.scale_weight

    # # plot heatmap
    # data = pd.DataFrame(data={'x': states[:, 0], 'y': states[:, 1], 'z': np.squeeze(weights)})
    # data = data.pivot(index='x', columns='y', values='z')
    # sns.heatmap(data)
    # plt.title("estimated")
    # plt.show()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.tricontourf(states[:, 0], states[:, 1], correction-weights, 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('difference')
    # plt.show()
    return weights * (1-args.gamma) * 500

def bias_compare(env,all_frames,d_pi,correction,est):
    ## this method counts the number of times of each state shown in one buffer.
    states = env.get_states().tolist()
    states = [[round(key, 2) for key in item] for item in states]

    discounted = correction * d_pi
    count = np.zeros(len(states))
    for i in range(all_frames.shape[0]):
        s = np.around(all_frames[i], 2)
        idx = states.index(s.tolist())
        count[idx]+=1
    sampling = count/ all_frames.shape[0]
    indices = np.argwhere(count)
    err_in_buffer = np.matmul(np.transpose(sampling),np.abs(correction -est))
    approx_bias = np.sum(np.abs(discounted[indices] * count[indices] -est[indices] * sampling[indices] * count[indices]))
    miss_bias = np.sum(np.abs(discounted[indices] * count[indices] - sampling[indices] * count[indices]))
    # print(indices.shape[0],approx_bias, miss_bias)
    return approx_bias/miss_bias,err_in_buffer

def tune():
    args = argsparser()
    logger.configure(args.log_dir,['csv'], log_suffix='Reacher-repeated-shared')
    ratio = []
    args.agent = 'batch_ac_shared_gc'
    param = {'buffer':[1,5,25,45],'lr':[0.001,0.004,0.008],'lr_weight':[0.0003,0.007,0.001,0.004],
             'LAMBDA_2':[1,10],'scale_weight':[1,10,50,100],'gamma':[0.8,0.9,0.95,0.99]}
    args.hidden = 8
    args.hidden_weight = 64

    agent = ['batch_ac_shared_gc']
    critic_hidden = [8,16,32,64]
    checkpoint = 1000

    for values in list(itertools.product(param['buffer'],param['lr'],param['lr_weight'],param['scale_weight'],
                                         param['LAMBDA_2'],param['gamma'])):
        args.buffer = values[0]
        args.batch_size = values[0]
        args.lr = values[1]
        args.lr_weight = values[2]
        args.scale_weight = values[4]
        args.LAMBDA_2 = values[3]
        args.gamma = values[5]
        seeds = range(10)
        for seed in seeds:
            args.seed = seed
            avgrets,avgerr,avgerr_buffer,avgerr_ratio = train(args,stepsize = 0.2)
            ratio.append(avgerr_ratio)

            name = [str(k) for k in values]
            name.append(str(seed))
            logger.logkv("hyperparam", '-'.join(name)+'-rets')
            for n in range(len(avgrets)):
                logger.logkv(str((n + 1) * checkpoint), avgrets[n])
            logger.dumpkvs()

            logger.logkv("hyperparam", '-'.join(name)+'-err-ratios')
            for n in range(len(avgrets)):
                logger.logkv(str((n + 1) * checkpoint), avgerr_ratio[n])
            logger.dumpkvs()

        ratio = np.array(ratio)
        mean = np.mean(ratio,axis=0)
        logger.logkv("hyperparam", '-'.join(name) + '-ratio-mean')
        for n in range(mean.shape[0]):
            logger.logkv(str((n + 1) * checkpoint), mean[n])
        logger.dumpkvs()
        ratio=[]

def plot_ratio():
    plt.figure(figsize=(13, 5.5), dpi=60)
    setsizes()
    args = argsparser()
    seeds = range(10)
    checkpoint = 1000

    args.agent = 'batch_ac_shared_gc'
    args.naive= False
    if args.agent == 'batch_ac' and args.naive == True:
        name = 'naive'
    elif args.agent == 'batch_ac' and args.naive == False:
        name = 'biased'
    else:
        name = args.agent
    hyperparam = repeated[name]
    args.buffer = hyperparam['buffer']
    args.batch_size = hyperparam['buffer']
    args.lr = hyperparam['lr']
    args.lr_weight = hyperparam['lr_weight']
    args.scale_weight = hyperparam['scale']
    args.LAMBDA_2 = hyperparam['gamma_coef']
    args.gamma = hyperparam['gamma']
    args.hidden = hyperparam['hid']
    args.hidden_weight = hyperparam['critic_hid']

    ratio = []
    logger.configure('./', ['csv'], log_suffix='batch-ac-err-in-buffer' + str(args.seed))
    # store err_in_buffer for both sampling and our correction
    for seed in seeds:
        args.seed = seed
        avgrets, avgerr, avgerr_buffer, avgerr_ratio = train(args, stepsize=0.2)
        ratio.append(avgerr_ratio)
        logger.logkv("name", 'err-' + str(seed))
        for n in range(len(avgerr_buffer)):
            logger.logkv(str((n + 1) * checkpoint), avgerr_buffer[n])
        logger.dumpkvs()

    ratio = np.array(ratio)
    mean = np.mean(ratio, axis=0)
    std = np.std(ratio,axis=0)/np.sqrt(10)
    plt.plot(range(0,checkpoint*mean.shape[0],checkpoint), mean, color='tab:orange', label='shared')
    plt.fill_between(range(0,checkpoint*mean.shape[0],checkpoint), mean + std, mean - std,
                     color='tab:orange', alpha=0.2, linewidth=0.9)

    args.agent = 'weighted_batch_ac'
    args.naive = False
    if args.agent == 'batch_ac' and args.naive == True:
        name = 'naive'
    elif args.agent == 'batch_ac' and args.naive == False:
        name = 'biased'
    else:
        name = args.agent
    hyperparam = repeated[name]
    args.buffer = hyperparam['buffer']
    args.batch_size = hyperparam['buffer']
    args.lr = hyperparam['lr']
    args.lr_weight = hyperparam['lr_weight']
    args.scale_weight = hyperparam['scale']
    args.LAMBDA_2 = hyperparam['gamma_coef']
    args.gamma = hyperparam['gamma']
    args.hidden = hyperparam['hid']
    args.hidden_weight = hyperparam['critic_hid']

    ratio = []
    for seed in seeds:
        args.seed = seed
        avgrets, avgerr, avgerr_buffer, avgerr_ratio = train(args, stepsize=0.2)
        ratio.append(avgerr_ratio)

    ratio = np.array(ratio)
    mean = np.mean(ratio, axis=0)
    std = np.std(ratio, axis=0) / np.sqrt(10)
    plt.plot(range(0, checkpoint * mean.shape[0], checkpoint), mean, color='tab:purple', label='non-shared')
    plt.fill_between(range(0, checkpoint * mean.shape[0], checkpoint), mean + std, mean - std,
                     color='tab:purple',
                     alpha=0.2, linewidth=0.9)
    plt.legend(prop={"size": 17})
    plt.xlabel("number of samples", fontsize=19)
    plt.ylabel("Ratio between biases", fontsize=19)
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=17, rotation=45)
    plt.title("Bias Ratio of the State Emphasis", fontsize=19)
    setaxes()
    plt.xlim(0,None)
    plt.show()

plot_ratio()