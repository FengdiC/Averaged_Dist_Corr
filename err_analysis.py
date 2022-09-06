import torch
from Components.env import TaskWrapper
import numpy as np
from Components.utils import argsparser
import gym
from config import agents_dict
import matplotlib.pyplot as plt
from Envs.reacher import DotReacher
from Components import logger
import itertools
import pandas as pd
import seaborn as sns

def train(args,stepsize=0.4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    seed = args.seed

    # Create Env
    env = DotReacher(stepsize=stepsize)
    torch.manual_seed(seed)
    np.random.seed(seed)
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

    num_steps = 20000
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
            est = plot_est_corr(env, agent, device, correction)
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

def plot_correction(env,agent,gamma,device):
    # get the policy
    states = env.get_states()
    policy = agent.network.get_policy(torch.from_numpy(states).to(device)).detach().cpu().numpy()
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
    while err > 1.2:
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


def plot_est_corr(env,agent,device,correction):
    # get the weights
    states = env.get_states()
    weights = agent.weight_network.forward(torch.from_numpy(states).to(device)).detach().cpu().numpy() * \
              agent.buffer_size*(1-agent.gamma)

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
    return weights

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

args = argsparser()
logger.configure(args.log_dir,['csv'], log_suffix='-Reacher_activation-hyperparam')
ratio = []
err = []
err_buffer = []
ret = []
buffer_size = [5,25,45,64]
lr = [0.001,0.0006,0.0003,0.0001]
weight_lr = [0.01,0.003,0.0003]
weight_epoch = [1,10,15]
weight_scale = [1,10,20]
activation = ['sigmoid','ReLU','tanh']
checkpoint = 1000

for values in list(itertools.product(buffer_size,lr,weight_lr,weight_epoch,weight_scale,activation)):
    print(values)
    args.buffer = values[0]
    args.batch_size = values[0]
    args.lr = values[1]
    args.lr_weight = values[2]
    args.epoch_weight = values[3]
    args.scale_weight = values[4]
    args.weight_activation = values[5]
    if values[5]!= 'ReLU' and values[4]>1:
        continue
    seeds = range(5)
    for seed in seeds:
        avgrets,avgerr,avgerr_buffer,avgerr_ratio = train(args,stepsize = 0.2)
        ratio.append(np.mean(avgerr_ratio))
        ret.append(np.mean(avgrets))
        err.append(np.mean(avgerr))
        err_buffer.append(np.mean(avgerr_buffer))

        name = [str(k) for k in values]
        name.append(str(seed))
        logger.logkv("hyperparam-rets", '-'.join(name))
        for n in range(len(avgrets)):
            logger.logkv(str((n + 1) * checkpoint), avgrets[n])
        logger.logkv("hyperparam-errs", '-'.join(name))
        for n in range(len(avgrets)):
            logger.logkv(str((n + 1) * checkpoint), avgerr[n])
        logger.logkv("hyperparam-err-ratio", '-'.join(name))
        for n in range(len(avgrets)):
            logger.logkv(str((n + 1) * checkpoint), avgerr_ratio[n])
        logger.logkv("hyperparam-err-buffer", '-'.join(name))
        for n in range(len(avgrets)):
            logger.logkv(str((n + 1) * checkpoint), avgerr_buffer[n])
plt.figure()
plt.subplot(211)
plt.plot(buffer_size,err_buffer)
plt.xlabel("buffer size")
plt.ylabel("averaged bias comparison ratio")
plt.subplot(212)
plt.plot(buffer_size,err)
plt.xlabel("buffer size")
plt.ylabel("averaged returns")
plt.show()
