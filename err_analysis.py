import torch
from Components.env import TaskWrapper
import numpy as np
from Components.utils import argsparser
import gym
from config import agents_dict
import matplotlib.pyplot as plt
from Envs.reacher import DotReacher
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
        agent = network(args.lr, args.gamma, args.batch_size, o_dim, a_dim, args.hidden,device,continuous=True)
    else:
        agent = network(args.lr,args.gamma,args.batch_size,o_dim,a_dim,args.hidden,device)

    # Experiment block starts
    # Create the buffer
    agent.create_buffer(env, args, args.buffer)

    ret = 0
    rets = []
    errs= []
    err_ratios= []
    avgrets = []
    losses = []
    avglos = []
    avgerr = []
    avgerr_ratio = []
    op = env.reset()

    num_steps = 100000
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
            correction, d_pi = plot_correction(env, agent, args.gamma, device)
            est = plot_est_corr(env, agent, device, correction)
            err = np.matmul(d_pi, np.abs(correction - est))
            all_frames = agent.buffer.all_frames()
            err_ratio = bias_compare(env, all_frames, d_pi, correction, est)
            errs.append(err)
            err_ratios.append(err_ratio)

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

            # print(np.matmul(d_pi,correction)) #should be close to 1
            avgerr.append(np.mean(errs))
            avgerr_ratio.append(np.mean(err_ratios))
            avgrets.append(np.mean(rets))
            avglos.append(np.mean(losses))
            rets = []
            losses = []
            errs= []
            err_ratios = []
            plt.clf()
            plt.subplot(311)
            plt.ylim([-0.5, -0.0])
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            plt.subplot(312)
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avglos)
            plt.subplot(313)
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgerr_ratio)
            # plt.savefig('Hopper_hyper_graph/hopper_ppo_lr_' + floatToString(args.lr) + "_seed_" + str(
            #     args.seed) + "_agent_" + str(args.agent)  + "_var_" + floatToString(args.var))
            plt.pause(0.001)
    return avgrets,avgerr_ratio

def plot_correction(env,agent,gamma,device):
    # get the policy
    states = env.get_states()
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
    err = np.sum(err)
    while err > 0.00001:
        power+=1
        err = np.matmul(np.ones(n**2), np.linalg.matrix_power(P,  power + 1)) - \
              np.matmul(np.ones(n**2), np.linalg.matrix_power(P, power))
        err = np.sum(err)
    # print(np.sum(np.linalg.matrix_power(P, 3),axis=1))
    d_pi = np.matmul(np.ones(n**2)/float(n**2), np.linalg.matrix_power(P, power + 1))
    # print(d_pi,np.sum(d_pi))
    if np.sum(d_pi - np.matmul(np.transpose(d_pi),P))>0.0001:
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
    weights = agent.weight_network.forward(torch.from_numpy(states).to(device)).detach().cpu().numpy()

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
    count = np.zeros(len(states))
    for i in range(all_frames.shape[0]):
        s = np.around(all_frames[i], 2)
        idx = states.index(s.tolist())
        count[idx]+=1
    indices = np.argwhere(count)
    approx_bias = np.dot(np.transpose(d_pi[indices]),np.abs(correction[indices]-est[indices]))
    miss_bias = np.dot(np.transpose(d_pi[indices]),np.abs(correction[indices]-count[indices]))
    # print(indices.shape[0],approx_bias, miss_bias)
    return approx_bias/miss_bias

buffer_size = [5,15,25,45,64,128]
ratio = []
for buffer in buffer_size:
    args = argsparser()
    args.buffer = buffer
    args.batch_szie = buffer
    _, avgratio = train(args)
    ratio.append(np.mean(avgratio))
plt.figure()
plt.plot(buffer_size,ratio)
plt.xlabel("buffer size")
plt.ylabel("averaged bias comparison ratio")
plt.show()
