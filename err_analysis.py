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

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    seed = args.seed

    # Create Env
    env = DotReacher()
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
    avgrets = []
    losses = []
    avglos = []
    avgerr = []
    op = env.reset()

    num_steps = 50000
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
            correction, d_pi = plot_correction(env, agent, args.gamma, device)
            est = plot_est_corr(env, agent, device, correction)
            err = np.matmul(d_pi, np.abs(correction - est))
            avgerr.append(err)
            avgrets.append(np.mean(rets))
            avglos.append(np.mean(losses))
            rets = []
            losses = []
            plt.clf()
            plt.subplot(311)
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            plt.subplot(312)
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avglos)
            plt.subplot(313)
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgerr)
            # plt.savefig('Hopper_hyper_graph/hopper_ppo_lr_' + floatToString(args.lr) + "_seed_" + str(
            #     args.seed) + "_agent_" + str(args.agent)  + "_var_" + floatToString(args.var))
            plt.pause(0.001)
    return avgrets

def plot_correction(env,agent,gamma,device):
    # get the policy
    states = env.get_states()
    policy = agent.network.get_policy(torch.from_numpy(states).to(device))
    # policy = np.ones((25,8))/8.0

    # get transition matrix P
    P = env.transition_matrix(policy)
    # # check if the matrix is a transition matrix
    # print(np.sum(P,axis=1))
    power = 1
    err = np.matmul(np.ones(25),np.linalg.matrix_power(P,10*(power+1)))-\
          np.matmul(np.ones(25), np.linalg.matrix_power(P, 10*power))
    err = np.sum(err)
    while err > 0.00001:
        power+=1
        err = np.matmul(np.ones(25), np.linalg.matrix_power(P, 10 * (power + 1))) - \
              np.matmul(np.ones(25), np.linalg.matrix_power(P, 10 * power))
        err = np.sum(err)
    # print(np.sum(np.linalg.matrix_power(P, 3),axis=1))
    d_pi = np.matmul(np.ones(25)/25.0, np.linalg.matrix_power(P, 10 * (power + 1)))
    # print(d_pi,np.sum(d_pi))
    if np.sum(d_pi - np.matmul(np.transpose(d_pi),P))>0.0001:
        print("not the stationary distribution")

    correction = np.matmul(np.linalg.inv(np.eye(25)-gamma* np.transpose(P)) , (1-gamma) * 1/(d_pi*25))
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

args = argsparser()
train(args)
