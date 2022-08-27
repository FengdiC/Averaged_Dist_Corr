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
    seed = args.seed

    # Create Env
    # env = gym.make(args.env)
    # env.seed(seed)
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
    op = env.reset()

    num_steps = 500000
    checkpoint = 10000
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
            # plot_correction(env, agent, args.gamma, device)
            # plot_est_corr(env, agent, device)
            avgrets.append(np.mean(rets))
            avglos.append(np.mean(losses))
            rets = []
            losses = []
            # plt.clf()
            # plt.subplot(211)
            # plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            # plt.subplot(212)
            # plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avglos)
            # # plt.savefig('Hopper_hyper_graph/hopper_ppo_lr_' + floatToString(args.lr) + "_seed_" + str(
            # #     args.seed) + "_agent_" + str(args.agent)  + "_var_" + floatToString(args.var))
            # plt.pause(0.001)
    return avgrets

def plot_correction(env,agent,gamma,device):
    # get the policy
    states = env.get_states()
    policy = agent.network.get_policy(torch.from_numpy(states).to(device))
    # get transition matrix P
    P = env.transition_matrix(policy)
    # compute stationary distribution
    d_pi = np.linalg.solve(np.eye(25)-P, np.zeros(25))
    print(np.sum(d_pi))

    correction = np.linalg.inv(np.eye(25)-gamma* np.transpose(P)) * (1-gamma) * 1/(d_pi*25)
    print(correction.shape)

    # plot heatmap
    data = pd.DataFrame(data={'x': states[:,0], 'y': states[:,1], 'z': np.squeeze(correction)})
    data = data.pivot(index='x', columns='y', values='z')
    sns.heatmap(data)
    plt.show()
    return correction


def plot_est_corr(env,agent,device):
    # get the weights
    states = env.get_states()
    weights = agent.weight_network.forward(torch.from_numpy(states).to(device))

    # plot heatmap
    data = pd.DataFrame(data={'x': states[:, 0], 'y': states[:, 1], 'z': np.squeeze(weights)})
    data = data.pivot(index='x', columns='y', values='z')
    sns.heatmap(data)
    plt.title("estimated")
    plt.show()
    return weights

args = argsparser()
train(args)
