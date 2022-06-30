import torch
from torch import nn
import numpy as np
from Components.utils import argsparser
import gym
from config import agents_dict
import matplotlib.pyplot as plt

def train(args):
    seed = args.seed

    # Create Env
    env = gym.make(args.env)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    # Set the agent
    network = agents_dict[args.agent]
    agent = network(args.lr,args.gamma,args.batch_size,o_dim,a_dim,args.hidden)

    # Experiment block starts
    # Create the buffer
    agent.create_buffer(env, args, args.buffer)

    ret = 0
    rets = []
    avgrets = []
    losses = []
    avglos = []
    # op = meanstdnormalizaer(env.reset())
    op = env.reset()

    num_steps = 5000000
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
        # op = meanstdnormalizaer(obs)
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
            # op =  meanstdnormalizaer(env.reset())
            op = env.reset()

        if (steps + 1) % checkpoint == 0:
            # plt.rcParams["figure.figsize"]
            # gymdisplay(env,MAIN)
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


args = argsparser()
train(args)
