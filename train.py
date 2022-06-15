import torch
from torch import nn
import numpy as np
from Components.utils import argsparser
import gym
from Agents.batch_ac import BatchActorCritic

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
    agent = BatchActorCritic(args.lr,args.gamma,args.batch_size,o_dim,a_dim,args.hidden)

    # Experiment block starts
    num_steps = 500000
    avgrets = agent.train(env,args,num_steps,args.buffer)


args = argsparser()
train(args)
