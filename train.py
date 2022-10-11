import torch
import gym
import time

import numpy as np
import matplotlib.pyplot as plt

from config import agents_dict
from Components.utils import argsparser
from Envs.gym_repeat import RepeatEnvWrapper
from Components.running_stat import ZFilter

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    seed = args.seed
    print(device)

    # Create Env
    env = gym.make(args.env)
    # env = RepeatEnvWrapper(args.env)
    env.seed(seed)
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
        agent = network(lr=args.lr, gamma=args.gamma, BS=args.batch_size, 
                        o_dim=o_dim, n_actions=a_dim, hidden=args.hidden, args=args,
                        device=device, continuous=True)
    else:
        agent = network(lr=args.lr, gamma=args.gamma, BS=args.batch_size,
                        o_dim= o_dim, n_actions=a_dim, hidden=args.hidden, args=args, device=device)

    # Experiment block starts
    # Create the buffer
    agent.create_buffer(env)

    obfilter = None
    if args.continuous:
        obfilter = ZFilter(env.observation_space.shape)
    
    ret = step = 0
    rets = []
    ep_lens = []
    avgrets = []
    losses = []
    avglos = []
    op = env.reset()
    if obfilter is not None:
        op = obfilter(op)

    num_steps = 1000000
    checkpoint = 10000
    num_episode = 0
    count = 0
    time = 0
    for steps in range(num_steps):
        # does torch need expand_dims
        a, lprob = agent.act(op)
        if args.continuous:    
            scaled_ac = env.action_space.low + (a + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
            scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
            obs, r, done, infos = env.step(scaled_ac)
        else:
            obs, r, done, infos = env.step(a)
        agent.store(op,r,done,a,lprob,time)

        # Observe        
        op = obs
        if obfilter is not None:
            op = obfilter(op)
        time += 1
        count += 1
        
        loss, count = agent.learn(count, obs)
        losses.append(loss)

        # End of Episode
        # ret += r * args.gamma**time
        # For convience
        ret += r
        step += 1
        if done:
            num_episode += 1
            rets.append(ret)
            ep_lens.append(step)
            print("Episode {} ended with return {:.2f} in {} steps. Total steps: {}".format(num_episode, ret, step, steps))

            ret = 0
            step = 0
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
            # plt.subplot(211)
            # plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            # plt.subplot(212)
            # plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avglos)
            # # plt.savefig('Hopper_hyper_graph/hopper_ppo_lr_' + floatToString(args.lr) + "_seed_" + str(
            # #     args.seed) + "_agent_" + str(args.agent)  + "_var_" + floatToString(args.var))
            # plt.pause(0.001)
    return avgrets

if __name__ == "__main__":
    args = argsparser()
    tic = time.time()
    avgrets = train(args)
    print("Run took:", time.time() - tic)
    np.savetxt("./results/{}_{}_{}-{}.txt".format(args.env, args.agent, args.naive, args.seed), avgrets)
