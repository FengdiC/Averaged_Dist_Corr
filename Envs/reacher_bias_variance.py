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
from Components.buffer import Buffer
from Networks.actor_critic import NNGammaCritic
import gym
from config import agents_dict
import matplotlib.pyplot as plt
from Envs.reacher import DotReacher, DotReacherRepeat
from Components import logger
import itertools
import pandas as pd
import seaborn as sns


def emphasis(agent,V,env,frames,actions,times,device,gamma,closs_weight,scale_weight,timeout=100):
    """
        Output:
        mean-dist of size |S| to be stored in a matric of size |seeds| x |S| x D
        var-dist of size |S| to be stored in a matric of size |seeds| x |S| x D
        """
    # update the weight for the critic
    states = [[round(key, 2) for key in item] for item in env._states]
    q_values = np.zeros(frames.shape[0])
    for i in range(frames.shape[0]):
        frame = np.around(frames[i], 2)
        idx = states.index(frame.tolist())
        q_values[i] = V[idx]
    new_lprobs, _ = agent.network.forward(torch.from_numpy(frames).to(device),
                                          torch.from_numpy(actions).to(device))
    # compute shared gradient
    values, weights = agent.weight_critic(torch.from_numpy(frames).to(device))
    closs = closs_weight * torch.mean((torch.from_numpy(q_values).to(device) - values) ** 2)
    ## Weight
    labels = gamma ** times
    wloss = torch.mean((torch.from_numpy(labels).to(device) * scale_weight - weights) ** 2)
    agent.weight_critic_opt.zero_grad()
    (closs + wloss).backward()
    agent.weight_critic_opt.step()


    indices = np.zeros(frames.shape[0])
    states = [[round(key, 2) for key in item] for item in env._states]
    counts = np.zeros(len(states))
    naive = np.zeros(len(states))
    for i in range(frames.shape[0]):
        frame = np.around(frames[i], 2)
        idx = states.index(frame.tolist())
        naive[idx] = naive[idx]*counts[idx] + gamma**times[i]
        counts[idx] +=1
        naive[idx] /= counts[idx]
        indices[i] = idx
    _, weights = agent.weight_critic(torch.from_numpy(env._obs).to(device))

    counts /= frames.shape[0]
    # # compared with discounted state distribution
    # shared = counts * weights.detach().numpy()/scale_weight
    # biased = counts
    # naive = naive * counts

    # compare the state distribution ratio
    shared = weights.detach().numpy() / scale_weight *(1-gamma) * timeout
    biased = np.ones(len(states))
    naive *= (1-gamma) * timeout

    # compute varaince of the naive estimation
    naive_var = np.zeros(len(states))
    for i in range(len(states)):
        if np.any(indices==i) == False:
            continue
        idx = np.where(indices==i)

        # values = gamma**np.array(times[idx]) *counts[i]

        values = gamma ** np.array(times[idx]) * (1-gamma) * timeout
        if np.abs(np.mean(values) - naive[i])>0.001:
            print("Computation Error!!!!!!")
        naive_var[i] = np.var(values)
    return biased,shared,naive, naive_var


def compute_gradient(agent,Q,env,frames,actions,times,device,gamma,scale_weight,closs_weight):
    states = [[round(key,2) for key in item] for item in env._states]
    q_values = np.zeros(frames.shape[0])
    for i in range(frames.shape[0]):
        frame = np.around(frames[i], 2)
        idx = states.index(frame.tolist())
        q_values[i] = Q[idx,int(actions[i])]
    new_lprobs, _ = agent.network.forward(torch.from_numpy(frames).to(device),
                                                           torch.from_numpy(actions).to(device))
    # compute shared gradient
    values, weights = agent.weight_critic(torch.from_numpy(frames).to(device))
    pobj = new_lprobs * torch.from_numpy(q_values).to(device)* weights.detach() / scale_weight
    ploss = -torch.mean(pobj)* (1-gamma) * 500
    agent.opt.zero_grad()
    ploss.backward(retain_graph=True)
    shared_grad = [p.grad.detach().numpy().flatten() for p in list(agent.network.parameters())]
    ## Critic
    closs = closs_weight * torch.mean((torch.from_numpy(q_values).to(device) - values) ** 2)
    ## Weight
    labels = gamma ** times
    wloss = torch.mean((torch.from_numpy(labels).to(device) * scale_weight - weights) ** 2)
    agent.weight_critic_opt.zero_grad()
    (closs + wloss).backward()
    agent.weight_critic_opt.step()

    # compute naive gradient
    pobj = gamma**torch.from_numpy(times).to(device)*new_lprobs * torch.from_numpy(q_values).to(device)
    ploss = -torch.mean(pobj)* (1-gamma) * 500
    agent.opt.zero_grad()
    ploss.backward(retain_graph=True)
    naive_grad = [p.grad.detach().numpy().flatten() for p in list(agent.network.parameters())]

    # compute biased gradient
    pobj =  new_lprobs * torch.from_numpy(q_values).to(device)
    ploss = -torch.mean(pobj)
    agent.opt.zero_grad()
    ploss.backward(retain_graph=True)
    biased_grad = [p.grad.detach().numpy().flatten() for p in list(agent.network.parameters())]
    return shared_grad, naive_grad, biased_grad

def compute_stat_dist(env,agent,gamma,device,policy=np.array([None,None])):
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
    return correction,d_pi

def log_prob_for_all(states,agent,device):
    """
    Gain the lprob value for all states and actions. Return a matrix of shape 25*8.
    """
    lprobs = []
    for a in range(8):
        new_lprobs, _ = agent.network(torch.from_numpy(states).to(device),
                                                 torch.from_numpy(a+np.zeros(states.shape[0])).to(device))
        lprobs.append(new_lprobs)
    return torch.stack(lprobs,dim=1)


def fixed_policy_check(stepsize = 0.2):
    # env._obs is an numpy array, while env._states is a list. But they both represent all states.
    args = argsparser()
    seed = args.seed
    args.agent = 'batch_ac_shared_gc'
    args.buffer = 25
    args.batch_size = 25
    args.lr = 0.0041
    args.lr_weight = 0.0036
    args.scale_weight = 6
    args.LAMBDA_2 = 11.65
    args.gamma = 0.95
    args.hidden = 8
    args.hidden_weight = 64
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed((seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = DotReacherRepeat(stepsize=stepsize)
    env = DotReacher(stepsize=stepsize)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    # Set the agent
    network = agents_dict[args.agent]
    agent = network(args.lr, args.gamma, args.batch_size, o_dim, a_dim, args.hidden, args, device)

    # Experiment block starts
    # Create the buffer
    agent.create_buffer(env)

    op = env.reset()
    # run an agent for a random steps
    num_steps = 9500
    num_episode = 0
    count = 0
    time = 0
    for steps in range(num_steps):
        # does torch need expand_dims
        a, lprob = agent.act(op)
        obs, r, done, infos = env.step(a)
        agent.store(op, r, done, a, lprob, time)

        # Observe
        op = obs
        time += 1
        count += 1

        loss, count = agent.learn(count, obs)
        if done:
            num_episode += 1
            time = 0
            op = env.reset()

    states = env.get_states()
    policy = agent.network.get_policy(torch.from_numpy(states).to(device))

    # compute the true gradient: requiring discounted state distribution, q-values and log-grad of policies
    corr,d_pi = compute_stat_dist(env,agent,args.gamma,device)
    print("corr: ",corr)
    d_pi_gamma = corr* d_pi
    print(np.sum(d_pi),"::should equal one")
    Q = env.q_values(policy,args.gamma)
    V = np.sum(Q * policy,axis=1)
    # log_probs = log_prob_for_all(states,agent,device)
    # ploss = torch.sum(torch.from_numpy(Q).to(device) * log_probs,dim=1)
    # ploss = torch.sum(torch.from_numpy(d_pi_gamma) * ploss)
    # agent.opt.zero_grad()
    # ploss.backward()
    # true_grad = [p.grad.detach().numpy().flatten() for p in list(agent.network.parameters())]
    # true_grad = np.concatenate(true_grad)
    # print(true_grad.shape)

    # set the total number of samples I would like to use
    buffers = []
    args.buffer = 1000
    for seed in range(10):
        # initial data buffers for 10 random seeds
        buffers.append(Buffer(args.gamma, args.lam, o_dim, 0, args.buffer))
    checkpoint= 50
    """
    mean-dist for all three should be a matric of size |seeds| x D x |S|
    var-dist for naive only is a matric of size |seeds| x D x |S|
    """

    shared_mean = [[] for s in range(10)]
    biased_mean = [[] for s in range(10)]
    naive_mean = [[] for s in range(10)]
    naive_var = [[] for s in range(10)]
    for seed in range(10):
        buffer = buffers[seed]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed((seed))
        op = env.reset()
        time = 0
        for num_samples in range(args.buffer):
            a, lprob = agent.act(op)
            obs, r, done, infos = env.step(a)
            buffer.add(op, r, done, a, lprob.item(), time)
            # Observe
            op = obs
            time += 1
            if done:
                time = 0
                op = env.reset()
            if num_samples % checkpoint==0 and num_samples>0:
                # get all frames, actions, times
                frames, actions, times = buffer.all_info()
                biased, shared, naive, naive_var_step = emphasis(agent,V,env,frames,actions,times,device,args.gamma,
                                                            args.LAMBDA_2,args.scale_weight,timeout=100)
                shared_mean[seed].append(shared)
                biased_mean[seed].append(biased)
                naive_mean[seed].append(naive)
                naive_var[seed].append(naive_var_step)
                # # compute gradients for all estimators
                # shared_grad,naive_grad,biased_grad = compute_gradient(agent,Q,env,frames,actions,times,device,
                #                                        args.gamma,args.scale_weight,args.LAMBDA_2)
                # shared[seed].append(np.concatenate(shared_grad))
                # naive[seed].append(np.concatenate(naive_grad))
                # biased[seed].append(np.concatenate(biased_grad))

    # # compute the bias and variance
    # shared = np.array(shared)
    # print(shared.shape)
    # naive = np.array(naive)
    # biased = np.array(biased)
    # true_grad = np.tile(true_grad,(shared.shape[1],1))
    # shared_mean = np.mean(shared, axis=0)
    # shared_var = np.var(shared, axis=0)
    # shared_var = np.mean(shared_var, axis=1)
    # shared_bias = np.sum((true_grad - shared_mean) ** 2, axis=1)

    """
    The variance of state emphasis is computed as
        var(F_D(S))=var(E[F_D(S)|D]) + E[var(F_D(S)|D)] by the total variance law
        Here, the state S is sampled from the buffer
    """
    shared = np.array(shared_mean)
    weight = np.tile(corr, (shared.shape[1], 1))

    shared_mean = np.mean(shared, axis=0)
    shared_bias = np.sum( weight* (shared_mean - weight)**2 ,axis=1 )
    shared_var = np.var(shared, axis=0)
    shared_var = np.sum(weight* shared_var, axis=1)

    biased = np.array(biased_mean)
    biased_mean = np.mean(biased, axis=0)
    biased_bias = np.sum(weight * (biased_mean - weight) ** 2, axis=1)
    biased_var = np.var(biased, axis=0)
    biased_var = np.sum(weight * biased_var, axis=1)

    naive = np.array(naive_mean)
    naive_var = np.array(naive_var)
    naive_var = np.var(naive, axis=0) + np.mean(naive_var,axis=0)
    naive_var = np.sum(weight * naive_var, axis=1)
    naive_mean = np.mean(naive, axis=0)
    naive_bias = np.sum(weight * (naive_mean - weight) ** 2, axis=1)


    plt.subplot(121)
    plt.plot(range(shared_bias.shape[0]),shared_bias,label='shared')
    plt.plot(range(shared_bias.shape[0]), naive_bias, label='naive')
    plt.plot(range(shared_bias.shape[0]), biased_bias, label='biased')
    plt.legend()
    plt.subplot(122)
    plt.plot(range(shared_bias.shape[0]), shared_var, label='shared')
    plt.plot(range(shared_bias.shape[0]), naive_var, label='naive')
    plt.plot(range(shared_bias.shape[0]), biased_var, label='biased')
    plt.legend()
    plt.show()
    plt.pause(0.01)

fixed_policy_check()
