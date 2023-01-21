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

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 17.0
    plt.rcParams['ytick.labelsize'] = 17.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['ytick.minor.pad'] = 50.0


def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.axes.set_ylim(-3,None)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
                   labelsize=16, pad=8)
    ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
                   labelsize=16, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())


def emphasis(agent,V,env,frames,actions,times,device,gamma,closs_weight,scale_weight,timeout=100):
    """
        Output:
        mean-dist of size |S| to be stored in a matric of size |seeds| x |S| x D
        var-dist of size |S| to be stored in a matric of size |seeds| x |S| x D
        """
    # # update the weight for the critic
    # states = [[round(key, 2) for key in item] for item in env._states]
    # q_values = np.zeros(frames.shape[0])
    # for i in range(frames.shape[0]):
    #     frame = np.around(frames[i], 2)
    #     idx = states.index(frame.tolist())
    #     q_values[i] = V[idx]
    # new_lprobs, _ = agent.network.forward(torch.from_numpy(frames).to(device),
    #                                       torch.from_numpy(actions).to(device))
    # # compute shared gradient
    # values, weights = agent.weight_critic(torch.from_numpy(frames).to(device))
    # closs = closs_weight * torch.mean((torch.from_numpy(q_values).to(device) - values) ** 2)
    # ## Weight
    # labels = gamma ** times
    # wloss = torch.mean((torch.from_numpy(labels).to(device) * scale_weight - weights) ** 2)
    # agent.weight_critic_opt.zero_grad()
    # (closs + wloss).backward()
    # agent.weight_critic_opt.step()


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
    # compared with discounted state distribution
    shared = counts * weights.detach().numpy()/ scale_weight * (1-gamma) * timeout
    biased = counts
    naive = naive * counts * (1-gamma) * timeout

    # # compare the state distribution ratio
    # shared = weights.detach().numpy() / scale_weight *(1-gamma) * timeout
    # biased = np.ones(len(states))
    # naive *= (1-gamma) * timeout

    # compute varaince of the naive estimation
    naive_var = np.zeros(len(states))
    for i in range(len(states)):
        if np.any(indices==i) == False:
            continue
        idx = np.where(indices==i)

        values = gamma**np.array(times[idx]) *counts[i] * (1-gamma) * timeout

        # values = gamma ** np.array(times[idx]) * (1-gamma) * timeout
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
    args.lr = 0.0042
    args.lr_weight = 0.0005
    args.scale_weight =143
    args.LAMBDA_2 = 18.94
    args.gamma = 0.99
    args.hidden = 8
    args.hidden_weight = 64
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed((seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = DotReacherRepeat(stepsize=stepsize)
    # env = DotReacher(stepsize=stepsize)

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
    num_steps = 40000
    num_episode = 0
    count = 0
    time = 0
    plotpoint=1000

    shared_bias = []
    shared_var = []
    biased_bias = []
    biased_var=[]
    naive_bias = []
    naive_var = []
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
        if steps % plotpoint==0:
            shared_mean_per_step = []
            biased_mean_per_step = []
            naive_mean_per_step = []
            naive_var_per_step = []
            states = env.get_states()
            policy = agent.network.get_policy(torch.from_numpy(states).to(device))

            # compute the true gradient: requiring discounted state distribution, q-values and log-grad of policies
            corr,d_pi = compute_stat_dist(env,agent,args.gamma,device)
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
            args.buffer = 25
            for seed in range(10):
                # initial data buffers for 10 random seeds
                buffers.append(Buffer(args.gamma, args.lam, o_dim, 0, args.buffer))
            checkpoint= 24
            """
            mean-dist for all three should be a matric of size |seeds| x D x |S|
            var-dist for naive only is a matric of size |seeds| x D x |S|
            """
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
                                                                    args.LAMBDA_2,args.scale_weight,timeout=500)
                        shared_mean_per_step.append(shared)
                        biased_mean_per_step.append(biased)
                        naive_mean_per_step.append(naive)
                        naive_var_per_step.append(naive_var_step)
            shared = np.array(shared_mean_per_step)
            print(shared.shape)
            biased = np.array(biased_mean_per_step)
            weight = biased

            shared_mean_per_step = np.mean(shared, axis=0)
            shared_bias.append( np.sum(weight * (shared_mean_per_step - d_pi_gamma) ** 2) )
            # print(np.var(shared, axis=0))
            shared_var.append( np.sum(weight * np.var(shared, axis=0)) )

            biased = np.array(biased_mean_per_step)
            biased_mean_per_step = np.mean(biased, axis=0)
            biased_bias.append( np.sum(weight * (biased_mean_per_step - d_pi_gamma) ** 2) )
            biased_var.append( np.sum(weight * np.var(biased, axis=0)) )

            naive = np.array(naive_mean_per_step)
            naive_var_per_step = np.array(naive_var_per_step)
            naive_var_per_step = np.var(naive, axis=0) + np.mean(naive_var_per_step, axis=0)
            naive_var.append( np.sum(weight * naive_var_per_step) )
            naive_mean_per_step = np.mean(naive, axis=0)
            naive_bias.append(np.sum(weight * (naive_mean_per_step - d_pi_gamma) ** 2))


    """
    The variance of state emphasis is computed as
        var(F_D(S))=var(E[F_D(S)|D]) + E[var(F_D(S)|D)] by the total variance law
        Here, the state S is sampled from the buffer
    """

    if agent == 'our correction':
        color = 'tab:orange'
        args.buffer = 1
        args.lr_weight = 0.005
        args.lr = 0.5
    elif agent == 'existing correction':
        color = 'tab:blue'
        args.buffer = 8
        args.lr = 0.95
    else:
        color = 'tab:green'
        args.buffer = 8
        args.lr = 0.95

    plt.figure(figsize=(13,7),dpi=60)
    shared_bias = np.array(shared_bias)
    setaxes()
    setsizes()
    plt.subplots_adjust(left=0,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.subplot(121)
    plt.plot(range(0,checkpoint*shared_bias.shape[0],checkpoint),shared_bias,label='our correction',color = 'tab:orange')
    plt.plot(range(0,checkpoint*shared_bias.shape[0],checkpoint), naive_bias, label='existing correction',color = 'tab:blue')
    plt.plot(range(0,checkpoint*shared_bias.shape[0],checkpoint), biased_bias, label='biased',color = 'tab:green')
    plt.legend(prop={"size": 17})
    plt.xlabel("number of samples", fontsize=19)
    plt.ylabel("squared error of estimated state distribution ratios", fontsize=19)
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=17, rotation=45)
    plt.title("Bias of the state emphasis", fontsize=19)
    setaxes()

    plt.subplot(122)
    plt.plot(range(0,checkpoint*shared_bias.shape[0],checkpoint), shared_var, label='our correction',color = 'tab:orange')
    plt.plot(range(0,checkpoint*shared_bias.shape[0],checkpoint), naive_var, label='existing correction',color = 'tab:blue')
    plt.plot(range(0,checkpoint*shared_bias.shape[0],checkpoint), biased_var, label='biased',color = 'tab:green')
    plt.legend(prop={"size": 17})
    plt.xlabel("number of samples", fontsize=19)
    plt.ylabel("variance of estimated state distribution ratios", fontsize=19)
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=17, rotation=45)
    plt.title("Variance of the state emphasis", fontsize=19)
    setaxes()
    plt.show()
    plt.pause(0.01)

fixed_policy_check()
