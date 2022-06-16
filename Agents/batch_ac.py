import torch
from torch import nn
import numpy as np
from Networks.actor_critic import MLPCategoricalActor
import gym
from Components.utils import meanstdnormalizaer, A
from Components.buffer import Buffer
import matplotlib.pyplot as plt

class BatchActorCritic(A):
    # the current code works for shared networks with categorical actions only
    def __init__(self,lr,gamma,BS,o_dim,n_actions,hidden,shared=False):
        super(BatchActorCritic,self).__init__(lr=lr,gamma=gamma,BS=BS,o_dim=o_dim,n_actions=n_actions,
                                              hidden=hidden,shared=shared)
        self.network = MLPCategoricalActor(o_dim,n_actions,hidden,shared)
        self.opt = torch.optim.Adam(self.network.parameters(),lr=lr)  #decay schedule?
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=100000, gamma=0.9)

    def update(self,closs_weight):
        # input: data  Job: finish one round of gradient update
        _, self.next_values = self.network.forward(torch.from_numpy(self.next_frames),torch.from_numpy(self.actions))
        self.new_probs, self.values = self.network.forward(torch.from_numpy(self.frames),torch.from_numpy(self.actions))

        self.dones = torch.from_numpy(self.dones)
        returns =  torch.from_numpy(self.rewards) + self.gamma*(1-self.dones)*self.next_values.detach()
        self.closs = closs_weight*torch.mean((returns-self.values)**2)
        pobj = self.new_probs * (returns - self.values).detach()
        self.ploss = -torch.mean(pobj)
        self.opt.zero_grad()
        self.ploss.backward()
        self.closs.backward()
        self.opt.step()

    def train(self,env,args,num_steps,buffer_size):
        # Create the buffer
        o_dim = env.observation_space.shape[0]
        self.buffer = Buffer(args, o_dim, 0, buffer_size)

        ret = 0
        rets = []
        avgrets = []
        losses = []
        avglos = []
        # op = meanstdnormalizaer(env.reset())
        op = env.reset()

        checkpoint = 10000
        num_episode = 0
        count = 0
        time = 0
        for steps in range(num_steps):
            # does torch need expand_dims
            a,lprob = self.network.act(torch.from_numpy(op))
            obs, r, done, infos = env.step(int(a.detach()))
            self.buffer.add(op,r , done, int(a.detach().item()), lprob.detach().item(), time)

            # Observe
            # op = meanstdnormalizaer(obs)
            op = obs
            time += 1
            count += 1

            # Update
            if count == buffer_size:
                self.buffer.add_last(obs)
                for epoch in range(args.epoch):
                    self.buffer.shuffle()
                    for turn in range(1):   # buffer_size//self.BS
                        # value functions may not be well learnt
                        self.frames, self.rewards, self.dones, self.actions, self.old_probs, self.times, self.next_frames \
                            = self.buffer.sample(self.BS, turn)
                        self.update(args.LAMBDA_2)
                        # self.scheduler.step()
                self.buffer.empty()

                print("ploss is: ", self.ploss.detach().numpy(),":::", self.closs.detach().numpy())
                losses.append(float(self.ploss.detach().numpy()+self.closs.detach().numpy()))
                count = 0

            # End of Episode
            ret += r
            if done:
                # add in the terminal state or not

                # a,lprob = self.network.act(torch.from_numpy(op))
                # self.buffer.add(op,0.0 , done, int(a.detach().item()), lprob.detach().item(), time)
                # count+=1
                num_episode += 1
                rets.append(ret)
                # assert ret<=200
                # if ret > 200:
                #     print("Large rewards: ", ret, ":::", r)
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




