import torch

import torch.nn as nn
import numpy as np

from Agents.kfac import KFACOptimizer
from Components.buffer import Buffer
from Networks.actor_critic import MLPGaussianActor, MLPCategoricalActor


class ACKTR():
    def __init__(self, args, o_dim, n_actions, hidden, device, shared=False, **kwargs) -> None:
        if args.continuous:
            self.actor_critic = MLPGaussianActor(o_dim, n_actions, hidden, shared, device)        
        else:
            self.actor_critic = MLPCategoricalActor(o_dim, n_actions, hidden, shared=False)

        self.args = args
        self.o_dim  = o_dim
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.lmbda = args.lam
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.continuous = args.continuous
        self.batch_size = args.batch_size
        self.device = device
        self.BS = args.batch_size

        self.optimizer = KFACOptimizer(model=self.actor_critic, lr=args.lr, kl_clip=args.kfac_clip, max_grad_norm=args.max_grad_norm)     

    def create_buffer(self, env):
        # Create the buffer
        self.buffer_size = self.args.batch_size
        o_dim = env.observation_space.shape[0]
        if self.continuous:
            self.buffer = Buffer(self.args.gamma, self.args.lam, o_dim, self.n_actions, self.args.batch_size)
        else:
            self.buffer = Buffer(self.args.gamma,self.args.lam, o_dim, 0, self.args.batch_size)
      
    def store(self, op, r, done, a, lprob, time):
        self.buffer.add(op, r, done, a, lprob.item(), time)
 
    def act(self, op):
        with torch.no_grad():        
            a, lprob = self.actor_critic.act(torch.from_numpy(op).unsqueeze(0).to(self.device))
            if self.continuous:
                a = a.reshape((-1))
        
        return a, lprob
              
    def learn(self, count, obs):
        if count == self.buffer_size:            
            frames, rewards, dones, actions, old_lprobs, times, next_frames = self.buffer.sample(self.BS)            
            if self.continuous:
                action_log_probs, values, dist_entropy = self.actor_critic(torch.from_numpy(frames).to(self.device), 
                                                                           torch.from_numpy(actions).to(self.device))
            else:
                action_log_probs, values, dist_entropy = self.actor_critic(torch.from_numpy(frames).to(self.device),
                                                                           torch.from_numpy(actions).to(self.device))           
            
            ###
            value_preds = torch.zeros(self.BS + 1)
            rets = torch.zeros(self.BS)
            _, value_preds[-1], _ = self.actor_critic(torch.from_numpy(obs).unsqueeze(0).to(self.device), torch.zeros(1, 1).to(self.device))
            gae = 0
            for step in reversed(range(rewards.size)):
                delta = rewards[step] + (1 - dones[step]) * self.gamma * value_preds[step + 1] - value_preds[step]
                gae = delta + (1 - dones[step]) * self.gamma * self.lmbda * gae
                rets[step] = gae + value_preds[step]           

            advantages = rets - values            
            dist_entropy = dist_entropy.mean()                  
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()

            if self.optimizer.steps % self.optimizer.Ts == 0:
                # Compute fisher, see Martens 2014
                self.actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = torch.randn(values.size())
                if values.is_cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self.optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.optimizer.acc_stats = False

            self.optimizer.zero_grad()
            print(value_loss, action_loss, dist_entropy)
            (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
            self.optimizer.step()
            ###

            self.buffer.empty()
            count = 0
            return value_loss.item() + action_loss.item(), count
        
        return 0, count
