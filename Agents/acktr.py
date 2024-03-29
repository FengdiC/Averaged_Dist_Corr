import torch
import itertools

import torch.nn as nn
import numpy as np

from Agents.kfac import KFACOptimizer, WeightedKFACOptimizer
from Components.buffer import Buffer
from Networks.actor_critic import MLPGaussianActor, MLPCategoricalActor, NNCategoricalActor, NNGaussianActor, NNGammaCritic


class ACKTR():
    def __init__(self, args, o_dim, n_actions, hidden, device, shared=False, **kwargs) -> None:
        self.args = args
        self.o_dim  = o_dim
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.lmbda = args.lam
        self.value_loss_coef = args.value_loss_coef
        self.value_fisher_coef = args.value_fisher_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.continuous = args.continuous
        self.batch_size = args.batch_size
        self.device = device
        self.BS = args.batch_size

        if args.continuous:
            self.actor_critic = MLPGaussianActor(o_dim, n_actions, hidden, shared, device)        
        else:
            self.actor_critic = MLPCategoricalActor(o_dim, n_actions, hidden, shared)
        self.actor_critic.to(device)
        
        self.optimizer = KFACOptimizer(model=self.actor_critic, lr=args.lr, kl_clip=args.kfac_clip, max_grad_norm=args.max_grad_norm)     
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer.optim, step_size=10000, gamma=0.9)

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
                _, last_val, _ = self.actor_critic(torch.from_numpy(obs).unsqueeze(0).to(self.device), torch.zeros(1, self.n_actions).to(self.device)) 
            else:
                action_log_probs, values, dist_entropy = self.actor_critic(torch.from_numpy(frames).to(self.device),
                                                                           torch.from_numpy(actions).to(self.device))           
                _, last_val, _ = self.actor_critic(torch.from_numpy(obs).unsqueeze(0).to(self.device), torch.zeros(1, 1).to(self.device)) 
            
            # GAE Estimation
            value_preds = torch.zeros(self.BS + 1)
            value_preds[-1] = last_val
            rets = torch.zeros(self.BS)
           
            gae = 0
            for step in reversed(range(rewards.size)):
                delta = rewards[step] + (1 - dones[step]) * self.gamma * value_preds[step + 1] - value_preds[step]
                gae = delta + (1 - dones[step]) * self.gamma * self.lmbda * gae
                rets[step] = gae + value_preds[step]
            
            rets, value_preds = rets.to(self.device), value_preds.to(self.device)

            # Value and Action loss
            advantages = rets - values
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-6)            
            dist_entropy = dist_entropy.mean()                  
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()
            
            # Fisher loss
            if self.optimizer.steps % self.optimizer.Ts == 0:
                # Compute fisher, see Martens 2014
                self.actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = torch.randn(values.size())
                if values.is_cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean() * self.value_fisher_coef

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self.optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.optimizer.acc_stats = False

            # Apply gradients
            self.optimizer.zero_grad()            
            (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
            self.optimizer.step()
            # self.scheduler.step()            

            self.buffer.empty()
            count = 0
            return value_loss.item() + action_loss.item(), count
        
        return 0, count


class SimpleWeightedACKTR(ACKTR):
    """ Uses KFAC only for Actor. GammaCritic uses Adam """
    def __init__(self, args, o_dim, n_actions, hidden, device, shared=False, **kwargs) -> None:
        self.args = args
        self.o_dim  = o_dim
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.lmbda = args.lam
        self.value_loss_coef = args.value_loss_coef
        self.value_fisher_coef = args.value_fisher_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.continuous = args.continuous
        self.batch_size = args.batch_size
        self.device = device        
        self.BS = args.batch_size        

        if args.continuous:
            self.actor = NNGaussianActor(o_dim, n_actions, hidden, device)        
        else:
            self.actor = NNCategoricalActor(o_dim, n_actions, hidden)
        self.actor.to(device)

        self.weight_critic = NNGammaCritic(o_dim, hidden, args.scale_weight)
        self.weight_critic.to(device)

        self.actor_opt = WeightedKFACOptimizer(model=self.actor, lr=args.lr, kl_clip=args.kfac_clip, max_grad_norm=args.max_grad_norm)             
        self.weight_critic_opt = torch.optim.Adam(self.weight_critic.parameters(), lr=args.lr_weight)

    def act(self, op):
        with torch.no_grad():        
            a, lprob = self.actor.act(torch.from_numpy(op).unsqueeze(0).to(self.device))
            if self.continuous:
                a = a.reshape((-1))
        
        return a, lprob

    def update_weight(self, frames, times):
        values, weights = self.weight_critic(torch.from_numpy(frames).to(self.device))
        
        self.actor_opt.weights = weights.detach()/self.args.scale_weight
        
        # Discount correction
        labels = self.gamma ** times
        weight_loss = torch.mean((torch.from_numpy(labels).to(self.device) * self.args.scale_weight - weights)**2)
      
        self.weight_critic_opt.zero_grad()
        weight_loss.backward()
        self.weight_critic_opt.step()
        
        return weight_loss.item()

    def learn(self, count, obs):
        loss = 0
        if count == self.buffer_size:            
            frames, rewards, dones, actions, old_lprobs, times, next_frames = self.buffer.sample(self.BS)            
            
            # Discount correction
            weight_loss = self.update_weight(frames, times)
            
            action_log_probs, dist_entropy = self.actor(torch.from_numpy(frames).to(self.device), torch.from_numpy(actions).to(self.device))
            values, weights = self.weight_critic(torch.from_numpy(frames).to(self.device))
            last_val, _ = self.weight_critic(torch.from_numpy(obs).unsqueeze(0).to(self.device)) 

            # GAE Estimation
            value_preds = torch.zeros(self.BS + 1)
            value_preds[-1] = last_val
            rets = torch.zeros(self.BS)
           
            gae = 0
            for step in reversed(range(rewards.size)):
                delta = rewards[step] + (1 - dones[step]) * self.gamma * value_preds[step + 1] - value_preds[step]
                gae = delta + (1 - dones[step]) * self.gamma * self.lmbda * gae
                rets[step] = gae + value_preds[step]           

            # Action loss
            advantages = rets - values            
            dist_entropy = dist_entropy.mean()                              
            action_loss = -(advantages.detach() * action_log_probs).mean()
            
            # Critic loss
            value_loss = self.value_loss_coef * advantages.pow(2).mean()            

            # Fisher loss for actor
            self.actor_opt.weights = weights.detach()/self.args.scale_weight            
            if self.actor_opt.steps % self.actor_opt.Ts == 0:                
                # Compute fisher, see Martens 2014
                self.actor.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()
                
                self.actor_opt.acc_stats = True
                pg_fisher_loss.backward(retain_graph=True)
                self.actor_opt.acc_stats = False
        
            # Apply gradients
            self.actor_opt.zero_grad()            
            (action_loss - dist_entropy * self.entropy_coef).backward()
            self.actor_opt.step()

            self.weight_critic_opt.zero_grad()
            value_loss.backward()
            self.weight_critic_opt.step()
            
            self.buffer.empty()
            count = 0
            return value_loss.item() + action_loss.item(), count
        
        return 0, count
