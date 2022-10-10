import torch
import numpy as np

from Agents.acktr import ACKTR
from Agents.kfac import WeightedKFACOptimizer
from Networks.actor_critic import MLPCategoricalActor, MLPGaussianActor
from Networks.weight import AvgDiscount


class WeightedACKTR(ACKTR):
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
        self.naive = args.naive           

        if args.continuous:
            self.actor_critic = MLPGaussianActor(o_dim, n_actions, hidden, shared, device)        
        else:
            self.actor_critic = MLPCategoricalActor(o_dim, n_actions, hidden, shared)
        self.actor_critic.to(device)

        self.weight_nn = AvgDiscount(o_dim, hidden, args.scale_weight)
        self.weight_nn.to(device)

        self.opt = WeightedKFACOptimizer(model=self.actor_critic, lr=args.lr, kl_clip=args.kfac_clip, max_grad_norm=args.max_grad_norm)             
        self.weight_opt = torch.optim.Adam(self.weight_nn.parameters(), lr=args.lr_weight)

    def update_weight(self, frames, times):        
        # Discount correction
        weights = self.weight_nn(torch.from_numpy(frames).to(self.device))        
        labels = self.gamma ** times
        weight_loss = 0.1 *  torch.mean((torch.from_numpy(labels).to(self.device) * self.args.scale_weight - weights)**2)
        
        self.weight_opt.zero_grad()
        self.opt.acc_stats = True
        weight_loss.backward()
        self.opt.acc_stats = False
        self.weight_opt.step()        

        return weight_loss.item()

    def learn(self, count, obs):
        loss = 0
        if count == self.buffer_size:            
            frames, rewards, dones, actions, old_lprobs, times, next_frames = self.buffer.sample(self.BS)            

            if self.naive:                
                self.opt.weights = self.gamma**torch.from_numpy(times.astype(np.float32)).to(self.device)
                weights = self.gamma**torch.from_numpy(times.astype(np.float32)).to(self.device)
            else:
                # Discount correction            
                weight_loss = self.update_weight(frames, times)        
                weights = self.weight_nn(torch.from_numpy(frames).to(self.device))        
                self.opt.weights = weights.detach()/self.args.scale_weight    
            
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

            # Action loss
            advantages = rets - values            
            dist_entropy = dist_entropy.mean()                              
            action_loss = -(advantages.detach() * action_log_probs * weights.detach()/self.args.scale_weight).mean()            

            # Critic loss
            value_loss = self.value_loss_coef * advantages.pow(2).mean()            

            # Fisher loss for actor                       
            if self.opt.steps % self.opt.Ts == 0:                
                # Compute fisher, see Martens 2014
                self.actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = torch.randn(values.size())
                if values.is_cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean() * self.value_fisher_coef

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self.opt.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.opt.acc_stats = False
        
            # Apply gradients
            self.opt.zero_grad()            
            (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
            self.opt.step()

            self.buffer.empty()
            count = 0
            return value_loss.item() + action_loss.item(), count
        
        return 0, count
