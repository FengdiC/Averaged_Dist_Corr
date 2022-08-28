import torch
from torch import nn

class MLPCategoricalActor(nn.Module):
    def __init__(self,o_dim,n_actions,hidden,shared=False):
        super(MLPCategoricalActor,self).__init__()
        self.body = nn.Sequential(nn.Linear(o_dim,hidden),nn.ReLU()
                              ,nn.Linear(hidden,hidden),nn.ReLU())
        self.prob = nn.Sequential(nn.Linear(hidden,n_actions),nn.Softmax())
        self.shared = shared
        if self.shared:
            self.critic = nn.Linear(hidden,1)
        else:
            self.critic_body = nn.Sequential(nn.Linear(o_dim,hidden),nn.ReLU()
                              ,nn.Linear(hidden,hidden),nn.ReLU())
            self.critic = nn.Linear(hidden, 1)


    def forward(self,obs,actions):
        obs = obs.float()
        body = self.body(obs)
        prob = self.prob(body)
        dist = torch.distributions.Categorical(prob)
        if self.shared:
            value = self.critic(body)
            return dist.log_prob(actions),torch.squeeze(value),dist.entropy()
        else:
            critic_body = self.critic_body(obs)
            value = self.critic(critic_body)
            return dist.log_prob(actions), torch.squeeze(value),dist.entropy()

    def act(self,obs):
        obs=obs.float()
        body = self.body(obs)
        prob = self.prob(body)
        dist = torch.distributions.Categorical(prob)
        a = dist.sample()
        return int(a), dist.log_prob(torch.as_tensor(a))

    def get_policy(self,obs):
        obs = obs.float()
        body = self.body(obs)
        prob = self.prob(body)
        dist = torch.distributions.Categorical(prob)
        return dist.probs


class MLPGaussianActor(nn.Module):
    def __init__(self,o_dim,a_dim,hidden,shared=False):
        super(MLPGaussianActor,self).__init__()
        self.body = nn.Sequential(nn.Linear(o_dim,hidden),nn.ReLU()
                              ,nn.Linear(hidden,hidden),nn.ReLU())
        self.mu = nn.Linear(hidden,a_dim)
        self.std = torch.ones(a_dim,requires_grad=True)
        self.shared = shared
        if self.shared:
            self.critic = nn.Linear(hidden,1)
        else:
            self.critic_body = nn.Sequential(nn.Linear(o_dim,hidden),nn.ReLU()
                              ,nn.Linear(hidden,hidden),nn.ReLU())
            self.critic = nn.Linear(hidden, 1)


    def forward(self,obs,actions):
        obs = obs.float()
        actions = actions.float()
        body = self.body(obs)
        mu = self.mu(body)
        std = self.std
        dist = torch.distributions.MultivariateNormal(mu,torch.diag(std))
        if self.shared:
            value = self.critic(body)
            return dist.log_prob(actions),torch.squeeze(value),dist.entropy()
        else:
            critic_body = self.critic_body(obs)
            value = self.critic(critic_body)
            return dist.log_prob(actions), torch.squeeze(value),dist.entropy()

    def act(self,obs):
        obs=obs.float()
        body = self.body(obs)
        mu = self.mu(body)
        std = self.std
        dist = torch.distributions.MultivariateNormal(mu,torch.diag(std))
        a = dist.sample()
        return a.detach().cpu().numpy(), dist.log_prob(torch.as_tensor(a))
