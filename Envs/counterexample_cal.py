import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from gym.core import Env
from gym import spaces

class TwoStates(Env):
    """
    Action space, Discrete(8)
    Observation space, Box(2), positions
    """
    def __init__(self,timeout=20):
        self._actions = ['top','bottom']
        self._states = [[0,1],[1,0]]
        self._obs = np.array(self._states,dtype=np.float32)
        self._timeout = timeout
        self.steps = 0

    def reset(self):
        self.steps = 0
        obs = self._obs[0]
        return obs,0

    def step(self,action):
        self.steps += 1
        if self.steps%2==0:
            next_obs = self._obs[0]
            idx = 0
            if action==0:
                reward = -1
            else:
                reward = 1
        else:
            next_obs = self._obs[1]
            idx =1
            if action == 0:
                reward = 1
            else:
                reward = -1
        done = False or self.steps == self._timeout
        # Metadata
        info = {}
        return next_obs, idx,reward, done, info

    @property
    def action_space(self):
        return spaces.Discrete(2)

    @property
    def observation_space(self):
        return spaces.Discrete(2)

    def get_states(self):
        return self._obs

def pi(theta=0):
    return np.exp(theta)/(1+np.exp(theta))

def q_values(theta=0,gamma=0.8):
    q_a_top = (1+gamma* (1-2*gamma+(-2+2*gamma)*pi(theta) ) )/(1-gamma**2)
    q_a_bottom = q_a_top-2
    q_b_top = -1+gamma* (q_a_top - 2* (1-pi(theta)))
    q_b_bottom = q_b_top +2
    return q_a_top,q_a_bottom,q_b_top,q_b_bottom

def true_grad(theta,gamma=0.8):
    q_a_top, q_a_bottom, q_b_top, q_b_bottom = q_values(theta)
    log_grad = 1 - pi(theta)
    log_grad_neg = -pi(theta)

    dist_a = (1-gamma)/(1-gamma**2)
    dist_b = gamma * (1-gamma)/(1-gamma**2)

    grad = dist_a * (pi(theta)*log_grad*q_a_top + (1-pi(theta))*log_grad_neg*q_a_bottom) + \
           dist_b * (pi(theta)*log_grad*q_b_top + (1-pi(theta))*log_grad_neg*q_b_bottom)
    return grad

def batch_weighted_grad(theta,log_grad,correction,states_idx,actions,times,gamma):
    q_a_top, q_a_bottom, q_b_top, q_b_bottom = q_values(theta)
    Q = np.array([[q_a_top, q_a_bottom], [q_b_top, q_b_bottom]])
    values = Q[0,0] * (1-states_idx)*(1-actions) + Q[0,1] * (1-states_idx)*actions+Q[1,0] * states_idx*(1-actions) + Q[1,1] * states_idx*actions

    grad = correction * log_grad * torch.from_numpy(values)
    # grad = gamma**torch.from_numpy(times)*log_grad * torch.from_numpy(values)
    print('correction: ', correction.detach().numpy())
    print(gamma**times)
    return torch.mean(grad)

def batch_naive_grad(theta,log_grad,correction,states_idx,actions,times,gamma):
    q_a_top, q_a_bottom, q_b_top, q_b_bottom = q_values(theta)
    Q = np.array([[q_a_top, q_a_bottom], [q_b_top, q_b_bottom]])
    values = Q[0,0] * (1-states_idx)*(1-actions) + Q[0,1] * (1-states_idx)*actions+Q[1,0] * states_idx*(1-actions) + Q[1,1] * states_idx*actions

    # grad = correction * log_grad * torch.from_numpy(values)
    grad = gamma**torch.from_numpy(times)*log_grad * torch.from_numpy(values)
    return torch.mean(grad)

def batch_biased_grad(theta,log_grad,correction,states_idx,actions,times,gamma):
    q_a_top, q_a_bottom, q_b_top, q_b_bottom = q_values(theta)
    Q = np.array([[q_a_top, q_a_bottom], [q_b_top, q_b_bottom]])
    values = Q[0,0] * (1-states_idx)*(1-actions) + Q[0,1] * (1-states_idx)*actions+Q[1,0] * states_idx*(1-actions) + Q[1,1] * states_idx*actions

    # grad = correction * log_grad * torch.from_numpy(values)
    grad = log_grad * torch.from_numpy(values)
    return torch.mean(grad)

def plot_grad():
    gamma=0.8
    theta = 0
    last_theta=-100
    params = []
    policies = []
    lr =1
    plt.figure()
    while theta-last_theta> 0.001:
        last_theta = theta
        params.append(last_theta)
        policies.append(pi(theta))
        grad = true_grad(theta)
        theta = theta + lr * grad
        # plt.arrow(last_theta, pi(last_theta), theta - last_theta, pi(theta) - pi(last_theta))
        print("new update: ",grad," policy: ",pi(theta))

    # plt.plot(params,policies,'-o')
    plt.subplot(211)
    plt.plot(range(len(params)),params)
    # plt.tick_params(labelbottom=False)
    plt.xlabel("training steps")
    plt.ylabel("parameter value")
    plt.subplot(212)
    plt.plot(range(len(policies)), policies)
    # plt.tick_params(labelbottom=False)
    plt.xlabel("training steps")
    plt.ylabel("policy probability of the top action")
    plt.show()


class Actor(torch.nn.Module):
    def __init__(self, device=None):
        super(Actor, self).__init__()
        self.to(device)
        self.theta = Variable(-0.5*torch.randn(1,).type(torch.FloatTensor), requires_grad=True)

    def forward(self, actions):
        policy = [1/(1+torch.exp(self.theta)), -torch.exp(self.theta)/(1+torch.exp(self.theta))]
        policies = actions*policy[1] + (1-actions)*policy[0]
        return policies

    def act(self):
        prob_a = torch.exp(self.theta)/(1+torch.exp(self.theta))
        prob_a = prob_a.detach().numpy()
        if prob_a[0]>0.5:
            return 0
        return 1

    def update(self,grad,lr):
        print('before update', self.theta)
        print(grad)
        self.theta = self.theta + lr* grad