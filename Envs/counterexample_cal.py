import numpy as np
import matplotlib.pyplot as plt

gamma = 0.8

def pi(theta=0):
    return np.exp(theta)/(1+np.exp(theta))

def q_values(theta=0):
    q_a_top = (1+gamma* (1-2*gamma+(-2+2*gamma)*pi(theta) ) )/(1-gamma**2)
    q_a_bottom = q_a_top-2
    q_b_top = -1+gamma* (q_a_top - 2* (1-pi(theta)))
    q_b_bottom = q_b_top +2
    return q_a_top,q_a_bottom,q_b_top,q_b_bottom

def true_grad(theta):
    q_a_top, q_a_bottom, q_b_top, q_b_bottom = q_values(theta)
    log_grad = 1 - pi(theta)
    log_grad_neg = -pi(theta)

    dist_a = (1-gamma)/(1-gamma**2)
    dist_b = gamma * (1-gamma)/(1-gamma**2)

    grad = dist_a * (pi(theta)*log_grad*q_a_top + (1-pi(theta))*log_grad_neg*q_a_bottom) + \
           dist_b * (pi(theta)*log_grad*q_b_top + (1-pi(theta))*log_grad_neg*q_b_bottom)
    return grad

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
    plt.arrow(last_theta, pi(last_theta), theta - last_theta, pi(theta) - pi(last_theta))
    print("new update: ",grad," policy: ",pi(theta))

# plt.plot(params,policies,'-o')
plt.xticks(params)
plt.tick_params(labelbottom=False)
plt.xlabel("parameter value")
plt.ylabel("policy probability of the top action")
plt.show()


