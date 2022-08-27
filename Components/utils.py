import numpy as np
import os
import random
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')

import argparse
def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DQN")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--agent', help='the trainer used', type=str, default='batch_ac')
    parser.add_argument('--env', help='Environment name', type=str, default='CartPole-v1')
    parser.add_argument('--log_dir', help='log direction', type=str, default='./logs/')

    parser.add_argument('--hidden', type=int, help='Number of hidden units', default=64)
    parser.add_argument('--var', type=float, help='Initialization value for action probability variacne', default=1)
    parser.add_argument('--grad_norm', type=int, help='The bar to clip down gradients', default=2.5)
    parser.add_argument('--lr_decay', type=float, help='The decay step for learning rate schedule', default=1e5)
    parser.add_argument('--batch_size', type=int, help='Max Episode Length', default=64)
    parser.add_argument('--buffer', type=int, help='Buffer size', default=64)
    parser.add_argument('--epoch', type=int, help='Epoches for mini-batches', default=10)
    parser.add_argument('--epoch_weight', type=int, help='Epoches for mini-batches', default=10)

    parser.add_argument('--gamma', type=float, help='Max Episode Length', default=0.99)
    parser.add_argument('--lam', type=float, help='Max Episode Length', default=0.95)
    parser.add_argument('--lr', type=float, help='Max Episode Length', default=0.0003)
    parser.add_argument('--LAMBDA_1', type=float, help='Lambda 1 for entropy', default=0)
    parser.add_argument('--LAMBDA_2', type=float, help='Lambda 2 for mse', default=10)
    parser.add_argument('--naive', type=bool, help='IF add on naive gamma power correction', default=True)
    parser.add_argument('--continuous', type=bool, help='whether actions are continuous', default=False)
    return parser.parse_args()

class TargetNetworkUpdater:
    """Copies the parameters of the main DQN to the target DQN"""

    def __init__(self, main_dqn_vars, target_dqn_vars):
        """
        Args:
            main_dqn_vars: A list of tensorflow variables belonging to the main DQN network
            target_dqn_vars: A list of tensorflow variables belonging to the target DQN network
        """
        self.main_dqn_vars = main_dqn_vars
        self.target_dqn_vars = target_dqn_vars

    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops

    def update_networks(self, sess):
        """
        Args:
            sess: A Tensorflow session object
        Assigns the values of the parameters of the main network to the
        parameters of the target network
        """
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)
        print("target update")

def meanstdnormalizaer(sequ):
    mean = np.mean(sequ)
    std = np.std(sequ)
    x = (sequ - mean)/(std+.001)
    return x

# def gymdisplay(env,MAIN,continuous=True):
#     op = meanstdnormalizaer(env.reset())
#     done=False
#     step = 0
#     while not done and step<300:
#         obs = tf.expand_dims(tf.constant(op), 0)
#         if continuous:
#             mean, var, a, value = MAIN(obs)
#             obs, r, done, infos = env.step(a)
#         else:
#             action_prob, a, value = MAIN(obs)
#             obs, r, done, infos = env.step(int(a))
#
#         # Observe
#         op = meanstdnormalizaer(obs)
#         img = env.render(mode="rgb_array")
#         plt.imshow(img)
#         plt.pause(0.00001)
#         step+=1
#     img = env.render(mode="rgb_array")
#     plt.imshow(img)
#     plt.close()

class A:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)