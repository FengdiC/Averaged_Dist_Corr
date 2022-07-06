import gym
from Components.utils import meanstdnormalizaer

class TaskWrapper:
    """Wrapper for the environment provided by gym"""

    def __init__(self, envName):
        self.env = gym.make(envName)
        self.observation_space =self.env.observation_space
        self.action_space = self.env.action_space

    def seed(self,number):
        self.env.seed(number)

    def reset(self):
        self.frame = self.env.reset()
        self.midstep= False
        # op = meanstdnormalizaer(env.reset())
        return self.frame

    def step(self, action):
        # Keep track whether the original task has terminated
        if self.midstep:
            self.frame = self.reset()
            self.reward = 0
        else:
            self.frame, self.reward, self.midstep, info = self.env.step(action)
        return self.frame, self.reward, self.midstep, None
