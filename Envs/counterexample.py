import numpy as np

class MixTime:
    """Wrapper for the environment provided by gym"""

    def __init__(self, envName):
        self.observation_space = np.zeros(1)
        self.action_space = DiscreteActions(n=2)

    def seed(self,number):
        return None

    def reset(self):
        self.state=1
        self.frame = np.array([0])
        return self.frame

    def step(self, action):
        # Keep track whether the original task has terminated
        if self.midstep:
            self.frame = self.reset()
            self.reward = 0
        else:
            self.frame, self.reward, self.midstep, info = self.env.step(action)
        return self.frame, self.reward, self.midstep, None

class DiscreteActions():
    def __init__(self,n):
        self.n = n