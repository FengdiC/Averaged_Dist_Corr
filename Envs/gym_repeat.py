from gym.core import Env
import gym

import numpy as np

class RepeatEnvWrapper(Env):
    def __init__(self, name='Pendulum-v1',timeout=200):
        # actions: up down left right
        # actions: upleft upright downleft downright
        self._timeout = timeout
        self.steps = 0
        self.env=gym.make(name)
        self.next_step_restart= False

    def reset(self):
        self.obs = self.env.reset()
        self.step = 0
        return self.obs

    def _restart(self):
        self.obs = self.env.reset()
        return self.obs

    def step(self, action):
        self.steps += 1
        if self.next_step_restart:
            self.obs = self._restart()
            reward = 0
        else:
            obs, r, done, infos = self.env.step(action)
            self.next_step_restart = done
            self.obs = obs
            # Reward
            reward = 0
        # Reach goal
        if done:
            reward = 1

        # Check termiation
        done = self.steps == self._timeout
        # Metadata
        info = {}
        return self.obs, reward, done, info