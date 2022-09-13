from gym.core import Env
import gym

import numpy as np

class RepeatEnvWrapper(Env):
    def __init__(self, name='Pendulum-v1',timeout=500):
        # actions: up down left right
        # actions: upleft upright downleft downright
        self._timeout = timeout
        self.steps = 0
        self.env=gym.make(name)
        self.next_step_restart= False

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        self.next_step_restart = False
        self.obs = self.env.reset()
        self.steps = 0
        return self.obs

    def seed(self,number):
        self.env.seed(number)

    def _restart(self):
        self.next_step_restart = False
        self.obs = self.env.reset()
        return self.obs

    def step(self, action):
        self.steps +=1
        if self.next_step_restart:
            self.obs = self._restart()
            reward = 0
            done = False
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