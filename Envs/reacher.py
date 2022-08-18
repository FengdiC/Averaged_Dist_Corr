from gym.core import Env
from gym import spaces
import torch
import numpy as np
import itertools

class DotReacher(Env):
    """
    Action space, Discrete(8)
    Observation space, Box(2), positions
    """
    def __init__(self,timeout=200):
        # actions: up down left right
        # actions: upleft upright downleft downright
        self._aval = 0.4 * np.array([[0,1], [0,-1], [-1, 0], [1,0], [-1,1], [1,1], [-1, -1], [1,-1]
                                            ], dtype=np.float32)

        states = [-0.8,-0.4,0,0.4,0.8]
        self._states = list(itertools.product(states,states))
        self._obs = np.array(self._states,dtype=np.float32)

        self._pos_tol = 0.2
        self._LB = np.array([-1., -1.])
        self._UB = np.array([1., 1.])
        self._timeout = timeout
        self.steps = 0

    def reset(self):
        self.steps = 0
        self.pos = self._obs[np.random.randint(0,25)]
        obs = self.pos
        return obs

    def step(self,action):
        self.steps += 1
        self.pos = np.clip(self.pos + self._aval[int(action-1)] ,self._LB, self._UB)
        next_obs = self.pos
        # Reward
        reward = -0.01
        # Done
        done = np.allclose(self.pos, np.zeros(2), atol=self._pos_tol)
        done = done or self.steps == self._timeout
        # Metadata
        info = {}
        return next_obs, reward, done, info

    @property
    def action_space(self):
        return spaces.Discrete(9)

    @property
    def observation_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,))

    def transition_matrix(self,policy):
        """
        Label states by self._states
        Inputs: policy is a numpy of shape 25 x 8, containing softmax policies for each state
        Outputs: transition matrix 25 x 25
        """

        # find out the next state with the state and the action
        next_state = []
        for i in range(len(self._states)):
            obs = self._obs[i]
            next = np.clip(self._aval + obs, self._LB, self._UB)
            next_idx = [self._states.index(next[i].tolist()) for i in range(8)]
            next_state.append(next_idx)

        next_state = np.array(next_state).astype(np.int32)
        P = np.zeros((25,25))
        for i in range(len(self._states)):
            for j in range(8):
                P[i,next_state[i,j]] = policy[i,j]

        return P






