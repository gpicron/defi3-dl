import random
from collections.__init__ import deque
import numpy as np


class Experience:

    def __init__(self, state, action, next_state, reward, episode=None):
        self.reward = reward
        self.next_state = next_state
        self.action = action
        self.state = state
        self.episode = episode

class ExperienceMemory:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def append(self, exp):
        self.buffer.append(exp)

    def __len__(self):
        return len(self.buffer)

    def sample(self, size, weightComputingFunction=None):
        if weightComputingFunction is None:
            return random.sample(self.buffer, min(size, len(self.buffer)))
        else:
            weights = weightComputingFunction(self.buffer)
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            indeces = np.random.choice(len(weights), size, replace=False, p=weights)

            return [self.buffer[i] for i in indeces]

    def last(self):
        return self.buffer[-1]