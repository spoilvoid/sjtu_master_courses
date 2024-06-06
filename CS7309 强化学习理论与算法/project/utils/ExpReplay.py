from typing import Tuple
import collections
import random
import numpy as np


class NormalReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        #print(state)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    

class RandomReplayBuffer:
    def __init__(self, state_size: tuple, max_size: int = 10000):
        self.states = np.empty((max_size, *state_size), dtype=float)
        self.actions = np.empty(max_size, dtype=int)
        self.rewards = np.empty(max_size, dtype=float)
        self.next_states = np.empty((max_size, *state_size), dtype=float)
        self.dones = np.empty(max_size, dtype=bool)
        self.cur_i = 0
        self.max_i = 0
        self.max_size = max_size

    def append(self, data: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """(s, a, r, s', done)"""
        self.states[self.cur_i, :] = data[0]
        self.actions[self.cur_i] = data[1]
        self.rewards[self.cur_i] = data[2]
        self.next_states[self.cur_i] = data[3]
        self.dones[self.cur_i] = data[4]
        self.cur_i = (self.cur_i + 1) % self.max_size
        self.max_i = min(self.max_i + 1, self.max_size - 1)

    def clear(self):
        self.cur_i = 0
        self.max_i = 0

    def sample(self, batch_size: int = 100):
        indices = np.random.randint(0, self.max_i, size=batch_size)
        return \
            self.states.take(indices, axis=0),\
            self.actions.take(indices),\
            self.rewards.take(indices),\
            self.next_states.take(indices, axis=0),\
            self.dones.take(indices)


class PriorityReplayBuffer(RandomReplayBuffer):
    def __init__(self, state_size: tuple, max_size: int = 10000):
        super().__init__(state_size, max_size)
        self.weights = np.empty(max_size, dtype=bool)
        self.alpha = 0.5
        self.beta = 0.5

    def append(self, data: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """(s, a, r, s', done)"""
        self.weights[self.cur_i] = self.max_priority() ** self.alpha
        super().append(data)

    def max_priority(self):
        if self.max_i == 0:
            return 1.0
        return np.max(self.weights[:self.max_i])

    def sample(self, batch_size: int = 100):
        sum_weights = np.sum(self.weights[:self.max_i])
        probs = self.weights[:self.max_i] / sum_weights
        indices = np.random.choice(np.arange(self.max_i), size=batch_size, p=probs)
        weights = np.abs((1 / self.max_i) * (1/probs[indices])) ** self.beta

        return indices,\
            self.states.take(indices, axis=0),\
            self.actions.take(indices),\
            self.rewards.take(indices),\
            self.next_states.take(indices, axis=0),\
            self.dones.take(indices),\
            weights

    def update(self, indices, tderrors):
        self.weights[indices] = np.abs(tderrors) ** self.alpha + 0.01