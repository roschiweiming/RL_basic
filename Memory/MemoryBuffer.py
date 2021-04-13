import numpy as np
from collections import deque
import random


# -- Memory Buffer -- #
class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxsize = size
        self.len = 0    # current size

    def sample(self, count):
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])

        return s_array, a_array, r_array, new_s_array, done_array

    def len(self):
        return self.len

    def store_transition(self, s, a, r, new_s, done):
        transition = (s, [a], [r], new_s, [done])
        self.len += 1
        if self.len > self.maxsize:
            self.len = self.maxsize
        self.buffer.append(transition)