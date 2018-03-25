import numpy as np


class MemoryBank:

    def __init__(self, size):
        self.size = size
        self.index = 0
        self.memory_bank = np.empty((size, 5), dtype=object)

    def add(self, state, action, reward, new_state, done):
        memory_index = self.index % self.size
        self.memory_bank[memory_index] = np.array((state, action, reward, new_state, done))

        self.index += 1

    def get_mini_batch(self, batch_size):
        minibatch = self.memory_bank[np.random.randint(self.size, size=batch_size)]
        return minibatch

    def save_memory(self, path):
        np.save(path, self.memory_bank)

    def load_memory(self, path):
        self.memory_bank = np.load(path)
