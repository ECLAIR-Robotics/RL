# Modified from the original via https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import random
import pickle as pkl
import os

# Experience Replay Memory our agent uses to store (s, a, r, s') transitions
class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def __len__(self):
        return len(self.memory)

    # Store a transition sequence
    def add(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    # Sample a batch of transition sequences
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return state, action, reward, next_state, done

    # Serialize and save the entire experience replay buffer
    def save(self, direc='agent_saves'):
        with open(direc + '/mem.pkl', 'wb') as file:
            pkl.dump(self.memory, file)

    # Load a serialized replay memory buffer that was saved via this.save()
    def load(self, direc='agent_saves'):
        print(os.path.exists(direc+'/mem.pkl'))
        with open(direc + '/mem.pkl', 'rb') as file:
            self.memory = pkl.load(file)