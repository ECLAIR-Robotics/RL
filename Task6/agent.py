import os
import torch
import numpy as np
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import matplotlib as plt

import model
import memory

class Agent:
    def __init__(self, memory_capacity=1000000, gamma=0.99, input_dims=(4, 84, 84), output_dim=6, lr=0.000025):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # DQN model
        self.model = model.DQN(input_dims, output_dim, lr).to(self.device)
        # DQN target
        self.target = model.DQN(input_dims, output_dim, lr).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        # Loss function is Huber loss, https://en.wikipedia.org/wiki/Huber_loss, feel free to change to MSE
        self.loss = lambda expected, target: F.smooth_l1_loss(expected, target)

        # Agents Experience Replay Memory
        self.memory = memory.ExperienceReplay(memory_capacity)

        # gamma hyperparam for calculating loss
        self.gamma = gamma

    # Depending on epsilon, we either take a random action, or the action corresponding
    # to the maximum q_values calculated by our neural network
    def get_action(self, state, epsilon, num_actions):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(range(num_actions))
        else:
            # when passed through network it should receive (batch_size, frame_stack, 84, 84), so we wrap state in a list i.e. [state] before making the tensor
            state = torch.tensor([state], dtype=torch.float32, device=self.device)
            q_values = self.model.forward(state)
            action = q_values.max(1)[1].item() # gets the index of the max q value which corresponds to the action chosen
        return action

    # get action corresponding to the maximum q_values calculated by our neural network, used for evaluation
    def _get_action(self, state):
        # when passed through network it should receive (batch_size, frame_stack, 84, 84), so we wrap state in a list i.e. [state] before making the tensor
        state = torch.tensor([state], dtype=torch.float32, device=self.device)
        q_values = self.model.forward(state)
        action = q_values.max(1)[1].item() # gets the index of the max q value which corresponds to the action chosen
        return action

    # Refresh our target model weights with our current model weights
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    # Computes the loss function and backpropagates
    def compute_td_loss(self, batch_size):
        # Sample batch_size number of transitions from memory
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        # Make Everything Tensors, load on device being used (cpu/gpu)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        ##################################################################
        # YOUR CODE GOES HERE
        ##################################################################

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss

    # Saves the model weights, the optimizer state, and the experience replay buffer to agent_saves/DQN_model.pt, agent_saves/Adam_optimizer.pt, and agent_saves/mem.pkl
    # Will overwrite existing files with the same name
    def save_variables(self, direc='agent_saves'):
        if not os.path.exists(direc):
            os.makedirs(direc)
        self.save_model(direc=direc)
        self.save_optim(direc=direc)
        self.memory.save(direc=direc)

    # Saves the model weights at dir/name, default: 'agent_saves/DQN_model.pt'
    def save_model(self, direc='agent_saves', name='DQN_model.pt'):
        if not os.path.exists(direc):
            os.makedirs(direc)
        torch.save(self.model.state_dict(), os.path.join(direc, name))

    # Saves the optimizer state at dir/name, default: 'agent_saves/Adam_optimizer.pt'
    def save_optim(self, direc='agent_saves', name='Adam_optimizer.pt'):
        if not os.path.exists(direc):
            os.makedirs(direc)
        torch.save(self.model.optimizer.state_dict(), os.path.join(direc, name))

    # Restores the states of the model from default path used in save_variables
    # copies the model state dict to the target model state dict if copy_model_to_target is True, used to resume training
    def load_variables(self, direc='agent_saves', copy_model_to_target=False, load_mem=False):
        self.load_model(path= direc + '/DQN_model.pt')
        self.load_optim(path= direc + '/Adam_optimizer.pt')
        if copy_model_to_target:
            self.target.load_state_dict(self.model.state_dict())
            self.target.to(self.device)
        if load_mem:
            self.memory.load(direc=direc)

    # Load the model weights at PATH
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    # Load the optimizer weights at PATH
    def load_optim(self, path):
        self.model.optimizer.load_state_dict(torch.load(path))

    # Load model weights at PATH, and set to eval() mode
    def eval(self, path='agent_saves/DQN_model.pt'):
        self.load_model(path)
        self.model.eval() # model in eval mode