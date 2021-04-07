import torch
import torch.nn as nn

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_dims, output_dim, lr):
        super(DQN, self).__init__()

        self.input_dims = input_dims
        self.output_dim = output_dim

        # Feature extraction section of the network
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Fully connected layer that maps to actions in the environment
        self.fully_connected = nn.Sequential(
            nn.Linear(self.extracted_feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # Optimizer function
        self.lr = lr
        # feel free to change to RMSProp or other
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.train()

    # Determines the input size for the fully connected of linear layers
    def extracted_feature_size(self):
        return self.feature_extraction(torch.zeros(1, *(self.input_dims[0], 84,84))).view(1, -1).size(1)

    # One forward pass through the network
    def forward(self, state):
        state = self.feature_extraction(state)
        # flatten after pass through convolutions
        state = state.view(state.size(0), -1)
        q_values = self.fully_connected(state)
        return q_values