import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SlitherCNN(BaseFeaturesExtractor):
    """
    A custom CNN that can handle (channels, height, width) = (5, 20, 20).
    """
    def __init__(self, observation_space, features_dim=128):
        # We assume observation_space is a gym.spaces.Box with shape (5, 20, 20)
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # should be 5

        # Define your CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.zeros((1, n_input_channels, 20, 20))
            n_flatten = self.cnn(sample_input).shape[1]

        # Final linear layer to get the desired feature_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.linear(x)
