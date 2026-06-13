import torch
import torch.nn as nn

class SpiralModelV0(nn.Module):
    def __init__(self, input_size=2, hidden_dim=10, output_size=3):
        super().__init__()
        
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_dim),   # nn.Linear(2, 10)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # nn.Linear(10, 10)
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)  # nn.Linear(10, 3)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)