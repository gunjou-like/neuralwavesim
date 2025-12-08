# app/model.py
import torch.nn as nn

class WavePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WavePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)