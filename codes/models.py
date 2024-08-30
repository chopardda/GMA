import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size = 467, hidden_size = 128, num_layers = 2, num_classes = 1, wandb_config=None):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size if wandb_config is None else wandb_config.hidden_size
        self.num_layers = num_layers if wandb_config is None else wandb_config.num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class CNN1D(nn.Module):
    def __init__(self, sequence_length = 467, input_size = 34, num_classes = 1, wandb_config=None):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        if wandb_config is not None:
            self.fc1 = nn.Linear(128 * (sequence_length // 4), wandb_config.out_features)  # Adjust this based on the pooling
            self.fc2 = nn.Linear(wandb_config.out_features, num_classes)
        else:
            self.fc1 = nn.Linear(128 * (sequence_length // 4), 100)  # Adjust this based on the pooling
            self.fc2 = nn.Linear(100, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x