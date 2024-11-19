import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class Eeg_lstm_fc(nn.Module):
    def __init__(
            self, resolution, num_channels, seq_length, hidden_size, num_layers, num_classes, dropout=0.5
    ):
        super(Eeg_lstm_fc, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
             num_channels, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(x.size(0), x.size(1), -1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        # out = self.softmax(out)
        return out

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, predictions = torch.max(outputs, 1)
        return predictions
