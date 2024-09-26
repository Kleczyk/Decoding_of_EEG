import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

class EEGClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, seq_length, dropout=0.5):
        super(EEGClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Warstwa LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # Warstwa w pełni połączona
        self.fc = nn.Linear(hidden_size, num_classes)

        # Funkcja aktywacji
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)

        # Inicjalizacja ukrytych stanów
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Przepuszczenie przez warstwę LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pobranie ostatniego ukrytego stanu
        out = out[:, -1, :]

        # Przepuszczenie przez warstwę w pełni połączoną
        out = self.fc(out)

        # Zastosowanie funkcji aktywacji
        out = self.softmax(out)
        return out

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, predictions = torch.max(outputs, 1)
        return predictions
