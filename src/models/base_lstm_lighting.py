
import lightning.pytorch as pl
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class LSTM_base_lighting(pl.LightningModule):
    def __init__(self, sequence_length, hidden_size, num_layers, dropout, learning_rate, num_classes,num_channels):
        super().__init__()
        self.save_hyperparameters()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        logits = self.fc(lstm_out[:, -1, :])
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        y_pred = torch.argmax(y_hat, dim=1)
        acc = accuracy_score(y.cpu(), y_pred.cpu())
        precision = precision_score(y.cpu(), y_pred.cpu(), average='macro', zero_division=0)
        recall = recall_score(y.cpu(), y_pred.cpu(), average='macro', zero_division=0)
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='macro')

        try:
            y_prob = torch.softmax(y_hat, dim=1).cpu().detach().numpy()
            y_true = y.cpu().numpy()
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            auc = float('nan') 

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
