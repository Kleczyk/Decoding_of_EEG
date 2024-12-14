import lightning.pytorch as pl
import torch
from torch import nn
from typing import Tuple, Any
from utils.calculate_metrics import calculate_metrics


class LSTMBaseLighting(pl.LightningModule):
    """
    A LightningModule implementation of an LSTM-based model for EEG classification.

    Attributes:
        num_layers (int): Number of LSTM layers.
        hidden_size (int): Number of hidden units in each LSTM layer.
        lstm (nn.LSTM): LSTM layer for sequential data processing.
        fc (nn.Linear): Fully connected layer for classification.
        criterion (nn.Module): Loss function.
    """

    def __init__(
        self,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        num_classes: int,
        num_channels: int
    ) -> None:
        """
        Initialize the LSTMBaseLighting model.

        :param sequence_length: Length of the input sequences.
        :param hidden_size: Size of the hidden layer in LSTM.
        :param num_layers: Number of LSTM layers.
        :param dropout: Dropout rate.
        :param learning_rate: Learning rate for the optimizer.
        :param num_classes: Number of output classes.
        :param num_channels: Number of input features (EEG channels).
        :return: None
        """
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

    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param input_sequences: Input tensor of shape (batch_size, seq_length, num_channels).
        :return: Logits of shape (batch_size, num_classes).
        """
        batch_size = input_sequences.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input_sequences.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input_sequences.device)

        lstm_out, _ = self.lstm(input_sequences, (h0, c0))
        logits = self.fc(lstm_out[:, -1, :])
        return logits

    def compute_metrics(self, true_labels: torch.Tensor, predicted_logits: torch.Tensor) -> dict[str, Any]:
        """
        Compute performance metrics.

        :param true_labels: True labels.
        :param predicted_logits: Predicted logits.
        :return: Dictionary containing performance metrics.
        """
        return calculate_metrics(true_labels, predicted_logits)

    def step(self, batch_data: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """
        Generic step method for training and validation.

        :param batch_data: Batch of data (features, labels).
        :param stage: Stage of the step ('train' or 'val').
        :return: Loss value.
        """
        features, labels = batch_data
        predictions = self(features)
        loss = self.criterion(predictions, labels)
        metrics = self.compute_metrics(labels, predictions)

        for metric_name, metric_value in metrics.items():
            self.log(f"{stage}_{metric_name}", metric_value, prog_bar=True)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        :param batch_data: Batch of data (features, labels).
        :param batch_idx: Index of the batch.
        :return: Loss value.
        """
        return self.step(batch_data, stage="train")

    def validation_step(self, batch_data: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        :param batch_data: Batch of data (features, labels).
        :param batch_idx: Index of the batch.
        :return: Loss value.
        """
        return self.step(batch_data, stage="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers.

        :return: Configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
