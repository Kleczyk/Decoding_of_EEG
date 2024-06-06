import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
from pytorch_lightning import LightningModule, Trainer
from torchmetrics.functional.classification.accuracy import accuracy
from src.data.CWTDataset import CWTDataset
from src.data.CWTSubset import CWTSubset


class CWT_EEG(LightningModule):
    """
    Class for training the model from EEG data after CWT transformation.
    Attributes:
        batch_size: int - number of samples in a batch
        sequence_length: int - length of the sequence
        input_size: int - number of features in the input
        hidden_size: int - number of features in the hidden state
        num_layers: int - number of LSTM layers
        lr: float - learning rate
        label_smoothing: float - label smoothing factor
        num_of_classes: int - number of classes
        val_percent: float - percentage of validation samples
        loss: torch.nn.CrossEntropyLoss - loss function
        lstm: nn.LSTM - LSTM layer
        fc: nn.Linear - fully connected layer
        ds: CWTDataset - dataset
        num_val_samples: int - number of validation samples
        train_set: CWTSubset - training dataset
        val_set: CWTSubset - validation dataset
    Methods:
        __init__ - constructor for the class
        forward - forward pass of the model
        count_parameters - counts the number of parameters in the model
        on_train_start - logs hyperparameters
        training_step - training step
        configure_optimizers - configures the optimizer
        validation_step - validation step
        generate_validation_indices - generates validation indices
        generate_train_indices - generates training indices
        setup - sets up the training and validation datasets
        train_dataloader - returns the training dataloader
        val_dataloader - returns the validation dataloader
        get_len_train_val - returns the length of the training and validation datasets
    """
    def __init__(
            self,
            batch_size,
            sequence_length,
            input_size,
            hidden_size,
            num_layers,
            lr,
            label_smoothing=0,
            conn_train=None,
            conn_val=None,

    ):
        super().__init__()
        self.save_hyperparameters('batch_size', 'sequence_length', 'input_size', 'hidden_size', 'num_layers', 'lr',
                                  'label_smoothing')
        self.conn_train = conn_train
        self.conn_val = conn_val
        self.hparams.batch_size = batch_size
        self.hparams.input_size = input_size
        self.hparams.sequence_length = sequence_length
        self.hparams.lr = lr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_of_classes = 3
        self.val_percent = 0.01
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.num_of_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = hn[-1, :, :]
        out = self.fc(out)

        return out

    # custom
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # only for HP
    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/train_loss": float("nan"),
                "hp/train_acc": float("nan"),
                "hp/val_loss": float("nan"),
                "hp/val_acc": float("nan"),
            },
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_of_classes)

        self.log("hp/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("hp/train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_of_classes)

        self.log("hp/val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("hp/val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def generate_validation_indices(
            self, data_length, num_of_val_samples, sequence_length
    ):
        available_indices = set(range(data_length))
        val_indices = []
        for _ in range(num_of_val_samples):
            if len(available_indices) == 0:
                raise ValueError(
                    "Nie można wygenerować więcej próbek z uwzględnieniem minimalnego dystansu"
                )
            chosen_index = int(np.random.choice(list(available_indices)))
            val_indices.append(chosen_index)
            indices_to_remove = set(
                range(
                    max(0, chosen_index - (2 * sequence_length) - 3),
                    min(data_length, chosen_index + (2 * sequence_length) + 3),
                )
            )
            available_indices.difference_update(indices_to_remove)

        return val_indices

    def generate_train_indices(self, data_length, val_i, sequence_length):
        min_distance = sequence_length + 1
        mask = np.ones(data_length, dtype=bool)
        for index in val_i:
            start = max(0, index - min_distance)
            end = min(data_length, index + min_distance + 1)

            mask[start:end] = False

        training_indices = list(np.where(mask)[0])
        return training_indices

    def setup(self, stage=None):
        self.ds = CWTDataset("df_train_cwt_data.db", self.hparams.sequence_length)
        self.num_val_samples = int(
            len(self.ds) / (4 * self.hparams.sequence_length + 6)
        )

        val_indices = self.generate_validation_indices(
            len(self.ds), self.num_val_samples, self.hparams.sequence_length
        )
        train_indices = self.generate_train_indices(
            len(self.ds), val_indices, self.hparams.sequence_length
        )
        print(
            "percent of val samples",
            len(val_indices) / (len(val_indices) + len(train_indices)),
        )
        self.train_set = CWTSubset(self.ds, train_indices)
        self.val_set = CWTSubset(self.ds, val_indices)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=7,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set, batch_size=self.hparams.batch_size, num_workers=7
        )

    def get_len_train_val(self):
        self.setup()
        return len(self.train_set), len(self.val_set)
