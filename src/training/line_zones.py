import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

# 1. Definicja modelu LSTM z warstwą FC
class LSTMLitModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout, learning_rate, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
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

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# 2. Funkcja celu dla Bayesowskiej optymalizacji

def bayesian_optimization():
    from ray import tune
    from ray.tune.integration.pytorch_lightning import TuneReportCallback

    # Define the search space for hyperparameters
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64]),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.randint(1, 4),
        "dropout": tune.uniform(0.1, 0.7),
        "input_size": tune.choice([32, 64, 128]),
        "num_classes": 3,
        "seq_length": tune.choice([32, 64, 128])
    }

    def train_model(config):
        # Dane treningowe
        input_size = config["input_size"]
        seq_length = config["seq_length"]
        num_classes = config["num_classes"]

        X = torch.rand(1000, seq_length, input_size)
        y = torch.randint(0, num_classes, (1000,))

        dataset = TensorDataset(X, y)
        train_set, val_set = random_split(dataset, [800, 200])
        train_loader = DataLoader(train_set, batch_size=config["batch_size"])
        val_loader = DataLoader(val_set, batch_size=config["batch_size"])

        # Model
        model = LSTMLitModel(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            learning_rate=config["lr"],
            num_classes=config["num_classes"]
        )

        # Logger dla Weight and Biases
        wandb_logger = WandbLogger(project="EEG_Classification")

        # Trener PyTorch Lightning
        trainer = pl.Trainer(
            max_epochs=10,
            logger=wandb_logger,
            callbacks=[
                TuneReportCallback({
                    "val_loss": "val_loss",
                    "val_acc": "val_acc",
                    "val_f1": "val_f1"
                })
            ]
        )

        trainer.fit(model, train_loader, val_loader)

    # Start tuning
    analysis = tune.run(
        train_model,
        config=search_space,
        resources_per_trial={"cpu": 1, "gpu": 1},  # GPU dla każdego trialu
        num_samples=10,
        metric="val_loss",
        mode="min"
    )

    print("Best hyperparameters found:", analysis.best_config)

if __name__ == "__main__":
    bayesian_optimization()
