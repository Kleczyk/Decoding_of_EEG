import optuna
import lightning.pytorch as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# 2. Funkcja celu dla Optuny

def objective(trial):
    # Hiperparametry do optymalizacji
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.7)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    batch_size = trial.suggest_int("batch_size", 8, 64)
    input_size = trial.suggest_int("input_size", 32, 128)
    seq_length = trial.suggest_int("seq_length", 32, 128)
    num_classes = 3

    # Dane treningowe
    X = torch.rand(1000, seq_length, input_size)
    y = torch.randint(0, num_classes, (1000,))

    dataset = TensorDataset(X, y)
    train_set, val_set = random_split(dataset, [800, 200])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Model
    model = LSTMLitModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        num_classes=num_classes
    )

    # Logger dla Weight and Biases
    wandb_logger = WandbLogger(project="EEG_Classification")

    # Trener PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        enable_progress_bar=True
    )

    # Trening i walidacja
    trainer.fit(model, train_loader, val_loader)

    # Zwrócenie wyniku walidacji
    metrics = trainer.callback_metrics
    return metrics.get("val_loss", float("inf")).item()

# 3. Optymalizacja za pomocą Optuny
if __name__ == "__main__":
    import wandb
    wandb.login()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # 4. Najlepsze hiperparametry
    print("Najlepsze hiperparametry:")
    print(study.best_params)
