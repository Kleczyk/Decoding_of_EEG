import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.metrics_fn import compute_accuracy, compute_auc
import numpy as np
from src.models.eeg_lstm_fn_cwt import EEGClassifier
from src.data.cwt_dataset import CwtDataset
from src.data.db_contlorer import DbController
from src.data.data_handler import DataHandler
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.air import session


def train_func(config):

    # Initialize wandb
    wandb.init(project=config.get("wandb_project", "EEG_Classification"), config=config)

    db = DbController(
        dbname="my_db", user="user", password="1234", host="localhost", port="5433"
    )

    train_dataset = CwtDataset(
        table="training_data", db_controller=db, sequence_length=config["seq_length"]
    )
    val_dataset = CwtDataset(
        table="validation_data", db_controller=db, sequence_length=config["seq_length"]
    )

    # Użyj nowo utworzonego datasetu
    config["train_dataset"] = train_dataset
    config["val_dataset"] = val_dataset
    model = EEGClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        seq_length=config["seq_length"],
        dropout=config["dropout"],
    )

    # Przeniesienie modelu na odpowiednie urządzenie
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Definicja funkcji kosztu i optymalizatora
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Przygotowanie danych
    train_loader = DataLoader(
        config["train_dataset"], batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        config["val_dataset"], batch_size=config["batch_size"], shuffle=False
    )

    # Training loop with Ray
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        train_accuracy = compute_accuracy(all_labels, all_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_probs = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs, dim=1)
                probs = torch.exp(outputs)  # Convert log-softmax to probabilities
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = compute_accuracy(val_labels, val_preds)
        val_auc = compute_auc(val_labels, np.array(val_probs), config["num_classes"])

        # Log metrics to wandb
        session.report(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_auc": val_auc,
            }
        )

        print(
            f"Epoch [{epoch + 1}/{config['epochs']}], "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}"
        )

    # Zapisywanie modelu
    torch.save(model.state_dict(), "model.pth")
    wandb.save("model.pth")


config = {
    "input_size": 120,
    "hidden_size": 64,
    "num_layers": 2,
    "num_classes": 4,
    "seq_length": 256,
    "dropout": 0.5,
    "learning_rate": 0.001,
    "batch_size": 10,
    "epochs": 10,
    "wandb_project": "EEG_Classification",
}
scaling_config = ScalingConfig(
    num_workers=1, use_gpu=True, resources_per_worker={"GPU": 0.2}
)
trainer = TorchTrainer(
    train_func, scaling_config=scaling_config, train_loop_config=config
)
result = trainer.fit()
