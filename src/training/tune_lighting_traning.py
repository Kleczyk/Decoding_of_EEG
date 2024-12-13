import os
from pathlib import Path

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
import torch
from ray.tune.schedulers import ASHAScheduler
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

import data.read_data as rd
from data.dataset import Dataset
from models.base_lstm_lighting import LSTM_base_lighting
from data import DATA_PATH

global_channels_names = [
    "Fc5.",
    "Fc3.",
    "Fc1.",
    "Af3.",
    "Afz.",
    "Af4.",
    "Af8.",
    "F7..",
    "F5..",
    "F3..",
    "F1..",
    "Fz..",
    "F2..",
    "F4..",
    "F6..",
    "F8..",
    "Ft7.",
    "Ft8.",
    "T7..",
    "T8..",
    "T9..",
    "T10.",
    "Tp7.",
    "Tp8.",
    "P7..",
    "P5..",
    "P3..",
    "P1..",
    "Pz..",
    "P2..",
    "P4..",
    "P6..",
    "P8..",
    "Po7.",
    "Po3.",
    "Poz.",
    "Po4.",
    "Po8.",
    "O1..",
    "Oz..",
    "O2..",
    "Iz..",
]


def get_dataloaders(config: dict) -> (DataLoader, DataLoader):
    df_train = rd.read_all_file_df(channels_names=global_channels_names, idx_people=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   idx_exp=range(3, 13), path=DATA_PATH)
    df_val = rd.read_all_file_df(channels_names=global_channels_names, idx_people=[10, 11, 12, 13],
                                 idx_exp=range(3, 13), path=DATA_PATH)

    train_dataset = Dataset(
        df=df_train, sequence_length=config["seq_length"]
    )
    val_dataset = Dataset(
        df=df_val, sequence_length=config["seq_length"]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"])

    return train_dataloader, test_dataloader


def bayesian_optimization():
    from ray import tune
    from ray.tune.integration.pytorch_lightning import TuneReportCallback

    # Define the search space for hyperparameters

    search_space = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64, 128, 256, 512]),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.uniform(1, 100),
        "dropout": tune.uniform(0, 0.4),
        "input_size": tune.qrandint(8, 1600),
        "num_classes": 3,
        "seq_length": tune.qrandint(8, 1600),
    }

    def train_model(config):
        train_set, val_set = get_dataloaders(config)
        train_loader = DataLoader(train_set, batch_size=config["batch_size"])
        val_loader = DataLoader(val_set, batch_size=config["batch_size"])

        # Model
        model = LSTM_base_lighting(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            learning_rate=config["lr"],
            num_classes=config["num_classes"]
        )

        # Logger dla Weight and Biases
        wandb_logger = WandbLogger(project="EEG_Classification_final", config=config)

        # Trener PyTorch Lightning
        trainer = pl.Trainer(
            storage_path=f"{DATA_PATH}/raw",
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

    scheduler = ASHAScheduler(

        time_attr="training_iteration",  # Atrybut czasowy, np. liczba epok
        max_t=100,  # Maksymalna liczba iteracji
        grace_period=10,  # Minimalna liczba iteracji przed pruningiem
        reduction_factor=3,  # Co 3-krotne zmniejszenie liczby prób
    )
    analysis = tune.run(
        train_model,
        config=search_space,
        scheduler=scheduler,
        num_samples=100,
        metric="val_acc",
        mode="max"
    )

    print("Best hyperparameters found:", analysis.best_config)


if __name__ == "__main__":
    bayesian_optimization()
