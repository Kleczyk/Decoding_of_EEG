import os
from pathlib import Path

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import wandb

import data.read_data as rd
from data.dataset import Dataset
from models.base_lstm_lighting import LSTM_base_lighting
from data import DATA_PATH

global_channels_names = [
    "Fc5.", "Fc3.", "Fc1.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "Ft7.", "Ft8.", "T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.",
    "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..",
    "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz..",
]


def get_dataloaders(config: dict) -> (DataLoader, DataLoader):
    df_train = rd.read_all_file_df(channels_names=global_channels_names, idx_people=[1, 2, 8, 9],
                                   idx_exp=[3, 7, 11], path=DATA_PATH)
    df_val = rd.read_all_file_df(channels_names=global_channels_names, idx_people=[10, 13],
                                 idx_exp=[3, 7, 11], path=DATA_PATH)

    train_dataset = Dataset(
        df=df_train, sequence_length=config["seq_length"]
    )
    val_dataset = Dataset(
        df=df_val, sequence_length=config["seq_length"]
    )
    print(f"Train dataset size: {train_dataset[0][0].shape}")
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"])

    return train_dataloader, val_dataloader


def train_model(config):
    wandb.init(project="EEG_Classification_finale", reinit=True)

    train_loader, val_loader = get_dataloaders(config)

   
    model = LSTM_base_lighting(
        sequence_length=config["seq_length"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        learning_rate=config["lr"],
        num_classes=config["num_classes"],
        num_channels=len(global_channels_names)
    )

    
    wandb_logger = WandbLogger(project="EEG_Classification_finale")

    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[]
    )

    trainer.fit(model, train_loader, val_loader)

    wandb.finish()


def bayesian_optimization():
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64, 128]),
        "hidden_size": tune.lograndint(100, 10000),
        "num_layers": tune.randint(1, 4),
        "dropout": tune.uniform(0, 0.4),
        # "input_size": tune.randint(8, 1600),
        "num_classes": 3,
        "seq_length": tune.randint(8, 800),
    }
    optuna_search = OptunaSearch(
        metric="val_acc",  
        mode="max"  
    )
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=10,
        grace_period=2,
        reduction_factor=2,
    )

    analysis = tune.run(
        train_model,
        config=search_space,
        scheduler=scheduler,
        search_alg=optuna_search,
        num_samples=100,
        metric="val_acc",
        mode="max",
        resources_per_trial={"cpu": 1, "gpu": 0.25},
    )

    print("Best hyperparameters found:", analysis.best_config)


if __name__ == "__main__":
    wandb.login()
    bayesian_optimization()
