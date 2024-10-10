import os
from typing import Dict

import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

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
from src.models.base_model_eeg import EEGClassifier
from src.data.cwt_dataset import CwtDataset
from src.data.db_contlorer import DbController
from src.data.data_handler import DataHandler
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.air import session


def get_dataloaders(batch_size, config):
    db = DbController(
        dbname="my_db", user="user", password="1234", host="localhost", port="5433"
    )

    train_dataset = CwtDataset(
        table="training_data", db_controller=db, sequence_length=config["seq_length"]
    )
    val_dataset = CwtDataset(
        table="validation_data", db_controller=db, sequence_length=config["seq_length"]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"])

    return train_dataloader, test_dataloader


def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    # Get dataloaders inside the worker training function
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size, config=config)

    # [1] Prepare Dataloader for distributed training
    # Shard the datasets among workers and move batches to the correct device
    # =======================================================================
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    model = EEGClassifier(input_size=config["input_size"],
                          hidden_size=config["hidden_size"],
                          num_layers=config["num_layers"],
                          num_classes=config["num_classes"],
                          seq_length=config["seq_length"],
                          dropout=config["dropout"], )

    # [2] Prepare and wrap your model with DistributedDataParallel
    # Move the model to the correct GPU/CPU device
    # ============================================================
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Model training loop
    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1:
            # Required for the distributed sampler to shuffle properly across epochs.
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            print(y.shape)
            pred = model(X)
            print(pred.shape)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                pred = model(X)
                loss = loss_fn(pred, y)

                test_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()

        test_loss /= len(test_dataloader)
        accuracy = num_correct / num_total

        # [3] Report metrics to Ray Train
        # ===============================
        ray.train.report(metrics={"loss": test_loss, "accuracy": accuracy})


def train_fn(num_workers=2, use_gpu=True):
    global_batch_size = 32


    train_config = {
        "lr": 1e-3,
        "epochs": 10,
        "batch_size_per_worker": global_batch_size // num_workers,
        "input_size": 120, # make more general
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

    # Configure computation resources
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    # [4] Start distributed training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result = trainer.fit()
    print(f"Training result: {result}")


if __name__ == "__main__":
    train_fn(num_workers=1, use_gpu=True)
