import os
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import ray
from ray import tune
from ray.air import session
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

# Import your modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.metrics_fn import compute_accuracy, compute_auc
from src.models.base_model_eeg import EEGClassifier
from src.data.cwt_dataset import CwtDataset
from src.data.db_contlorer import DbController

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
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader

def train_func_per_worker(config: Dict):
    # Cast hyperparameters to appropriate types
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = int(config["batch_size_per_worker"])
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_layers"])
    dropout = config["dropout"]

    # Get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size, config=config)

    # Prepare data loaders for distributed training
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    # Initialize model
    model = EEGClassifier(
        input_size=config["input_size"],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=config["num_classes"],
        seq_length=config["seq_length"],
        dropout=dropout,
    )

    # Prepare model for distributed training
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1:
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
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

        # Report metrics to Ray Tune
        session.report({"loss": test_loss, "accuracy": accuracy})

def main():
    # Define the search space
    search_space = {
        "train_loop_config": {
            "lr": tune.loguniform(1e-5, 1e-1),
            "batch_size_per_worker": tune.loguniform(8, 64),
            "hidden_size": tune.loguniform(32, 128),
            "num_layers": tune.loguniform(1, 3),
            "dropout": tune.uniform(0.1, 0.7),
            "epochs": 10,
            "input_size": 120,
            "num_classes": 3,
            "seq_length": 256,
            "wandb_project": "EEG_Classification",
        },
        "scaling_config": {
            "num_workers": 1,
            "use_gpu": True,
        },
    }

    # Configure the scheduler and search algorithm
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    search_alg = BayesOptSearch()

    # Create a base trainer
    base_trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config={},  # Will be overridden by param_space
        scaling_config=ScalingConfig(),
    )

    tuner = tune.Tuner(
        base_trainer.as_trainable(),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=20,
        ),
    )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

if __name__ == "__main__":
    main()