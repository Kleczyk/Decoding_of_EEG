import multiprocessing
from typing import Tuple

import numpy as np
from sympy.combinatorics import Subset

multiprocessing.set_start_method('spawn')

import lightning.pytorch as pl
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import DataLoader, SubsetRandomSampler
from lightning.pytorch.loggers import WandbLogger
import wandb

from data.read_data import read_all_file_df
from data.base_eeg_dataset import BaseEEGDataset
from models.base_lstm_lighting import LSTMBaseLighting
from data import DATA_PATH
from data.utils.all_channels_names import ALL_CHANNEL_NAMES


def split_indices(dataset_length: int, seq_length: int, test_ratio: float = 0.2) -> Tuple[list, list]:
    """
    Splits dataset indices into training and testing subsets, ensuring no sequence overlap in the test set.

    Args:
        dataset_length (int): Total length of the dataset.
        seq_length (int): Length of each sequence.
        test_ratio (float): Proportion of data to be used for the test set.

    Returns:
        Tuple[list, list]: Lists of training and testing indices.
    """
    if dataset_length < seq_length:
        raise ValueError("Dataset length must be greater than the sequence length!")

    # Valid start indices for sequences
    valid_indices = np.arange(dataset_length - seq_length + 1)

    # Calculate number of test samples based on ratio
    num_test_samples = int(len(valid_indices) * test_ratio)

    # Ensure indices are selected within valid range
    test_indices = np.random.choice(valid_indices, num_test_samples, replace=False).tolist()
    train_indices = [idx for idx in valid_indices if idx not in test_indices]

    # Validate ranges
    if any(idx + seq_length > dataset_length for idx in train_indices):
        raise ValueError(
            f"Train index out of range detected! Max valid index: {dataset_length - seq_length}, "
            f"First invalid index: {next(idx for idx in train_indices if idx + seq_length > dataset_length)}"
        )

    if any(idx + seq_length > dataset_length for idx in test_indices):
        raise ValueError(
            f"Test index out of range detected! Max valid index: {dataset_length - seq_length}, "
            f"First invalid index: {next(idx for idx in test_indices if idx + seq_length > dataset_length)}"
        )

    return train_indices, test_indices


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Prepares training and testing DataLoaders for EEG sequential data.

    Args:
        config (dict): Configuration dictionary containing:
            - seq_length (int): Length of each sequence.
            - batch_size (int): Batch size for DataLoaders.
            - debug (bool, optional): Whether to print sample batches for debugging.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing DataLoaders.
    """
    normalized_data = read_all_file_df(
        channels_names=ALL_CHANNEL_NAMES,
        idx_people=[1, 2, 8, 9],
        idx_exp=[3],
        path=DATA_PATH,
        normalize_z_score=True
    )

    dataset = BaseEEGDataset(df=normalized_data, sequence_length=config["seq_length"])





    train_indices, test_indices = split_indices(len(dataset), config["seq_length"])

    # Debugging index ranges
    print("Dataset length:", len(dataset))
    print("Train indices count:", len(train_indices))
    print("Test indices count:", len(test_indices))
    print("Max train index:", max(train_indices) if train_indices else "Empty")
    print("Max test index:", max(test_indices) if test_indices else "Empty")
    print("Min train index:", min(train_indices) if train_indices else "Empty")
    print("Min test index:", min(test_indices) if test_indices else "Empty")

    train_Sampler = SubsetRandomSampler(train_indices)
    test_Sampler= SubsetRandomSampler(test_indices)
    train_loader = DataLoader(
        dataset,
        sampler=train_Sampler,
        batch_size=config['batch_size'],
    )
    test_loader = DataLoader(
        dataset,
        sampler=test_Sampler,
        batch_size=config['batch_size'],
    )
    if config.get("debug", False):
        print("Train Batch Example:")
        for batch in train_loader:
            print(batch)
            break

        print("\nTest Batch Example:")
        for batch in test_loader:
            print(batch)
            break

    return train_loader, test_loader


def train_model(config: dict) -> dict:
    run_name = f"{config['model_name']}_exp-{config['exp_type']}_{wandb.util.generate_id()}"
    wandb.init(project="EEG_Classification_finale", name=run_name, reinit=True)

    train_loader, val_loader = get_dataloaders(config)

    model = LSTMBaseLighting(
        sequence_length=config["seq_length"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        learning_rate=config["lr"],
        num_classes=config["num_classes"],
        num_channels=len(ALL_CHANNEL_NAMES),
    )

    wandb_logger = WandbLogger(project="EEG_Classification_finale", name=run_name)

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(dirpath="best_model", filename="best_model", monitor="val_acc", mode="max")]
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

    return {"model_path": "best_model/best_model.ckpt"}


def optimize_hyperparameters() -> None:
    max_epochs = 100

    search_space = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64, 128]),
        "hidden_size": tune.lograndint(100, 10000),
        "num_layers": tune.randint(1, 4),
        "dropout": tune.uniform(0, 0.4),
        "num_classes": 3,
        "seq_length": tune.randint(8, 800),
        "max_epochs": max_epochs,
        "model_name": tune.choice(["LSTMBase"]),
        "exp_type": tune.choice(["motor_imagery", "rest_state"]),
    }

    optuna_search = OptunaSearch(metric="val_acc", mode="max")
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=max_epochs,
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
        resources_per_trial={"cpu": 0.25, "gpu": 0.12},
    )

    best_config = analysis.best_config
    print("Best hyperparameters found:", best_config)

    # Save best model path
    best_model_path = train_model(best_config)["model_path"]
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    search_space = {
        "lr": 1,
        "batch_size": 3,
        "hidden_size": 3,
        "num_layers": 3,
        "dropout": 3,
        "num_classes": 3,
        "seq_length": 50,
        "max_epochs": 3,
        "model_name": tune.choice(["LSTMBase"]),
        "exp_type": tune.choice(["motor_imagery", "rest_state"]),
    }
    x = get_dataloaders(search_space)

    optimize_hyperparameters()