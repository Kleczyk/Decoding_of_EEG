import multiprocessing

from ray.tune.integration.pytorch_lightning import TuneReportCallback

multiprocessing.set_start_method('spawn')

import lightning.pytorch as pl
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import wandb
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

from data.read_data import read_all_file_df
from data.test_dataset import TestEEGDataset, get_test_dataloaders_via_dataset
from data.dataset import Dataset
from models.base_lstm_lighting import LSTMBaseLighting
from data import DATA_PATH

# Define global channel names
GLOBAL_CHANNEL_NAMES = [
    "Fc5.", "Fc3.", "Fc1.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "Ft7.", "Ft8.", "T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.",
    "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..",
    "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz..",
]


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    def normalize_except_last_column(df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        features = df.iloc[:, :-1]
        normalized_features = scaler.fit_transform(features)
        df.iloc[:, :-1] = normalized_features
        return df

    df_train = read_all_file_df(
        channels_names=GLOBAL_CHANNEL_NAMES,
        idx_people=[1, 2, 8, 9],
        idx_exp=[3],
        path=DATA_PATH,
    )
    df_train = normalize_except_last_column(df_train)

    df_val = read_all_file_df(
        channels_names=GLOBAL_CHANNEL_NAMES,
        idx_people=[10, 13],
        idx_exp=[3],
        path=DATA_PATH,
    )
    df_val = normalize_except_last_column(df_val)

    train_dataset = Dataset(df=df_train, sequence_length=config["seq_length"])
    val_dataset = Dataset(df=df_val, sequence_length=config["seq_length"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    return train_loader, val_loader


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
        num_channels=len(GLOBAL_CHANNEL_NAMES),
    )

    wandb_logger = WandbLogger(project="EEG_Classification_test", name=run_name)

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[
            TuneReportCallback(),  # Report `val_auc` to Tune
            pl.callbacks.ModelCheckpoint(dirpath="best_model", filename="best_model", monitor="val_auc", mode="max"),
        ], )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

    return {"model_path": "best_model/best_model.ckpt"}


def optimize_hyperparameters() -> None:
    max_epochs = 50

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

    optuna_search = OptunaSearch(metric="val_auc", mode="max")
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
        metric="val_auc",
        mode="max",
        resources_per_trial={"cpu": 0.25, "gpu": 0.12},
    )

    best_config = analysis.best_config
    print("Best hyperparameters found:", best_config)

    # Save best model path
    best_model_path = train_model(best_config)["model_path"]
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    optimize_hyperparameters()
