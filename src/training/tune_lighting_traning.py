import multiprocessing
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

from data.read_data import read_all_file_df
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
    """
    Create PyTorch DataLoaders for training and validation datasets.

    Args:
        config (dict): Configuration dictionary with parameters.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    def normalize_except_last_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all columns except the last one using Z-score normalization.

        Args:
            df (pd.DataFrame): Input DataFrame to normalize.

        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        scaler = StandardScaler()
        features = df.iloc[:, :-1]  # All columns except the last one
        normalized_features = scaler.fit_transform(features)
        df.iloc[:, :-1] = normalized_features  # Replace with normalized values
        return df

    # Read and normalize data
    df_train = read_all_file_df(
        channels_names=GLOBAL_CHANNEL_NAMES,
        idx_people=[1, 2, 8, 9],
        idx_exp=[3, 7, 11],
        path=DATA_PATH,
    )
    df_train = normalize_except_last_column(df_train)

    df_val = read_all_file_df(
        channels_names=GLOBAL_CHANNEL_NAMES,
        idx_people=[10, 13],
        idx_exp=[3, 7, 11],
        path=DATA_PATH,
    )
    df_val = normalize_except_last_column(df_val)

    # Create datasets and dataloaders
    train_dataset = Dataset(df=df_train, sequence_length=config["seq_length"])
    val_dataset = Dataset(df=df_val, sequence_length=config["seq_length"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    return train_loader, val_loader

def train_model(config: dict) -> None:
    """
    Train the LSTM model using PyTorch Lightning.

    Args:
        config (dict): Configuration dictionary for model and training parameters.
    """
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

    wandb_logger = WandbLogger(project="EEG_Classification_finale", name=run_name)

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        logger=wandb_logger,
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

def optimize_hyperparameters() -> None:
    """
    Perform hyperparameter optimization using Ray Tune with OptunaSearch.
    """
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
        max_t=max_epochs,  # Use max_epochs here to ensure consistency
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

    print("Best hyperparameters found:", analysis.best_config)

if __name__ == "__main__":
    optimize_hyperparameters()
