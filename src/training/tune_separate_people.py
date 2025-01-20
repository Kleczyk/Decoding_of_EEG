import multiprocessing

multiprocessing.set_start_method('spawn')

import lightning.pytorch as pl
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import wandb

from data.read_data import read_all_file_df
from data.base_eeg_dataset import BaseEEGDataset
from models.base_lstm_lighting import LSTMBaseLighting
from data import DATA_PATH



def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:

    df_train = read_all_file_df(
        channels_names=GLOBAL_CHANNEL_NAMES,
        idx_people=[1, 2, 8, 9],
        idx_exp=[3],
        path=DATA_PATH,
        normalize_z_score=True
    )
    df_val = read_all_file_df(
        channels_names=GLOBAL_CHANNEL_NAMES,
        idx_people=[10, 13],
        idx_exp=[3],
        path=DATA_PATH,
        normalize_z_score=True
    )

    train_dataset = BaseEEGDataset(df=df_train, sequence_length=config["seq_length"])
    val_dataset = BaseEEGDataset(df=df_val, sequence_length=config["seq_length"])

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
