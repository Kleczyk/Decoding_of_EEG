import multiprocessing



#asdasd
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    import sys
    sys.path.insert(0, '/home/danielkleczykkleczynski/repos/Decoding_of_EEG/src')

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

    from data.base_eeg_dataset import Dataset
    from models.metrics_fn import compute_accuracy, compute_auc
    from models.eeg_lstm_fn_cwt import Eeg_lstm_fn_cwt
    from models.eeg_lstm_fc import Eeg_lstm_fc
    from data.cwt_dataset_polars import CwtDatasetPolars
    from data.dataset_polars import DatasetPolars
    from data.cwt_dataset import CwtDataset
    import data.read_data as rd
    from data.db_contlorer import DbController

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
    def get_dataloaders(batch_size, config):

        df = rd.read_all_file_df(channels_names=global_channels_names)

        train_dataset = Dataset(
            df=df, sequence_length=config["seq_length"]
        )
        val_dataset = Dataset(
            df=df, sequence_length=config["seq_length"]
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        train_dataloader, test_dataloader = get_dataloaders(
            batch_size=batch_size, config=config
        )

        # Prepare data loaders for distributed training
        train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
        test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

        # Initialize model
        model = Eeg_lstm_fc(
            resolution=config["resolution"],
            num_channels=config["num_channels"],
            seq_length=config["seq_length"],
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=config["num_classes"],
            dropout=dropout,
        )

        # Prepare model for distributed training
        model = ray.train.torch.prepare_model(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            if ray.train.get_context().get_world_size() > 1:
                train_dataloader.sampler.set_epoch(epoch)

            model.train()
            for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
                pred = model(X)
                y = y.squeeze()
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
                    y = y.squeeze()
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
                "dropout": tune.uniform(0.0, 0.4),
                "epochs": 10,
                "resolution": 20, #TODO make it dynamic for CWT transform resolution
                "num_channels": len(global_channels_names),
                "seq_length": 256,
                "num_classes": 3,
                "wandb_project": "EEG_Classification",
            },
            "scaling_config": {
                "num_workers": 1,
                "use_gpu": True,
            },
        }
        #new search space
        # Configure the scheduler and search algorithm
        scheduler = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)

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

    main()
