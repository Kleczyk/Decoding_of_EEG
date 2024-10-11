from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.ray_training_tune import train_fn  # Ensure 'train_fn' is correctly imported

def bayesian_optimization():
    # Define the search space for hyperparameters
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
    bayesian_optimization()
