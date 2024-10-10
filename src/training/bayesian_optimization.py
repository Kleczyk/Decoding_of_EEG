
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.ray_training_tune import train_fn  # Make sure to replace 'your_existing_module' with the module where train_fn is defined

def bayesian_optimization():
    # Define the search space for hyperparameters
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size_per_worker": tune.qloguniform(8, 64, 1),  # Using a continuous representation
        "hidden_size": tune.qloguniform(32, 128, 1),  # Continuous range for hidden sizes
        "num_layers": tune.qloguniform(1, 3, 1),  # Converted to a continuous range
        "dropout": tune.uniform(0.1, 0.7),
    }

    # Configure the scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    # Configure the Bayesian optimization search algorithm
    search_algo = BayesOptSearch(metric="loss", mode="min")

    # Run the optimization process using Ray Tune
    tuner = tune.Tuner(
        tune.with_resources(train_fn, resources={"cpu": 4, "gpu": 1}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=search_algo,
            scheduler=scheduler,
            num_samples=20  # Number of hyperparameter configurations to try
        )
    )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

if __name__ == "__main__":
    bayesian_optimization()
