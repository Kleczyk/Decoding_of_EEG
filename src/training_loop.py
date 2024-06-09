from models.CWT_EEG_CrossPersonValidation import CWT_EEG_CrossPersonValidation

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ray import tune
import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import Trainer
import psycopg2
from pytorch_lightning.loggers import WandbLogger
import sqlalchemy as create_engine
frfom ray.tune.search.bayesopt import BayesOptSearch


def train_cwt_eeg(config):
    wandb_logger = WandbLogger(project="EEG",)
    wandb_logger.experiment.config.update(config)

    DATABASE_URL_TRAIN = "postgresql+psycopg2://user:1234@0.0.0.0:5433/dbtrain"
    DATABASE_URL_VAL = "postgresql+psycopg2://user:1234@0.0.0.0:5434/dbval"

    engine_train = create_engine(DATABASE_URL_TRAIN, echo=True, pool_size=10, max_overflow=20)
    engine_val = create_engine(DATABASE_URL_VAL, echo=True, pool_size=10, max_overflow=20)



    # conn_train = psycopg2.connect(database="dbtrain", host="0.0.0.0", user="user", password="1234", port="5433")
    # conn_val = psycopg2.connect(database="dbval", host="0.0.0.0", user="user", password="1234", port="5434")
    model = CWT_EEG_CrossPersonValidation(
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        lr=config['lr'],
        label_smoothing=config.get('label_smoothing', 0),
        engine_train=engine_train,
        engine_val=engine_val
        # conn_train=conn_train,
        # conn_val=conn_val
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )


    trainer = Trainer(
        max_epochs=10,
        logger=wandb_logger,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, early_stop_callback,
                   TuneReportCallback({"loss": "ptl/val_loss"}, on="validation_end")],
    )
    trainer.fit(model)


if __name__ == "__main__":
    ray.init(num_cpus=1, num_gpus=1)  # Adjust based on your system's resources

    search_space = {
        'batch_size': tune.choice([8, 16, 32, 64]),
        'sequence_length': tune.choice([10, 100, 200]),
        'input_size': 640,  # Fixed for our dataset
        'hidden_size': tune.choice([256, 512, 1024]),
        'num_layers': tune.choice([1, 2, 3]),
        'lr': tune.loguniform(1e-5, 1e-1),
        'label_smoothing': 0,
    }
    algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
    # algo = ConcurrencyLimiter(algo, max_concurrent=4)

    analysis = tune.run(
        train_cwt_eeg,
        search_alg=algo,
        config=search_space,
        num_samples=10,
        resources_per_trial={"cpu": 1, "gpu": 1}
    )

    print("Best hyperparameters found were: ", analysis.best_config)
