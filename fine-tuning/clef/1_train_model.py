import logging
import os
import sys
from configparser import ConfigParser
from datetime import datetime
from typing import Dict

import optuna
import torch
import typer
from optuna.trial import Trial
from toolbox.datasets.sc_dataset import ScDataset
from toolbox.training.sequence_classification import ScConfig, ScManager

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def hyperopt(
    config_file: str,
    stdout=True,
    num_trials: int = 30,
):
    """
    Finetune models with hyperparameter optimization on CLEF dataset

    \b
    Args:\b
        config_file: Path to config file\b
        stdout: Log to stdout
        num_trials: Number of trials to optimize
    \b
    Usage:\b
        python 1_train_model.py data/config/default.ini\b
    """
    config = ConfigParser()
    config.read(config_file)

    model_id_string = config["Model"]["name"]
    if stdout:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    else:
        log_file = f"logs/{model_id_string}_{datetime.now().strftime('%Y%m%d')}.txt"
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            level=logging.DEBUG,
            format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        )

    mod = __import__("transformers")
    mod_func = getattr(mod, config["Model"]["model_function"])
    tok_func = getattr(mod, config["Model"]["tokenizer_function"])

    train_data: ScDataset = torch.load(config["Data"]["training_data"])
    val_data: ScDataset = torch.load(config["Data"]["validation_data"])
    test_data: ScDataset = torch.load(config["Data"]["testing_data"])

    sc_config = ScConfig(
        output_directory=os.path.join(
            "models",
            f"{model_id_string}_full"
            if "full" in config["Data"]["training_data"]
            else f"{model_id_string}_part",
        ),
        experiment_name=config["Logging"]["experiment_name"],
        model_name=model_id_string,
        model_function=mod_func,
        tokenizer_function=tok_func,
        max_epochs=int(config["Training"]["epochs"]),
        learning_rate=float(config["Training"]["learning_rate"]),
        early_stopping_patience=config.getint("Training", "early_stopping_patience"),
        early_stopping_threshold=config.getfloat(
            "Training", "early_stopping_threshold"
        ),
        train_batch_size=config["Training"].getint("train_batch_size"),
        eval_batch_size=config["Training"].getint("eval_batch_size"),
        eval_metric=config["Training"]["metric"],
        greater_is_better=config.getboolean("Training", "greater_is_better"),
        results_dir=config["Storage"]["results"],
        tokenizer_type=config["Model"]["tokenizer_type"],
        ncv=False,
        tracking_uri=config["Logging"]["tracking_uri"],
    )

    def hyperparameter_space(trial: Trial) -> Dict[str, float]:
        hps = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8, 16, 32]
            ),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-10, 1e-1),
        }
        return hps

    manager = ScManager(
        model_path=config["Model"]["model"],
        tokenizer_path=config["Model"]["tokenizer"],
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=sc_config,
        hp_search_space=hyperparameter_space,
    )
    logger.info("Initialized manager")
    manager.hyperopt(
        n_trials=num_trials,
        storage=f"sqlite:///data/databases/{model_id_string}.db",
        study_name=model_id_string,
        run_name=config["Model"]["name"],
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )


if __name__ == "__main__":
    typer.run(hyperopt)
