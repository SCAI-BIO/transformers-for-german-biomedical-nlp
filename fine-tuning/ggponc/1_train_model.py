# -*- coding: utf-8 -*-

# imports
import logging
import os
import sys
from configparser import ConfigParser

import click
import pandas as pd
import torch

from toolbox.training import OptunaTrainer
from toolbox.training.ner import NerConfig, NerManager
from toolbox.utils.optimization import hyperparameter_space

# globals

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# main
@click.command()
@click.argument("config-file", type=str)
def finetune(config_file: str):
    """
    Finetune models with hyperparameter optimization on GGPONC dataset

    \b
    Args:\b
        config_file: Path to config file\b
    \b
    Usage:\b
        python 1_train_model.py data/config/default.ini\b
    """
    config = ConfigParser()
    config.read(config_file)

    transformer_module = __import__("transformers")
    model_function = getattr(transformer_module, config["Model"]["model_function"])
    tokenizer_function = getattr(
        transformer_module, config["Model"]["tokenizer_function"]
    )

    training_data = torch.load(config["Data"]["training_data"])
    validation_data = torch.load(config["Data"]["validation_data"])

    model_id_string = config["Model"]["name"]
    ner_config = NerConfig(
        output_directory=os.path.join("models", model_id_string),
        experiment_name=config["Logging"]["experiment_name"],
        model_name=model_id_string,
        model_function=model_function,
        tokenizer_function=tokenizer_function,
        trainer_function=OptunaTrainer,
        max_epochs=config.getint("Training", "epochs"),
        label_set=config["Data"]["label_set"],
        learning_rate=config.getfloat("Training", "learning_rate"),
        early_stopping_patience=config.getint("Training", "early_stopping_patience"),
        early_stopping_threshold=config.getfloat(
            "Training", "early_stopping_threshold"
        ),
        eval_batch_size=config.getint("Training", "eval_batch_size"),
        train_batch_size=config.getint("Training", "train_batch_size"),
        warmup_steps=config.getint("Training", "warmup_steps"),
        eval_metric=config["Training"]["metric"],
        greater_is_better=config.getboolean("Training", "greater_is_better"),
        results_dir=config["Storage"]["results"],
        tokenizer_type=config["Model"]["tokenizer_type"],
    )

    manager = NerManager(
        model_path=config["Model"]["model"],
        tokenizer_path=config["Model"]["tokenizer"],
        train_data=training_data,
        val_data=validation_data,
        config=ner_config,
        hp_search_space=hyperparameter_space,
    )
    manager.finetune(
        n_trials=3,
        direction="maximize"
        if config.getboolean("Training", "greater_is_better")
        else "minimize",
        storage=f"sqlite:///data/databases/{model_id_string}.db",
        study_name=model_id_string,
        load_if_exists=True,
    )

    testing_data = torch.load(config["Data"]["testing_data"])
    average_results, entity_results = manager.evaluate(testing_data)
    average_results = pd.DataFrame([average_results])
    average_results.to_csv(ner_config.average_results_file)
    entity_results.to_csv(ner_config.entity_results_file)


if __name__ == "__main__":
    finetune()
