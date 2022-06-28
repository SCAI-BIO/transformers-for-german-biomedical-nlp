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

# globals
from toolbox.utils.optimization import hyperparameter_space

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# functions


@click.command()
@click.argument("config_file")
def fine_tuning(config_file: str):
    """
    Perform fine_tuning with hyperparameter optimization for BRONCO data.

    \b
    Args:\b
        config_file: str = Path to config file\b
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
    ner_config = NerConfig(
        output_directory=os.path.join("models", config["Model"]["name"]),
        experiment_name=config["Logging"]["experiment_name"],
        model_name=config["Model"]["name"],
        model_function=model_function,
        tokenizer_function=tokenizer_function,
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
        trainer_function=OptunaTrainer,
        max_length=512,
    )

    manager = NerManager(
        model_path=config["Model"]["model"],
        tokenizer_path=config["Model"]["tokenizer"],
        train_data=training_data,
        val_data=validation_data,
        config=ner_config,
        hp_search_space=hyperparameter_space,
        use_mlflow=False,
    )
    manager.finetune(
        n_trials=3,
        direction="maximize"
        if config.getboolean("Training", "greater_is_better")
        else "minimize",
        storage=f"sqlite:///data/databases/{ner_config.model_name}.db",
        study_name=ner_config.model_name,
        load_if_exists=True,
    )

    testing_data = torch.load(config["Data"]["testing_data"])
    average_results, entity_results = manager.evaluate(testing_data)
    average_results = pd.DataFrame([average_results])
    average_results.to_csv(ner_config.average_results_file)
    entity_results.to_csv(ner_config.entity_results_file)


if __name__ == "__main__":
    fine_tuning()
