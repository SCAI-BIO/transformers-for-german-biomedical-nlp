# -*- coding: utf-8 -*-
# imports ----------------------------------------------------------------------

import logging
import os
import pickle
import sys
from configparser import ConfigParser
from datetime import datetime
from typing import List, Tuple

import click
from transformers import set_seed

from helpers import SEED
from toolbox.training import OptunaTrainer
from toolbox.training.ner import NerConfig, NerManager

# global vars ------------------------------------------------------------------
from toolbox.utils.optimization import hyperparameter_space

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# functions --------------------------------------------------------------------


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_file")
@click.argument("fold_idx", type=int)
@click.option("--stdout", is_flag=True, default=False)
def distributed_ncv(config_file: str, fold_idx: int, stdout):
    """
    Run nested cross-validation with predefined splits

    \b
    Args:\b
        config_file: str = Path to config file\b
        fold_idx: int = Index of outer fold\b
        stdout: bool = Print to stdout\b
    """
    set_seed(SEED)
    config = ConfigParser()
    config.read(config_file)

    model_id_string = config["Model"]["name"]
    if stdout:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    else:
        log_file = f"logs/{model_id_string}_{str(fold_idx)}_{datetime.now().strftime('%Y%m%d')}.txt"
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            level=logging.DEBUG,
            format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        )

    transformer_module = __import__("transformers")
    model_function = getattr(transformer_module, config["Model"]["model_function"])
    tokenizer_function = getattr(
        transformer_module, config["Model"]["tokenizer_function"]
    )

    with open(
        config["Data"]["whole_data"].replace("whole_dataset.pt", "ncv_dataset.pkl"),
        "rb",
    ) as f:
        dataset: Tuple[
            "NerDataset", "NerDataset", List[Tuple["NerDataset", "NerDataset"]]
        ] = pickle.load(f)

    ner_config = NerConfig(
        output_directory=os.path.join("models", config["Model"]["name"]),
        experiment_name=config["Logging"]["experiment_name"] + "_ncv",
        model_name=config["Model"]["name"],
        model_function=model_function,
        tokenizer_function=tokenizer_function,
        trainer_function=OptunaTrainer,
        max_epochs=int(config["Training"]["epochs"]),
        label_set=config["Data"]["label_set"],
        outer_folds=5,
        inner_folds=5,
        learning_rate=float(config["Training"]["learning_rate"]),
        early_stopping_patience=int(config["Training"]["early_stopping_patience"]),
        early_stopping_threshold=float(config["Training"]["early_stopping_threshold"]),
        train_batch_size=int(config["Training"]["train_batch_size"]),
        eval_batch_size=int(config["Training"]["eval_batch_size"]),
        eval_metric=config["Training"]["metric"],
        greater_is_better=bool(config["Training"]["greater_is_better"]),
        results_dir=config["Storage"]["results"],
        tokenizer_type=config["Model"]["tokenizer_type"],
        warmup_steps=int(config["Training"]["warmup_steps"]),
    )

    manager = NerManager(
        model_path=config["Model"]["model"],
        tokenizer_path=config["Model"]["tokenizer"],
        data=dataset,  # type: ignore
        config=ner_config,
        hp_search_space=hyperparameter_space,
    )
    manager.run_inner_cv_loop(
        outer_idx=fold_idx,
        n_trials=2,
        storage=f"sqlite:///data/databases/{model_id_string}_ner.db",
        study_name="chadl_" + config["Model"]["name"] + f"_{fold_idx}_ncv",
        run_name=config["Model"]["name"],
    )


if __name__ == "__main__":
    cli(obj={})
