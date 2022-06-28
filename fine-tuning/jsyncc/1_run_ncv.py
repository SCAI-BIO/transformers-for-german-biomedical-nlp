# -*- coding: utf-8 -*-
# imports
import logging
import os
import pickle
import sys
from configparser import ConfigParser
from typing import List, Tuple

import click

from toolbox.training.sequence_classification import ScConfig, ScManager
from toolbox.utils.optimization import hyperparameter_space

# globals
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# functions


@click.command()
@click.argument("config-file", type=str)
@click.argument("fold-idx", type=int)
@click.option("--use-mlflow", is_flag=True)
def distributed_ncv(config_file: str, fold_idx: int, use_mlflow: bool):
    """
    Run distributed nested cross validation for JSynCC

    \b
    Args:\b
        config_file: Path to the respective config file\b
        fold_idx: Index of the respective outer fold\b
        use_mlflow: Boolean variable if mlflow should be used\b

    \b
    Usage:\b
        python 1_run_ncv.py data/config/default.ini 0\b
        python 1_run_ncv.py data/config/default.ini 1\b
        python 1_run_ncv.py data/config/default.ini 2\b
        python 1_run_ncv.py data/config/default.ini 3\b
        python 1_run_ncv.py data/config/default.ini 4\b
    """
    config = ConfigParser()
    config.read(config_file)

    model_id_string = config["Model"]["name"]
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    transformers_module = __import__("transformers")
    model_function = getattr(transformers_module, config["Model"]["model_function"])
    tokenizer_function = getattr(
        transformers_module, config["Model"]["tokenizer_function"]
    )

    with open(
        "data/processed/ncv_datasets.pkl",
        "rb",
    ) as f:
        dataset: Tuple[
            "ScDataset", "ScDataset", List[Tuple["ScDataset", "ScDataset"]]
        ] = pickle.load(f)

    sc_config = ScConfig(
        output_directory=os.path.join("models", model_id_string),
        experiment_name=config["Logging"]["experiment_name"] + "_ncv",
        model_name=model_id_string,
        model_function=model_function,
        tokenizer_function=tokenizer_function,
        max_epochs=int(config["Training"]["epochs"]),
        learning_rate=float(config["Training"]["learning_rate"]),
        early_stopping_patience=int(config["Training"]["early_stopping_patience"]),
        early_stopping_threshold=float(config["Training"]["early_stopping_threshold"]),
        train_batch_size=config["Training"].getint("train_batch_size"),
        eval_batch_size=config["Training"].getint("eval_batch_size"),
        eval_metric=config["Training"]["metric"],
        greater_is_better=bool(config["Training"]["greater_is_better"]),
        results_dir=config["Storage"]["results"],
        tokenizer_type=config["Model"]["tokenizer_type"],
    )

    manager = ScManager(
        model_path=config["Model"]["model"],
        tokenizer_path=config["Model"]["tokenizer"],
        data=dataset,  # type: ignore
        config=sc_config,
        hp_search_space=hyperparameter_space,
        use_mlflow=use_mlflow,
    )
    logger.info("Initialized manager")
    manager.run_inner_ncv_loop(
        outer_idx=fold_idx,
        n_trials=30,
        storage=f"sqlite:///data/databases/{model_id_string}-{str(fold_idx)}.db",
        study_name="jsyncc_" + config["Model"]["name"],
        run_name=config["Model"]["name"],
    )


if __name__ == "__main__":
    distributed_ncv()
