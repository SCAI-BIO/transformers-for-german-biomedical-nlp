# -*- coding: utf-8 -*-

# imports
import logging
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
import torch
from optuna import Trial
from toolbox.datasets.ner_dataset import LabelSet
from toolbox.utils.callbacks import MaxTrialCallback
from toolbox.utils.helpers import CallbackDict, TrainerUtils
from toolbox.utils.metrics import NerMetrics
from torch.nn import Module
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import BestRun

# globals

logger = logging.getLogger(__name__)


# classes
class NerTrainer(object):
    def __init__(
        self,
        run_id: Union[int, None],
        trainer_function: Callable,
        model: Union[Module, Callable],
        callbacks: List[TrainerCallback],
        training_arguments: TrainingArguments,
        metric_function: Callable,
        labels_set: LabelSet,
        training_data: Dataset = None,
        validation_data: Dataset = None,
        callable_functions: CallbackDict = None,
        arch: torch.device = torch.device("cpu"),
        hp_search_space: Callable = None,
        compute_objective: Callable[[Dict[str, float]], float] = None,
    ) -> None:
        """

        Args:
            run_id (int): Run id; used for CV and NCV.
            trainer_function (Callable): Trainer function. E.g. huggingface trainer or inherited variant.
            model (Union[Module, Callable]): model instance or function that returns a model (hp-opt)
            callbacks (List[TrainerCallback]): Callbacks for Trainer
            training_arguments (TrainingArguments): arguments for trainer
            training_data (Dataset): training dataset
            validation_data (Dataset): validation dataset
            metric_function (Callable): function to calculate models
            callable_functions (CallbackDict, optional): Removed/added callbacks for Trainer instance. Defaults to None.
            arch (torch.device, optional): cpu or cuda. Defaults to torch.device("cpu").
            hp_search_space (Callable, optional): Hyperparameter search space. Defaults to None.
            compute_objective (Callable[[Dict[str, float]], float], optional): Objective function for hyperparameter
                optimization. Defaults to None.
        """

        super().__init__()

        self.run_id = run_id
        self.model = model
        self.callbacks = callbacks
        self.training_arguments = training_arguments
        self.training_data = training_data
        self.validation_data = validation_data
        self.metric_function = metric_function
        self.label_set = labels_set
        self.arch = arch
        self.hp_search_space = hp_search_space
        self.objective = compute_objective

        self.trainer: Trainer = trainer_function(
            model_init=self.model if hp_search_space is not None else None,
            model=self.model if hp_search_space is None else None,
            args=self.training_arguments,
            train_dataset=self.training_data,
            eval_dataset=self.validation_data,
            compute_metrics=metric_function,
            callbacks=callbacks,
        )

        if callable_functions is not None:
            TrainerUtils.prepare_mlflow_callbacks(self.trainer, callable_functions)
        logger.info("Initialized Trainer instance")

    def train(self, trial: Trial = None, **kwargs):
        """Calls train function of trainer instance"""
        logger.info("Start training")
        self.trainer.train(trial=trial, **kwargs)

    def hyperparameter_search(
        self,
        n_trials: int,
        direction: str,
        storage: str,
        study_name: str,
        load_if_exists: bool = False,
        timeout: int = 100000,
        **kwargs
    ) -> BestRun:
        """Calls hyperparameter_search of trainer instance"""
        return self.trainer.hyperparameter_search(
            hp_space=self.hp_search_space,
            compute_objective=self.objective,
            n_trials=n_trials,
            direction=direction,
            storage=storage,
            study_name=study_name,
            load_if_exists=load_if_exists,
            timeout=timeout,
            callbacks=[MaxTrialCallback(n_trials)],
            **kwargs
        )

    def evaluate(self, dataset: Dataset = None) -> Tuple[Dict, pd.DataFrame]:
        """Evaluation with specified data"""
        logger.info("Start evaluation")
        if dataset is None:
            dataset = self.validation_data
        prediction_output = self.trainer.predict(dataset)
        compute_metrics = NerMetrics(label_set=self.label_set)
        tmp_entity = pd.DataFrame(
            compute_metrics.compute_classification_report(
                prediction_output.label_ids, prediction_output.predictions
            )
        )
        if id is not None:
            run_name = "Run" + str(self.run_id)
            tmp_entity["run"] = [run_name] * tmp_entity.shape[0]

        average_results = self.trainer.evaluate(dataset)
        return average_results, tmp_entity
