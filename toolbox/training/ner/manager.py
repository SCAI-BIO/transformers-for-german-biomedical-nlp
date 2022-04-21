# -*- coding: utf-8 -*-
# imports ----------------------------------------------------------------------

import logging
import math
import os
import shutil
from collections import Counter
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union, Any
from uuid import uuid1

import mlflow
import numpy as np
import optuna
import optuna.visualization as ov
import pandas as pd
import transformers
from optuna.pruners import HyperbandPruner
from pandas import DataFrame
from torch.nn import Module
from transformers import AutoConfig, PreTrainedTokenizerFast, TrainingArguments
from transformers.integrations import MLflowCallback
from transformers.trainer_utils import HPSearchBackend

from toolbox.datasets.ner_dataset import LabelSet, NerDataset
from toolbox.training.hyperopt import ComputeObjective
from toolbox.utils.callbacks import EarlyStoppingCallback, LogCallback
from toolbox.utils.helpers import CallbackDict, ModelLoader, TrainerUtils
from toolbox.utils.metrics import NerMetrics
from .config import NerConfig
from .trainer import NerTrainer

# global vars ------------------------------------------------------------------

logger = logging.getLogger(__name__)
CVDATA = List[Tuple[NerDataset, NerDataset]]
NCVDATA = List[Tuple[NerDataset, NerDataset, CVDATA]]


# functions --------------------------------------------------------------------

# classes ----------------------------------------------------------------------


class NerManager(object):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config: NerConfig,
        data: Union[NerDataset, CVDATA, NCVDATA] = None,
        train_data: NerDataset = None,
        val_data: NerDataset = None,
        hp_search_space: Callable = None,
        additional_train_data: NerDataset = None,
        additional_model_config: object = None,
        scheme: str = "BILOU",
        mode: str = "strict",
        use_mlflow: bool = False,
    ) -> None:
        """
        Manager to run different NER training/evaluation methods

        Args:
            model_path (str): path to local model or name of huggingface model
            tokenizer_path (str): path to local tokenizer or name of huggingface tokenizer
            config (NerConfig): config instance for NER
            data (NerDataset, optional): dataset for CV/NCV. Defaults to None.
            train_data (NerDataset, optional): training data for fine-tuning. Defaults to None.
            val_data (NerDataset, optional): validation data for evaluation. Defaults to None.
            hp_search_space: (Callable, optional): hyperparameters for optuna
            additional_train_data (NerDataset, optional): Additional unlabeled data for GanBERT. Defaults to None.
            additional_model_config (object, optional): Additional model configuration for GanBERT. Defaults to None.
            scheme (str): either BILOU or BIO
            mode (str): either strict or default
        """
        super().__init__()

        if data is not None and (train_data is not None or val_data is not None):
            raise Exception(
                "Data or training/validation data should be specified, not both."
            )
        elif data is None and train_data is None and val_data is None:
            raise Exception("Data must be provided.")

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.data = data
        self.train_data = train_data
        self.val_data = val_data
        self.hp_search_space = hp_search_space
        self.additional_training_data = additional_train_data
        self.config = config
        self.additional_model_config = additional_model_config

        self.tokenizer: PreTrainedTokenizerFast = None  # type: ignore
        self.trainer_instance: NerTrainer = None  # type: ignore
        self.scheme = scheme
        self.mode = mode

        transformers.set_seed(config.seed)
        if use_mlflow:
            TrainerUtils.prepare_mlflow_logging(config)
        self.use_mlflow = use_mlflow

    def finetune(
        self,
        n_trials: int = None,
        direction: str = None,
        storage: str = None,
        study_name: str = None,
        load_if_exists: bool = False,
        timeout: int = 10000,
    ):
        """Finetune a model with provided data and save it to mlflow"""
        train_data = self.train_data
        val_data = self.val_data
        config = self.config

        model_config = AutoConfig.from_pretrained(self.model_path)
        model_config.num_labels = train_data.num_labels

        model_fct = config.model_function
        tokenizer_fct = config.tokenizer_function

        logger.info("Try to load tokenizer.")
        if self.tokenizer_path is not None:
            if config.tokenizer_type == "pretrained":
                self.tokenizer = tokenizer_fct.from_pretrained(self.tokenizer_path)
            elif config.tokenizer_type == "file":
                logger.info("Load tokenizer from tokenizer file")
                self.tokenizer = tokenizer_fct(self.tokenizer_path)
            else:
                logger.error("No adequate tokenizer_type specified.")
        else:
            self.tokenizer = tokenizer_fct.from_pretrained(self.model_path)
            logger.info("Load tokenizer from pretrained model.")

        label_set = LabelSet.load(config.label_set)
        train_data.retokenize(self.tokenizer)
        val_data.retokenize(self.tokenizer)

        if config.warmup_steps is not None:
            num_warmup = int(config.warmup_steps)
        else:
            num_warmup = math.ceil(
                config.warmup_ratio
                * (len(train_data) / config.train_batch_size)
                * int(config.early_stopping_patience)
            )

        training_arguments = TrainingArguments(
            output_dir=config.output_directory,
            do_train=True,
            do_eval=True,
            evaluation_strategy=config.eval_strategy,  # type: ignore
            save_strategy=config.eval_strategy,  # type: ignore
            per_device_eval_batch_size=config.eval_batch_size,
            per_device_train_batch_size=config.train_batch_size,
            num_train_epochs=config.max_epochs,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model=config.eval_metric,
            greater_is_better=config.greater_is_better,
            weight_decay=config.weight_decay,
            warmup_steps=num_warmup,
            seed=config.seed,
            dataloader_num_workers=int(
                os.getenv("SLURM_CPUS_PER_TASK")
                if int(os.getenv("SLURM_CPUS_PER_TASK", 0)) > 1
                else 0
            ),
            fp16=False,
        )

        metric_func = NerMetrics(label_set, scheme=self.scheme, mode=self.mode)

        if self.additional_training_data is not None:
            self.additional_training_data.retokenize(self.tokenizer)
            train_data += self.additional_training_data

        if self.hp_search_space is not None:
            model = ModelLoader(
                model_fct, model_config, self.model_path, self.additional_model_config
            )
        else:
            if self.additional_model_config is None:
                model = model_fct.from_pretrained(self.model_path, config=model_config)
            else:
                model = model_fct.from_pretrained(
                    self.model_path, config=(model_config, self.additional_model_config)
                )
        logger.debug(f"Model of type: {type(model)}")

        early_stopping_callback = EarlyStoppingCallback(
            int(config.early_stopping_patience), float(config.early_stopping_threshold)
        )

        if self.hp_search_space is not None:
            self.trainer_instance = NerTrainer(
                run_id=None,
                trainer_function=config.trainer_function,
                model=model,
                callbacks=[deepcopy(early_stopping_callback)],
                training_arguments=training_arguments,
                training_data=train_data,
                validation_data=val_data,
                metric_function=metric_func,
                labels_set=label_set,
                callable_functions=CallbackDict(
                    [MLflowCallback], [LogCallback(False)] if self.use_mlflow else []
                ),
                arch=config.arch,
                hp_search_space=self.hp_search_space,
                compute_objective=ComputeObjective(config.eval_metric),
            )

            best_run = self.trainer_instance.hyperparameter_search(
                n_trials=n_trials,
                direction=direction,
                storage=storage,
                study_name=study_name,
                load_if_exists=load_if_exists,
                timeout=timeout,
                pruner=HyperbandPruner(),
                # sampler=BaseSampler(),
            )
            best_hyperparameters = best_run.hyperparameters
            hg_training_arguments: Dict = training_arguments.to_dict()
            hg_training_arguments = {
                key: value
                for key, value in hg_training_arguments.items()
                if key[0] != "_" and value != -1
            }
            for parameter, value in best_hyperparameters.items():
                hg_training_arguments[parameter] = value

            training_arguments = TrainingArguments(**hg_training_arguments)

        self.trainer_instance = NerTrainer(
            run_id=None,
            trainer_function=config.trainer_function,
            model=model() if isinstance(model, ModelLoader) else model,
            callbacks=[early_stopping_callback],
            training_arguments=training_arguments,
            training_data=train_data,
            validation_data=val_data,
            metric_function=metric_func,
            labels_set=label_set,
            callable_functions=CallbackDict(
                [MLflowCallback], [LogCallback(True)] if self.use_mlflow else []
            ),
            arch=config.arch,
        )
        self.trainer_instance.train()
        shutil.rmtree(config.output_directory)
        logger.info("Fine-tuning finished.")

    def evaluate(self, eval_dataset: NerDataset):
        """Evaluate the associated model with the given dataset"""
        eval_dataset.retokenize(self.tokenizer)  # type: ignore
        average_results, entity_results = self.trainer_instance.evaluate(eval_dataset)
        return average_results, entity_results

    @staticmethod
    def set_params(input_trial: optuna.Trial, params: Dict, dist):

        storage = input_trial.storage
        trial_id = input_trial._trial_id

        for name, param_value in params.items():
            distribution = dist[name]
            param_value_in_internal_repr = distribution.to_internal_repr(param_value)
            storage.set_trial_param(
                trial_id, name, param_value_in_internal_repr, distribution
            )

    def run_inner_cv_loop(
        self,
        outer_idx: int,
        n_trials: int,
        storage: str,
        study_name: str,
        load_if_exists: bool = True,
        run_name: str = None,
        pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner(),
        sampler: optuna.samplers.BaseSampler = None,
    ) -> tuple[DataFrame, DataFrame, Module | Callable[..., Any]]:

        assert isinstance(self.data, list), "List of dataset splits must be provided"
        assert isinstance(
            self.data[0][0], NerDataset
        ), "Dataset format does not seem to be correct"
        assert isinstance(
            self.data[0][2], list
        ), "Dataset format does not seem to be correct"
        assert (
            len(self.data) == self.config.outer_folds
        ), "Number of folds must match supplied splits"

        train, val, inner_data = self.data[outer_idx]
        assert (
            len(inner_data) == self.config.inner_folds
        ), "Number of folds must match supplied splits"
        config = self.config

        model_config = AutoConfig.from_pretrained(self.model_path)
        model_config.num_labels = train.num_labels

        model_fct = config.model_function
        tokenizer_fct = config.tokenizer_function
        tokenizer = self.load_tokenizer(tokenizer_fct)
        label_set = train.labels_set

        logger.info("Start re-tokenization")
        train.retokenize(tokenizer, split=False)
        val.retokenize(tokenizer, split=False)
        for (inner_train, inner_val) in inner_data:
            inner_train.retokenize(tokenizer, split=False)
            inner_val.retokenize(tokenizer, split=False)
        logger.info(f"Datasets were re-tokenized.")

        if config.warmup_steps is not None:
            num_warmup = int(config.warmup_steps)
        else:
            num_warmup = int((len(train) / config.train_batch_size) * config.max_epochs)

        metric_func = NerMetrics(label_set, scheme=self.scheme, mode=self.mode)
        compute_objective = ComputeObjective(config.eval_metric)
        model_initiator = ModelLoader(
            model_fct, model_config, self.model_path, self.additional_model_config
        )
        early_stopping_callback = EarlyStoppingCallback(
            config.early_stopping_patience, config.early_stopping_threshold
        )

        checkpoint_folder = f"split-{outer_idx}"
        out_folder = os.path.join(config.output_directory, checkpoint_folder)
        training_arguments = TrainingArguments(
            output_dir=out_folder,
            do_train=True,
            do_eval=True,
            evaluation_strategy=config.eval_strategy,  # type: ignore
            save_strategy=config.eval_strategy,  # type: ignore
            per_device_eval_batch_size=config.eval_batch_size,
            per_device_train_batch_size=config.train_batch_size,
            num_train_epochs=config.max_epochs,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model=config.eval_metric,
            greater_is_better=config.greater_is_better,
            weight_decay=config.weight_decay,
            warmup_steps=num_warmup,
            seed=config.seed,
            dataloader_num_workers=int(
                os.getenv("SLURM_CPUS_PER_TASK")
                if os.getenv("SLURM_CPUS_PER_TASK") is not None
                else 0
            ),
            fp16=False,
        )
        logger.info(f"Start training round {outer_idx}")

        direction = "maximize" if config.greater_is_better else "minimize"
        study = optuna.create_study(
            direction=direction,
            storage=storage,
            load_if_exists=load_if_exists,
            pruner=pruner,
            sampler=sampler,
            study_name=study_name,
        )

        assess_function = max if config.greater_is_better else min

        if self.use_mlflow:
            o = mlflow.start_run(run_name=f"{run_name}-{outer_idx}")

        # trial config
        try:
            trial_history = {
                k: v
                for k, v in Counter(
                    [t.user_attrs.get("trial_idx", None) for t in study.trials]
                ).items()
                if v >= 5
            }
            done_trials = len(trial_history)
        except:
            done_trials = 0

        try:

            best_params = study.best_params
            best_metrics = study.best_value

            int_values = study.best_trial.intermediate_values
            best_epochs = assess_function(int_values, key=int_values.get) + 1
            del int_values
            all_metrics = []
            all_epochs = []
            all_params = []

            trial_dict = dict()
            for trial in study.trials:
                trial_index = trial.user_attrs.get("trial_idx", None)
                if trial_index is not None and trial_index not in trial_dict.keys():
                    trial_dict[trial_index] = [trial]
                elif trial_index is not None:
                    trial_dict[trial_index].append(trial)

            for trial_index, tlist in trial_dict.items():
                if len(tlist) == config.inner_folds:
                    metrics = []
                    epochs = []
                    trial_params = None
                    for inner_trial in tlist:
                        if trial_params is None:
                            trial_params = inner_trial.params

                        assert inner_trial.params == trial_params
                        vals = inner_trial.intermediate_values
                        epochs.append(assess_function(vals, key=vals.get))
                        metrics.append(max(vals.values()))
                    all_metrics.append(np.mean(metrics))
                    all_epochs.append(int(np.mean(epochs)))
                    all_params.append(trial_params)
        except:
            all_metrics = []
            all_epochs = []
            all_params = []
            best_params = None
            best_metrics = -1 if config.greater_is_better else float("Inf")
            best_epochs = None

        # hyperparameter optimization for n trials over all inner folds
        while done_trials < n_trials:
            trials = [study.ask() for _ in range(len(inner_data))]

            fold_params = None
            fold_dist = None
            fold_metrics = []
            fold_epochs = []

            for i, ((inner_train_data, inner_val_data), current_trial) in enumerate(
                zip(inner_data, trials)
            ):
                exp_name = f"{study_name}: Trial {outer_idx}-{i}-{done_trials}-{uuid1().hex[:6]}"
                if fold_params is None:
                    fold_params = self.hp_search_space(current_trial)
                    fold_dist = current_trial.distributions
                else:
                    self.set_params(current_trial, fold_params, fold_dist)

                current_trial.set_user_attr("inner_idx", i)
                current_trial.set_user_attr("trial_idx", done_trials)
                current_trial.set_user_attr("outer_ids", outer_idx)

                trainer_instance = NerTrainer(
                    run_id=i,
                    trainer_function=config.trainer_function,
                    model=model_initiator,
                    callbacks=[deepcopy(early_stopping_callback)],
                    training_arguments=training_arguments,
                    training_data=inner_train_data,
                    validation_data=inner_val_data,
                    metric_function=metric_func,
                    labels_set=label_set,
                    callable_functions=CallbackDict(
                        [MLflowCallback],
                        [
                            LogCallback(
                                save_model=False,
                                parent_run=o,
                                run_name=exp_name,
                            )
                        ]
                        if self.use_mlflow
                        else [],
                    ),
                    arch=config.arch,
                    compute_objective=compute_objective,
                    hp_search_space=lambda x: x,
                )
                trainer_instance.trainer.hp_search_backend = HPSearchBackend.OPTUNA
                trainer_instance.trainer.hp_space = lambda x: x.params
                trainer_instance.trainer.compute_objective = compute_objective

                try:
                    trainer_instance.train(current_trial)
                except optuna.TrialPruned:
                    logger.info(f"Trial was pruned (State: {current_trial}")

                metrics = trainer_instance.evaluate(inner_val_data)
                trainer_instance.objective = compute_objective(metrics[0])

                intermediate_values = current_trial.storage.get_trial(
                    current_trial._trial_id
                ).intermediate_values
                best_epoch = (
                    assess_function(intermediate_values, key=intermediate_values.get)
                    + 1
                )
                fold_epochs.append(best_epoch)
                current_trial.set_user_attr("best_epoch", best_epoch)
                fold_metrics.append(trainer_instance.objective)

                shutil.rmtree(out_folder)
                logger.info(
                    f"hyperparameters of best run: {fold_metrics[-1]} at {outer_idx}-{i}"
                )

            for i, (trial, metric) in enumerate(zip(trials, fold_metrics)):
                if i == 0:
                    study.tell(trial, metric)
                else:
                    study.tell(trial, metric)

            avg_metrics = np.mean(fold_metrics)
            avg_epoch = int(np.mean(fold_epochs))

            all_metrics.append(avg_metrics)
            all_epochs.append(avg_epoch)
            all_params.append(trial.params)

            if (config.greater_is_better and avg_metrics > best_metrics) or (
                not config.greater_is_better and avg_metrics < best_metrics
            ):
                best_metrics = avg_metrics
                best_epochs = avg_epoch
                best_params = fold_params

            done_trials += 1

        # training of final model on whole data
        # preparation of training parameters
        hf_training_args: Dict = training_arguments.to_dict()
        hf_training_args = {
            key: value
            for key, value in hf_training_args.items()
            if key[0] != "_" and value != -1
        }
        hf_training_args["save_strategy"] = "no"
        hf_training_args["evaluation_strategy"] = "no"
        hf_training_args["do_eval"] = False
        hf_training_args["num_train_epochs"] = max(best_epochs, 1)

        for parameter, value in best_params.items():
            hf_training_args[parameter] = value

        training_arguments = TrainingArguments(**hf_training_args)
        logger.info(f"{outer_idx}: {training_arguments}")

        # training
        trainer_instance = NerTrainer(
            run_id=outer_idx,
            trainer_function=config.trainer_function,
            model=model_initiator(),
            training_arguments=training_arguments,
            training_data=train,
            labels_set=label_set,
            callable_functions=CallbackDict(
                [MLflowCallback],
                [
                    LogCallback(
                        save_model=True,
                        parent_run=o,
                        run_name=f"Trial {str(outer_idx)}-final",
                    )
                ]
                if self.use_mlflow
                else [],
            ),
            callbacks=None,  # type: ignore
            metric_function=None,  # type: ignore
            arch=config.arch,
        )

        trainer_instance.train()

        # evaluation
        (
            average_results,
            entity_results,
        ) = trainer_instance.evaluate(val)

        # storage of avg results
        average_results = pd.DataFrame(average_results, index=[0])
        average_results.to_csv(config.average_results_file + f".split{outer_idx}")
        entity_results.to_csv(config.entity_results_file + f".split{outer_idx}")

        optuna_history_plot = ov.plot_optimization_history(study)
        optuna_history_plot.write_image(os.path.join(out_folder, "history.png"))
        optuna_slice_plot = ov.plot_slice(study)
        optuna_slice_plot.write_image(os.path.join(out_folder, "slice.png"))
        optuna_importance_plot = ov.plot_param_importances(study)
        optuna_importance_plot.write_image(
            os.path.join(out_folder, "param_importance.png")
        )

        all_metrics = np.array(all_metrics)
        np.savetxt(os.path.join(out_folder, "all_metrics.txt"), all_metrics)  # type: ignore
        all_epochs = np.array(all_epochs)
        np.savetxt(os.path.join(out_folder, "all_epochs.txt"), all_epochs)  # type: ignore

        if self.use_mlflow:
            mlflow.log_artifact(local_path=os.path.join(out_folder, "all_metrics.txt"))
            mlflow.log_artifact(local_path=os.path.join(out_folder, "all_epochs.txt"))

            # log to mlflow
            mlflow.log_artifact(
                local_path=config.average_results_file + f".split{outer_idx}"
            )
            mlflow.log_artifact(
                local_path=config.entity_results_file + f".split{outer_idx}"
            )
            mlflow.log_artifact(local_path=os.path.join(out_folder, "history.png"))
            mlflow.log_artifact(local_path=os.path.join(out_folder, "slice.png"))
        try:
            mlflow.end_run()
        except:
            pass
        shutil.rmtree(out_folder)
        return average_results, entity_results, trainer_instance.model

    def load_tokenizer(self, tokenizer_fct):
        logger.info("Try to load tokenizer.")
        if self.tokenizer_path is not None:
            if self.config.tokenizer_type == "pretrained":
                tokenizer = tokenizer_fct.from_pretrained(self.tokenizer_path)
            elif self.config.tokenizer_type == "file":
                logger.info("Load tokenizer from tokenizer file")
                tokenizer = tokenizer_fct(self.tokenizer_path)
            else:
                logger.error("No adequate tokenizer_type specified.")
        else:
            tokenizer = tokenizer_fct.from_pretrained(self.model_path)
            logger.info("Load tokenizer from pretrained model.")
        return tokenizer
