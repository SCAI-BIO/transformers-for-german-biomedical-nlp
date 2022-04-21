# -*- coding: utf-8 -*-
import logging
import os
import shutil
from collections import Counter
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union
from uuid import uuid1

import mlflow
import numpy as np
import optuna
import optuna.visualization as ov
import pandas as pd
import transformers
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback
from transformers.trainer_utils import HPSearchBackend, IntervalStrategy

from toolbox.datasets.sc_dataset import ScDataset
from toolbox.training.hyperopt import ComputeObjective
from toolbox.utils import CallbackDict, ScMetrics, TrainerUtils
from toolbox.utils.callbacks import EarlyStoppingCallback, LogCallback
from toolbox.utils.helpers import ModelLoader
from .config import ScConfig
from .trainer import ScMiTrainer, ScTrainer

# global vars ------------------------------------------------------------------

logger = logging.getLogger(__name__)
CVDATA = List[Tuple[ScDataset, ScDataset]]
NCVDATA = List[Tuple[ScDataset, ScDataset, CVDATA]]


# functions --------------------------------------------------------------------

# classes ----------------------------------------------------------------------


class ScManager(object):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config: ScConfig,
        data: Union[ScDataset, CVDATA, NCVDATA] = None,
        train_data: ScDataset = None,
        val_data: ScDataset = None,
        hp_search_space: Callable = None,
        scheme: str = "BILOU",
        mode: str = "strict",
        use_mlflow: bool = True,
    ) -> None:
        """[summary]

        Args:
            model_path (str): path to local model or name of huggingface model
            tokenizer_path (str): path to local tokenizer or name of huggingface tokenizer
            config (ScConfig): config instance for SC
            data (ScDataset, optional): dataset for CV/NCV. Defaults to None.
            train_data (ScDataset, optional): training data for fine-tuning. Defaults to None.
            val_data (ScDataset, optional): validation data for evaluation. Defaults to None.
            hp_search_space: (Callable, optional): optional search space
        """
        super().__init__()

        if data is not None and (train_data is not None or val_data is not None):
            raise Exception(
                "Data or training/validation data should be specified, not both."
            )
        elif data is None and train_data is None and val_data is None:
            raise Exception("Data must be provided.")

        self.model = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.data = data
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.hp_search_space = hp_search_space

        self.tokenizer: PreTrainedTokenizerFast = None  # type: ignore
        self.trainer_instance: ScTrainer = None  # type: ignore
        self.scheme = scheme
        self.mode = mode

        transformers.set_seed(config.seed)
        if use_mlflow:
            TrainerUtils.prepare_mlflow_logging(config)
        self.use_mlflow = use_mlflow

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

    def run_inner_ncv_loop(
        self,
        outer_idx: int,
        n_trials: int,
        storage: str,
        study_name: str,
        load_if_exists: bool = True,
        run_name: str = None,
        pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner(),
        sampler: optuna.samplers.BaseSampler = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, PreTrainedModel]:

        assert isinstance(self.data, list), "List of dataset splits must be provided"
        assert isinstance(
            self.data[0][0], ScDataset
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

        logger.info("Start re-tokenization")
        train.retokenize(tokenizer, config.max_length)
        val.retokenize(tokenizer, config.max_length)
        for (inner_train, inner_val) in inner_data:
            inner_train.retokenize(tokenizer, config.max_length)
            inner_val.retokenize(tokenizer, config.max_length)
        logger.info(f"Datasets were re-tokenized.")

        if config.warmup_steps is not None:
            num_warmup = int(config.warmup_steps)
        else:
            num_warmup = int((len(train) / config.train_batch_size) * config.max_epochs)

        metric_func = ScMetrics()
        compute_objective = ComputeObjective(config.eval_metric)
        model_initiator = ModelLoader(model_fct, model_config, self.model_path)
        early_stopping_callback = EarlyStoppingCallback(
            config.early_stopping_patience, config.early_stopping_threshold
        )

        checkpoint_folder = f"split-{outer_idx}"
        out_folder = os.path.join(config.output_directory, checkpoint_folder)
        hf_training_arguments = TrainingArguments(
            output_dir=out_folder,
            do_train=True,
            do_eval=True,
            evaluation_strategy=config.eval_strategy,  # type: ignore
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
                if os.getenv("SLURM_CPUS_PER_TASK", None) is not None
                else 0
            ),
            fp16=False,
            save_strategy=IntervalStrategy.EPOCH,
            use_legacy_prediction_loop=True,
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

        asses_function = max if config.greater_is_better else min

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
            best_epochs = asses_function(int_values, key=int_values.get) + 1
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
                        epochs.append(asses_function(vals, key=vals.get))
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

                trainer_instance = ScTrainer(
                    run_id=i,
                    trainer_function=ScMiTrainer,
                    model=model_initiator,
                    callbacks=[deepcopy(early_stopping_callback)],
                    training_arguments=hf_training_arguments,
                    training_data=inner_train_data,
                    validation_data=inner_val_data,
                    metric_function=metric_func,
                    callable_functions=CallbackDict(
                        [MLflowCallback],
                        [  # type: ignore
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
                    asses_function(intermediate_values, key=intermediate_values.get) + 1
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
            training_arguments: Dict = hf_training_arguments.to_dict()
            training_arguments = {
                key: value
                for key, value in training_arguments.items()
                if key[0] != "_" and value != -1
            }
            training_arguments["save_strategy"] = "no"
            training_arguments["evaluation_strategy"] = "no"
            training_arguments["do_eval"] = False
            training_arguments["num_train_epochs"] = max(best_epochs, 1)

            for parameter, value in best_params.items():
                training_arguments[parameter] = value

            hf_training_arguments = TrainingArguments(**training_arguments)
            logger.info(f"{outer_idx}: {hf_training_arguments}")

            # training
            trainer_instance = ScTrainer(
                run_id=outer_idx,
                trainer_function=ScMiTrainer,
                model=model_initiator(),
                training_arguments=hf_training_arguments,
                training_data=train,
                callable_functions=CallbackDict(
                    [MLflowCallback],
                    [  # type: ignore
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

            optuna_optimization_history = ov.plot_optimization_history(study)
            optuna_optimization_history.write_image(
                os.path.join(out_folder, "history.png")
            )
            optuna_slice_plot = ov.plot_slice(study)
            optuna_slice_plot.write_image(os.path.join(out_folder, "slice.png"))
            optuna_param_importance = ov.plot_param_importances(study)
            optuna_param_importance.write_image(
                os.path.join(out_folder, "param_importance.png")
            )

            all_metrics = np.array(all_metrics)
            np.savetxt(os.path.join(out_folder, "all_metrics.txt"), all_metrics)  # type: ignore
            all_epochs = np.array(all_epochs)
            np.savetxt(os.path.join(out_folder, "all_epochs.txt"), all_epochs)  # type: ignore

            if self.use_mlflow:
                mlflow.log_artifact(
                    local_path=os.path.join(out_folder, "all_metrics.txt")
                )
                mlflow.log_artifact(
                    local_path=os.path.join(out_folder, "all_epochs.txt")
                )

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
        return average_results, entity_results, trainer_instance.model  # type: ignore

    def load_tokenizer(self, tokenizer_fct) -> transformers.PreTrainedTokenizer:
        logger.info("Try to load tokenizer.")
        tokenizer = None
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
