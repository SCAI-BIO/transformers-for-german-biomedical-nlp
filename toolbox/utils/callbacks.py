# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime
from typing import Optional

import mlflow
import numpy as np
import optuna
from transformers.integrations import MLflowCallback
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvaluationStrategy

logger = logging.getLogger(__name__)


class LogCallback(MLflowCallback):
    def __init__(
        self,
        save_model: bool = False,
        parent_run: mlflow.ActiveRun = None,
        run_name: str = "Run",
        add_uuid: bool = True,
    ) -> None:
        super().__init__()
        self.model = None
        self.save_model = save_model
        self.run_name = run_name
        self.nested = True if parent_run is not None else False
        self.parent_run = parent_run
        self.add_uuid = add_uuid

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            if self._log_artifacts:
                logger.info("Logging artifacts. This may take time.")
                mlflow.log_artifacts(args.output_dir)
            if self.save_model:
                logger.info("Log model to mlflow server.")
                mlflow.pytorch.log_model(self.model, "models")
            self._initialized = False
            mlflow.end_run()
        else:
            try:
                mlflow.end_run()
            except:
                logger.warning("Unsuccessfully tried to end mlflow run.")

    def setup(self, args, state, model):
        """
        Set up the optional MLflow integration.

        Environment:
            HF_MLFLOW_LOG_ARTIFACTS (:obj:`str`, `optional`):
                Whether to use MLflow .log_artifact() facility to log artifacts.

                This only makes sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or `1`, will copy
                whatever is in TrainerArgument's output_dir to the local or remote artifact storage. Using it without a
                remote storage will just copy the files to your artifact location.
        """
        log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper()
        if log_artifacts in {"TRUE", "1"}:
            self._log_artifacts = True
        # if state.is_world_process_zero:
        self._ml_flow.start_run(
            run_name=f"{self.run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            if self.add_uuid
            else self.run_name,
            nested=self.nested,
            tags={
                mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: self.parent_run.info.run_id
            }
            if self.parent_run is not None
            else None,
        )
        combined_dict = args.to_dict()
        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}
        # remove params that are too long for MLflow
        for name, value in list(combined_dict.items()):
            # internally, all values are converted to str in MLflow
            if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                logger.warning(
                    f"Trainer is attempting to log a value of "
                    f'"{value}" for key "{name}" as a parameter. '
                    f"MLflow's log_param() only accepts values no longer than "
                    f"250 characters so we dropped this attribute."
                )
                del combined_dict[name]
        # MLflow cannot log more than 100 values in one go, so we have to split it
        combined_dict_items = list(combined_dict.items())
        try:
            for i in range(
                0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH
            ):
                self._ml_flow.log_params(
                    dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH])
                )
        except mlflow.exceptions.RestException as e:
            logger.warning(f"Could not update mlflow run")
            logger.warning(e.args[0])

        self._initialized = True
        self.model = model

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if mlflow.active_run is not None:
            try:
                mlflow.end_run(status="KILLED")
            except:
                logging.warning("Mlflow session was not killed properly.")


class EarlyStoppingCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles early stopping.
    Args:
       early_stopping_patience (:obj:`int`):
            Use with :obj:`metric_for_best_model` to stop training when the specified metric worsens for
            :obj:`early_stopping_patience` evaluation calls.
       early_stopping_threshold(:obj:`float`, `optional`):
            Use with TrainingArguments :obj:`metric_for_best_model` and :obj:`early_stopping_patience` to denote how
            much the specified metric must improve to satisfy early stopping conditions. `
    This callback depends on :class:`~transformers.TrainingArguments` argument `load_best_model_at_end` functionality
    to set best_metric in :class:`~transformers.TrainerState`.
    """

    def __init__(
        self,
        early_stopping_patience: int = 1,
        early_stopping_threshold: Optional[float] = 0.0,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        assert (
            args.load_best_model_at_end
        ), "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            args.evaluation_strategy != EvaluationStrategy.NO
        ), "EarlyStoppingCallback requires EvaluationStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping "
                f"is disabled "
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True


class MaxTrialCallback:
    def __init__(self, n_trials: int) -> None:
        self.n_trials = n_trials

    def __call__(self, study, trial) -> None:
        n_complete = len(
            [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                or t.state == optuna.trial.TrialState.RUNNING
                or t.state == optuna.trial.TrialState.PRUNED
            ]
        )
        if n_complete >= self.n_trials:
            study.stop()
