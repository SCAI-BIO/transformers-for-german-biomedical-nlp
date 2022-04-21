# -*- coding: utf-8 -*-

# imports
import collections
import datetime
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.integrations import (
    default_hp_search_backend,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_ray,
)
from transformers.trainer_pt_utils import (
    nested_concat,
    nested_detach,
    torch_pad_and_concatenate,
)
from transformers.trainer_utils import (
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    default_compute_objective,
    default_hp_space_optuna,
)

from toolbox.datasets.sc_dataset import FixedSampler, collate_multi
from toolbox.training.modified_trainer import run_hp_search_optuna

# globals
from toolbox.utils import CallbackDict, ScMetrics, TrainerUtils
from toolbox.utils.callbacks import MaxTrialCallback

logger = logging.getLogger(__name__)


# functions

# classes
class ScTrainer(object):
    def __init__(
        self,
        run_id: int,
        trainer_function: Callable,
        model: Union[Module, Callable],
        callbacks: List[TrainerCallback],
        training_arguments: TrainingArguments,
        training_data: Dataset,
        metric_function: Callable,
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
            compute_objective (Callable[[Dict[str, float]], float], optional): Objective function for hyperparameter optimization. Defaults to None.
        """
        super().__init__()

        self.run_id = run_id
        self.model = model
        self.callbacks = callbacks
        self.training_arguments = training_arguments
        self.training_data = training_data
        self.validation_data = validation_data
        self.metric_function = metric_function
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

    def train(self, trial=None):
        """Calls train function of trainer instance"""
        logger.info("Start training")
        self.trainer.train(trial=trial)

    def hyperparameter_search(
        self,
        n_trials: int,
        direction: str,
        storage: str,
        study_name: str,
        load_if_exists: bool = True,
        timeout: int = False,
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
        )

    def evaluate(self, dataset):
        """Evaluation with specified data"""
        logger.info("Start evaluation")
        predictions = self.trainer.predict(dataset)
        logits = predictions.predictions
        labels, bag_ids = predictions.label_ids
        compute_metrics = ScMetrics()
        tmp_group = pd.DataFrame(
            compute_metrics.compute_group_metrics(
                logits, bag_ids, labels, dataset.num_labels
            )
        )
        if id is not None:
            run_name = "Run" + str(self.run_id)
            tmp_group["run"] = [run_name] * tmp_group.shape[0]
        return self.trainer.evaluate(dataset), tmp_group


class ScMiTrainer(Trainer):
    """Trainer class for multi-instance learning"""

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
    ):
        """

        Args:
            model (Union[PreTrainedModel, torch.nn.Module], optional): model instance. Defaults to None.
            args (TrainingArguments, optional): huggingface TrainingArguments. Defaults to None.
            data_collator (Optional[DataCollator], optional): data_collator for multi-instance learning.
                Defaults to None.
            train_dataset (Optional[Dataset], optional): Defaults to None.
            eval_dataset (Optional[Dataset], optional): Defaults to None.
            tokenizer (PreTrainedTokenizerBase): Tokenizer instance. Defaults to None.
            model_init (Callable[[], PreTrainedModel], optional): Function to init a model. Has to be specified if no
                model is provided. Defaults to None.
            compute_metrics (Optional[Callable[[EvalPrediction], Dict]], optional): Function for metric calculation.
                Defaults to None.
            callbacks (Optional[List[TrainerCallback]], optional): Callbacks to be added to the Trainer instance.
                Defaults to None.
            optimizers (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], optional): Optimizers for
                Trainer instance. Defaults to ( None, None, ).
        """
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
        )
        self.hp_search_backend = None
        self.compute_objective = None
        self.hp_name = None
        self.hp_space = None
        self.control = None
        self._past = None
        self._past = None
        self.loss_function = nn.BCEWithLogitsLoss()

    def _calculate_loss(self, logits, bag_ids, labels) -> torch.Tensor:
        """Wrapper around loss functions with different signatures"""

        tensors = []
        for bag_id in bag_ids.unique():
            idx = torch.where(bag_ids == bag_id)[0]  # type: ignore
            subset = logits[idx].float()
            sub_max, _ = torch.max(subset, dim=0)
            tensors.append(sub_max)
        logits = torch.stack(tensors)

        return self.loss_function(logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if "bag_ids" in inputs.keys():
            bag_ids = inputs.pop("bag_ids")
        else:
            bag_ids = None

        if "labels" in inputs.keys():
            labels = inputs.pop("labels")
        else:
            labels = None

        logits = model(**inputs)["logits"]
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            raise NotImplementedError("Not available in this trainer")

        loss = None
        if labels is not None:
            assert bag_ids is not None, "Bag ids should be passed"
            loss = self._calculate_loss(logits, bag_ids, labels)

        return (loss, logits) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        with torch.no_grad():
            if has_labels:
                label_ids = inputs["labels"]
                bag_ids = inputs["bag_ids"]
                loss, logits = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys
                    )
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if logits.dim() == 1:
            logits = logits.view(-1, logits.shape[0])

        if has_labels:
            labels = nested_detach((label_ids, bag_ids))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return loss, logits, labels

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_sampler=FixedSampler(self.train_dataset, self.args.train_batch_size),
            collate_fn=collate_multi,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        logger.debug("Eval dataloader is provided.")

        return DataLoader(
            eval_dataset,
            batch_sampler=FixedSampler(eval_dataset, self.args.eval_batch_size),
            collate_fn=collate_multi,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        return DataLoader(
            test_dataset,
            batch_sampler=FixedSampler(test_dataset, self.args.eval_batch_size),
            collate_fn=collate_multi,
            num_workers=self.args.dataloader_num_workers,
        )

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        start = datetime.datetime.now()
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            raise NotImplementedError("Not available in this setting")
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        losses_host: torch.Tensor = None  # type: ignore
        host_predictions: Union[torch.Tensor, List[torch.Tensor]] = None  # type: ignore
        host_labels: Union[torch.Tensor, List[torch.Tensor]] = None  # type: ignore
        host_bag: Union[torch.Tensor, List[torch.Tensor]] = None  # type: ignore

        eval_losses = []
        if not prediction_loss_only:
            prediction_gatherer = []
            labels_gatherer = []
            bag_ids_gatherer = []
        else:
            prediction_gatherer = None
            labels_gatherer = None
            bag_ids_gatherer = None

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            num_batch = len(inputs["bag_ids"].unique())
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            if loss is not None:
                losses = loss.repeat(num_batch)  # type: ignore
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if logits is not None:
                host_predictions = (
                    logits
                    if host_predictions is None
                    else nested_concat(host_predictions, logits, padding_index=-100)
                )
            if labels is not None:
                label_ids, bag_ids = labels
                host_labels = (  # type: ignore
                    label_ids
                    if host_labels is None
                    else torch_pad_and_concatenate(
                        host_labels[0], label_ids, padding_index=-100
                    ),
                )
                if host_bag is not None:
                    bag_ids += max(host_bag[0]) + 1
                host_bag = (  # type: ignore
                    bag_ids
                    if host_bag is None
                    else torch_pad_and_concatenate(
                        host_bag[0], bag_ids, padding_index=-100
                    ),
                )
            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                self.args.eval_accumulation_steps is not None
                and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                eval_losses.append(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    prediction_gatherer.append(
                        self._gather_and_numpify(host_predictions, "eval_ " "preds")
                    )
                    labels_gatherer.append(
                        self._gather_and_numpify(host_labels, "eval_label_ids")
                    )
                    bag_ids_gatherer.append(
                        self._gather_and_numpify(host_bag, "eval_bag_ids")
                    )
                    print("LabelsG", labels_gatherer)

                # Set back to None to begin a new accumulation
                losses_host, host_predictions, host_labels = None, None, None  # type: ignore

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses.append(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            prediction_gatherer.append(
                self._gather_and_numpify(host_predictions, "eval_preds")
            )
            labels_gatherer.append(
                self._gather_and_numpify(host_labels, "eval_label_ids")
            )
            bag_ids_gatherer.append(self._gather_and_numpify(host_bag, "eval_bag_ids"))

        eval_loss = np.stack(eval_losses)
        preds = np.stack(prediction_gatherer) if not prediction_loss_only else None
        label_ids = np.stack(labels_gatherer) if not prediction_loss_only else None
        bag_ids = np.stack(bag_ids_gatherer) if not prediction_loss_only else None

        label_ids = (label_ids, bag_ids)

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            try:
                num_out = model.classifier.out_features
            except:
                num_out = model.classifier.out_proj.out_features
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids),  # type: ignore
                range=num_out,
            )
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        logger.info(datetime.datetime.now() - start)

        return EvalLoopOutput(
            predictions=preds,
            label_ids=label_ids,
            metrics=metrics,
            num_samples=num_examples,
        )

    def hyperparameter_search(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ) -> BestRun:
        """
        Launch a hyperparameter search using ``optuna`` or ``Ray Tune``. The optimized quantity is determined by
        :obj:`compute_objective`, which defaults to a function returning the evaluation loss when no metric is
        provided, the sum of all metrics otherwise.
        .. warning::
            To use this method, you need to have provided a ``model_init`` when initializing your
            :class:`~transformers.Trainer`: we need to reinitialize the model at each new run. This is incompatible
            with the ``optimizers`` argument, so you need to subclass :class:`~transformers.Trainer` and override the
            method :meth:`~transformers.Trainer.create_optimizer_and_scheduler` for custom optimizer/scheduler.
        Args:
            hp_name:
            hp_space (:obj:`Callable[["optuna.Trial"], Dict[str, float]]`, `optional`):
                A function that defines the hyperparameter search space. Will default to
                :func:`~transformers.utils.default_hp_space_optuna` or
                :func:`~transformers.utils.default_hp_space_ray` depending on your backend.
            compute_objective (:obj:`Callable[[Dict[str, float]], float]`, `optional`):
                A function computing the objective to minimize or maximize from the metrics returned by the
                :obj:`evaluate` method. Will default to :func:`~transformers.utils.default_compute_objective`.
            n_trials (:obj:`int`, `optional`, defaults to 100):
                The number of trial runs to test.
            direction(:obj:`str`, `optional`, defaults to :obj:`"minimize"`):
                Whether to optimize greater or lower objects. Can be :obj:`"minimize"` or :obj:`"maximize"`, you should
                pick :obj:`"minimize"` when optimizing the validation loss, :obj:`"maximize"` when optimizing one or
                several metrics.
            backend(:obj:`str` or :class:`~transformers.training_utils.HPSearchBackend`, `optional`):
                The backend to use for hyperparameter search. Will default to optuna or Ray Tune, depending on which
                one is installed. If both are installed, will default to optuna.
            kwargs:
                Additional keyword arguments passed along to :obj:`optuna.create_study` or :obj:`ray.tune.run`. For
                more information see:
                - the documentation of `optuna.create_study
                  <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html>`__
                - the documentation of `tune.run
                  <https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run>`__
        Returns:
            :class:`transformers.trainer_utils.BestRun`: All the information about the best run.
        """
        if backend is None:
            backend = default_hp_search_backend()
            if backend is None:
                raise RuntimeError(
                    "At least one of optuna or ray should be installed. "
                    "To install optuna run `pip install optuna`."
                    "To install ray run `pip install ray[tune]`."
                )
        backend = HPSearchBackend(backend)
        if backend == HPSearchBackend.OPTUNA and not is_optuna_available():
            raise RuntimeError(
                "You picked the optuna backend, but it is not installed. Use `pip install optuna`."
            )
        if backend == HPSearchBackend.RAY and not is_ray_tune_available():
            raise RuntimeError(
                "You picked the Ray Tune backend, but it is not installed. Use `pip install 'ray[tune]'`."
            )
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = default_hp_space_optuna if hp_space is None else hp_space
        self.hp_name = hp_name
        self.compute_objective = (
            default_compute_objective
            if compute_objective is None
            else compute_objective
        )

        run_hp_search = (
            run_hp_search_optuna
            if backend == HPSearchBackend.OPTUNA
            else run_hp_search_ray
        )
        best_run = run_hp_search(self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run
