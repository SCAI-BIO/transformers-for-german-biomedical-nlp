# -*- coding: utf-8 -*-

# imports
import os
from typing import Callable, Dict, Optional, Union

from transformers import Trainer
from transformers.integrations import (
    default_hp_search_backend,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_ray,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    HPSearchBackend,
    default_compute_objective,
    default_hp_space_optuna,
)


# functions
def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna

    def _objective(trial, checkpoint_dir=None):
        checkpoint = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint = os.path.join(checkpoint_dir, subdir)
        trainer.objective = None
        trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
        return trainer.objective

    timeout = kwargs.pop("timeout", None)
    callbacks = kwargs.pop("callbacks", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        callbacks=callbacks,
    )
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params)


# classes
class OptunaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.hp_search_backend = None
        self.compute_objective = None
        self.hp_name = None
        self.hp_space = None

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
        Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py and adapted to own
        needs.


        Launch a hyperparameter search using ``optuna`` or ``Ray Tune``. The optimized quantity is determined by
        :obj:`compute_objective`, which defaults to a function returning the evaluation loss when no metric is
        provided, the sum of all metrics otherwise.
        .. warning::
            To use this method, you need to have provided a ``model_init`` when initializing your
            :class:`~transformers.Trainer`: we need to reinitialize the model at each new run. This is incompatible
            with the ``optimizers`` argument, so you need to subclass :class:`~transformers.Trainer` and override the
            method :meth:`~transformers.Trainer.create_optimizer_and_scheduler` for custom optimizer/scheduler.
        Args:
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
            hp_name:
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

        self.hp_space = (
            default_hp_space_optuna[backend] if hp_space is None else hp_space
        )
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

        return best_run
