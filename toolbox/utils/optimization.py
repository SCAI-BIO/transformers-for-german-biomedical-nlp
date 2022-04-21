# -*- coding: utf-8 -*-
from typing import Dict

from optuna import Trial


def hyperparameter_space(trial: Trial) -> Dict[str, float]:
    hps = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-10, 1e-1),
    }
    return hps
