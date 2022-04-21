# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Callable

from torch import cuda, device
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
)


@dataclass
class BaseConfig:
    seed: int = 20210217

    mlflow_tracking_uri: str = "localhost:5000"
    experiment_name: str = "default"
    arch: device = device("cuda") if cuda.is_available() else device("cpu")


@dataclass
class _TrainingParameters:
    model_name: str = field(default="model")
    model_function: PreTrainedModel = field(default=AutoModel)
    tokenizer_function: PreTrainedTokenizer = field(default=AutoTokenizer)
    tokenizer_type: str = "pretrained"
    trainer_function: Callable = field(default=Trainer)
    max_length: int = 512

    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    train_batch_size: int = 16
    eval_batch_size: int = 16
    max_epochs: int = 20
    warmup_ratio: float = 0.1
    warmup_steps: int = None

    output_directory: str = "models/"
    logging_steps: int = 200


@dataclass
class TrainingConfig(_TrainingParameters, BaseConfig):
    pass
