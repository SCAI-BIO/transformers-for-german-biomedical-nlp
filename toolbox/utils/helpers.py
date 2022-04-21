# -*- coding: utf-8 -*-
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import mlflow
import torch
# from toolbox.training.ganbert import GanBertForTokenClassification
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedModel, Trainer
from transformers.configuration_utils import PretrainedConfig

from toolbox.training.base_config import BaseConfig

logger = logging.getLogger(__name__)


class CallbackDict(object):
    def __init__(self, remove: List[type] = None, add: List[Callable] = None) -> None:
        super().__init__()

        self.remove = remove
        self.add = add

    def __call__(self) -> Dict[str, List[Callable]]:
        return {"remove": self.remove, "add": self.add}

    def __len__(self):
        to_remove = len(self.remove) if self.remove is not None else 0
        to_add = len(self.add) if self.add is not None else 0
        return to_remove + to_add


class TrainerUtils(object):
    @staticmethod
    def prepare_mlflow_logging(config: BaseConfig):
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    @staticmethod
    def prepare_mlflow_callbacks(trainer: Trainer, callback_dict: CallbackDict):
        if callback_dict is None:
            return None
        elif len(callback_dict) == 0:
            return None

        for log_type in callback_dict.remove:
            trainer.remove_callback(log_type)

        for instance in callback_dict.add:
            trainer.add_callback(instance)

    @staticmethod
    def calculate_splits(
        dataset: Dataset, folds: int, shuffle: bool = True
    ) -> List[Tuple[int]]:
        kf = KFold(folds, shuffle=shuffle)
        return kf.split(range(len(dataset)))  # type: ignore

    @staticmethod
    def prepare_inputs(
        inputs: Dict[str, Union[torch.Tensor, Any]], device=torch.device("cuda")
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        return inputs


class ModelLoader:
    def __init__(
        self,
        model_fct: Callable,
        model_config: PretrainedConfig,
        model_path: str,
        additional_model_config=None,
    ):
        self.model_fct = model_fct
        self.model_config = model_config
        self.model_path = model_path
        self.additional_model_config = additional_model_config

    def __call__(self) -> PreTrainedModel:
        if self.additional_model_config is None:
            model = self.model_fct.from_pretrained(
                self.model_path, config=self.model_config
            )
        else:
            model = self.model_fct.from_pretrained(
                self.model_path,
                config=(self.model_config, self.additional_model_config),
            )
        logger.info(f"Model of type: {type(model)}")
        return model
