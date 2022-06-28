# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path

from toolbox.training.base_config import TrainingConfig


@dataclass
class NerConfig(TrainingConfig):
    """Configuration for NER"""

    results_dir: str = None
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 0.1
    early_stopping_metric: str = None
    early_stopping_greater_is_better: str = None
    greater_is_better: bool = True
    eval_metric: str = "f1"
    eval_strategy: str = "epoch"

    label_set: str = "data/interim/label_set.pt"
    results_prefix = "results"
    hyperopt: bool = False

    outer_folds: int = None
    inner_folds: int = None

    def __post_init__(self):
        results_dir = Path(self.results_dir)
        self.average_results_file = str(
            results_dir / f"{self.results_prefix}_{self.model_name}_average.csv",
        )
        self.entity_results_file = str(
            results_dir / f"{self.results_prefix}_{self.model_name}_groups.csv",
        )

        if self.early_stopping_metric is None:
            self.early_stopping_metric = self.eval_metric
            self.early_stopping_greater_is_better = self.greater_is_better
