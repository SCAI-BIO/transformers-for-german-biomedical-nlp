# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path

from toolbox.training.base_config import TrainingConfig


@dataclass
class ScConfig(TrainingConfig):
    """Configuration for Multi-Instance sequence classification"""

    results_dir: str = None
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 0.1
    greater_is_better: bool = True
    eval_metric: str = "f1"
    eval_strategy: str = "epoch"

    results_prefix = "results"
    outer_folds = 5
    inner_folds = 5

    def __post_init__(self):
        results_dir = Path(self.results_dir)
        self.average_results_file = str(
            results_dir / f"{self.results_prefix}_{self.model_name}_average.csv",
        )
        self.entity_results_file = str(
            results_dir / f"{self.results_prefix}_{self.model_name}_groups.csv",
        )
