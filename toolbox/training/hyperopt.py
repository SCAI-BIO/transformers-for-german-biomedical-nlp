# -*- coding: utf-8 -*-
from typing import Dict


class ComputeObjective:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name

    def __call__(self, metrics: Dict[str, float]) -> float:
        """
        Compute objective to maximize/minimize when doing a hyperparameter search. It is the f1-score.
        Args:
            metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.
        Return:
            :obj:`float`: The objective to minimize or maximize
        """
        assert (
            self.metric_name in metrics
        ), f"Key {self.metric_name} should exists in metrics"
        metric = metrics.get(self.metric_name)
        return metric if isinstance(metric, float) else -1.0
