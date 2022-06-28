# -*- coding: utf-8 -*-
import logging
from typing import Dict, Union

import numpy as np
import sklearn.metrics as sm
import torch
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import (
    precision_recall_fscore_support,
    precision_recall_fscore_support_v1,
)
from seqeval.scheme import BILOU, IOB2
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


class NerMetrics:
    def __init__(self, label_set, scheme: str = "BILOU", mode: str = "strict") -> None:
        self.label_set = label_set
        self.scheme = BILOU if scheme == "BILOU" else IOB2
        if self.scheme == BILOU and mode != "strict":
            logger.warning("Can only use strict mode for BILOU")
        self.mode = mode

    def get_categories(self, labels, predictions):
        return (
            self.label_set.ids_to_label(labels),
            self.label_set.ids_to_label(predictions),
        )

    def compute_classification_report(
        self,
        labels,
        logits,
        digits: int = 2,
    ) -> Dict:
        """
        Generate classification report (seqeval) for the prediction/label_ids.

        Args:
            labels ([type]): 2-D tensor, label ids
            logits ([type]): 3-D tensor, softmax probabilities
            digits (int, optional): numerical precision. Defaults to 2.

        Returns:
            Dict: Entity wise metrics
        """
        if logits.ndim == 3:
            predictions = np.argmax(logits, axis=2)
        elif logits.ndim == 2:
            predictions = logits
        else:
            raise Exception("Unknown dims")
        truth, pred = self.get_categories(labels, predictions)
        return classification_report(
            truth,
            pred,
            digits=digits,
            output_dict=True,
            zero_division="0",
            scheme=self.scheme,
            mode=self.mode,
        )

    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute macro/micro f1, precision and recall for the given input.

        Args:
            pred (EvalPrediction): prediction and label_ids

        Returns:
            Dict[str, float]: [description]
        """

        # NER ids are also passed through prediction loop; therefore it is a tuple
        # instead of a tensor
        logits = pred.predictions
        if logits.ndim == 3:
            prediction = np.argmax(logits, axis=2)
        elif logits.ndim == 2:
            prediction = logits
        else:
            raise Exception("Unknown dims")
        labels = pred.label_ids

        # get label names to work with seqeval
        truth, pred = self.get_categories(labels, prediction)

        if self.mode == "strict" and self.scheme:
            (
                macro_precision,
                macro_recall,
                macro_f1,
                _,
            ) = precision_recall_fscore_support_v1(
                truth,
                pred,
                scheme=self.scheme,
                average="macro",
                zero_division="0",
                mode=self.mode,
            )
            (
                micro_precision,
                micro_recall,
                micro_f1,
                _,
            ) = precision_recall_fscore_support_v1(
                truth,
                pred,
                scheme=self.scheme,
                average="micro",
                zero_division="0",
                mode=self.mode,
            )
        else:
            logger.warning(f"{self.scheme} is not considered for calculation.")
            (
                macro_precision,
                macro_recall,
                macro_f1,
                _,
            ) = precision_recall_fscore_support(
                truth,
                pred,
                average="macro",
                zero_division="0",
            )
            (
                micro_precision,
                micro_recall,
                micro_f1,
                _,
            ) = precision_recall_fscore_support(
                truth,
                pred,
                average="micro",
                zero_division="0",
            )

        results = {
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "micro_f1": micro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
        }
        return results

    def __call__(self, predictions: EvalPrediction) -> Dict[str, float]:
        """
        Compute macro/micro f1, precision and recall for the given input.

        Args:
            predictions (EvalPrediction): prediction and label_ids

        Returns:
            Dict[str, float]: [description]
        """
        return self.compute_metrics(pred=predictions)


class ScMetrics:
    """Metrics for sequence classification"""

    def __init__(self, eps: float = 1e-9, average: str = "micro") -> None:
        self.eps = eps
        self.average = average

    def __call__(self, predictions: EvalPrediction, num_labels: int) -> Dict[str, any]:
        logits = predictions.predictions
        labels, bag_ids = predictions.label_ids
        labels = labels.reshape(-1, num_labels)
        bag_ids = bag_ids.reshape(
            -1,
        )
        logits = logits.reshape(-1, num_labels)
        instance_logits = np.stack(
            [
                np.max(logits[bag_ids == instance_id], axis=0)
                for instance_id in np.unique(bag_ids)
            ]
        )
        instance_logits = sigmoid(instance_logits)
        preds = np.where(instance_logits > 0.5, 1, 0)
        precision, recall, f1, support = sm.precision_recall_fscore_support(
            labels, preds, average=self.average
        )
        return {"f1": f1, "recall": recall, "precision": precision, "support": support}

    @staticmethod
    def compute_group_metrics(
        logits: Union[torch.Tensor, np.ndarray],
        bag_ids: torch.Tensor,
        labels: torch.Tensor,
        label_range: int,
    ):
        labels = labels.reshape(-1, label_range)
        bag_ids = bag_ids.reshape(
            -1,
        )
        logits = logits.reshape(-1, label_range)
        instance_logits = np.stack(
            [
                np.max(logits[bag_ids == instance_id], axis=0)
                for instance_id in np.unique(bag_ids)
            ]
        )
        probs = sigmoid(instance_logits)
        preds = np.where(probs > 0.5, 1, 0)
        return (
            sm.classification_report(labels, preds, output_dict=True),
            instance_logits,
        )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calc_mlm_accuracy_from_predictions(
    predictions: torch.Tensor, labels: torch.Tensor
) -> float:
    """For BERT and ELECTRA. Calculate MLM accuracy"""
    masked_tokens = np.where(labels != -100, 1, 0)
    print(predictions.shape)
    print(labels.shape)
    print(masked_tokens.shape)
    true = torch.sum(predictions.__eq__(labels) * masked_tokens)
    total = np.sum(masked_tokens)
    acc = float(true / total)
    return acc if acc > 0 else 0.0


def calc_discriminator_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate the discriminator accuracy based on logits and labels"""
    probs = torch.sigmoid(logits)
    t = torch.autograd.Variable(torch.Tensor([0.5])).to(labels.device)  # threshold
    cls = (probs > t).int()
    true = cls.eq(labels).int()
    return float(torch.sum(true).float() / labels.numel())


def calc_electra_accuracy(preds: EvalPrediction) -> Dict:
    disc_logits = torch.FloatTensor(preds.predictions[0])
    gen_preds = torch.FloatTensor(preds.predictions[1])
    disc_labels = torch.FloatTensor(preds.label_ids[0])
    gen_labels = torch.FloatTensor(preds.label_ids[1])

    return {
        "discriminator accuracy": calc_discriminator_accuracy(disc_logits, disc_labels),
        "generator_accuracy": calc_mlm_accuracy_from_predictions(gen_preds, gen_labels),
    }


def calc_bert_accuracy(preds: EvalPrediction) -> Dict:
    mlm_accuracy = calc_mlm_accuracy_from_predictions(
        torch.FloatTensor(preds.predictions), torch.FloatTensor(preds.label_ids)
    )
    return {"mlm_accuracy": mlm_accuracy}
