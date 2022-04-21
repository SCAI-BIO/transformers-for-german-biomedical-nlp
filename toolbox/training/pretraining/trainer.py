# -*- coding: utf-8 -*-
"""Training class for pretraining"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_detach


class InheritedTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        **kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(**inputs, return_dict=True)
            else:
                outputs = model(**inputs, return_dict=True)
            loss = outputs["loss"].mean().detach()
            logits = outputs["logits"]

            if self.args.past_index >= 0:
                raise NotImplementedError

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        logits = torch.argmax(logits, 2)

        labels = tuple(inputs.get(name) for name in self.label_names)
        if len(labels) == 1:
            labels = labels[0]

        return loss, logits, labels
