# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers import (ElectraConfig, ElectraForMaskedLM,
                          ElectraForPreTraining)


class ElectraModelForPreTraining(nn.Module):
    """
    Model used for Pre-Training of BioELECTRA small and base
    """

    def __init__(
        self,
        init_model: bool = False,
        pretrained_generator: str = None,
        pretrained_discriminator: str = None,
        generator_config: ElectraConfig = None,
        discriminator_config: ElectraConfig = None,
        compatible_with_hf_trainer: bool = True,
        lambda_coefficient: int = 1,
    ) -> None:
        super().__init__()

        if init_model:
            assert generator_config is not None and discriminator_config is not None

        self.compatible_with_hf_trainer = compatible_with_hf_trainer
        if init_model:
            self.generator = ElectraForMaskedLM(generator_config)
            self.discriminator = ElectraForPreTraining(discriminator_config)
        else:
            self.generator = ElectraForMaskedLM.from_pretrained(pretrained_generator)
            self.discriminator = ElectraForPreTraining.from_pretrained(
                pretrained_discriminator
            )

        # share embeddings between generator and discriminator
        self.discriminator.electra.embeddings = self.generator.electra.embeddings
        # Tie input and output token embeddings
        self.generator.generator_lm_head.weight = (
            self.generator.electra.embeddings.word_embeddings.weight
        )
        self.lambda_coefficient = lambda_coefficient

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels,
        return_dict: bool = False,
    ):
        generator_output = self.generator(
            input_ids, attention_mask, token_type_ids, labels=labels, return_dict=True
        )
        disc_labels = self.create_disc_labels(generator_output.logits, labels)
        disc_output = self.discriminator(
            input_ids,
            attention_mask,
            token_type_ids,
            labels=disc_labels,
            return_dict=True,
        )

        total_loss = generator_output.loss + self.lambda_coefficient * disc_output.loss

        if self.compatible_with_hf_trainer and not return_dict:
            return total_loss, disc_output.logits
        elif self.compatible_with_hf_trainer and return_dict:
            return {
                "loss": total_loss,
                "logits": disc_output.logits,
                "labels": labels,
                "disc_labels": disc_labels,
                "generator_logits": generator_output.logits,
            }
        else:
            return total_loss, generator_output, disc_output

    @staticmethod
    def create_disc_labels(
        generator_logits: torch.Tensor, generator_labels: torch.Tensor
    ) -> torch.Tensor:
        """Creates the appropriate labels for the discriminator"""
        assert generator_logits.ndim == 3

        predictions = torch.argmax(generator_logits, 2)
        replacement = predictions.eq(generator_labels)  # True if replaced
        return replacement.int()
