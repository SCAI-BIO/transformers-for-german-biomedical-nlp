# -*- coding: utf-8 -*-
"""
Script to pretrain transformer-based models with huggingface's Trainer class
"""

# imports

import logging
from pathlib import Path

import click
import torch
from torch.utils.data.dataset import Dataset
from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    IntervalStrategy,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import SchedulerType, get_last_checkpoint

from toolbox.datasets.pretraining_dataset import RobertaDataset
from toolbox.models.electra_pretraining import ElectraModelForPreTraining
from toolbox.training.pretraining.trainer import InheritedTrainer
from toolbox.utils.metrics import calc_bert_accuracy, calc_electra_accuracy

# global variables

logger = logging.getLogger(__name__)


# functions


@click.group()
def cli():
    pass


@cli.command()
@click.argument("output-dir")
@click.argument("training-data")
@click.argument("validation-data")
@click.argument("model-name")
@click.option(
    "--tokenizer",
    help="Path to tokenizer files. Only needed if different to model-name",
    default=None,
)
@click.option("--gradient-accumulation-steps", type=int, default=18)
@click.option("--batch-size", type=int, default=28)
@click.option("--learning-rate", type=float, default=4e-4)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--max-steps", type=int, default=50_000)
@click.option("--warmup-steps", type=int, default=2_000)
def domain_adapted_roberta(
    output_dir: str,
    training_data: str,
    validation_data: str,
    model_name: str,
    tokenizer: str,
    gradient_accumulation_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    warmup_steps: int,
):
    if tokenizer is None:
        tokenizer = model_name

    # Create output folder if it does not exist yet
    output_dir = Path(output_dir)
    (output_dir / "model").mkdir(exist_ok=True, parents=True)

    # load model and setup Trainer
    model = RobertaForMaskedLM.from_pretrained(model_name)
    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        lr_scheduler_type=SchedulerType.POLYNOMIAL,
        do_train=True,
        do_eval=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        logging_steps=100,
        eval_steps=250,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        save_steps=500,
        save_total_limit=50,
        logging_dir="logs/",
        label_names=["labels"],
        fp16=False,
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer)
    training: Dataset = torch.load(training_data)
    validation: Dataset = torch.load(validation_data)

    if isinstance(model, RobertaForMaskedLM) and not isinstance(
        training, RobertaDataset
    ):
        training = RobertaDataset.from_bert_dataset(training)  # type: ignore
        validation = RobertaDataset.from_bert_dataset(validation)  # type: ignore

    trainer = InheritedTrainer(
        model=model,
        args=training_arguments,
        train_dataset=training,
        eval_dataset=validation,
        compute_metrics=calc_bert_accuracy,
        tokenizer=tokenizer,
    )

    if get_last_checkpoint(output_dir) is not None:
        # Load model and proceed training if checkpoint is found
        checkpoint_path = get_last_checkpoint(output_dir)
        logger.info(f"Load model from '{checkpoint_path}'")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        # Start training if no checkpoint is found
        trainer.train()
        logger.info(f"Start domain-adaption")

    trainer.save_model(str(output_dir / "model"))


@cli.command()
@click.argument("output-dir")
@click.argument("training-data")
@click.argument("validation-data")
@click.argument("model-vocab")
@click.option("--gradient-accumulation-steps", type=int, default=1)
@click.option("--batch-size", type=int, default=64)
@click.option("--learning-rate", type=float, default=5e-4)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--max-steps", type=int, default=1_000_000)
@click.option("--warmup-steps", type=int, default=10_000)
def electra_small(
    output_dir: str,
    training_data: str,
    validation_data: str,
    model_vocab: str,
    gradient_accumulation_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    warmup_steps: int,
):
    # Create output folder if it does not exist yet
    output_dir = Path(output_dir)
    (output_dir / "model").mkdir(exist_ok=True, parents=True)

    generator_config = ElectraConfig(
        hidden_size=64, intermediate_size=256, num_attention_heads=1
    )
    discriminator_config = ElectraConfig()

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        logging_steps=100,
        eval_steps=250,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        save_steps=1000,
        save_total_limit=50,
        logging_dir="logs/",
        label_names=["labels"],
    )

    model = ElectraModelForPreTraining(
        init_model=True,
        generator_config=generator_config,
        discriminator_config=discriminator_config,
        compatible_with_hf_trainer=True,
    )

    tokenizer = ElectraTokenizer(model_vocab)
    training = torch.load(training_data)
    validation = torch.load(validation_data)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=training,
        eval_dataset=validation,
        compute_metrics=calc_electra_accuracy,
        tokenizer=tokenizer,
    )

    if get_last_checkpoint(output_dir) is not None:
        # Load model and proceed training if checkpoint is found
        checkpoint_path = get_last_checkpoint(output_dir)
        logger.info(f"Load model from '{checkpoint_path}'")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        # Start training if no checkpoint is found
        trainer.train()
        logger.info(f"Start training of ELECTRA small model")

    trained_model: ElectraModelForPreTraining = trainer.model  # type: ignore
    trained_model.generator.save_pretrained(str(output_dir / "generator"))
    trained_model.discriminator.save_pretrained(str(output_dir / "discriminator"))


@cli.command()
@click.argument("output-dir")
@click.argument("training-data")
@click.argument("validation-data")
@click.argument("model-vocab")
@click.option("--gradient-accumulation-steps", type=int, default=1)
@click.option("--batch-size", type=int, default=32)
@click.option("--learning-rate", type=float, default=5e-4)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--max-steps", type=int, default=1_000_000)
@click.option("--warmup-steps", type=int, default=10_000)
def electra_base(
    output_dir: str,
    training_data: str,
    validation_data: str,
    model_vocab: str,
    gradient_accumulation_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    warmup_steps: int,
):
    # Create output folder if it does not exist yet
    output_dir = Path(output_dir)
    (output_dir / "model").mkdir(exist_ok=True, parents=True)

    # Configure model and setup trainer
    generator_config = ElectraConfig(
        num_attention_heads=4,
        num_hidden_layers=12,
        hidden_size=256,
        intermediate_size=1028,
        embedding_size=768,
    )
    discriminator_config = ElectraConfig(
        num_attention_heads=12,
        num_hidden_layers=12,
        hidden_size=768,
        intermediate_size=3072,
        embedding_size=768,
    )

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        logging_steps=100,
        eval_steps=250,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        save_steps=1000,
        save_total_limit=50,
        logging_dir="logs/",
        label_names=["labels"],
    )

    model = ElectraModelForPreTraining(
        init_model=True,
        generator_config=generator_config,
        discriminator_config=discriminator_config,
        compatible_with_hf_trainer=True,
    )

    tokenizer = ElectraTokenizer(model_vocab)
    training = torch.load(training_data)
    validation = torch.load(validation_data)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=training,
        eval_dataset=validation,
        compute_metrics=calc_electra_accuracy,
        tokenizer=tokenizer,
    )

    if get_last_checkpoint(output_dir) is not None:
        # Load model and proceed training if checkpoint is found
        checkpoint_path = get_last_checkpoint(output_dir)
        logger.info(f"Load model from '{checkpoint_path}'")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        # Start training if no checkpoint is found
        trainer.train()
        logger.info(f"Start training of ELECTRA base model")

    trained_model: ElectraModelForPreTraining = trainer.model  # type: ignore
    trained_model.generator.save_pretrained(str(output_dir / "generator"))
    trained_model.discriminator.save_pretrained(str(output_dir / "discriminator"))


if __name__ == "__main__":
    cli()
