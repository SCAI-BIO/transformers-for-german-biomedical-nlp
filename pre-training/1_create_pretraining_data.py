# -*- coding: utf-8 -*-

# imports
import logging
import random
import sys
from os.path import join
from typing import List

import click
import torch
from toolbox.datasets.pretraining_dataset import create_datasets, process_documents
from transformers import BertTokenizerFast, ElectraTokenizerFast, RobertaTokenizerFast

# globals
logger = logging.getLogger(__name__)


# functions


@click.command()
@click.argument("sequence-length", type=int)
@click.argument("dataset-name")
@click.option("--output-dir", default="./data/processed")
@click.option("--short-seq-prob", default=0.1, type=float)
@click.option("--masked-lm-prob", default=0.15, type=float)
@click.option("--max-predictions-per-seq", default=20, type=int)
@click.option("--threads", default=1, type=int)
@click.option("-v", "--vocab", default=None, help="Vocab file")
@click.option(
    "-t",
    "--tokenizer",
    help="Path to custom vocab file or huggingface tokenizer",
    default=None,
)
@click.option(
    "-i",
    "--input-files",
    help="Path to input files",
    multiple=True,
)
@click.option("--stdout", is_flag=True)
@click.option("--model", default="BERT", help="BERT, ELECTRA, or RoBERTa")
@click.option("--eval-ratio", default=0.05, type=float)
def create_pretraining_data(
    sequence_length: int,
    dataset_name: str,
    output_dir: str,
    short_seq_prob: float,
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    threads: int,
    vocab: str,
    tokenizer: str,
    input_files: List[str],
    stdout: bool,
    model: str,
    eval_ratio: float,
):
    assert input_files is not None

    # Init logging
    if stdout:
        logging.basicConfig(format="%(asctime)s: %(message)s", stream=sys.stdout)

    # init tokenizer

    if model == "BERT":
        tok_function = BertTokenizerFast
    elif model == "ELECTRA":
        tok_function = ElectraTokenizerFast
    elif model == "RoBERTa":
        tok_function = RobertaTokenizerFast
    else:
        raise NotImplemented("Tokenizer is not available")

    if vocab is not None:
        logging.info(f"Read custom tokenizer from file: {vocab}.")
        try:
            tokenizer = tok_function(vocab)
        except FileNotFoundError:
            logging.error("Vocab file was not found.")
            raise
    else:
        logging.info(f"Initialize tokenizer from pretrained model {tokenizer}")
        tokenizer = tok_function.from_pretrained(tokenizer)

    # create datasets

    rng = random.Random()
    training, val = create_datasets(
        process_documents(
            input_files,
            tokenizer,
            sequence_length,
            short_seq_prob,
            masked_lm_prob,
            max_predictions_per_seq,
            tokenizer.get_vocab(),
            rng,
            threads,
        ),
        model_type=model,
        do_eval=True,
        eval_size=eval_ratio,
    )
    training_name = join(
        output_dir,
        model + "_" + dataset_name + "-" + str(sequence_length) + "_training.pt",
    )
    validation_name = join(
        output_dir,
        model + "_" + dataset_name + "-" + str(sequence_length) + "_validation.pt",
    )

    torch.save(training, training_name)
    torch.save(val, validation_name)
    logging.info("Saved dataset to %s and %s", training_name, validation_name)


if __name__ == "__main__":
    create_pretraining_data()
