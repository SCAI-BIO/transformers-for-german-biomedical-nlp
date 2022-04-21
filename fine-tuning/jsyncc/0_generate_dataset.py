# -*- coding: utf-8 -*-
"""
Create training, validation and testing data for JSynCC.
"""

# imports

import logging
import os
import pickle
import random
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import List

import click
import torch
import transformers
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import BertTokenizerFast

# globals
from toolbox.datasets.sc_dataset import Document, ScDataset

SEED = 20201120
AMBIGUOUS_TOPICS = {
    "emergency": "Notfallmedizin",
    "anaesthetics": "Anästhesie",
    "ophthalmology": "Augenheilkunde",
}


# functions


def get_tensors_from_dataset(dataset: ScDataset):
    input_ids = torch.stack(
        [
            i["input_ids"]
            if len(i["input_ids"]) == 1
            else i["input_ids"][0].view(1, -1)
            for i in dataset
        ]
    )
    labels = torch.stack([i["labels"] for i in dataset])
    return input_ids, labels


def load_corpus(file: str) -> List[Document]:
    """
    Load and process corpus.

    Args:
        file: path to corpus file

    Returns: List of processed documents
    """
    assert os.path.exists(file)

    instances = []

    root = ElementTree.parse(file).getroot()
    for document in root:
        # Init a Training instance for document
        doc_id = document.find("id").text
        doc_topic = [topic.text for topic in document.findall("topic")]
        if len(doc_topic) != 1:
            logging.info("Document has more/less than one topic")
        for i, topic in enumerate(doc_topic):
            if topic in AMBIGUOUS_TOPICS.keys():
                doc_topic[i] = AMBIGUOUS_TOPICS[topic]
        doc_text = document.find("text").text
        doc_inst = Document(doc_text, doc_id, doc_topic)
        instances.append(doc_inst)
    logging.info("Found %g in the file.", len(instances))

    # Add randomness
    random.shuffle(instances)

    return instances


# main


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input-file", type=click.Path(exists=True))
@click.argument("output-directory", type=click.Path())
@click.option(
    "--cutoff", type=int, default=50, help="Minimum frequency for topics to be included"
)
@click.option("-t", "--tokenizer", type=str, default="bert-base-german-cased")
@click.option("-l", "--log", is_flag=True)
def generate_dataset(
    input_file: str, output_directory: str, cutoff: int, tokenizer: str, log: bool
):
    """Generate dataset for JSynCC"""
    transformers.set_seed(SEED)

    # Prepare logging
    if log:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    # Tokenize text
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer)

    instances = load_corpus(input_file)
    dataset = ScDataset(instances)

    dataset.prepare_data(tokenizer, 512, cutoff)
    torch.save(dataset, str(output_directory / "whole_data.pt"))

    ids, labels = get_tensors_from_dataset(dataset)
    n_test = int(0.2 * len(dataset))
    n_val = int(0.1 * len(dataset))

    stratified_shuffler = StratifiedShuffleSplit(test_size=n_test, n_splits=1)
    remain_idx, test_idx = next(iter(stratified_shuffler.split(ids, labels)))
    test_set = dataset.subset(test_idx)
    remaining_set = dataset.subset(remain_idx)

    ids, labels = get_tensors_from_dataset(remaining_set)
    stratified_shuffler = StratifiedShuffleSplit(test_size=n_val, n_splits=1)
    train_idx, val_idx = next(iter(stratified_shuffler.split(ids, labels)))
    val_set = remaining_set.subset(val_idx)
    train_set = remaining_set.subset(train_idx)

    torch.save(train_set, str(output_directory / "training_data.pt"))
    torch.save(val_set, str(output_directory / "validation_data.pt"))
    torch.save(test_set, str(output_directory / "testing_data.pt"))


@cli.command()
@click.argument("dataset-path", type=click.Path())
@click.argument("outer-folds", type=int)
@click.argument("inner-folds", type=int)
@click.argument("output-file", type=click.Path())
def split_for_ncv(
    dataset_path: str, outer_folds: int, inner_folds: int, output_file: str
):
    """
    Create fixed splits for nested cross-validation

    \b
    Args:\b
        dataset_path: Path to whole dataset file\b
        outer_folds: Number of outer folds\b
        inner_folds: Number of inner folds\b
        output_file: Output file\b
    \b
    Usage:\b
        python 0_create_dataset.py split-for-ncv data/processed/whole_data.pt 5 5 data/processed/ncv_split.pkl\b
    """
    dataset = torch.load(dataset_path)
    datasets = ScDataset.create_ncv_split(outer_folds, inner_folds, dataset)
    with open(output_file, "wb") as f:
        pickle.dump(datasets, f)


if __name__ == "__main__":
    cli()
