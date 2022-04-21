# -*- coding: utf-8 -*-

# imports
import glob
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import List, Union

import click
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# globals
from toolbox.datasets.ner_dataset import (
    LabelSet,
    NerDataset,
    NerTrainingInstance,
    clean_annotations,
    multi_instance_generator,
)

logger = logging.getLogger(__name__)
ENTITIES = frozenset(
    [
        "Living_Beings",
        "Disorders",
        "Procedures",
        "Physiology",
        "Anatomical_Structure",
        "Devices",
        "Chemicals_Drugs",
        "TNM",
    ]
)
SEED = 20201124


# functions


def read_annotations(file: str) -> Union[pd.DataFrame, None]:
    """Read annotations from file and returns pandas DataFrame object"""

    df = pd.read_csv(file, sep="\t", names=["type", "desc", "word"])
    if df.shape[0] == 0:
        return None

    # drop na entries
    df.dropna()

    # keep only text-bound annotations
    df = clean_annotations(df)

    if df.shape[0] == 0:
        return None

    # unpack the entries
    df[["entity", "start", "end"]] = df.desc.str.split(" ", expand=True)
    df = df.astype({"start": int, "end": int})

    # remove genes (not included in the GGPONC paper)
    df = df[df.entity != "Genes"]

    return df.sort_values(by=["start", "end"])


def generate_training_instance(file_path: str) -> NerTrainingInstance:
    """Generate TrainingInstance for a given txt/ann combination"""

    ann_path = file_path.replace("txt", "ann")
    assert os.path.exists(ann_path)
    annotations = read_annotations(ann_path)

    with open(file_path, "r") as f:
        segment = f.read()

    return NerTrainingInstance(text=segment, label_df=annotations)


def generate_training_instances_with_overflow(
    file_path: str,
    tokenizer=PreTrainedTokenizerFast,
    label_set=LabelSet,
    max_length: int = 512,
) -> List[NerTrainingInstance]:
    """
    Generate multiple training instances if sequence is too long

    \b
    Args:\b
        file_path: Path to the text file\b
        tokenizer: Name or path of the tokenizer\b
        label_set: LabelSet instance\b
        max_length: Maximum sequence length\b
    """

    ann_path = file_path.replace("txt", "ann")
    assert os.path.exists(ann_path), "Annotation file does not exists"
    annotations = read_annotations(ann_path)

    with open(file_path, "r") as f:
        segment = f.read()

    return [
        instance
        for instance in multi_instance_generator(  # type: ignore
            segment, annotations, tokenizer, label_set, max_length  # type: ignore
        )
    ]


# main


@click.command()
@click.option("--output-directory", default="data/processed", type=click.Path())
@click.option("--tokenizer-string", default="bert-base-german-cased", type=str)
@click.option(
    "--label-set-path", default="data/processed/label_set.pt", type=click.Path()
)
@click.option("--labeled-input", default="data/raw/Evaluation/Gold", type=click.Path())
@click.option("--split-training-instances", is_flag=True)
def generate_dataset(
    output_directory: str,
    tokenizer_string: str,
    label_set_path: str,
    labeled_input: str,
    split_training_instances: bool,
):
    """
    Generates the NerDataset for GGPONC

    \b
    Args:\b
        output_directory: Output directory\b
        tokenizer_string: Name or path of the tokenizer\b
        label_set_path: Path where the LabelSet instance should be saved\b
        labeled_input: Path of the directory that contains the gold standard annotations\b
        split_training_instances: Specify whether instances should be split\b
    \b
    Usage:\b
        python 0_generate_dataset.py --labeled-input=data/raw/Gold\b
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    transformers.set_seed(SEED)
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_string, use_fast=True)
    label_set = LabelSet(ENTITIES, SEED)
    label_set.save(label_set_path)

    hash_strings = []
    instances: List[NerTrainingInstance] = []
    for guideline in os.listdir(labeled_input):
        guideline_path = os.path.join(labeled_input, guideline)
        logging.debug(f"Reading files in {guideline_path}")
        for file in glob.glob(guideline_path + "/*.txt"):
            if not split_training_instances:
                instances.append(generate_training_instance(file))
            else:
                instances.extend(
                    generate_training_instances_with_overflow(
                        file_path=file,
                        tokenizer=tokenizer,
                        label_set=label_set,  # type: ignore
                        max_length=512,
                    )
                )

    for instance in instances:
        hash_strings.append(hashlib.sha1(instance.text.encode()).digest())
        if not split_training_instances:
            instance.process(tokenizer, label_set)

    dataset = NerDataset(instances, label_set)
    torch.save(dataset, str(output_directory / "whole_data.pt"))

    logger.info(f"Whole dataset: {len(dataset)}")
    splits = dataset.split({"training": 0.7, "validation": 0.1, "testing": 0.2})
    for key, subset in splits.items():
        logger.info(f"{key}: {len(subset)}")
        torch.save(subset, str(output_directory / f"{key}_data.pt"))
    logger.info(f"Generated {len(hash_strings)} hash strings.")


if __name__ == "__main__":
    generate_dataset()
