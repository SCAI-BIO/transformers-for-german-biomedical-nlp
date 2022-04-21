# -*- coding: utf-8 -*-
# imports
import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import torch
from toolbox.datasets.ner_dataset import (
    LabelSet,
    NerDataset,
    NerTrainingInstance,
    get_unique_labels,
)
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed

# globals
logger = logging.getLogger(__name__)

set_seed(20211223)


# functions


def prepare_dataset(
    input_dir: str,
    tokenizer: str,
    output_dir: str,
    split_ratios: Dict[str, float] = {
        "training": 0.7,
        "validation": 0.1,
        "testing": 0.2,
    },
    split_instances: bool = False,
):
    """
    Generate datasets from annotation files

    Args:
        input_dir (str): Path to the annotation files
        tokenizer (PreTrainedTokenizerFast): Name or path of the tokenizer
        output_dir (str): Path to the output files
        split_ratios (Dict[str, int]): Dictionary with split names and ratios
        split_instances (bool): Indicate whether instances should be split if
                                they are longer than the maximum sequence length

    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    except:
        logger.error("Tokenizer could not be loaded. Check path or model name.")
        raise

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = []
    for file_dir in sorted([x[0] for x in os.walk(input_dir)]):
        logger.debug(f"Process {file_dir}")
        files = sorted(glob(file_dir + "/*.txt"))
        for section_txt in files:
            print("Process ", section_txt)
            try:
                instance = generate_instance(section_txt, split=split_instances)
                if instance is not None:
                    instances.append(instance)
            except FileNotFoundError:
                logger.error("Annotation file path is not correct!")
                raise

    label_list = get_unique_labels(instances)
    label_set = LabelSet(label_list)
    label_set.save(str(output_dir / "label_set.pt"))

    if split_instances:
        additional_instances = []
        for instance in instances:
            adds = instance.process(tokenizer, label_set)
            if adds is not None:
                additional_instances.extend(adds)

        logger.info(f"Generated {len(additional_instances)} by splitting.")
        instances.extend(additional_instances)
        del additional_instances

    dataset = NerDataset(instances, label_set)
    torch.save(dataset, str(output_dir / "whole_dataset.pt"))

    instance_text = []
    with open(str(output_dir / "whole_documents.json"), "w+") as f:
        for instance in dataset.instances:
            instance_text.append(
                {"text": instance.text, "continuation": instance.continuation}
            )
        f.write(json.dumps(instance_text))

    for i, instance in enumerate(dataset.instances):
        instance.label_df.to_csv(
            str(output_dir / f"label-df_{i}.csv"), index=False, header=True
        )

    if split_ratios is not None:
        logger.info("Generate splits.")
        splits = dataset.split(split_ratios)
        for key, subset in splits.items():
            if subset is not None:
                instance_text = []
                with open(str(output_dir / f"{key}_documents.json"), "w+") as f:
                    for instance in subset.instances:
                        instance_text.append(
                            {
                                "text": instance.text,
                                "continuation": instance.continuation,
                            }
                        )
                    f.write(json.dumps(instance_text))
                torch.save(subset, os.path.join(output_dir, f"{key}_dataset.pt"))
    logger.info("Finished dataset generation.")


def generate_instance(
    section_txt: str,
    text_file_str: str = "text",
    annotation_file_str: str = "annotations",
    split: bool = True,
) -> Union[NerTrainingInstance, None]:
    """Generate instance for given text and annotation file"""
    logger.info(f"Process {section_txt}")
    section_csv = section_txt.replace(text_file_str, annotation_file_str).replace(
        "txt", "csv"
    )
    try:
        annotations = pd.read_csv(section_csv)
    except FileNotFoundError:
        logger.warning(
            f"There does not seem to be an annotation file for {section_txt}"
        )
        return None
    text = Path(section_txt).read_text()
    logger.debug(f"Read files {section_txt} and {section_csv}. Generate instance.")
    logger.debug(annotations.shape)
    return NerTrainingInstance(text, annotations, split=split, schema="BILOU")
