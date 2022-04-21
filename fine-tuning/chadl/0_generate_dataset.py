# -*- coding: utf-8 -*-
"""
Functions to generate ChaDL dataset
"""

# imports
import glob
import logging
import pickle
import sys
from os.path import join
from pathlib import Path
from typing import List

import click
import jsonlines
import torch
from transformers import set_seed

from helpers import SEED
from helpers.annotation_processing import process_file
from helpers.dataset_generation import prepare_dataset
from helpers.raw_processing import extract_full_text
from toolbox.datasets.ner_dataset import NerDataset

# globals

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)
logger = logging.getLogger(__name__)
set_seed(SEED)


# functions


@click.group()
def entry_point():
    pass


@entry_point.command()
@click.option("--input-directory", default="data/raw", type=click.Path(exists=True))
@click.option(
    "--output-directory", default="data/interim/extraction", type=click.Path()
)
def process_raw_files(input_directory: str, output_directory: str):
    """
    Converts the docx files into txt files.

    \b
    Args:\b
        input_directory: str = Path to input folder which contains docx files\b
        output_directory: str = Path to output folder\b
    \b
    Usage:\b
        python 0_generate_dataset.py process-raw-files --input-directory=data/raw/docs\b
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    docs = []
    for file in glob.glob(join(input_directory, "*.docx")):
        out_name = file.split("/")[-1].replace("docx", "txt")
        with open(join(output_directory, out_name), "w") as f:
            extraction = extract_full_text(file)
            f.write(extraction)

        docs.append({"text": extraction})

    with open(str(output_directory / "discharge_letters.jsonl"), "w") as f:
        jsonlines.Writer(f).write_all(docs)


@entry_point.command()
@click.argument("annotation-directory", type=click.Path(exists=True))
@click.argument(
    "output-directory",
    type=click.Path(),
    default="data/interim/processed_annotations",
)
def process_annotated_files(annotation_directory, output_directory):
    """
    Extracts text and annotations from exported XML files.\b
    \b
    Args:\b
        annotation_directory: str = Path to directory which contains annotation files (XML; INCEpTION export)\b
        output_directory: str = Path to output directory\b
    \b
    Usage:\b
        python 0_generate_dataset.py process-annotated-files --annotation-directory=data/interim/annotations\b
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    for file in glob.glob(f"{annotation_directory}/*"):
        file_name = file.split("/")[-1].replace(".xml", "")
        try:
            section = process_file(file, global_offset=False)[0]
            section.write_files(file_name, str(output_directory))
        except:
            logger.debug(f"{file} was malformed.")


@entry_point.command()
@click.argument(
    "annotation-files",
    type=click.Path(exists=True),
    default="data/interim/processed_annotations",
)
@click.argument("output-dir", type=click.Path(), default="data/processed")
@click.option("--tokenizer", default="bert-base-german-cased")
@click.option("--split_ratio", multiple=True, default=None)
def create_dataset(
    annotation_files: str,
    output_dir: str,
    tokenizer: str,
    split_ratio: List[float],
):
    """
    Create datasets from processed annotations

    \b
    Args:\b
        annotation_files: str = Path to directory that contains processed annotation files \b
        output_dir: str = Output directory \b
        tokenizer: str = Huggingface uri for tokenizer \b
        split_ratio: float = 2 or 3 floats indicating the split ratio \b
    \b
    Usage: \b
        python 0_generate_dataset.py create-dataset\b
    """
    # TODO: remove label_df part
    if len(split_ratio) != 0:
        if isinstance(split_ratio, list):
            if len(split_ratio) == 2:
                split_ratio = {"training": split_ratio[0], "testing": split_ratio[1]}
            elif len(split_ratio) == 3:
                split_ratio = {
                    "training": split_ratio[0],
                    "validation": split_ratio[1],
                    "testing": split_ratio[2],
                }
            else:
                raise Exception("Dimensions to not fit. Check split_ratio.")
        else:
            raise NotImplementedError("This type is not supported.")

    if len(split_ratio) == 0:
        return prepare_dataset(
            annotation_files,
            tokenizer,
            output_dir,
        )
    else:
        return prepare_dataset(
            annotation_files,
            tokenizer,
            output_dir,
            split_ratio,
        )


@entry_point.command()
@click.argument("dataset-path", type=click.Path(exists=True), default="data/processed")
@click.argument("outer-folds", type=int, default=5)
@click.argument("inner-folds", type=int, default=5)
@click.argument(
    "output-path", type=click.Path(), default="data/processed/ncv_dataset.pkl"
)
def split_for_ncv(
    dataset_path: str, outer_folds: int, inner_folds: int, output_path: str
):
    """
    Generate fixed splits for nested cross-validation

    \b
    Args:\b
        dataset_path: str = Path to dataset files\b
        outer_folds: int = Number of outer folds\b
        inner_folds: int = Number of inner folds\b
        output_path: str = Output directory\b
    \b
    Usage:\b
        python 0_generate_dataset.py split-for-ncv
    """

    dataset = torch.load(dataset_path)
    datasets = NerDataset.create_ncv_split(outer_folds, inner_folds, dataset)
    with open(output_path, "wb") as f:
        pickle.dump(datasets, f)


if __name__ == "__main__":
    entry_point()
