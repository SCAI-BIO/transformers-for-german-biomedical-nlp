# -*- coding: utf-8 -*-
import dataclasses
import logging
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Union

import pandas as pd
import toolbox
import torch
import typer
from tqdm import tqdm
from transformers import BertTokenizerFast

# globals

logger = logging.getLogger(__name__)
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

# functions


def extract_code_list(code_string: Union[List[str], str]) -> List[str]:
    if isinstance(code_string, list):
        if len(code_string) > 1:
            raise Exception("Non unique annotation")
        if len(code_string) == 0:
            return []
        else:
            code_string = code_string[0]

    if len(code_string) > 0:
        if "|" in code_string:
            return code_string.split("|")
        else:
            return [code_string]
    else:
        return []


# class definitions


@dataclasses.dataclass
class Document:

    doc_id: int
    title: str
    goal: str
    harms: str
    replacement: str
    reduction: str
    refinement: str
    labels: List[str]
    added_labels = None

    @classmethod
    def read_file(cls, input_file: Path):
        doc_id = int((str(input_file).split("/")[-1]).replace(".txt", ""))
        with input_file.open("r") as f:
            return cls(doc_id, *f.read().splitlines(), [])

    def assign_labels(self, labels: List[str]):
        self.labels.extend(labels)
        self.added_labels = True

    def __str__(self) -> str:
        return f"""
        {self.title} ({self.doc_id})
        ---
        Goal: {self.goal}
        Harms: {self.harms}
        Replacement: {self.replacement}
        Reduction: {self.reduction}
        Refinement: {self.refinement}
        ---
        Labels: {", ".join(self.labels) if self.added_labels else "Not added"}
        """

    def convert_to_sc_instance(self) -> toolbox.datasets.sc_dataset.Document:
        text = "\n".join(
            [
                self.title,
                self.goal,
                self.harms,
                self.replacement,
                self.reduction,
                self.refinement,
            ]
        )
        return toolbox.datasets.sc_dataset.Document(
            text=text, doc_id=self.doc_id, label_names=self.labels
        )


#

app = typer.Typer()


@app.command()
def process_raw_data(
    input_filepath: Path = typer.Argument("data/raw/"),
    output_filepath: Path = typer.Argument("data/processed"),
    tokenizer_uri: str = typer.Option(
        "../bronco/data/external/bert-base-german-cased/"
    ),
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.info("making final data set from raw data")
    output_filepath.mkdir(exist_ok=True, parents=True)

    training_dir = input_filepath / "docs-training"
    test_dir = input_filepath / "docs"

    ann_train_dev = pd.read_csv(
        input_filepath / "anns_train_dev.txt",
        sep="\t",
        header=None,
        names=["id", "codes"],
    )
    ann_test = pd.read_csv(
        input_filepath / "anns_test.txt",
        sep="\t",
        header=None,
        names=["id", "codes"],
    )
    annotations = pd.concat([ann_train_dev, ann_test])

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_uri)

    train_ids = frozenset(
        [
            int(x)
            for x in (input_filepath / "ids_training.txt").open("r").read().splitlines()
        ]
    )
    dev_ids = frozenset(
        [
            int(x)
            for x in (input_filepath / "ids_development.txt")
            .open("r")
            .read()
            .splitlines()
        ]
    )
    test_ids = frozenset(
        [
            int(x)
            for x in (input_filepath / "ids_test.txt").open("r").read().splitlines()
        ]
    )

    files = [
        x for x in training_dir.glob("**/*") if x.is_file() and "id.txt" not in str(x)
    ]
    files.extend(
        [x for x in test_dir.glob("**/*") if x.is_file() and "id.txt" not in str(x)]
    )
    documents = []
    for file in tqdm(files):
        doc = Document.read_file(file)
        doc.assign_labels(
            extract_code_list(annotations.codes[annotations.id == doc.doc_id].tolist())
        )
        documents.append(doc.convert_to_sc_instance())

    train_documents = [x for x in documents if x.doc_id in train_ids]
    dev_documents = [x for x in documents if x.doc_id in dev_ids]
    test_documents = [x for x in documents if x.doc_id in test_ids]

    train_dataset = toolbox.datasets.ScDataset(train_documents)
    dev_dataset = toolbox.datasets.ScDataset(dev_documents)
    test_dataset = toolbox.datasets.ScDataset(test_documents)

    full_dataset = deepcopy(train_dataset)
    full_dataset += dev_dataset
    full_dataset += test_dataset
    full_dataset.generate_label_set()

    full_labels = full_dataset.label_set
    logger.info(f"Label set with {len(full_labels)} codes.")

    train_dataset.prepare_data(tokenizer, 512, label_set=full_labels)
    dev_dataset.prepare_data(tokenizer, 512, label_set=full_labels)
    test_dataset.prepare_data(tokenizer, 512, label_set=full_labels)

    logger.info(f"Full: {len(full_dataset)}, {len(full_dataset.label_set)}")
    logger.info(f"Train: {len(train_dataset)}, {len(train_dataset.label_set)}")
    logger.info(f"Dev: {len(dev_dataset)}, {len(dev_dataset.label_set)}")
    logger.info(f"Test: {len(test_dataset)}, {len(test_dataset.label_set)}")

    torch.save(train_dataset, output_filepath / "train_full.pt")
    torch.save(dev_dataset, output_filepath / "val_full.pt")
    torch.save(test_dataset, output_filepath / "test_full.pt")


if __name__ == "__main__":
    app()
