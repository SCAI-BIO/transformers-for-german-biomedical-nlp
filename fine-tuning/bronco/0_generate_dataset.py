# -*- coding: utf-8 -*-

# imports

import logging
from collections import namedtuple
from copy import deepcopy
from glob import glob
from pathlib import Path

import click
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer

from toolbox.datasets.ner_dataset import LabelSet, NerDataset, NerTrainingInstance

# globals


# classes

Annotation = namedtuple("Annotation", ["label", "text", "begin", "end"])


class Sentence(object):
    """Helper class store text and annotations for sentences in the BRONCO150 dataset"""

    def __init__(self, sentence_id: int):
        super().__init__()

        self.id = sentence_id
        self.text = ""
        self.annotations = []

    def add_text(self, text: str):
        start = len(self.text)
        self.text += text + " "
        return start, len(self.text) - 1

    def add_annotation(self, annotation: Annotation):
        self.annotations.append(annotation)

    def get_instance(self, max_length: int = 512):
        """Get a LabeledTrainingInstance"""
        df = pd.DataFrame(
            [
                {"start": anno.begin, "end": anno.end, "entity": anno.label}
                for anno in self.annotations
            ]
        )
        return NerTrainingInstance(self.text, df, max_length=max_length)

    def __str__(self):
        out = "\n"
        for anno in self.annotations:
            out += f"\t\t{anno.begin}, {anno.end}: {anno.label}\t{anno.text}\n"
        return f"""
        Document:
        {self.text.strip()}

        Annotations:
        {out}
        """


class AnnotationHelper(object):
    """Helper class to gather annotations"""

    def __init__(self, label):
        super().__init__()

        self.label = label
        self.text = []
        self.begin = None
        self.end = None

    def add_position(self, start, end, text):
        """Extend current annotation"""
        if len(self.text) == 0:
            self.begin = start
            self.end = end
        else:
            self.end = end

        self.text.append(text)

    def get_annotation_instance(self):
        """Get annotation instance"""
        return Annotation(self.label, " ".join(self.text), self.begin, self.end)


# functions


@click.command()
@click.argument("input-filepath", type=click.Path(exists=True))
@click.argument("output-filepath", type=click.Path())
@click.option("--seed", default=202107, type=int)
@click.option("--max-length", default=512, type=int)
@click.option("--default-tokenizer", default="bert-base-german-cased", type=str)
def main(
    input_filepath: str,
    output_filepath: str,
    seed: int,
    max_length: int,
    default_tokenizer: str,
):
    """
    Convert raw CONLL files into dataset for NER tasks.

    \b
    Args:\b
        input_filepath: str = Path to directory that contains CONLL files\b
        output_filepath: str = Path to the directory in which the dataset files should be saved\b
        seed: int = Seed\b
        max_length: int = maximum sequence length\b
        default_tokenizer: str = huggingface uri for the default tokenizer\b

    Usage:\b
        python 0_generate_dataset.py data/raw/BRONCO150/conllIOBTags data/processed\b
    """
    transformers.set_seed(seed)

    logger = logging.getLogger(__name__)
    logger.info("Generate dataset from raw data")
    logger.info(f"Input: {input_filepath}")
    logger.info(f"Output: {output_filepath}")

    output_directory = Path(output_filepath)
    output_directory.mkdir(parents=True, exist_ok=True)

    documents = {}
    for file in glob(f"{input_filepath}/*.CONLL"):
        logger.info(f"Read {file}")
        sentences = []
        with open(file, "r") as f:
            i = 0
            current_annotation = None
            sentence = Sentence(i)
            for line in tqdm(f.readlines(), desc="Processing file"):
                line = line.replace("\n", "")
                if len(line) > 0:
                    split = line.split("\t")
                    token = split[0]
                    start, end = sentence.add_text(token)

                    label = split[-1]
                    tag = label[0]
                    name = label[2:]

                    if tag == "B" and current_annotation is None:
                        current_annotation = AnnotationHelper(name)
                        current_annotation.add_position(start, end, token)
                    elif tag == "B" and current_annotation is not None:
                        sentence.add_annotation(
                            current_annotation.get_annotation_instance()
                        )
                        current_annotation = AnnotationHelper(name)
                        current_annotation.add_position(start, end, token)
                    elif current_annotation is not None and tag == "I":
                        current_annotation.add_position(start, end, token)
                    elif current_annotation is not None and tag == "O":
                        sentence.add_annotation(
                            current_annotation.get_annotation_instance()
                        )
                        current_annotation = None
                    else:
                        pass
                else:
                    current_annotation = None
                    i += 1
                    sentences.append(sentence)
                    sentence = Sentence(i)
            documents[file.split("/")[-1]] = sentences

    for key, content in documents.items():
        with open(str(output_directory / key.replace("CONLL", "txt")), "w+") as f:
            for sentence in content:
                f.write(f"{sentence.text}\n")

    logger.info("Gather labels")
    labels = []
    for split in documents.values():
        for sample in tqdm(split):
            for current_annotation in sample.annotations:
                if current_annotation.label not in labels:
                    labels.append(current_annotation.label)
    label_set = LabelSet(labels)
    label_set.save_to_txt(str(output_directory / "label_set.txt"))
    label_set.save(str(output_directory / "label_set.pt"))

    tokenizer = AutoTokenizer.from_pretrained(default_tokenizer)

    logger.info("Process instances")
    logger.info(f"Generate instances with {max_length} tokens.")
    instances = {}
    for key, sentences in documents.items():
        instances[key] = []
        for sentence in tqdm(sentences, desc="Generating instances"):
            sentence_instance = sentence.get_instance(max_length=max_length)
            sentence_instance.process(tokenizer=tokenizer, label_set=label_set)
            instances[key].append(sentence_instance)

    logger.info("Generate dataset")
    datasets = [
        NerDataset(instances=samples, labels_set=label_set)
        for samples in instances.values()
    ]

    # save datasets according to the predefined splits
    no_samples = 0
    for i, ds in enumerate(datasets):
        out_file = f"bronco-dataset_split{i}.pt"
        out_path = str(output_directory / out_file)
        logger.info(f"Saving dataset to {out_path}")
        no_samples += len(ds)
        torch.save(ds, out_path)
    logger.info(f"The full dataset has {no_samples} samples")

    # use first three datasets as training data, and split 4 and 5 as validation and testing splits respectively
    train = deepcopy(datasets[0])
    train += datasets[1]
    train += datasets[2]
    val = datasets[3]
    test = datasets[4]

    torch.save(train, str(output_directory / "training_data.pt"))
    torch.save(val, str(output_directory / "validation_data.pt"))
    torch.save(test, str(output_directory / "testing_data.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
