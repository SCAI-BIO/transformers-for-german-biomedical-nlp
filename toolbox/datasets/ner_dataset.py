# -*- coding: utf-8 -*-
# imports ----------------------------------------------------------------------

import copy
import itertools
import logging
import math
import pickle
import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set, Tuple, Union

import fuzzysearch
import nltk
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from tokenizers import Encoding
from toolbox.datasets import TOKENIZERS
from toolbox.utils.helpers import TrainerUtils
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

# global vars ------------------------------------------------------------------

logger = logging.getLogger(__name__)


# classes ----------------------------------------------------------------------


class LabelSet:
    """
    Handles the conversion of token labels to the respective ids and vice versa.
    """

    def __init__(
        self,
        labels: Union[Set[str], List[str], FrozenSet[str]],
        seed: int = 20210325,
        scheme: str = "BILOU",
    ):
        """
        Args:
            labels (List[str]): Requires a list of all entities
        """
        random.seed(seed)
        self._labels_to_id = {"O": 0}
        self._ids_to_label = {0: "O"}
        self._labels = labels

        iter_scheme = {
            "BILOU": "BILU",
            "BIO": "BI",
        }

        for _num, (label, s) in enumerate(
            itertools.product(labels, iter_scheme[scheme])
        ):
            num = _num + 1

            ner_label = f"{s}-{label}"
            self._labels_to_id[ner_label] = num
            self._ids_to_label[num] = ner_label

    def __eq__(self, o: "LabelSet") -> bool:
        if (
            self._labels_to_id == o._labels_to_id
            and self._ids_to_label == o._ids_to_label
            and self._labels == o._labels
        ):
            return True
        else:
            return False

    def __len__(self):
        return len(self._ids_to_label)

    def get_aligned_label_ids_from_annotations(
        self, tokenized_text: Encoding, annotations: pd.DataFrame, scheme="BILOU"
    ) -> Tensor:
        """
        Generates a tensor with the label ids for each token.

        Args:
            tokenized_text (Encoding): tokenized input
            annotations (pd.DataFrame): dataframe with the respective annotations
                                        for the input
            scheme: BILOU or BIO

        Returns:
            Tensor: Label ids for each token
        """
        raw_labels = align_annotations(tokenized_text, annotations, scheme=scheme)
        return torch.LongTensor(list(map(self._labels_to_id.get, raw_labels)))

    def save_to_txt(self, file_name: str):
        """
        Save the label_ids-label_name dict to a file.

        Args:
            file_name (str): Output file name
        """
        with open(file_name, "w") as f:
            for key, value in self._ids_to_label.items():
                f.write(f"{key}, {value}\n")

    def save(self, file_name: str):
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name: str) -> "LabelSet":
        """
        Loads the LabelSet from a previously saved object.

        Args:
            file_name (str): Input file name

        Returns:
            LabelSet
        """
        with open(file_name, "rb") as f:
            instance = pickle.load(f)
        return instance

    def ids_to_label(self, labels: np.ndarray) -> Union[List[str], List[List[str]]]:
        """
        Converts the label ids to the label name.

        Args:
            labels (np.ndarray): Array with label ids

        Returns:
            List[str]: List with label_names for each token
        """
        if labels.ndim == 2:
            return [self.ids_to_label(arr) for arr in labels]
        else:
            return [self._ids_to_label[entity_id] for entity_id in labels]


@dataclass
class NerTrainingInstance:
    """
    Class for labeled samples
    """

    text: str
    label_df: DataFrame
    encoding: Encoding = None
    encodings = None
    labels: Union[List[int], Tensor] = None
    continuation: bool = False
    schema: str = None

    max_length: int = 512
    split: bool = True

    @staticmethod
    def split_overlong(instance: "NerTrainingInstance", tokenizer: TOKENIZERS):
        tok = nltk.data.load("tokenizers/punkt/german.pickle")
        decoded_text = tokenizer.decode(
            instance.encodings[0].ids, skip_special_tokens=True
        )
        rest = ""
        for enc in instance.encodings[1:]:
            rest += tokenizer.decode(enc.ids, skip_special_tokens=True)
        sent_positions = list(tok.span_tokenize(decoded_text))
        i = len(sent_positions) - 1
        start = None
        while i >= 0:
            start, end = sent_positions[i]
            sent = decoded_text[start:end]
            match = fuzzysearch.find_near_matches(
                sent[:30], instance.text, max_l_dist=1
            )
            if len(match) == 0:
                i -= 1
                continue
            elif len(match) > 1:
                logger.warning(f"Found {len(match)} matches.")
                print(match)
                raise Exception("Found multiple matches")
            start = match[0].start
            break

        first_half = instance.text[:start]
        second_half = instance.text[start:]

        first_df = instance.label_df.loc[instance.label_df.start < start]
        second_df = instance.label_df.loc[instance.label_df.start >= start]
        second_df.loc[:, "start"] -= start
        second_df.loc[:, "end"] -= start

        first_instance = NerTrainingInstance(
            first_half,
            first_df,
            continuation=instance.continuation,
            schema=instance.schema,
        )
        second_instance = NerTrainingInstance(
            second_half,
            second_df,
            continuation=True,
            schema=instance.schema,
        )
        return first_instance, second_instance

    def process(
        self,
        tokenizer: TOKENIZERS,
        label_set: LabelSet,
        scheme: str = "BILOU",
        split: bool = None,
    ) -> Union[None, List["NerTrainingInstance"]]:
        self.schema = scheme
        self.encodings = tokenizer(
            self.text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
        ).encodings
        if (
            (split is not None and not split)
            or not self.split
            or len(self.encodings) == 1
        ):
            if len(self.encodings) > 1:
                logger.warning(f"More than 1 encoding {len(self.encodings)}.")
            self.encoding = self.encodings[0]
            self.labels = label_set.get_aligned_label_ids_from_annotations(
                self.encoding, self.label_df, scheme=scheme
            )
            if hasattr(self, "encodings"):
                del self.encodings
            return None
        else:
            logging.debug("Generated multiple encodings for sequence")
            try:
                first, second = self.split_overlong(self, tokenizer)
            except:
                print(self.encodings)
                print(self.encoding)
                print(self.split)
            first.process(tokenizer, label_set, scheme, self.split)
            self.text = first.text
            self.label_df = first.label_df
            self.labels = first.labels
            self.encoding = first.encoding
            del first

            additional_instances = [second]
            res = second.process(tokenizer, label_set, scheme, self.split)
            if res is not None:
                additional_instances.extend(res)  # type: ignore

        return additional_instances

    def __eq__(self, __o: "NerTrainingInstance") -> bool:
        if self.text == __o.text:
            return True
        else:
            return False


class NerDataset(Dataset):
    """Dataset for NER data"""

    def __init__(
        self, instances: List[NerTrainingInstance], labels_set: LabelSet = None
    ) -> None:
        self.instances = instances
        self.num_labels = len(labels_set) if labels_set is not None else None
        self.labels_set = labels_set

    def __eq__(self, __o: "NerDataset") -> bool:
        if all([x == b for x, b in zip(self.instances, __o.instances)]) and (
            self.labels_set == __o.labels_set
        ):
            return True
        else:
            return False

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.instances)

    def __add__(self, other: "NerDataset") -> "NerDataset":
        dset = copy.deepcopy(self)
        dset.instances.extend(other.instances)
        return dset

    def subset(self, indices: List[int]) -> "NerDataset":
        """
        Adds the possibility to generate of a subset of the given dataset.

        Args:
            indices (List[int]): indices of the respective samples

        Returns:
            NerDataset: Subset of the given dataset
        """
        instances = [self.instances[i] for i in indices]
        return NerDataset(instances, self.labels_set)

    def split(self, ratios: Dict[str, float]) -> Dict[str, "NerDataset"]:
        """
        Perform e.g. a train/val/test split of the given dataset.

        Args:
            ratios (Dict[str, float]): Ratios of the new subsets. The sum of the
                                       ratios should be 1.

        Returns:
            Dict[str, NerDataset]
        """
        splits = {}
        index = list(range(len(self.instances)))
        random.shuffle(index)

        for key, value in ratios.items():
            num_samples = math.ceil(value * len(self.instances))
            if num_samples > len(index):
                num_samples = len(index)

            subset_indices = []
            while len(subset_indices) < num_samples:
                subset_indices.append(index.pop())
            if len(subset_indices) == 0:
                splits[key] = None
                logger.warning("Could not split the data correctly. To few samples.")
            else:
                splits[key] = self.subset(subset_indices)

        return splits

    def __getitem__(self, item):
        """Returns the respective item from the dataset"""

        training_instance: NerTrainingInstance = self.instances[item]
        tokens = torch.tensor(training_instance.encoding.ids)
        attn = torch.tensor(training_instance.encoding.attention_mask)
        labels = training_instance.labels

        instance_dict = {
            "input_ids": tokens.long(),
            "attention_mask": attn.long(),
            "labels": labels.long(),
        }
        return instance_dict

    def retokenize(self, tokenizer: TOKENIZERS, split: bool = None):
        """Apply the tokenizer again to all instances of the dataset"""
        for instance in self.instances:
            instance.process(
                tokenizer, self.labels_set, scheme=instance.schema, split=split
            )
        logging.info("Tokenized data again.")

    def get_sequence_lengths(self, tokenizer: TOKENIZERS):
        """Get sequence lengths of all instances (raw input, words)"""
        sequence_lengths = []
        for instance in self.instances:
            encoding = tokenizer(instance.text)[0].tokens
            sequence_lengths.append(len(encoding))
        return sequence_lengths

    @staticmethod
    def create_ncv_split(
        folds: int, inner_folds: int, dataset: "NerDataset"
    ) -> List[
        Tuple["NerDataset", "NerDataset", List[Tuple["NerDataset", "NerDataset"]]]
    ]:
        """Split dataset into outer and inner folds for NCV"""

        datasets = []

        for train_idx, val_idx in TrainerUtils.calculate_splits(
            dataset, folds, shuffle=True
        ):
            train_data = dataset.subset(train_idx)
            val_data = dataset.subset(val_idx)

            inner_datasets = []
            for inner_train_idx, inner_val_idx in TrainerUtils.calculate_splits(
                train_data, inner_folds
            ):
                inner_train_data = train_data.subset(inner_train_idx)
                inner_val_data = train_data.subset(inner_val_idx)
                inner_datasets.append((inner_train_data, inner_val_data))

            datasets.append((train_data, val_data, inner_datasets))

        return datasets

    def to_dictionaries(self) -> List[Dict]:
        """Extract sentences from list of training instances"""
        annotations = []
        for instance in self.instances:
            text = instance.text
            if instance.label_df is not None and instance.label_df.shape[0] > 0:
                annotation_dict = []
                annotation_text = []
                for i, row in instance.label_df.iterrows():
                    annotation_dict.append([row["start"], row["end"], row["entity"]])
                    current_annotation_text = text[row["start"] : row["end"]]
                    annotation_text.append(current_annotation_text)
                    if "text" in row and annotation_text[-1] != row["text"]:
                        print(row)
                        raise Exception
            else:
                annotation_dict = []
                annotation_text = []
            annotations.append(
                {
                    "text": text,
                    "labels": annotation_dict,
                    "label_text": annotation_text,
                }
            )
        return annotations


# functions --------------------------------------------------------------------


def get_unique_labels(instances: List[NerTrainingInstance]) -> Set[str]:
    """Find unique labels in list of instances"""
    labels = []
    for instance in instances:
        df = instance.label_df
        if df.shape[0] > 0:
            labels.extend(df.entity.unique())

    return set(list(labels))


def align_annotations(
    encoding: Encoding, annotation_df: pd.DataFrame, scheme: str = "BILOU"
) -> List[str]:
    """Aligns the annotations with the encoding of the respective text

    Args:
        encoding (Encoding): Encoded text
        annotation_df (pd.DataFrame): Dataframe with the entity annotations (start, end, entity)
        scheme (str) : Either BILOU or BIO

    Returns:
        List[str]: List of token annotations according to the BILOU/BIO scheme
    """
    assert scheme == "BILOU" or scheme == "BIO", "scheme has to be either BILOU or BIO"
    tokens = encoding.tokens
    aligned_labels = ["O"] * len(tokens)

    if annotation_df is not None:
        for _, row in annotation_df.iterrows():
            token_idx_set = set()
            for char_idx in range(row["start"], row["end"]):

                token_idx = encoding.char_to_token(char_idx)
                if token_idx is not None:
                    token_idx_set.add(token_idx)
            if len(token_idx_set) == 1:
                token_idx = token_idx_set.pop()
                prefix = "U" if scheme == "BILOU" else "B"
                aligned_labels[token_idx] = f"{prefix}-{row.entity}"

            else:

                last_token = len(token_idx_set) - 1
                for num, token_idx in enumerate(sorted(token_idx_set)):
                    if num == 0:
                        prefix = "B"
                    elif num == last_token and scheme == "BILOU":
                        prefix = "L"
                    else:
                        prefix = "I"
                    aligned_labels[token_idx] = f"{prefix}-{row.entity}"
    return aligned_labels


def clean_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe from insecure and non-text bound annotations(BRAT FORMAT)"""
    insecure = df.loc[df.type.str.match("^A"), :].copy()
    filtered = df.loc[df.type.str.match("^T"), :].copy()
    if insecure.shape[0] == 0:
        return filtered

    insecure[["desc", "target"]] = insecure.desc.str.split(" ", expand=True)
    insecure = insecure.drop(columns="word")
    included = [row not in insecure.target.to_list() for row in filtered.type.to_list()]
    filtered = filtered.loc[included, :]
    return filtered


def multi_instance_generator(
    text: str,
    label_df: DataFrame,
    tokenizer: TOKENIZERS,
    label_set: LabelSet,
    max_length: int = 512,
    scheme: str = "BILOU",
) -> NerTrainingInstance:
    assert isinstance(
        tokenizer, BertTokenizerFast
    ), "Tokenizer must be a BERT tokenizer"
    batch_encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_overflowing_tokens=True,
        padding="max_length",
    )
    for encoding in batch_encoding.encodings:
        labels = label_set.get_aligned_label_ids_from_annotations(
            encoding, label_df, scheme=scheme
        )
        tuples = [
            offset_tuple for offset_tuple in encoding.offsets if offset_tuple != (0, 0)
        ]
        begin = tuples[0][0]
        end = tuples[-1][1]
        enc_text = text[begin:end]
        if label_df is not None:
            enc_df = label_df.loc[
                (label_df["start"] >= begin) & (label_df["end"] <= end), :
            ].copy()
            enc_df["start"] -= begin
            enc_df["end"] -= begin
        else:
            enc_df = None
        instance = NerTrainingInstance(
            text=enc_text,
            label_df=enc_df,
            encoding=encoding,
            labels=labels,
            max_length=max_length,
            continuation=True if begin > 0 else False,
            schema=scheme,
            split=False,
        )
        yield instance
