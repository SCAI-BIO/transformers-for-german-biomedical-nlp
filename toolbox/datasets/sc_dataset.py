# -*- coding: utf-8 -*-
"""Module for handling JSynCC data"""

# imports

import logging
import math
import random
import re
from copy import deepcopy
from os import path
from typing import Dict, List, Tuple

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from toolbox.datasets import TOKENIZERS
from toolbox.utils import TrainerUtils
from torch.utils.data import Dataset, Sampler

# globals


cm = 1 / 2.54  # centimeters in inches


class Document:
    """Single document out of the JSynCC Corpus"""

    def __init__(self, text: str, doc_id: str, label_names: list):
        assert isinstance(label_names, list)
        self.text: str = text
        self.doc_id: str = doc_id
        self.label_names: List[str] = [
            re.sub(r" ", "_", topic) for topic in label_names
        ]

        self.raw_tokens = None
        self.tokens = None
        self.attention_mask = None
        self.label = None
        self.label_ids = None

    def tokenize(self, tokenizer: TOKENIZERS, max_length: int) -> None:
        """Tokenizes the text from the instance"""
        batch_encoding = tokenizer(
            text=str(self.text),
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=True,
            padding="max_length",
        )

        self.tokens = []
        self.attention_mask = []
        for encoding in batch_encoding.encodings:
            self.tokens.append(encoding.ids)
            self.attention_mask.append(encoding.attention_mask)

    def set_label(self, topic_dict: Dict[str, int]) -> torch.Tensor:
        """Generates the label for this instance"""

        self.label_ids = []

        for label_name in self.label_names:
            try:
                self.label_ids.append(topic_dict[label_name])
            except KeyError:
                logging.error("Label was not found in the dictionary.")
                raise

        self.label = self._label_list_to_tensor(self.label_ids, len(topic_dict))
        return self.label

    @staticmethod
    def _label_list_to_tensor(topic_ids: list, num_classes: int) -> torch.Tensor:
        """Generates a multi-label encoding for a given list of input ids"""
        class_encoder = MultiLabelBinarizer(classes=range(num_classes))
        return torch.FloatTensor(class_encoder.fit_transform([topic_ids])).view(-1)


class ScDataset(Dataset):
    """Dataset for sequence classification data"""

    def __init__(self, instances: List[Document]):
        self.instances = instances

        self.label_set = self.generate_label_set()
        self.num_labels = len(self.label_set)

        self.min_freq = None
        self.max_length = None

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.instances)

    def __add__(self, other: "ScDataset") -> "ScDataset":
        self.instances.extend(other.instances)

        if self.label_set is None:
            self.label_set = self.generate_label_set()
            self.num_labels = len(self.label_set)
        return self

    def __getitem__(self, item):
        """Returns the respective item from the dataset"""
        tokens = torch.tensor(self.instances[item].tokens)
        attn = torch.tensor(self.instances[item].attention_mask)
        labels = self.instances[item].label
        out = {
            "input_ids": tokens,
            "attention_mask": attn,
            "labels": labels,
            "doc_id": self.instances[item].doc_id,
        }
        return out

    def subset(self, indices: List[int]) -> "ScDataset":
        docs = [self.instances[i] for i in indices]
        subset = ScDataset(docs)
        subset.label_set = self.label_set
        subset.num_labels = len(self.label_set)
        return subset

    def prepare_data(
        self,
        tokenizer: TOKENIZERS,
        max_length: int,
        min_freq: int = None,
        unique_labels: bool = False,
        label_set: Dict[str, int] = None,
    ):
        """Prepares the data for training after the object has been initialized"""
        self.min_freq = min_freq
        self.max_length = max_length

        if unique_labels:
            for instance in self.instances:
                if len(instance.label_names) > 1:  # type: ignore
                    instance.label_names = None
            self._remove_instances()
        if min_freq is not None:
            label_count = self.count_labels()
            labels_to_keep = [
                label for label, count in label_count.items() if count >= min_freq
            ]
            self._remove_labels(labels_to_keep)

        if label_set is not None:
            self.label_set = label_set
            self.num_labels = len(label_set)
        else:
            self.label_set = self.generate_label_set()
        for instance in self.instances:
            instance.tokenize(tokenizer, max_length)
            instance.set_label(self.label_set)

    def retokenize(self, tokenizer: TOKENIZERS, max_length: int):
        """Apply the tokenizer again to all instances of the dataset"""
        for instance in self.instances:
            instance.tokenize(tokenizer, max_length)
        logging.info("Tokenized data again.")

    def _remove_labels(self, topics_keep):
        """Remove labels from training instances which are not in the dict."""
        for instance in self.instances:
            to_remove = []
            for i, topic in enumerate(instance.label_names):  # type: ignore
                if topic not in topics_keep and len(instance.label_names) == 1:  # type: ignore
                    instance.label_names = None
                elif topic not in topics_keep and len(instance.label_names) > 1:  # type: ignore
                    to_remove.append(i)
            for i in reversed(to_remove):
                del instance.label_names[i]  # type: ignore
            if instance.label_names is not None and len(instance.label_names) == 0:
                instance.label_names = None
        self._remove_instances()
        logging.info("Removed the obsolete labels.")

    def _remove_instances(self):
        """Remove instances which are no longer needed."""
        to_remove = [i for i, _ in enumerate(self.instances) if _.label_names is None]
        for i in reversed(to_remove):
            del self.instances[i]
        logging.info("Removed %g instances from the dataset.", len(to_remove))

    def count_labels(self):
        """Generate a dictionary with the labels and their frequency"""
        labels = {}
        for instance in self.instances:
            for label in instance.label_names:
                if label not in labels.keys():
                    labels[label] = 1
                else:
                    labels[label] += 1

        return {
            k: v
            for k, v in sorted(labels.items(), key=lambda item: item[1], reverse=True)
        }

    def generate_label_set(self):
        """Generate a dictionary with the topic names and their ids"""
        topics = self.count_labels()
        topic_dict = {topic: i for i, topic in enumerate(topics)}
        self.label_set = topic_dict
        self.num_labels = len(topic_dict)
        return topic_dict

    def split_data(self, ratios: List[float] = [0.7, 0.1, 0.2]):
        """Split the dataset into training, validation, and testing."""
        assert sum(ratios) == 1.0

        dset = deepcopy(self)
        subsets = []
        for topic in dset.label_set.keys():
            tmp = self.make_subsets(dset, topic)
            if len(tmp) > 0:
                if isinstance(tmp, ScDataset):
                    subsets.append(tmp)
                elif isinstance(tmp, list):
                    subsets.extend(tmp)

        training, validation, testing = self.split_subset(subsets, ratios)  # type: ignore
        logging.info("Collected %g samples for training", len(training))
        logging.info("Collected %g samples for validation", len(validation))
        logging.info("Collected %g samples for testing", len(testing))

        return training, validation, testing

    def save_datasets(
        self, output_directory: str = "./data", ratios: List[float] = [0.7, 0.1, 0.2]
    ):
        """Save all datasets to the respective output directory"""
        train, val, test = self.split_data(ratios)
        torch.save(train, path.join(output_directory, "training_data.pt"))
        torch.save(val, path.join(output_directory, "validation_data.pt"))
        torch.save(test, path.join(output_directory, "testing_data.pt"))

    def _subset(self, label_name):
        """Make subset for one topic"""
        ids = []
        for i, instance in enumerate(self.instances):
            if label_name in instance.label_names:
                ids.append(i)
        sub = [self.instances.pop(i) for i in reversed(ids)]
        return ScDataset(sub)

    def make_subsets(self, dataset, topic_name: str):
        orig_len = len(dataset)
        subset = dataset._subset(topic_name)
        if orig_len == len(subset):
            return subset
        labels = subset.count_labels()
        if len(labels.keys()) > 1:
            subset_list = [subset]
            labels = list(subset.count_labels().keys())
            del labels[0]
            for label in labels:
                subsubset = self.make_subsets(subset, label)
                if isinstance(subsubset, ScDataset):
                    subset_list.append(subsubset)
                else:
                    subset_list.extend(subsubset)
            subset_list = [sub for sub in subset_list if len(sub) > 0]
            return subset_list
        else:
            return subset

    @staticmethod
    def split_subset(
        subset: List, ratios: Tuple[float, float, float] = (0.75, 0.1, 0.15)
    ):
        training = []
        validation = []
        testing = []

        rnd = random.Random()

        for sub in subset:
            num_val = ratios[1] * len(sub)
            num_test = ratios[2] * len(sub)

            count_val = 0
            while count_val < num_val:
                validation.append(sub.instances.pop(rnd.randint(0, len(sub) - 1)))
                count_val += 1

            count_test = 0
            while count_test < num_test:
                testing.append(sub.instances.pop(rnd.randint(0, len(sub) - 1)))
                count_test += 1

            training.extend(sub.instances)

        return (
            ScDataset(training),
            ScDataset(validation),
            ScDataset(testing),
        )

    @staticmethod
    def create_ncv_split(
        folds: int, inner_folds: int, dataset: "ScDataset"
    ) -> List[Tuple["ScDataset", "ScDataset", List[Tuple["ScDataset", "ScDataset"]]]]:
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
        instances = []
        for instance in self.instances:
            labels = instance.label_names
            text = instance.text
            instances.append(
                {
                    "text": text,
                    "labels": labels,
                }
            )
        return instances


class FixedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.len_per_doc = self.determine_length_per_doc(dataset, range(len(dataset)))

    @staticmethod
    def determine_length_per_doc(dataset, indices):
        return torch.tensor([dataset[idx]["input_ids"].shape[0] for idx in indices])

    def __iter__(self):
        indices = list(torch.randperm(len(self.dataset)))
        len_per_doc = list(self.determine_length_per_doc(self.dataset, indices))

        num_sequences = sum(len_per_doc)
        num_batches = float(num_sequences) / int(self.batch_size)

        while num_batches > 0:
            error = False
            sampled = []
            len_of_samples = 0
            if sum(len_per_doc) < self.batch_size:
                sampled = indices
            else:
                i = 0
                while len_of_samples < self.batch_size:
                    if len(indices) == 0 or i > (len(indices) - 1):
                        error = True
                        break
                    dataset_idx = indices[i]
                    sample_len = len_per_doc[i]
                    if (len_of_samples + sample_len) <= self.batch_size:
                        sampled.append(dataset_idx)
                        len_of_samples += self.dataset[dataset_idx]["input_ids"].shape[
                            0
                        ]
                        del indices[i]
                        del len_per_doc[i]
                    else:
                        i += 1
            if not error:
                yield sampled
            num_batches -= 1

    def __len__(self):
        return math.ceil(float(sum(self.len_per_doc)) / self.batch_size)


def collate_multi(batch):
    batch_tokens = []
    batch_attentions = []
    batch_ids = []
    batch_labels = []
    batch_dids = []

    for i, sample in enumerate(batch):
        batch_tokens.append(sample["input_ids"])
        batch_attentions.append(sample["attention_mask"])
        batch_ids.extend([i] * len(sample["input_ids"]))
        batch_labels.append(sample["labels"])
        batch_dids.extend([int(sample["doc_id"])] * len(sample["input_ids"]))

    out_tokens = torch.cat(batch_tokens, dim=0)
    out_attentions = torch.cat(batch_attentions, dim=0)
    out_ids = torch.tensor(batch_ids)
    out_labels = torch.stack(batch_labels)
    torch.tensor(batch_dids)

    return {
        "input_ids": out_tokens,
        "attention_mask": out_attentions,
        "bag_ids": out_ids,
        "labels": out_labels,
    }
