# -*- coding: utf-8 -*-
"""
    This file contains all functions needed to handle the pretraining data
"""

# imports
import collections
import logging
import math
import random
from concurrent import futures
from typing import Dict, List, Tuple, Union

import torch

# Class definitions
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast, ElectraTokenizerFast, RobertaTokenizerFast

# classes
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


class TrainingInstance(object):
    """
    Contains required information for training for one sample.
    """

    def __init__(
        self, tokens, ids, segment_ids, attention_mask, masked_lm_labels, is_random_next
    ):
        self.tokens = tokens  # tokens
        self.ids = ids  # token ids
        self.segment_ids = segment_ids  # A or B
        self.attention_mask = attention_mask  # info about padding
        if is_random_next:
            self.next_sentence_label = 1
        else:
            self.next_sentence_label = 0
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        for tok in self.tokens:
            s += tok + " "
        return s

    def __repr__(self):
        return self.__str__()


class BertDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for BERT training.
    Returns all information for MLM and NSP training.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels,
        next_sentence_labels=None,
        use_nsp: bool = True,
    ):
        num = input_ids.shape[0]
        assert num == attention_mask.shape[0]
        assert num == token_type_ids.shape[0]
        assert num == labels.shape[0]
        if use_nsp:
            assert num == len(next_sentence_labels)
        assert input_ids.shape[1] == token_type_ids.shape[1]
        assert input_ids.shape[1] == labels.shape[1]
        assert input_ids.shape[1] == labels.shape[1]

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.next_sentence_labels = next_sentence_labels
        logging.info("Initialized BertDataset object.")

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx, :],
            "attention_mask": self.attention_mask[idx, :],
            "token_type_ids": self.token_type_ids[idx, :],
            "labels": self.labels[idx, :],
            "next_sentence_label": self.next_sentence_labels[idx],
        }
        return item

    def __len__(self):
        return self.labels.shape[0]


class RobertaDataset(BertDataset):
    """
    Pytorch dataset for RoBERTa training
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        super().__init__(
            input_ids, attention_mask, token_type_ids, labels, use_nsp=False
        )

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx, :],
            "attention_mask": self.attention_mask[idx, :],
            "labels": self.labels[idx, :],
        }
        return item

    @classmethod
    def from_bert_dataset(cls, dataset: Dataset) -> "RobertaDataset":
        if not isinstance(dataset, BertDataset):
            raise NotImplementedError

        input_ids = dataset.input_ids
        attention_masks = dataset.attention_mask
        token_type_ids = dataset.token_type_ids
        labels = dataset.labels

        return RobertaDataset(input_ids, attention_masks, token_type_ids, labels)


class ElectraDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for ELECTRA training.
    Returns all information for MLM training.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        num = input_ids.shape[0]
        assert num == attention_mask.shape[0]
        assert num == token_type_ids.shape[0]
        assert num == labels.shape[0]
        assert input_ids.shape[1] == token_type_ids.shape[1]
        assert input_ids.shape[1] == labels.shape[1]
        assert input_ids.shape[1] == labels.shape[1]

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        logging.info("Initialized ElectraDataset object.")

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx, :],
            "attention_mask": self.attention_mask[idx, :],
            "token_type_ids": self.token_type_ids[idx, :],
            "labels": self.labels[idx, :],
        }
        return item

    def __len__(self):
        return self.labels.shape[0]


DATASETS = Union[ElectraDataset, BertDataset, RobertaDataset]


# functions


def process_documents(
    files: List[str],
    tokenizer: Union[BertTokenizerFast, ElectraTokenizerFast, RobertaTokenizerFast],
    max_seq_len: int,
    short_seq_prob: float,
    masked_lm_prob: float,
    max_predictions_per_seq: float,
    vocab_words: Dict,
    rng: random.Random,
    threads: int,
):
    """
    Processes all input files and returns a list of training instances.
    Mainly based on create_pretraining_data.py from the official BERT GitHub repository.

    :param files: List of paths to input files
    :param tokenizer: BertTokenizerFast object
    :param max_seq_len: maximum length of a sequence (the rest is padded)
    :param short_seq_prob: probability of short sequences to avoid bias
    :param masked_lm_prob: probability of masking
    :param max_predictions_per_seq: maximum number of tokens that have been masked
    :param vocab_words: list of tokens in the vocab
    :param rng: random number generator
    :param threads: number of threads
    :return: List of TrainingInstances
    """
    documents = [[]]
    for file in tqdm(files, desc="Processing files"):
        start_doc_count = len(documents)
        logging.debug(f"Start processing document {file}.")
        with open(file, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()

                if not line:
                    documents.append([])
                else:
                    documents[-1].append(
                        tokenizer.convert_ids_to_tokens(
                            tokenizer.encode(line, add_special_tokens=False)
                        )
                    )
        logging.debug(
            f"Found {len(documents) - start_doc_count} documents in file {file}."
        )

    # Remove empty documents
    documents = [x for x in documents if x]

    # Introduce randomness
    random.shuffle(documents)

    # Create training sequences
    instances = []

    e = futures.ThreadPoolExecutor(threads)
    for document_index in range(len(documents)):
        future = e.submit(
            create_instances_from_document,
            documents,
            document_index,
            max_seq_len,
            short_seq_prob,
            masked_lm_prob,
            max_predictions_per_seq,
            rng,
            vocab_words,
            tokenizer,
        )
        instances.extend(future.result())

    # Introduce further randomness
    random.shuffle(instances)

    return instances


def create_instances_from_document(
    documents: List[List[int]],
    document_index: int,
    max_seq_len: int,
    short_seq_prob: float,
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    rng: random.Random,
    vocab_words: Dict,
    tokenizer: Union[BertTokenizerFast, ElectraTokenizerFast],
):
    """
    Takes a list of documents (List[str]) and an index to returns TrainingInstances

    :param documents: List of documents (List[str]; list of sentences)
    :param document_index: index of the current document
    :param max_seq_len: maximum length of a sequence (the rest is padded)
    :param short_seq_prob: probability of short sequences to avoid bias
    :param masked_lm_prob: probability of masking
    :param max_predictions_per_seq: maximum number of tokens that have been masked
    :param vocab_words: list of tokens in the vocab
    :param rng: random number generator
    :param tokenizer: initialized BERT tokenizer
    :return: List[TrainingInstance]
    """
    logging.debug(f"Load document {document_index}.")
    current_document = documents[document_index]

    # Account for special tokens
    max_num_tokens = max_seq_len - 3
    target_seq_length = max_num_tokens

    # Create shorter sequences in some cases (still padded to max_len)
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    logging.debug(f"Try to find a sequence with {target_seq_length} tokens.")

    # Create training instances for the whole document
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(current_document):
        logging.debug(f"Add document {i} to current chunk.")
        segment = current_document[i]
        current_chunk.append(segment)
        current_length += len(segment)  # type: ignore

        # Proceed if the target length is reached; go to next sentence if not
        if i == len(current_document) - 1 or current_length >= target_seq_length:
            if current_chunk:

                # Randomly select the last sentence in sequence A
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])  # type: ignore

                # Continue with the elements from the current chunk or select random elements for sequence B
                # Select a random document for sequence B if document[i] contains only one sentence
                tokens_b = []
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    logging.debug("Randomly extend sequence A.")
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Avoid i==j --> repeat until another document is selected
                    random_document_index = None
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(documents) - 1)
                        if random_document_index != document_index:
                            break
                        else:
                            logging.debug("Same document selected. Try again.")

                    logging.debug(f"Select {random_document_index} as counterpart.")
                    random_document = documents[random_document_index]

                    # Select a random start sentence to add randomness
                    random_start = rng.randint(0, len(random_document) - 1)

                    # Add sentences until target length is reached
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])  # type: ignore
                        if len(tokens_b) >= target_b_length:
                            break

                    # Give the unused sentences back to the pool to avoid loosing training data
                    logging.debug("Return unused sentences.")
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    # Continue with the elements from the current chunk --> continuous sequence
                    logging.debug("Extend sequence A continuously in sequence B.")
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])  # type: ignore

                # Clip sequence to desired length
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                # Assert that both sequences are not empty
                if len(tokens_a) < 1:
                    i += 1
                    current_chunk = []
                    current_length = 0
                    logging.warning(
                        f"Sequence A does not contain enough tokens. "
                        f"Skip sentences {i - current_length} to {i}."
                    )
                    continue
                if len(tokens_b) < 1:
                    i -= current_length
                    current_chunk = []
                    current_length = 0
                    logging.warning(
                        f"Sequence B does not contain enough tokens. Retry for sentence {i}."
                    )
                    continue

                # Create the full sequence for training
                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)

                tokens.append("[SEP]")
                segment_ids.append(1)

                logging.debug("Combined sequences.")

                # Mask the sequence
                (
                    tokens,
                    masked_lm_positions,
                    masked_lm_tokens,
                ) = create_masked_lm_predictions(
                    tokens,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    vocab_words,
                    rng,
                    whole_word_mask=False,
                )

                # Create the mask needed for the trainer function
                masked_lm_labels = create_mlm_mask(
                    masked_lm_positions, masked_lm_tokens, max_seq_len, tokenizer  # type: ignore
                )

                # Add padding if the length of the sequence is smaller than the target
                tokens, attention_mask, segment_ids = add_padding(
                    tokens, segment_ids, max_seq_len
                )

                # Init a TrainingInstance object and add it to the list
                instance = TrainingInstance(
                    tokens=tokens,
                    ids=tokenizer.convert_tokens_to_ids(tokens),
                    segment_ids=segment_ids,
                    attention_mask=attention_mask,
                    is_random_next=is_random_next,
                    masked_lm_labels=masked_lm_labels,
                )
                instances.append(instance)

            # Reset for next iteration
            current_chunk = []
            current_length = 0

        i += 1

    logging.debug(
        f"Generated {len(instances)} training instances for document {document_index}."
    )
    return instances


def add_padding(tokens: List[str], segment_ids: List[int], max_seq_len: int):
    """
    Adds padding to token list and returns attention mask.

    :param tokens: List of tokens
    :param segment_ids: List of [0,1] that indicate membership to sequence A or B
    :param max_seq_len: length of sequences
    :return: list of tokens, attention mask (indicates which indices are padding), and segment ids
    """
    attention_mask = [1] * len(tokens)
    if len(tokens) < max_seq_len:
        diff = max_seq_len - len(tokens)
        attention_mask.extend([0] * diff)
        segment_ids.extend([1] * diff)
        tokens.extend(["[PAD]"] * diff)
        logging.debug("Added padding to sequence.")

    return tokens, attention_mask, segment_ids


def create_masked_lm_predictions(
    tokens: List[str],
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    vocab_words: Dict,
    rng: random.Random,
    whole_word_mask: bool = True,
):
    """
    Masks the sequence according to the BERT paper. The code is mainly based/copied from the official BERT GitHub
    repository.

    :param tokens: List of tokens
    :param masked_lm_prob: probability of masking
    :param max_predictions_per_seq: maximum number of masked tokens per sequence
    :param vocab_words: vocabulary of the tokenizer
    :param rng: random number generator
    :param whole_word_mask: boolean; indicates whether whole words or single tokens should be masked
    :return: Tuple(List of tokens, positions that have been masked, labels of masked tokens)
    """
    # Generate list of candidates
    logging.debug("Create masked version of sequence.")
    candidate_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if whole_word_mask and len(candidate_indices) >= 1 and token.startswith("##"):
            candidate_indices[-1].append(i)
        else:
            candidate_indices.append([i])

    # Add additional randomness
    rng.shuffle(candidate_indices)

    output_tokens = list(tokens)

    num_to_predict = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )

    masked_lms = []
    covered_indices = set()
    for index_set in candidate_indices:
        if len(masked_lms) >= num_to_predict:
            break

        # Check if the addition of the candidate would exceed the number of masked tokens
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue

        is_any_index_covered = False
        for index in index_set:
            if index in covered_indices:
                is_any_index_covered = True
                break

        if is_any_index_covered:
            continue

        for index in index_set:
            covered_indices.add(index)
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = rng.sample(list(vocab_words)[5:], 1)[0]
            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_tokens = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_tokens.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_tokens


def create_mlm_mask(
    masked_lm_positions: List[int],
    masked_lm_tokens: List[str],
    max_seq_len: int,
    tokenizer: BertTokenizerFast,
) -> List[int]:
    """
    Creates a mask for the transformers' trainer. If a word is masked, the actual token id is given. If not, the value
    is -100. Tokens with this value will not be included in the masked language modelling task.

    :param masked_lm_positions: positions of the masked tokens in the sequence
    :param masked_lm_tokens: ids of the masked tokens in the same order as masked_lm_positions
    :param max_seq_len: maximum number of tokens/target length
    :param tokenizer: BertTokenizerFast object
    :return: List of integers
    """
    logging.debug("Create mlm mask for training instance.")
    masked_lm_label = [-100] * max_seq_len
    for i, label in zip(masked_lm_positions, masked_lm_tokens):
        masked_lm_label[i] = tokenizer.convert_tokens_to_ids(label)

    return masked_lm_label


def truncate_seq_pair(
    tokens_a: List[int], tokens_b: List[int], max_num_tokens: int, rng: random.Random
) -> None:
    """
    Removes tokens from the beginning/end of sequence A/B to reduce it to the target sequence length

    :param tokens_a: List of tokens from sequence A
    :param tokens_b: List of tokens from sequence B
    :param max_num_tokens: maximum number of tokens (A + B)
    :param rng: random number generator
    """
    logging.debug(
        f"Clip sequence to fit max length of {max_num_tokens} "
        f"(currently: {len(tokens_a) + len(tokens_b)})."
    )
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_datasets(
    instances: List[TrainingInstance],
    do_eval: bool = False,
    eval_size: float = 0.1,
    model_type: str = "BERT",
) -> Union[DATASETS, Tuple[DATASETS, DATASETS]]:
    """
    Creates a BertDataset object for the pretraining of a BERT model.

    Args:
        instances: List of TrainingInstances
        do_eval: indicate whether evaluation dataset should be generated
        eval_size: Fraction of data used for evaluation during pretraining
        model_type: BERT, ELECTRA, or RoBERTa

    Returns:

    """

    def make_dataset(dataset_instances: List[TrainingInstance]):
        ids = torch.LongTensor([x.ids for x in dataset_instances])
        attentions = torch.FloatTensor([x.attention_mask for x in dataset_instances])
        token_types = torch.LongTensor([x.segment_ids for x in dataset_instances])
        labels = torch.LongTensor([x.masked_lm_labels for x in dataset_instances])
        nsl = torch.LongTensor([x.next_sentence_label for x in dataset_instances])

        if model_type == "BERT":
            return BertDataset(ids, attentions, token_types, labels, nsl)
        elif model_type == "ELECTRA":
            return ElectraDataset(ids, attentions, token_types, labels)
        elif model_type == "RoBERTa":
            return RobertaDataset(ids, attentions, token_types, labels)

    logging.debug("Create Dataset object.")
    if do_eval:
        num_samples = math.ceil(eval_size * len(instances))
        val_subset = [
            instances.pop(random.randrange(len(instances))) for _ in range(num_samples)
        ]
        train = instances
        return make_dataset(train), make_dataset(val_subset)
    else:
        return make_dataset(instances)
