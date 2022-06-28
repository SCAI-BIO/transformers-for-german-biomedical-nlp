import re
from typing import Any, List

from flair.data import Sentence
from flair.datasets.sequence_labeling import JsonlDataset


class ModJsonlDataset(JsonlDataset):
    def _add_label_to_sentence(
        self, text: str, sentence: Sentence, start: int, end: int, label: str
    ):
        """
        Adds a NE label to a given sentence.
        :param text: raw sentence (with all whitespaces etc.). Is used to determine the token indices.
        :param sentence: Tokenized flair Sentence.
        :param start: Start character index of the label.
        :param end: End character index of the label.
        :param label: Label to assign to the given range.
        :return: Nothing. Changes sentence as INOUT-param
        """

        annotated_part = text[start:end]

        # Remove leading and trailing whitespaces from annotated spans
        while re.search(r"^\s", annotated_part):
            start += 1
            annotated_part = text[start:end]

        while re.search(r"\s$", annotated_part):
            end -= 1
            annotated_part = text[start:end]

        # Search start and end token index for current span
        start_idx = -1
        end_idx = -1
        for token in sentence:
            if token.start_pos <= start <= token.end_pos and start_idx == -1:
                start_idx = token.idx - 1

            if token.start_pos <= end <= token.end_pos and end_idx == -1:
                end_idx = token.idx - 1

        # If end index is not found set to last token
        if end_idx == -1:
            end_idx = sentence[-1].idx - 1

        # Throw error if indices are not valid
        if start_idx == -1 or start_idx > end_idx:
            raise ValueError(
                f"Could not create token span from char span.\n\
                            Sen: {sentence}\nStart: {start}, End: {end}, Label: {label}\n\
                                Ann: {annotated_part}\nRaw: {text}\nCo: {start_idx}, {end_idx}"
            )

        # Add span notation
        sentence[start_idx : end_idx + 1].add_label(self.label_type, label)

    def _add_labels_to_sentence(
        self, raw_text: str, sentence: Sentence, labels: List[List[Any]]
    ):
        # Add tags for each annotated span
        for label in labels:
            self._add_label_to_sentence(
                raw_text, sentence, label[0], label[1], label[2]
            )


def load_dataset(path: str) -> ModJsonlDataset:
    return ModJsonlDataset(
        path_to_jsonl_file=path, text_column_name="text", label_column_name="labels"
    )
