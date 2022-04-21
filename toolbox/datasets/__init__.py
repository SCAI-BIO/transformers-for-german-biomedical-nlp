from typing import Union

from transformers import BertTokenizerFast, ElectraTokenizerFast, RobertaTokenizerFast

TOKENIZERS = Union[BertTokenizerFast, ElectraTokenizerFast, RobertaTokenizerFast]
