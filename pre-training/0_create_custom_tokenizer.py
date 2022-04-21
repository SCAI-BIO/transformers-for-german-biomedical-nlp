# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List

import click
from tokenizers.implementations import BertWordPieceTokenizer


@click.command()
@click.argument("output-dir", type=click.Path())
@click.option("-i", "--input-files", multiple=True, type=str)
@click.option("--vocab-size", type=int, default=30000)
@click.option("--min-freq", type=int, default=2)
def create_tokenizer(
    output_dir: str,
    input_files: List[str],
    vocab_size: int,
    min_freq: int,
):
    """
    Create a BERT tokenizer and save the vocab file.

    \b
    Usage:\b
        python 0_create_custom_tokenizer.py custom_tokenizer -i test.txt -i test2.txt --vocab-size=15000
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=input_files, vocab_size=vocab_size, min_frequency=min_freq)
    tokenizer.save_model(str(output_dir))


if __name__ == "__main__":
    create_tokenizer()
