# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2
# Community License Agreement.

import argparse
import json
import os
from pathlib import Path
from typing import List, Union

import sentencepiece as spm
import structlog
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

log = structlog.get_logger()


def train_vocab(vocab_size, model_dir: str, data_dir: Union[str, Path]):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = Path(model_dir, f"tok{vocab_size}")

    # # how many shards we'll use for vocab training, kept low for efficiency
    # num_shards = 10

    # 1) export a large chunk of text as a single text file tiny.txt
    data_dir = Path(data_dir)
    shard_filenames = sorted(data_dir.glob("*.json"))
    tmp_folder = Path("tmp")
    tmp_folder.mkdir(exist_ok=True)
    tmp_file = Path(tmp_folder, "training_tmp.txt")

    count_words = 0
    with open(tmp_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames):
            with open(shard, "r") as f:
                for line in f:
                    example = json.loads(line)
                    text = example["content"]
                    text = text.strip()
                    count_words += len(text.split())
                    of.write(text + "\n")

    # 2) train the sentencepiece model
    log.info(f"Training a tokenizer with {count_words} words")

    spm.SentencePieceTrainer.train(
        input=tmp_file,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
        max_sentence_length=16384,
    )

    # 3) optional cleanup, ask the user if they'd like to delete tiny.txt
    dec = input(f"Delete the temporary file {tmp_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tmp_file)
        log.info(f"Deleted {tmp_file}")

    log.info(f"Trained tokenizer is in {prefix}.model")
    log.info("Done.")


class Tokenizer:
    def __init__(self, tokenizer_model_path: Path) -> None:
        if isinstance(tokenizer_model_path, Path):
            self.tokenizer_model_path = tokenizer_model_path
        else:
            self.tokenizer_model_path = Path(tokenizer_model_path)

        self.sp_model = SentencePieceProcessor(
            model_file=self.tokenizer_model_path.as_posix()
        )
        self.vocab_size = self.sp_model.vocab_size()
        self.bos = self.sp_model.bos_id()
        self.eos = self.sp_model.eos_id()
        self.pad = self.sp_model.pad_id()

    def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
        encoded_text = self.sp_model.encode(text)
        if bos:
            encoded_text = [self.bos] + encoded_text
        if eos:
            encoded_text = encoded_text + [self.eos]
        return encoded_text

    def decode(self, encoded_text: List[int]) -> str:
        return self.sp_model.decode(encoded_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training a tokenizer with sentence piece (BPE)",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2000,
        help="Vocab size for the tokenizer",
    )

    parser.add_argument(
        "--model-output-dir",
        type=str,
        default="climateGPT/models/",
        help="Directory to save the trained tokenizer",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="climateGPT/data/shuffled_shards/",
        help="Directory containing the training data",
    )

    args = parser.parse_args()

    train_vocab(args.vocab_size, args.model_output_dir, args.data_dir)
