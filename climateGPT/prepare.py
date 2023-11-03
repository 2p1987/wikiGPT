# Taken and adapted from llama.c (Andrej Karpathy)


import argparse
import json
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import structlog
from tqdm import tqdm

from climateGPT.tokenize import Tokenizer

log = structlog.get_logger()

DATA_CACHE_DIR = Path("climateGPT/data")


class UserCancellationError(Exception):
    """Exception raised when a user cancels an operation."""

    pass


def clear_folder(path):
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)  # Recursively delete directories
        else:
            item.unlink()  # Delete files


def create_shuffled_shards(n_shards: int, seed: int) -> None:
    """
    Each original shard is obtained as a thematic search from wikipedia
    (see https://github.com/2p1987/wikicollect for more info).
    However, in the case of multiple workers for the batch data loading,
    we don't want workers to return batches from the same thematic search.
    This methods will load all avaiable thematic searches shards and save
    them as randomly generated shards.
    These shuffled shards will be our input to pretokenize the dataset.
    """
    shards_folder_path = Path(DATA_CACHE_DIR, "original_shards")
    shard_filenames = sorted(shards_folder_path.glob("*.json"))
    # load all the data in a long list
    full_data = []
    for shard in shard_filenames:
        with open(shard, "r") as f:
            for line in f:
                tmp = json.loads(line)
                full_data.append(tmp)
    # shuffle articles
    random.seed(seed)
    random.shuffle(full_data)
    # save in the number of required shards
    full_data_len = len(full_data)
    shard_len = full_data_len // n_shards
    extras = full_data_len % n_shards
    shuffled_shards_folder_path = Path(shards_folder_path.parent, "shuffled_shards")
    shuffled_shards_folder_path.mkdir(exist_ok=True)
    # ask to remove everything in the folder if it already exists
    if any(shuffled_shards_folder_path.iterdir()):
        response = (
            input(
                f"The folder {shuffled_shards_folder_path} is not empty.\n\
Do you want to delete its contents? (y/n): "
            )
            .strip()
            .lower()
        )

        if response == "y":
            clear_folder(shuffled_shards_folder_path)
            log.info(
                f"All pre-existing files in {shuffled_shards_folder_path} have been \
deleted."
            )
        else:
            raise UserCancellationError(
                f"Folder {shuffled_shards_folder_path.as_posix()} is not empty."
            )

    for i in range(n_shards):
        shuffle_shard_path = Path(shuffled_shards_folder_path, f"shard_{i}.json")
        start_idx = i * shard_len + min(i, extras)
        stop_idx = (i + 1) * shard_len + min(i + 1, extras)
        with open(shuffle_shard_path, "w", encoding="utf-8") as f:
            for line in full_data[start_idx:stop_idx]:
                json.dump(line, f)
                f.write("\n")
    log.info(
        f"Original shards have been shuffled and saved locally at \
{shuffled_shards_folder_path.as_posix()}"
    )


def process_shard(args, tokenizer_path: Path):
    shard_id, shard = args
    enc = Tokenizer(tokenizer_path)
    all_tokens = []
    with open(shard, "r") as f:
        for line in tqdm(f, position=shard_id):
            example = json.loads(line)
            text = example["content"]
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=True)  # encode the text
            all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens_np = np.array(all_tokens, dtype=np.uint16)
    # save .bin files into a new tok{N} directory
    bin_dir = Path(DATA_CACHE_DIR, f"tok{enc.vocab_size}")
    bin_dir.mkdir(exist_ok=True)
    bin_basename = shard.name.replace(".json", ".bin")
    tokenized_filename = Path(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens_np.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens_np.size / ((all_tokens_np == 1).sum())
    log.info(
        f"Saved {tokenized_filename} with average sequence length {avg_seq_len: .2f}"
    )


def pretokenize(tokenizer_path: Path):
    # iterate the shards and tokenize all of them one by one
    shuffled_shard_folder = Path(DATA_CACHE_DIR, "shuffled_shards")
    if shuffled_shard_folder.exists() is False:
        raise RuntimeError(
            "Shuffled shards do not exist. Please run 'create_shuffled_shards'"
        )
    shard_filenames = sorted(shuffled_shard_folder.glob("*.json"))
    log.info(f"Number of shards to be processed: {len(shard_filenames)}")
    # process all the shards in a process pool
    fun = partial(process_shard, tokenizer_path=tokenizer_path)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    log.info("Finished processing all shards.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepate and pre-tokenize data",
    )

    subparsers = parser.add_subparsers(
        dest="task",
        help="Either create shuffled shards as a first step if they don't exist, \
        or directly pretokenize the shuffled shards.",
    )

    parser_create_shuffled_shards = subparsers.add_parser(
        "shuffle",
        help="Creates the shuffled shards from data in the data/original_shards folder",
    )
    parser_create_shuffled_shards.add_argument("--n-shards", type=int, default=100)
    parser_create_shuffled_shards.add_argument("--seed", type=int, default=678)

    parser_pretokenize = subparsers.add_parser(
        "pretokenize",
        help="Pre-tokenize and store locally data from shuffled shards.",
    )

    parser_pretokenize.add_argument(
        "--tokenizer-path",
        type=str,
        help="path to the tokenizer model",
    )

    args = parser.parse_args()

    if args.task == "shuffle":
        log.info("Shuffling original shards.")
        create_shuffled_shards(n_shards=args.n_shards, seed=args.seed)

    elif args.task == "pretokenize":
        log.info("Pretokenizing shuffled shards")
        tokenizer_model_path = Path(args.tokenizer_path)
        if tokenizer_model_path.exists() is False:
            raise ValueError(
                f"Tokenizer model path {tokenizer_model_path} does not exist."
            )
        pretokenize(tokenizer_model_path)

    else:
        raise ValueError(f"Expected task 'shuffle' or 'pretokenize'. Got {args.task}")
