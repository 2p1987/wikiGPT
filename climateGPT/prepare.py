# Taken and adapted from llama.c (Andrej Karpathy)


import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from datasets import load_dataset
from tqdm import tqdm

from climateGPT.tokenize import Tokenizer

log = structlog.get_logger()

DATA_CACHE_DIR = Path("climateGPT/data/")


def load_wiki_sample():
    log.info("Attempting to load wiki sample from cache")
    cache_path = Path("climateGPT/data/wiki_cache/wiki_sample.parquet")
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    if cache_path.exists():
        log.info("Wiki sample found in cache")
        return pd.read_parquet(cache_path)
    else:
        log.info("Wiki sample not found in cache, loading from HF")
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train[:8%]",
            num_proc=8,
        )
        df = dataset.to_pandas()
        df = (
            df.rename(columns={"text": "content"})
            .drop(columns=["id", "url"])
            .reset_index(drop=True)
        )
        df.to_parquet(cache_path)
        log.info("Wiki sample saved to cache")
    return df


class UserCancellationError(Exception):
    """Exception raised when a user cancels an operation."""

    pass


def clear_folder(path):
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)  # Recursively delete directories
        else:
            item.unlink()  # Delete files


def create_shuffled_shards(
    df: pd.DataFrame, n_shards: int, seed: int, data_cache_dir: Path
) -> None:
    """
    Each original shard is obtained as a thematic search from wikipedia
    (see https://github.com/2p1987/wikicollect for more info).
    However, in the case of multiple workers for the batch data loading,
    we don't want workers to return batches from the same thematic search.
    This methods will load all avaiable thematic searches shards and save
    them as randomly generated shards.
    These shuffled shards will be our input to pretokenize the dataset.
    """
    # shuffle articles
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # save in the number of required shards
    shuffled_df_len = len(shuffled_df)
    shard_len = shuffled_df_len // n_shards
    extras = shuffled_df_len % n_shards
    shuffled_shards_folder_path = Path(data_cache_dir, "shuffled_shards")
    shuffled_shards_folder_path.mkdir(parents=True, exist_ok=True)
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
            for line in shuffled_df[start_idx:stop_idx].iterrows():
                json.dump({"title": line[1]["title"], "content": line[1]["content"]}, f)
                f.write("\n")
    log.info(
        f"Original shards have been shuffled and saved locally at \
{shuffled_shards_folder_path.as_posix()}"
    )


def process_shard(args, tokenizer_path: Path, data_cache_dir: Path):
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
    bin_dir = Path(data_cache_dir, f"tok{enc.vocab_size}")
    bin_dir.mkdir(exist_ok=True)
    bin_basename = shard.name.replace(".json", ".bin")
    tokenized_filename = Path(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens_np.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens_np.size / ((all_tokens_np == 1).sum())
    log.info(f"Saved {tokenized_filename}")
    log.info(f"Average sequence length: {avg_seq_len}")
    log.info(f"Number of tokens: {all_tokens_np.size}")


def pretokenize(tokenizer_path: str, data_cache_dir: str):
    # iterate the shards and tokenize all of them one by one
    shuffled_shard_folder = Path(data_cache_dir, "shuffled_shards")
    data_cache_dir_ = Path(data_cache_dir)
    tokenizer_path_ = Path(tokenizer_path)

    if tokenizer_path_.exists() is False:
        raise ValueError(
            f"Tokenizer model path {tokenizer_path_.as_posix()} does not exist."
        )
    if shuffled_shard_folder.exists() is False:
        raise RuntimeError(
            "Shuffled shards do not exist. Please run 'create_shuffled_shards'"
        )
    shard_filenames = sorted(shuffled_shard_folder.glob("*.json"))
    log.info(f"Number of shards to be processed: {len(shard_filenames)}")
    # process all the shards in a process pool
    fun = partial(
        process_shard, tokenizer_path=tokenizer_path_, data_cache_dir=data_cache_dir_
    )
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
    parser_create_shuffled_shards.add_argument(
        "--n-shards", type=int, default=100, help="Number of shards to be created."
    )
    parser_create_shuffled_shards.add_argument(
        "--seed", type=int, default=678, help="Random seed for shuffling"
    )
    parser_create_shuffled_shards.add_argument(
        "--data-cache-dir",
        type=str,
        default=DATA_CACHE_DIR,
        help="Data cache directory.",
    )

    parser_pretokenize = subparsers.add_parser(
        "pretokenize",
        help="Pre-tokenize and store locally data from shuffled shards.",
    )

    parser_pretokenize.add_argument(
        "--tokenizer-path",
        type=str,
        help="path to the tokenizer model",
    )
    parser_pretokenize.add_argument(
        "--data-cache-dir",
        type=str,
        default=DATA_CACHE_DIR,
        help="Data cache directory.",
    )

    args = parser.parse_args()

    if args.task == "shuffle":
        log.info("Shuffling original shards.")
        full_data = load_wiki_sample()
        data_cache_dir = Path(args.data_cache_dir)
        create_shuffled_shards(
            df=full_data,
            n_shards=args.n_shards,
            seed=args.seed,
            data_cache_dir=args.data_cache_dir,
        )

    elif args.task == "pretokenize":
        log.info("Pretokenizing shuffled shards")
        pretokenize(
            tokenizer_path=args.tokenizer_path, data_cache_dir=args.data_cache_dir
        )

    else:
        raise ValueError(f"Expected task 'shuffle' or 'pretokenize'. Got {args.task}")
