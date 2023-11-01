# Taken (and slightly adapted) from llama.c (Andrej Karpathy)


import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import structlog
from tqdm import tqdm

from climateGPT.tokenize import Tokenizer

log = structlog.get_logger()

DATA_CACHE_DIR = Path("climateGPT/data")


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
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # save .bin files into a new tok{N} directory
    bin_dir = Path(DATA_CACHE_DIR, f"tok{enc.vocab_size}")
    bin_dir.mkdir(exist_ok=True)
    bin_basename = shard.name.replace(".json", ".bin")
    tokenized_filename = Path(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    log.info(
        f"Saved {tokenized_filename} with average sequence length {avg_seq_len: .2f}"
    )


def pretokenize(tokenizer_path: Path):
    # iterate the shards and tokenize all of them one by one
    shard_filenames = sorted([file for file in DATA_CACHE_DIR.glob("*.json")])
    log.info(f"Number of shards to be processed: {len(shard_filenames)}")
    # process all the shards in a process pool
    fun = partial(process_shard, tokenizer_path=tokenizer_path)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    log.info("Finished processing all shards.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize input data and save it locally",
    )

    parser.add_argument(
        "tokenizer",
        type=str,
        help="path to the tokenizer model",
    )

    args = parser.parse_args()
    tokenizer_model_path = Path(args.tokenizer)
    if tokenizer_model_path.exists() is False:
        raise ValueError(f"Tokenizer model path {tokenizer_model_path} does not exist.")
    pretokenize(tokenizer_model_path)
