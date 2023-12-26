from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import structlog
from datasets import load_dataset
from tqdm import tqdm

from climateGPT.tokenize import Tokenizer

log = structlog.get_logger()

vocab_size = 2000
context_length = 512
save_path = Path(
    f"climateGPT/data/fine_tuning/vocab_{vocab_size}_context_{context_length}"
)
save_path.mkdir(exist_ok=True, parents=True)


def load_dataset_from_hf() -> pd.DataFrame:
    dataset = load_dataset("pierre-pessarossi/wikipedia-climate-data")

    return dataset["train"].to_pandas()


def tokenize_content(df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = Tokenizer(Path(f"climateGPT/models/tok{vocab_size}.model"))

    df["tokenized_content"] = df["content"].apply(
        lambda x: tokenizer.encode(x.strip(), bos=True, eos=True)
    )

    return df


def create_batches_from_tokenized_content(
    tokenized_content: List[int], context_length
) -> List[int]:
    one_content_batches = []
    for i in range(len(tokenized_content) - context_length - 1):
        tmp = tokenized_content[i : i + context_length + 1]
        one_content_batches.extend(tmp)

    return one_content_batches


def create_all_tokens(df: pd.DataFrame, context_length: int) -> List[int]:
    batches = []
    for r in tqdm(range(len(df))):
        tmp = create_batches_from_tokenized_content(
            df["tokenized_content"][r], context_length
        )
        batches.extend(tmp)

    return batches


def process_batches(shard_id, df):
    max_rows = len(df)
    start_rows = shard_id * 10
    end_rows = min([start_rows + 10, max_rows])

    all_tokens = create_all_tokens(
        df[start_rows:end_rows].reset_index(), context_length
    )
    all_tokens_np = np.array(all_tokens, dtype=np.uint16)
    log.info(f"Number of tokens: {all_tokens_np.size}")

    filename = Path(save_path, f"shard_{shard_id}.bin")

    with open(filename, "wb") as f:
        f.write(all_tokens_np.tobytes())


if __name__ == "__main__":
    df = load_dataset_from_hf()
    log.info("Loaded dataset from HuggingFace dataset hub.")
    df = tokenize_content(df)
    log.info("Tokenized content.")

    # process all the shards in a process pool
    fun = partial(
        process_batches,
        df=df,
    )
    with ProcessPoolExecutor() as executor:
        executor.map(fun, [i for i in range(len(df) // 10 + 1)])
    log.info("Finished processing all shards.")


# TODO: save batches as .bin files
# TODO: create dataset from .bin files with memmap and index based on context length + 1
# , adapt len method to take that into account
# TODO: modify training loop to use the dataset

# TODO: refactor prepare_batch to incorporate all in the prepare script
