from pathlib import Path

import pandas as pd
import structlog
from datasets import load_dataset

log = structlog.get_logger()


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
            num_proc=4,
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
