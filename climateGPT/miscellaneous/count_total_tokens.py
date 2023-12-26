from pathlib import Path

import pandas as pd

from climateGPT.tokenize import Tokenizer

# from datasets import load_dataset


DATA_CACHE_DIR = Path("climateGPT/data")


# climate_data = load_dataset("pierre-pessarossi/wikipedia-climate-data")
# df = climate_data["train"].to_pandas()
df = (
    pd.read_parquet("climateGPT/data/wiki_cache/wiki_sample.parquet")
    .sample(frac=0.01)
    .reset_index(drop=True)
)

# tokenize
vocab_sizes = [2000, 32000]

tokenizers = []
for vocab_size in vocab_sizes:
    tokenizer_path = f"climateGPT/models/tok{vocab_size}.model"
    tokenizers.append(Tokenizer(Path(tokenizer_path)))

# encode
for i, tokenizer in enumerate(tokenizers):
    total_tokens = 0
    for r in range(len(df)):
        tokenized_content = tokenizer.encode(df["content"][r], bos=True, eos=True)
        total_tokens += len(tokenized_content)

    print(
        f"Tokens (est.): {round(total_tokens * 100):,} for vocab size {vocab_sizes[i]}"
    )
