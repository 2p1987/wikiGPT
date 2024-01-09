from pathlib import Path

import numpy as np

# from datasets import load_dataset


# PRETOKENIZED_SOURCE = "climateGPT/data/fine_tuning/vocab_2000_context_512"
PRETOKENIZED_SOURCE = "climateGPT/data/tok32000"
PRETOKENIZED_SOURCE = Path(PRETOKENIZED_SOURCE)

total_filenames = sorted(PRETOKENIZED_SOURCE.glob("*.bin"))
memmaps = [np.memmap(f, dtype=np.uint16, mode="r") for f in total_filenames]
total_tokens = np.sum([len(m) for m in memmaps])

print("Total number of tokens:", total_tokens)
