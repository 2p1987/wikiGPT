import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import structlog
import torch
import torch.distributed as dist

log = structlog.get_logger()


# create an iterable dataset class to iterate over pre-tokenized data
class TokenIterator(torch.utils.data.IterableDataset):
    def __init__(
        self,
        pretokenized_source: Path,
        context_length: int,
        split: str,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.pretokenized_source = pretokenized_source
        self.context_length = context_length
        self.split = split
        self.verbose = verbose
        total_filenames = sorted(self.pretokenized_source.glob("*.bin"))
        self.filenames = (
            total_filenames[4:] if self.split == "train" else total_filenames[:4]
        )
        if len(self.filenames) == 0:
            raise ValueError(
                "The pretokenized source folder does not contain any .bin files."
            )

    def _get_start_and_stop_index(self, worker_info, worker_id: int) -> Tuple[int, int]:
        if worker_info is not None:
            len_filenames = len(self.filenames)
            per_worker = len_filenames // worker_info.num_workers
            extras = len_filenames % worker_info.num_workers

            start_idx = worker_id * per_worker + min(worker_id, extras)
            stop_idx = (worker_id + 1) * per_worker + min(worker_id + 1, extras)

        else:
            start_idx = 0
            stop_idx = len(self.filenames)

        return start_idx, stop_idx

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        start_idx, stop_idx = self._get_start_and_stop_index(worker_info, worker_id)
        if self.verbose:
            log.info(
                "Iter for", worker_id=worker_id, start_idx=start_idx, stop_idx=stop_idx
            )

        while True:
            for shard in self.filenames[start_idx:stop_idx]:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.context_length
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.context_length
                    end = start + self.context_length + 1
                    # calling .astype will copy the data into a new numpy array,
                    # now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y

    @classmethod
    def iter_batches(cls, batch_size, device, num_workers=0, **iterator_kwargs):
        ds = cls(**iterator_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


# create a dataset class to loop over batches of pre-tokenized data
class TokenBatches(torch.utils.data.Dataset):
    def __init__(
        self, pretokenized_source: Path, context_length: int, split: str
    ) -> None:
        super().__init__()
        self.pretokenized_source = pretokenized_source
        self.context_length = context_length
        self.split = split
        total_filenames = sorted(self.pretokenized_source.glob("*.bin"))
        self.filenames = (
            total_filenames[4:] if self.split == "train" else total_filenames[:4]
        )
        if len(self.filenames) == 0:
            raise ValueError(
                "The pretokenized source folder does not contain any .bin files."
            )
        # Store the lengths of the memmaps instead of the memmaps themselves
        self.memmap_lengths = [
            os.path.getsize(f) // 2 for f in self.filenames
        ]  # Assuming dtype=np.uint16 (2 bytes)
        self.total_batches = np.cumsum(
            [length // (self.context_length + 1) for length in self.memmap_lengths]
        )

    def _load_memmap(self, file_index):
        # Lazily load the memmap
        return np.memmap(self.filenames[file_index], dtype=np.uint16, mode="r")

    def __len__(self):
        return self.total_batches[-1]

    def __getitem__(self, idx):
        # Find which shard contains the desired index
        shard_idx = np.searchsorted(self.total_batches, idx, side="right")
        if shard_idx > 0:
            idx -= self.total_batches[
                shard_idx - 1
            ]  # Adjust index relative to the shard

        # Load the sample from the appropriate memmap file
        start_token_idx = idx * (self.context_length + 1)  # +1 for the target
        chunk = self._load_memmap(shard_idx)[
            start_token_idx : start_token_idx + self.context_length + 1
        ]
        # calling .astype will copy the data into a new numpy array, now in RAM
        chunk = torch.from_numpy(chunk.astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]

        return x, y

    @classmethod
    def iter_batches(cls, batch_size, device, num_workers, **iterator_kwargs):
        ds = cls(**iterator_kwargs)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            shuffle=True,
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    # ds = TokenIterator(Path("climateGPT/data/tok32000"), 6, "val")

    # dl = torch.utils.data.DataLoader(ds, batch_size=1, pin_memory=True, num_workers=2)
    # for i, (x, y) in enumerate(dl):
    #     if i > 1:
    #         break
    #     print(x)
    #     print(y)

    ds = TokenBatches(
        Path("climateGPT/data/fine_tuning/vocab_2000_context_512"), 512, "train"
    )

    from functools import partial

    iter_params = {
        "context_length": 512,
        "pretokenized_source": Path(
            "climateGPT/data/fine_tuning/vocab_2000_context_512"
        ),
    }

    iter_batches = partial(
        TokenBatches.iter_batches,
        batch_size=1,
        num_workers=4,
        device="cpu",
        **iter_params
    )

    # training
    train_batch_iter = iter_batches(split="train")
    for _ in range(10):
        X, Y = next(train_batch_iter)
        print(X.shape, Y.shape)
