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
