# TODO: test repartition between workers
# TODO: test encoding/decoding of batches
# TODO: test number of batches

from pathlib import Path

import torch

from climateGPT.iterate import TokenIterator


class TestTokenIterator:
    def test_iter_batches(self):
        iter_params = {
            "pretokenized_source": Path("tests/data/tok32000"),
            "context_length": 6,
            "split": "train",
        }

        iter_batches = TokenIterator.iter_batches(
            batch_size=1, device="cpu", num_workers=0, **iter_params
        )

        x, y = next(iter_batches)
        assert (x == torch.tensor([[259, 13, 29934, 5921, 2, 1]])).all()
        assert (y == torch.tensor([[13, 29934, 5921, 2, 1, 450]])).all()
