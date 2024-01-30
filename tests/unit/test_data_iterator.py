from pathlib import Path

import torch

from wikiGPT.iterate import TokenIterator


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
        assert (x == torch.tensor([[2582, 515, 1422, 3736, 1103, 3145]])).all()
        assert (y == torch.tensor([515, 1422, 3736, 1103, 3145, 373])).all()
