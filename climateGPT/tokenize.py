# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2
# Community License Agreement.

from pathlib import Path
from typing import List

from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, tokenizer_model_path: Path) -> None:
        if isinstance(tokenizer_model_path, Path):
            self.tokenizer_model_path = tokenizer_model_path
        else:
            self.tokenizer_model_path = Path(tokenizer_model_path)

        self.sp_model = SentencePieceProcessor(
            model_file=self.tokenizer_model_path.as_posix()
        )
        self.vocab_size = self.sp_model.vocab_size()
        self.bos = self.sp_model.bos_id()
        self.eos = self.sp_model.eos_id()
        self.pad = self.sp_model.pad_id()

    def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
        encoded_text = self.sp_model.encode(text)
        if bos:
            encoded_text = [self.bos] + encoded_text
        if eos:
            encoded_text = encoded_text + [self.eos]
        return encoded_text

    def decode(self, encoded_text: List[int]) -> str:
        return self.sp_model.decode(encoded_text)
