from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer with three special tokens."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    byte_offset = 3
    vocab_size = 259
    model_path = None

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        values = [
            token_id - self.byte_offset
            for token_id in token_ids
            if self.byte_offset <= token_id < self.vocab_size
        ]
        return bytes(values).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text("byte-tokenizer-v1\n", encoding="utf-8")


class SentencePieceTokenizer:
    """Small wrapper with the same interface as ByteTokenizer."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import sentencepiece as spm
        except ImportError as error:
            raise RuntimeError("sentencepiece is required for BPE tokenization") from error
        self.model_path = str(Path(model_path).resolve())
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)
        self.pad_token_id = self.processor.pad_id()
        self.unk_token_id = self.processor.unk_id()
        self.bos_token_id = self.processor.bos_id()
        self.eos_token_id = self.processor.eos_id()
        self.vocab_size = self.processor.vocab_size()

    @classmethod
    def train(
        cls,
        sentences: Iterable[str],
        model_prefix: str | Path,
        vocab_size: int = 8000,
    ) -> "SentencePieceTokenizer":
        try:
            import sentencepiece as spm
        except ImportError as error:
            raise RuntimeError("sentencepiece is required to train a BPE tokenizer") from error
        prefix = Path(model_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            sentence_iterator=(text for text in sentences if text.strip()),
            model_prefix=str(prefix),
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            hard_vocab_limit=False,
            shuffle_input_sentence=False,
        )
        return cls(prefix.with_suffix(".model"))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.processor.decode(token_ids)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.model_path, destination)


def load_tokenizer(path: str | Path | None):
    if path is None:
        return ByteTokenizer()
    path = Path(path)
    if path.suffix == ".model":
        return SentencePieceTokenizer(path)
    if path.read_text(encoding="utf-8").strip() == "byte-tokenizer-v1":
        return ByteTokenizer()
    raise ValueError(f"unsupported tokenizer file: {path}")
