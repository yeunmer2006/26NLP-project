from __future__ import annotations


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer with three special tokens."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    byte_offset = 3
    vocab_size = 259

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

