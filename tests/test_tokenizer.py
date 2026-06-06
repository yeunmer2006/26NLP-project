from src.data import ByteTokenizer


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "TinyLlama 测试"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_outside_byte_range() -> None:
    tokenizer = ByteTokenizer()
    token_ids = tokenizer.encode("A", add_bos=True, add_eos=True)
    assert token_ids == [tokenizer.bos_token_id, ord("A") + 3, tokenizer.eos_token_id]


def test_byte_tokenizer_save_and_load(tmp_path) -> None:
    from src.data import load_tokenizer

    path = tmp_path / "tokenizer.txt"
    ByteTokenizer().save(path)
    tokenizer = load_tokenizer(path)
    assert tokenizer.decode(tokenizer.encode("round trip")) == "round trip"
