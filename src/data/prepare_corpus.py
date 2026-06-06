from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from src.common import load_json, save_json, set_seed
from src.data.tokenizer import SentencePieceTokenizer


def iter_texts(config: dict, split: str) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("datasets is required to prepare Hugging Face corpora") from error
    dataset = load_dataset(
        config["dataset_name"],
        config.get("dataset_config"),
        split=split,
        streaming=config.get("streaming", True),
    )
    if config.get("shuffle", True):
        kwargs = {"seed": config["seed"]}
        if config.get("streaming", True):
            kwargs["buffer_size"] = config.get("shuffle_buffer", 10_000)
        dataset = dataset.shuffle(**kwargs)
    text_field = config.get("text_field", "text")
    for example in dataset:
        text = str(example.get(text_field, "")).strip()
        if text:
            yield text


def cache_documents(texts: Iterable[str], path: Path, max_documents: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    document_count = 0
    character_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            document_count += 1
            character_count += len(text)
            if document_count >= max_documents:
                break
    return {"documents": document_count, "characters": character_count}


def cached_texts(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)["text"]


def encode_to_npy(
    texts: Iterable[str],
    tokenizer: SentencePieceTokenizer,
    output: Path,
    token_budget: int,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    token_ids: list[int] = []
    documents = 0
    characters = 0
    for text in texts:
        remaining = token_budget - len(token_ids)
        if remaining <= 0:
            break
        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
        token_ids.extend(encoded[:remaining])
        documents += 1
        characters += len(text)
    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    np.save(output, np.asarray(token_ids, dtype=dtype))
    return {
        "documents": documents,
        "characters": characters,
        "tokens": len(token_ids),
        "average_tokens_per_document": len(token_ids) / max(1, documents),
        "path": str(output),
    }


def prepare(config: dict) -> None:
    set_seed(config["seed"])
    output_dir = Path(config["output_dir"])
    raw_dir = output_dir / "raw"
    tokenizer_prefix = output_dir / "tokenizer" / "tinystories_bpe"
    train_raw = raw_dir / "train.jsonl"
    valid_raw = raw_dir / "validation.jsonl"

    train_cache_stats = cache_documents(
        iter_texts(config, config["train_split"]),
        train_raw,
        config["max_train_documents"],
    )
    valid_cache_stats = cache_documents(
        iter_texts(config, config["validation_split"]),
        valid_raw,
        config["max_validation_documents"],
    )
    tokenizer = SentencePieceTokenizer.train(
        cached_texts(train_raw),
        tokenizer_prefix,
        config["vocab_size"],
    )
    train_stats = encode_to_npy(
        cached_texts(train_raw),
        tokenizer,
        output_dir / "train.npy",
        config["train_token_budget"],
    )
    valid_stats = encode_to_npy(
        cached_texts(valid_raw),
        tokenizer,
        output_dir / "validation.npy",
        config["validation_token_budget"],
    )
    save_json(
        {
            "config": config,
            "tokenizer": str(tokenizer_prefix.with_suffix(".model")),
            "vocab_size": tokenizer.vocab_size,
            "raw_cache": {"train": train_cache_stats, "validation": valid_cache_stats},
            "encoded": {"train": train_stats, "validation": valid_stats},
        },
        output_dir / "dataset_stats.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a token-limited Hugging Face corpus.")
    parser.add_argument("--config", default="configs/data_tinystories.json")
    args = parser.parse_args()
    prepare(load_json(args.config))


if __name__ == "__main__":
    main()
