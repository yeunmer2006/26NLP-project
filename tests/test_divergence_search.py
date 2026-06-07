import json

from src.determinism.divergence_search import sampled_documents


def test_sampled_documents_is_reproducible(tmp_path) -> None:
    path = tmp_path / "validation.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(20):
            handle.write(json.dumps({"text": f"story {index}"}) + "\n")
    assert sampled_documents(path, 5, 42) == sampled_documents(path, 5, 42)
    assert len(sampled_documents(path, 5, 42)) == 5
