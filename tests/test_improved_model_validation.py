import argparse

from src.determinism.improved_model_validation import (
    DEFAULT_BATCH_SIZES,
    GENERATION_COMPOSITIONS,
    LOGITS_COMPOSITIONS,
    build_summary,
    completed_keys,
    token_hash,
)


def test_validation_composition_sets_are_fixed() -> None:
    assert LOGITS_COMPOSITIONS == (
        "A_target_only",
        "C_seven_short",
        "E_mixed_lengths",
    )
    assert len(GENERATION_COMPOSITIONS) == 5
    assert DEFAULT_BATCH_SIZES == (1, 2, 4, 8)


def test_completed_keys_supports_resume() -> None:
    rows = [
        {
            "stage": "logits",
            "candidate_rank": "3",
            "composition": "C_seven_short",
            "status": "ok",
        },
        {
            "stage": "generation",
            "candidate_rank": "44",
            "composition": "E_mixed_lengths",
            "status": "error",
        },
    ]
    assert completed_keys(rows) == {("logits", 3, "C_seven_short")}


def test_token_hash_is_stable() -> None:
    assert token_hash([1, 2, 3]) == token_hash([1, 2, 3])
    assert token_hash([1, 2, 3]) != token_hash([1, 2, 4])


def test_build_summary_counts_results() -> None:
    rows = [
        {
            "stage": "logits",
            "status": "ok",
            "logits_bitwise_equal": "True",
            "max_abs_diff": "0.0",
            "top1_changed": "False",
            "top5_changed": "False",
        },
        {
            "stage": "generation",
            "status": "ok",
            "candidate_rank": "44",
            "composition": "A_target_only",
            "batch_size": "1",
            "generated_token_ids": "[1, 2]",
            "token_hash": "abc",
            "generated_text": "text",
            "output_identical": "True",
            "first_divergence_token": "",
        },
    ]
    args = argparse.Namespace(
        checkpoint="checkpoint.pt",
        candidates_input="candidates.csv",
        device="cuda",
        attention_fixed_split_size=64,
        linear_tile_m=16,
        linear_tile_n=256,
        linear_k_block_size=480,
        logits_candidates=20,
        generation_rank=44,
        batch_sizes=(1, 2, 4, 8),
        max_new_tokens=8,
    )
    summary = build_summary(rows, args)
    assert summary["logits_summary"]["bitwise_equal_cases"] == 1
    assert summary["generation_summary"]["all_outputs_identical"]
