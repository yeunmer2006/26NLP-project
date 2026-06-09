import argparse
import csv

from src.determinism.improved_model_validation import (
    DEFAULT_BATCH_SIZES,
    FIXED_ORDER_VARIANT,
    GENERATION_COMPOSITIONS,
    LOGITS_COMPOSITIONS,
    MODEL_VARIANTS,
    build_summary,
    completed_keys,
    read_existing_rows,
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
            "model_variant": "native",
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
    assert completed_keys(rows) == {
        ("native", "logits", 3, "C_seven_short")
    }


def test_read_existing_rows_treats_legacy_rows_as_fixed_order(tmp_path) -> None:
    output = tmp_path / "legacy.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("stage", "candidate_rank", "composition", "status"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "stage": "logits",
                "candidate_rank": 1,
                "composition": "A_target_only",
                "status": "ok",
            }
        )

    rows = read_existing_rows(output, resume=True)

    assert rows[0]["model_variant"] == FIXED_ORDER_VARIANT
    assert completed_keys(rows) == {
        (FIXED_ORDER_VARIANT, "logits", 1, "A_target_only")
    }


def test_token_hash_is_stable() -> None:
    assert token_hash([1, 2, 3]) == token_hash([1, 2, 3])
    assert token_hash([1, 2, 3]) != token_hash([1, 2, 4])


def test_build_summary_counts_results() -> None:
    rows = []
    for model_variant in MODEL_VARIANTS:
        for rank in range(1, 21):
            for composition in LOGITS_COMPOSITIONS:
                is_fixed = model_variant == FIXED_ORDER_VARIANT
                rows.append(
                    {
                        "model_variant": model_variant,
                        "stage": "logits",
                        "candidate_rank": str(rank),
                        "composition": composition,
                        "status": "ok",
                        "logits_bitwise_equal": str(is_fixed),
                        "max_abs_diff": "0.0" if is_fixed else "0.001",
                        "top1_changed": "False",
                        "top5_changed": "False",
                    }
                )
    rows.append(
        {
            "model_variant": FIXED_ORDER_VARIANT,
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
    )
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
    assert summary["configuration"]["logits_positions"] == "last_valid_token_only"
    assert summary["logits_comparison"]["native"] == {
        "tested_cases": 60,
        "bitwise_equal_cases": 0,
        "nonzero_difference_cases": 60,
        "top1_changed_cases": 0,
        "top5_changed_cases": 0,
        "maximum_absolute_difference": 0.001,
    }
    assert summary["logits_comparison"][FIXED_ORDER_VARIANT] == {
        "tested_cases": 60,
        "bitwise_equal_cases": 60,
        "nonzero_difference_cases": 0,
        "top1_changed_cases": 0,
        "top5_changed_cases": 0,
        "maximum_absolute_difference": 0.0,
    }
    assert summary["logits_summary"] == summary["logits_comparison"][FIXED_ORDER_VARIANT]
    assert summary["generation_summary"]["all_outputs_identical"]
