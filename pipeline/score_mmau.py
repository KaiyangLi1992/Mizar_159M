#!/usr/bin/env python3
"""Score free generations with MMAU's official content-matching rule only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import string


ARTICLES = {"a", "an", "the"}


def normalize_answer(text: object) -> str:
    value = "" if text is None else str(text).lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return " ".join(token for token in value.split() if token not in ARTICLES)


def tokenize(text: object) -> set[str]:
    return set(re.findall(r"\b\w+\b", normalize_answer(text)))


def official_correct(answer: str, prediction: str, choices: list[str]) -> bool:
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    if not prediction_tokens:
        return False
    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    return answer_tokens.issubset(prediction_tokens) and prediction_tokens.isdisjoint(incorrect_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    references = json.loads(args.references.read_text())
    frozen = json.loads((Path(__file__).resolve().parents[1] / "data/evaluation/mmau_mini_official.json").read_text())
    if len(references) != 1000 or references != frozen:
        raise ValueError("This CLI scores frozen MMAU-mini only; full9k requires the official service")
    predictions = json.loads(args.predictions.read_text())
    reference_by_id = {str(row["id"]): row for row in references}
    prediction_by_id = {str(row["id"]): row for row in predictions}
    if len(reference_by_id) != len(references) or len(prediction_by_id) != len(predictions):
        raise ValueError("duplicate sample ID")
    if set(reference_by_id) != set(prediction_by_id):
        raise ValueError("prediction/reference ID coverage mismatch")
    rows = []
    for identity in reference_by_id:
        reference = reference_by_id[identity]
        prediction = prediction_by_id[identity]
        raw = prediction.get("model_output", prediction.get("prediction"))
        if not isinstance(raw, str):
            raise ValueError(f"missing raw free generation for {identity}")
        rows.append({
            "id": identity,
            "correct": official_correct(reference["answer"], raw, reference["choices"]),
            "model_output": raw,
        })
    correct = sum(row["correct"] for row in rows)
    result = {
        "metric": "mmau_official_string_match",
        "decoding": "unrestricted_full_vocabulary_generation",
        "correct": correct,
        "total": len(rows),
        "accuracy_pct": 100 * correct / len(rows),
        "per_sample": rows,
    }
    if args.output.exists():
        raise FileExistsError(f"refuse to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"MMAU official accuracy: {result['accuracy_pct']:.4f}% ({correct}/{len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
