"""
Official-style evaluator for DCASE 2026 Task 5 ADQA dev set.

The challenge spec defines:
  - Metric: Top-1 Accuracy (fraction correct / total)
  - Submission CSV columns: question (sample id), answer (plain choice text)
  - Predicted answer must match one of multi_choice exactly after post-processing
    (strip option prefixes like A., B., (a), etc.)

This script implements that scoring rule on dev.jsonl + a predictions CSV or JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

_PREFIX_RE = re.compile(
    r"^\s*[\(\[]?\s*([A-Za-z])\s*[\)\]\.\:\-]\s*",
    re.IGNORECASE,
)


def load_ground_truth(jsonl_path: Path) -> dict[str, dict]:
    gt = {}
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            gt[row["id"]] = row
    return gt


def strip_option_prefix(text: str) -> str:
    s = (text or "").strip()
    while True:
        m = _PREFIX_RE.match(s)
        if not m:
            break
        s = s[m.end():].strip()
    return s


def normalize_for_match(text: str) -> str:
    return " ".join((text or "").strip().split()).casefold()


def extract_leading_letter(text: str) -> tuple[str | None, str]:
    s = (text or "").strip()
    m = re.match(r"^[\(\[]?\s*([a-zA-Z])\s*[\)\]\.\:\-]\s*(.*)$", s, re.DOTALL)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    m = re.match(r"^([a-zA-Z])\)?\s*(.*)$", s, re.DOTALL)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    return None, s


# A leading option label is only honored when followed by a real delimiter,
# e.g. "a) text", "B. text", "(c) text". A bare leading letter (as in the word
# "draw" or "actually") must NOT be treated as an option label, otherwise
# plain-text answers in the MMAU format get mis-mapped to the wrong choice.
_LABEL_RE = re.compile(r"^\s*[\(\[]?\s*([A-Za-z])\s*[\)\]\.\:\-]\s+(.*)$", re.DOTALL)


def resolve_to_choice(raw_pred: str, choices: list[str]) -> str | None:
    """Map a raw model string to one of the official choice strings, if possible.

    The DCASE 2026 ADQA dev set uses the MMAU presentation format (no letter
    labels), so the model is expected to answer with the choice text itself.
    Matching priority:
      1. exact normalized full-text match to a choice
      2. explicit "a) ... / a. ..." label (only with a delimiter) -> choice
      3. choice text uniquely contained in the output
      4. output uniquely contained in a choice (truncated generation)
    """
    if not (raw_pred or "").strip():
        return None

    choice_map = {normalize_for_match(c): c for c in choices}
    pred_norm = normalize_for_match(raw_pred)

    # 1) exact full-text match
    if pred_norm in choice_map:
        return choice_map[pred_norm]

    # 2) explicit labelled answer with a delimiter, e.g. "b) the cat"
    m = _LABEL_RE.match(raw_pred.strip())
    if m:
        body_norm = normalize_for_match(m.group(2))
        if body_norm in choice_map:
            return choice_map[body_norm]
        idx = ord(m.group(1).lower()) - ord("a")
        if 0 <= idx < len(choices):
            return choices[idx]

    # 3) a single choice appears verbatim inside the output
    hits = [c for c in choices if normalize_for_match(c) and normalize_for_match(c) in pred_norm]
    if len(hits) == 1:
        return hits[0]

    # 4) the output is a prefix/substring of exactly one choice (truncated gen)
    hits2 = [c for c in choices if pred_norm and pred_norm in normalize_for_match(c)]
    if len(hits2) == 1:
        return hits2[0]

    return None


def load_predictions_csv(path: Path) -> dict[str, str]:
    preds = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["question"].strip()
            preds[qid] = row["answer"]
    return preds


def load_predictions_json(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    preds = {}
    for row in data:
        qid = row.get("id") or row.get("question")
        pred = row.get("answer_prediction") or row.get("prediction") or row.get("model_output")
        if qid and pred is not None:
            preds[qid] = pred
    return preds


def evaluate(gt: dict[str, dict], preds: dict[str, str]) -> dict:
    total = 0
    correct = 0
    unresolved = 0
    per_sample = []

    for qid, ref in gt.items():
        if qid not in preds:
            continue
        total += 1
        resolved = resolve_to_choice(preds[qid], ref["multi_choice"])
        ok = resolved == ref["answer"]
        if resolved is None:
            unresolved += 1
        if ok:
            correct += 1
        per_sample.append({
            "id": qid,
            "gt": ref["answer"],
            "raw_pred": preds[qid],
            "resolved_pred": resolved,
            "correct": ok,
        })

    acc = correct / total if total else 0.0
    random_guess = sum(1.0 / len(r["multi_choice"]) for r in gt.values()) / len(gt)

    return {
        "accuracy": acc,
        "accuracy_pct": acc * 100,
        "correct": correct,
        "total_scored": total,
        "missing_predictions": len(gt) - total,
        "unresolved_predictions": unresolved,
        "random_guess_accuracy": random_guess,
        "per_sample": per_sample,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_jsonl", required=True)
    ap.add_argument("--predictions_csv", default=None)
    ap.add_argument("--predictions_json", default=None)
    ap.add_argument("--output", default=None, help="Optional summary JSON path")
    ap.add_argument("--write_submission_csv", default=None,
                    help="Write resolved plain-text answers to official CSV format")
    args = ap.parse_args()

    if not args.predictions_csv and not args.predictions_json:
        ap.error("Provide --predictions_csv or --predictions_json")

    gt = load_ground_truth(Path(args.dev_jsonl))
    if args.predictions_csv:
        preds = load_predictions_csv(Path(args.predictions_csv))
    else:
        preds = load_predictions_json(Path(args.predictions_json))

    summary = evaluate(gt, preds)
    print(f"Top-1 accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total_scored']})")
    print(f"Random guess baseline: {summary['random_guess_accuracy']:.4f}")
    if summary["unresolved_predictions"]:
        print(f"Unresolved predictions: {summary['unresolved_predictions']}")

    if args.write_submission_csv:
        out_path = Path(args.write_submission_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            for row in summary["per_sample"]:
                writer.writerow({
                    "question": row["id"],
                    "answer": row["resolved_pred"] or "",
                })
        print(f"Wrote submission CSV to {out_path}")

    if args.output:
        dump = dict(summary)
        dump.pop("per_sample")
        Path(args.output).write_text(json.dumps(dump, indent=2))
        print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
