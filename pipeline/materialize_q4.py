#!/usr/bin/env python3
"""Rebuild joint strongAC+AVQA Q4 from raw candidates and frozen generated labels."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random

from common import FROZEN_Q4_SEEDS, require, sha256_file


QUOTAS = {"A": 19200, "B": 13440, "C": 3840, "D": 1920}


def load_labels(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(f"strong-dual-v1-selection-{seed}")
    selected = []
    for quadrant, count in QUOTAS.items():
        pool = [row for row in rows if row["quadrant_gen4"] == quadrant]
        require(len(pool) >= count, f"quadrant {quadrant} has {len(pool)} < {count}")
        selected.extend(rng.sample(pool, count))
    rng.shuffle(selected)
    require(len(selected) == len({row["sample_key"] for row in selected}) == 38400,
            "Q4 selection must contain 38,400 unique identities")
    return selected


def render(selected: list[dict], seed: int) -> list[dict]:
    positions = list(range(4)) * (len(selected) // 4)
    random.Random(f"strong-dual-v1-presentation-{seed}").shuffle(positions)
    result = []
    for row, position in zip(selected, positions):
        rotation = (row["gold_option_index"] - position) % 4
        order = [(index + rotation) % 4 for index in range(4)]
        choices = [row["choices"][index] for index in order]
        gold = order.index(row["gold_option_index"])
        require(choices[gold] == row["gold_choice"], "gold content moved incorrectly")
        value = {
            **row,
            "sample_key": row["sample_key"] + f"::train_rotate{rotation}",
            "training_original_sample_key": row["sample_key"],
            "training_rotation": rotation,
            "training_permutation_new_to_old": order,
            "choices": choices,
            "gold_option_index": gold,
            "input": row["question"] + " " + " ".join(
                f"{letter}) {choice}" for letter, choice in zip("abcd", choices)
            ),
            "answer": f"{'abcd'[gold]}) {row['gold_choice']}",
        }
        # The executed manifest preserves source casing.  The frozen Stage-3
        # loader lowercases the rendered input immediately before tokenization.
        require("<|endoftext|>" not in value["input"], "question EOT is prohibited")
        result.append(value)
    require(Counter(row["gold_option_index"] for row in result) == Counter({0: 9600, 1: 9600, 2: 9600, 3: 9600}),
            "gold-position balance drift")
    return result


def rank_major(rows: list[dict], local_batch: int = 32, world: int = 2) -> list[dict]:
    global_batch = local_batch * world
    require(len(rows) % global_batch == 0, "nonintegral global batches")
    return [
        row
        for rank in range(world)
        for start in range(0, len(rows), global_batch)
        for row in rows[start + rank * local_batch:start + (rank + 1) * local_batch]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Frozen A2_RAW_DIRECT.json: original recordings, NOT S2 chunk pairs")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=FROZEN_Q4_SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"refuse to overwrite: {args.output}")

    ledger = json.loads((Path(__file__).resolve().parents[1] / "provenance/artifacts.json").read_text())
    require(sha256_file(args.candidates) == ledger['s3_candidates']['sha256'], 'candidate SHA drift')
    require(sha256_file(args.labels) == ledger['q4_labels']['sha256'], 'label SHA drift')
    with args.candidates.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    candidates = source
    require(Counter(r['source'] for r in candidates) == {'audiomcq':256077,'avqa':33875},
            'joint source universe drift')
    labels = load_labels(args.labels)
    label_by_id = {row["sample_key"]: row for row in labels}
    require(len(labels) == len(label_by_id) == 289952, "label identity coverage drift")
    require(set(label_by_id) == {row["sample_key"] for row in candidates}, "candidate/label ID mismatch")
    # The executed candidate artifact was hash-sharded before screening.  Its
    # order is preserved by the committed label list, while the S2 manifest has
    # a different order.  Iterate labels to reproduce random.sample exactly.
    source_by_id = {row["sample_key"]: row for row in candidates}
    rows = [
        {
            **source_by_id[label["sample_key"]],
            **label,
        }
        for label in labels
        if label["split_v1"] == "train"
    ]
    require(len(rows) == 261171, "joint train universe must contain 261,171 rows")
    for row in rows:
        t, s = row['teacher_correct_by_rotation'], row['student_correct_by_rotation']
        require(len(t) == len(s) == 4 and all(type(v) is bool for v in t+s), 'four generated outcomes required')
        q = ('B' if all(s) else 'A') if all(t) else ('C' if all(s) else 'D')
        require(q == row['quadrant_gen4'], 'inconsistent gen4 quadrant')
        require(not row['filepath2'] and len(row['choices']) == 4, 'single recording and four choices required')

    materialized = rank_major(render(choose(rows, args.seed), args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write('[')
        for index, row in enumerate(materialized):
            if index:
                handle.write(',\n')
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')))
        handle.write(']\n')
    digest = sha256_file(args.output)
    expected = ledger['s3_q4']['manifests'][str(args.seed)]['sha256']
    require(digest == expected, f"manifest SHA mismatch: expected {expected}, got {digest}")
    print(f"wrote {args.output} rows=38400 sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
