from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from common import FROZEN_Q4_SEEDS, reject_prohibited_config, sha256_file  # noqa: E402
from materialize_q4 import QUOTAS, rank_major  # noqa: E402
from score_mmau import official_correct  # noqa: E402


class ReleaseTest(unittest.TestCase):
    def test_artifact_ledger_and_committed_labels(self):
        ledger = json.loads((ROOT / "provenance/artifacts.json").read_text())
        labels = ROOT / ledger["q4_labels"]["path"]
        self.assertEqual(labels.stat().st_size, ledger["q4_labels"]["bytes"])
        self.assertEqual(sha256_file(labels), ledger["q4_labels"]["sha256"])
        with gzip.open(labels, "rt", encoding="utf-8") as handle:
            rows = json.load(handle)
        self.assertEqual(len(rows), 289952)
        self.assertEqual(len({row["sample_key"] for row in rows}), 289952)
        self.assertEqual({row["split_v1"] for row in rows}, {"train", "probe"})
        self.assertTrue(all(len(row["teacher_correct_by_rotation"]) == 4 for row in rows))
        self.assertTrue(all(len(row["student_correct_by_rotation"]) == 4 for row in rows))
        self.assertEqual(sum(row['split_v1']=='train' for row in rows), 261171)
        for row in rows:
            t, s = row['teacher_correct_by_rotation'], row['student_correct_by_rotation']
            self.assertTrue(all(type(v) is bool for v in t+s))
            q = ('B' if all(s) else 'A') if all(t) else ('C' if all(s) else 'D')
            self.assertEqual(row['quadrant_gen4'], q)

    def test_joint_release_not_relabelled_history(self):
        current=json.loads((ROOT/'provenance/artifacts.json').read_text())
        old=json.loads((ROOT/'provenance/strongac_only_v1/artifacts.json').read_text())
        self.assertNotEqual(current['q4_labels']['sha256'], old['q4_labels']['sha256'])
        for seed in FROZEN_Q4_SEEDS:
            self.assertNotEqual(current['s3_q4']['manifests'][str(seed)]['sha256'], old['s3_q4']['manifests'][str(seed)]['sha256'])
        for stage in ['s1','s2']:
            self.assertEqual(current[stage], old[stage])

    def test_stage_templates_pass_safety_guard(self):
        for stage, template in (("s1", "s1.yaml.in"), ("s2", "s2.yaml.in"), ("s3-q4", "s3_q4.yaml.in")):
            value = (ROOT / "configs" / template).read_text()
            replacements = {
                "HF_CACHE": "/tmp/hf", "SMOLLM2_135M": "/tmp/slm",
                "S1_MANIFEST": "/tmp/s1.json", "S1_CED_CACHE": "/tmp/s1-cache",
                "S1_CHECKPOINT": "/tmp/s1.ckpt", "S2_MANIFEST": "/tmp/s2.json",
                "S2_CHECKPOINT": "/tmp/s2.ckpt", "S3_Q4_MANIFEST": "/tmp/q4.json",
                "S3_Q4_MANIFEST_SHA256": "0" * 64,
                **{f"S2_CED_CACHE_{index}": f"/tmp/cache{index}" for index in range(4)},
            }
            for key, replacement in replacements.items():
                value = value.replace("{{" + key + "}}", replacement)
            config = yaml.safe_load(value)
            reject_prohibited_config(config, stage)

    def test_guard_rejects_prohibited_key(self):
        with self.assertRaises(ValueError):
            reject_prohibited_config({"forced_choice": True, "train": {}}, "s1")

    def test_q4_constants_and_rank_major(self):
        self.assertEqual(FROZEN_Q4_SEEDS, (20260905, 20260906, 20260907, 20260908, 20260909))
        self.assertEqual(QUOTAS, {"A": 19200, "B": 13440, "C": 3840, "D": 1920})
        rows = [{"id": index} for index in range(128)]
        arranged = rank_major(rows)
        self.assertEqual([row["id"] for row in arranged[:32]], list(range(32)))
        self.assertEqual([row["id"] for row in arranged[32:64]], list(range(64, 96)))

    def test_official_mmau_content_match(self):
        choices = ["dog barking", "piano music", "rain falling", "a car horn"]
        self.assertTrue(official_correct("piano music", "The answer is piano music.", choices))
        self.assertFalse(official_correct("piano music", "I hear piano music and a car horn.", choices))
        self.assertFalse(official_correct("piano music", "", choices))


if __name__ == "__main__":
    unittest.main()
