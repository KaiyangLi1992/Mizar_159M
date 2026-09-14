#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from common import FROZEN_Q4_SEEDS, reject_prohibited_config, require, sha256_file


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "s1": ROOT / "configs/s1.yaml.in",
    "s2": ROOT / "configs/s2.yaml.in",
    "s3-q4": ROOT / "configs/s3_q4.yaml.in",
}


def replace(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: replace(child, mapping) for key, child in value.items()}
    if isinstance(value, list):
        return [replace(child, mapping) for child in value]
    if isinstance(value, str):
        for key, replacement in mapping.items():
            value = value.replace("{{" + key + "}}", replacement)
        require("{{" not in value and "}}" not in value, f"unresolved placeholder: {value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Render one path-portable Mizar config")
    parser.add_argument("stage", choices=sorted(TEMPLATES))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    assets = yaml.safe_load(args.assets.read_text())
    caches = assets.get("s2_ced_cache_roots", [])
    require(len(caches) == 4, "assets.s2_ced_cache_roots must have four paths")
    mapping = {
        "HF_CACHE": str(assets["hf_cache"]),
        "SMOLLM2_135M": str(assets["smollm2_135m"]),
        "S1_MANIFEST": str(assets["s1_manifest"]),
        "S1_CED_CACHE": str(assets["s1_ced_cache"]),
        "S1_CHECKPOINT": str(assets["s1_checkpoint"]),
        "S2_MANIFEST": str(assets["s2_manifest"]),
        "S2_CHECKPOINT": str(assets["s2_checkpoint"]),
        **{f"S2_CED_CACHE_{index}": str(path) for index, path in enumerate(caches)},
    }

    config = yaml.safe_load(TEMPLATES[args.stage].read_text())
    if args.stage == "s3-q4":
        require(args.seed in FROZEN_Q4_SEEDS, f"S3-Q4 seed must be one of {FROZEN_Q4_SEEDS}")
        manifests = {int(key): value for key, value in assets["s3_q4_manifests"].items()}
        require(args.seed in manifests, f"missing Q4 manifest path for seed {args.seed}")
        mapping["S3_Q4_MANIFEST"] = str(manifests[args.seed])
        ledger = json.loads((ROOT / 'provenance/artifacts.json').read_text())
        mapping["S3_Q4_MANIFEST_SHA256"] = ledger['s3_q4']['manifests'][str(args.seed)]['sha256']
    else:
        require(args.seed is None, "--seed is only valid for s3-q4")

    config = replace(config, mapping)
    if args.stage == "s3-q4":
        seed = args.seed
        config["myconfig"] = f"S3STRONG_MERGED_Q4_S{seed}"
        config["data"]["order"].update(seed=seed, sampler_seed=seed)
        config["train"].update(random_seed=seed, data_order_seed=seed)
        config["experiment_notes"].update(token=f"merged_q4_s{seed}", seed=seed)

    config["model"]["encoder"]["pretrained_audioencoder_path"] = str(assets.get("ced_small", "mispeech/ced-small"))
    reject_prohibited_config(config, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"refuse to overwrite: {args.output}")
    args.output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    print(f"wrote {args.output} sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
