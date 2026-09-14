from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FROZEN_Q4_SEEDS = (20260905, 20260906, 20260907, 20260908, 20260909)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reject_prohibited_config(config: dict[str, Any], stage: str) -> None:
    """Fail closed on prohibited inference/selection or active auxiliary loss."""

    def walk(value: Any, path: tuple[str, ...] = ()):
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = key.lower()
                require("forced_choice" not in lowered, f"prohibited key: {'.'.join(path + (key,))}")
                walk(child, path + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + (str(index),))
        elif isinstance(value, str):
            lowered = value.lower().replace("-", "_")
            require("forced_choice" not in lowered, f"prohibited value at {'.'.join(path)}")

    walk(config)
    train = config.get("train", {})
    for key, value in train.items():
        lowered = key.lower()
        if any(token in lowered for token in ("kd", "replay", "quadrant", "reinforce")):
            enabled = value.get("enabled", False) if isinstance(value, dict) else bool(value)
            require(not enabled, f"active auxiliary objective is not part of release: train.{key}")
    if stage in {"s2", "s3-q4"}:
        notes = json.dumps(config.get("experiment_notes", {}), ensure_ascii=False).lower()
        require("cot" not in notes, f"{stage} must be direct-answer CE")
    if stage == "s3-q4":
        data = config["data"]
        require(data.get("audio_mode") == "normal", "S3 must use normal audio")
        require("ced_hidden_cache" not in data, "S3 must read raw first-20-second audio")
        require(config["train"]["optimizer"]["schedule_total_optimizer_steps"] == 600,
                "S3 LR horizon must remain 600")
        require(config["train"]["checkpoint_optimizer_steps"] == [100, 200, 400, 600],
                "S3 checkpoint grid drift")
