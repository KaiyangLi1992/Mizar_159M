#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from common import reject_prohibited_config, require


ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = {"s1": ROOT / "runtime/stage1", "s2": ROOT / "runtime/stage2", "s3-q4": ROOT / "runtime/stage3"}
WORLD = {"s1": 4, "s2": 2, "s3-q4": 2}


def main() -> int:
    parser = argparse.ArgumentParser(description="Print or execute one Mizar training stage")
    parser.add_argument("stage", choices=sorted(RUNTIMES))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    require(args.config.is_file(), f"missing config: {args.config}")
    config = yaml.safe_load(args.config.read_text())
    reject_prohibited_config(config, args.stage)
    require(config["train"]["world_size"] == WORLD[args.stage], "world-size drift")
    command = [args.python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={WORLD[args.stage]}", str(RUNTIMES[args.stage] / "train.py"),
               "--config", str(args.config.resolve()), "--save-dir", str(args.output.resolve()),
               "--distributed-backend", "nccl", "--reraise-exceptions"]
    print("command:")
    print(shlex.join(command))
    if not args.execute:
        print("dry-run only; add --execute inside the GPU allocation")
        return 0
    require(not args.output.exists() or not any(args.output.iterdir()),
            "Use a fresh output directory; stage continuation must not auto-resume optimizer state")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MELLOW_AUTO_RESUME_LATEST="0")
    if env.get("MIZAR_AUDIO_PATH_MAP"):
        env["MIZAR_AUDIO_PATH_MAP"] = str(Path(env["MIZAR_AUDIO_PATH_MAP"]).resolve())
    return subprocess.run(command, cwd=RUNTIMES[args.stage], env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
