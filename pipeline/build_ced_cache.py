#!/usr/bin/env python3
"""Precompute frozen CED-Small hidden tokens for Experiment C.

The cache is intentionally before the trainable mapper/resampler:

    waveform -> frozen CED encoder -> hidden [T, 384]

For CED last-4-layer finetuning, cache_point=last5_out stores the output after
the frozen lower CED blocks and before the final four trainable CED blocks.

Fixed 20s B/C and raw variable E2 use separate cache roots because their audio
length policy and hidden-token validity differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "runtime/stage1"
sys.path.insert(0, str(TRAIN_ROOT))

from models.ced import CEDSmallWrapper  # noqa: E402
from utils.audio_paths import load_audio


def stable_key(path1: str, path2: str, policy: str) -> str:
    raw = f"{policy}\n{path1}\n{path2}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def load_mono(path: str, sampling_rate: int) -> torch.Tensor:
    wav, sr = load_audio(path, channels_first=True)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sampling_rate:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sampling_rate)
    return wav[0].contiguous()


def fixed_len(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.numel() > target_len:
        start = (wav.numel() - target_len) // 2
        return wav[start : start + target_len].contiguous()
    if wav.numel() < target_len:
        return torch.nn.functional.pad(wav, (0, target_len - wav.numel()))
    return wav


def load_pair(path1: str, path2: str, args: argparse.Namespace) -> torch.Tensor:
    wav1 = load_mono(path1, args.sampling_rate)
    if not str(path2 or "").strip():
        if args.policy == "fixed20":
            target = int(args.segment_seconds * args.sampling_rate) * 2
            return fixed_len(wav1, target)
        return wav1
    wav2 = load_mono(path2, args.sampling_rate)
    if args.policy == "variable" and Path(path1).resolve() == Path(path2).resolve():
        return wav1
    if args.policy == "fixed20":
        target = int(args.segment_seconds * args.sampling_rate)
        wav1 = fixed_len(wav1, target)
        wav2 = fixed_len(wav2, target)
    return torch.cat((wav1, wav2), dim=0)


def unique_pairs(rows: list[dict], policy: str) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    index_by_key = {}
    keys: list[tuple[str, str, str]] = []
    row_to_cache = np.full(len(rows), -1, dtype=np.int64)
    for row_idx, row in enumerate(rows):
        path1 = str(row["filepath1"])
        path2 = str(row["filepath2"])
        key = stable_key(path1, path2, policy)
        cache_idx = index_by_key.get(key)
        if cache_idx is None:
            cache_idx = len(keys)
            index_by_key[key] = cache_idx
            keys.append((key, path1, path2))
        row_to_cache[row_idx] = cache_idx
    return keys, row_to_cache


def build_encoder(args: argparse.Namespace, device: torch.device) -> CEDSmallWrapper:
    encoder = CEDSmallWrapper(
        encoder_config={
            "pretrained_audioencoder_path": args.ced_model_name,
            "hf_cache_dir": args.hf_cache_dir,
            "input_sampling_rate": args.sampling_rate,
            "ced_sampling_rate": args.ced_sampling_rate,
            "freeze_audio_encoder_weights": True,
            "ced_mapper": "freq_merge_variable",
            "ced_variable_length": args.policy == "variable",
        },
        d_out=576,
    )
    encoder.eval()
    encoder.to(device)
    return encoder


def encode_one(
    encoder: CEDSmallWrapper,
    waveform: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    cache_point: str,
    finetune_last_n_layers: int,
) -> tuple[np.ndarray, list[int], bool]:
    def forward(amp_enabled: bool) -> tuple[torch.Tensor, list[int]]:
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=amp_enabled,
            dtype=torch.float16,
        ):
            mel = encoder._waveform_to_mel(waveform.unsqueeze(0).to(device))
            if cache_point == "last5_out":
                segments = encoder._forward_pre_last_n_segments(
                    mel,
                    finetune_last_n_layers,
                )
            else:
                segments = encoder._forward_hidden_segments(mel)
            hidden_tensor = torch.cat(segments, dim=1)[0]
            lengths = [int(seg.shape[1]) for seg in segments]
        return hidden_tensor, lengths

    hidden_tensor, segment_lengths = forward(use_amp)
    used_fp32_fallback = False
    if not bool(torch.isfinite(hidden_tensor).all()):
        if not use_amp:
            raise RuntimeError("CED produced non-finite hidden values in full precision")
        hidden_tensor, segment_lengths = forward(False)
        used_fp32_fallback = True
    if not bool(torch.isfinite(hidden_tensor).all()):
        raise RuntimeError("CED FP32 fallback still produced non-finite hidden values")

    hidden_f16 = hidden_tensor.detach().cpu().to(torch.float16)
    if not bool(torch.isfinite(hidden_f16).all()):
        raise RuntimeError("finite CED hidden values overflowed while casting to float16")
    return hidden_f16.numpy(), segment_lengths, used_fp32_fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--policy", choices=["fixed20", "variable"], required=True)
    parser.add_argument("--cache-point", choices=["final_hidden", "last5_out"], default="final_hidden")
    parser.add_argument("--finetune-last-n-layers", type=int, default=4)
    parser.add_argument("--ced-model-name", default="mispeech/ced-small")
    parser.add_argument("--hf-cache-dir", default=str(REPO_ROOT / "hf_cache"))
    parser.add_argument("--sampling-rate", type=int, default=32000)
    parser.add_argument("--ced-sampling-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--limit-unique", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--parent-num-shards", type=int, default=0)
    parser.add_argument("--parent-shard-index", type=int, default=0)
    parser.add_argument("--num-sub-shards", type=int, default=1)
    parser.add_argument("--sub-shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--write-exact20-fingerprints",
        action="store_true",
        help="write the canonical fixed20 float32 waveform hash into keys.jsonl",
    )
    args = parser.parse_args()
    if args.cache_point == "last5_out" and args.finetune_last_n_layers <= 0:
        raise SystemExit("--cache-point last5_out requires --finetune-last-n-layers > 0")

    if args.out_root.exists():
        raise FileExistsError(f"Refuse to overwrite cache: {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.train_json.read_text())
    keys, row_to_cache = unique_pairs(rows, args.policy)
    if args.parent_num_shards > 0:
        selected = []
        parent_local_idx = 0
        for idx, item in enumerate(keys):
            if idx % args.parent_num_shards != args.parent_shard_index:
                continue
            if parent_local_idx % args.num_sub_shards == args.sub_shard_index:
                selected.append(item)
            parent_local_idx += 1
    else:
        selected = [
            item for idx, item in enumerate(keys)
            if idx % args.num_shards == args.shard_index
        ]
    if args.limit_unique > 0:
        selected = selected[: args.limit_unique]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(args, device)
    use_amp = device.type == "cuda"

    hidden_path = args.out_root / "hidden.f16.dat"
    hidden_mm = np.memmap(
        hidden_path,
        dtype=np.float16,
        mode="w+",
        shape=(len(selected), args.max_tokens, encoder.hidden_dim),
    )
    valid_tokens = np.zeros(len(selected), dtype=np.int32)
    segment_lengths = np.zeros((len(selected), args.max_segments), dtype=np.int32)

    key_lines = []
    for out_idx, (key, path1, path2) in enumerate(selected):
        waveform = load_pair(path1, path2, args)
        audio_hash = None
        if args.write_exact20_fingerprints:
            canonical = waveform.detach().cpu().to(torch.float32).contiguous().numpy()
            audio_hash = "sha256-f32le-32000:" + hashlib.sha256(
                canonical.astype("<f4", copy=False).tobytes(order="C")
            ).hexdigest()
        hidden, seg_lengths, used_fp32_fallback = encode_one(
            encoder,
            waveform,
            device,
            use_amp,
            cache_point=args.cache_point,
            finetune_last_n_layers=args.finetune_last_n_layers,
        )
        if used_fp32_fallback:
            print(
                f"recomputed non-finite AMP row in FP32: key={key} local_index={out_idx}",
                flush=True,
            )
        if hidden.shape[0] > args.max_tokens:
            raise RuntimeError(
                f"{key} produced {hidden.shape[0]} tokens, max_tokens={args.max_tokens}"
            )
        if len(seg_lengths) > args.max_segments:
            raise RuntimeError(
                f"{key} produced {len(seg_lengths)} segments, max_segments={args.max_segments}"
            )
        hidden_mm[out_idx, : hidden.shape[0], :] = hidden
        valid_tokens[out_idx] = hidden.shape[0]
        segment_lengths[out_idx, : len(seg_lengths)] = seg_lengths
        key_record = {
            "local_index": out_idx,
            "global_key": key,
            "filepath1": path1,
            "filepath2": path2,
        }
        if audio_hash is not None:
            key_record.update({
                "audio_hash": audio_hash,
                "audio_hash_algorithm": "sha256_f32le_mono_32000_fixed20_model_input",
            })
        key_lines.append(json.dumps(key_record))
        if (out_idx + 1) % 100 == 0:
            print(f"cached {out_idx + 1}/{len(selected)}", flush=True)

    hidden_mm.flush()
    np.save(args.out_root / "valid_tokens.npy", valid_tokens)
    np.save(args.out_root / "segment_lengths.npy", segment_lengths)
    np.save(args.out_root / "row_to_global_cache.npy", row_to_cache)
    (args.out_root / "keys.jsonl").write_text("\n".join(key_lines) + "\n")
    (args.out_root / "metadata.json").write_text(json.dumps({
        "train_json": str(args.train_json),
        "policy": args.policy,
        "cache_point": args.cache_point,
        "finetune_last_n_layers": args.finetune_last_n_layers if args.cache_point == "last5_out" else 0,
        "num_rows": len(rows),
        "num_unique_pairs": len(keys),
        "num_cached_this_shard": len(selected),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "parent_num_shards": args.parent_num_shards,
        "parent_shard_index": args.parent_shard_index,
        "num_sub_shards": args.num_sub_shards,
        "sub_shard_index": args.sub_shard_index,
        "hidden_path": str(hidden_path),
        "shape": [len(selected), args.max_tokens, encoder.hidden_dim],
        "dtype": "float16",
        "max_tokens": args.max_tokens,
        "max_segments": args.max_segments,
        "sampling_rate": args.sampling_rate,
        "ced_sampling_rate": args.ced_sampling_rate,
        "segment_seconds": args.segment_seconds,
        "ced_model_name": args.ced_model_name,
        "write_exact20_fingerprints": args.write_exact20_fingerprints,
        "audio_hash_algorithm": (
            "sha256_f32le_mono_32000_fixed20_model_input"
            if args.write_exact20_fingerprints
            else None
        ),
    }, indent=2) + "\n")
    print(f"wrote cache shard to {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
