import json
import hashlib
import math
import random
import torch
import torchaudio
from utils.audio_paths import load_audio
from torch.utils.data import Dataset
import os
import random
from pathlib import Path
from tqdm import tqdm
from scipy.signal import fftconvolve
import numpy as np
from transformers import AutoTokenizer
from data.template import DETAIL, WORD, BOTH, FIRST, SECOND, EMOTION, LONGLINE, BOTH_SPK, FIRST_SPK, SECOND_SPK
from data.template import DETAILONLY, WORDONLY, LONGLINEONLY
from data.lazy_memmap import LazyMemmap
from collections import Counter, defaultdict


QSAFE_FAMILY_GENERAL = 0
QSAFE_FAMILY_SPEECH_CONTENT = 1
QSAFE_FAMILY_SPEAKER_PROSODY = 2
QSAFE_FAMILY_TEMPORAL = 3
QSAFE_FAMILY_QUALITY = 4


def classify_question_safe_family(row):
    """Conservatively route one MCQ to an acoustic-augmentation safety family."""

    text = " ".join(
        str(value or "")
        for value in (
            row.get("question"),
            row.get("input"),
            " ".join(str(item) for item in (row.get("choices") or [])),
        )
    ).lower()
    quality_terms = (
        "audio quality", "recording quality", "distortion", "distorted", "clipping",
        "noisy recording", "background noise", "signal-to-noise", "loudness",
        "volume level", "quiet or loud", "muffled", "clear recording",
    )
    temporal_terms = (
        "first", "last", "before", "after", "then", "followed", "precede",
        "order", "sequence", "how many", "number of", "count", "duration",
        "longer", "shorter", "start", "end", "progresses", "beginning",
    )
    speaker_terms = (
        "speaker", "voice", "gender", "male", "female", "emotion", "prosody",
        "tone of voice", "accent", "age of", "pitch of", "speaking style",
    )
    speech_terms = (
        "say", "says", "said", "spoken", "speech", "word", "phrase", "sentence",
        "language", "conversation", "talking", "describ", "mention", "utter",
    )
    if any(term in text for term in quality_terms):
        return QSAFE_FAMILY_QUALITY
    if any(term in text for term in temporal_terms):
        return QSAFE_FAMILY_TEMPORAL
    if any(term in text for term in speaker_terms):
        return QSAFE_FAMILY_SPEAKER_PROSODY
    if any(term in text for term in speech_terms):
        return QSAFE_FAMILY_SPEECH_CONTENT
    return QSAFE_FAMILY_GENERAL


def question_safe_audio_augment(waveform, family, seed):
    """Deterministic channel perturbation with family-specific safety limits.

    The transform never crops, shifts, stretches, reverses, pitch-shifts, or masks
    time spans.  Quality/loudness questions receive an identity view.
    """

    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"Q-SAFE expects mono [1,T], got {tuple(waveform.shape)}")
    family = int(family)
    if family == QSAFE_FAMILY_QUALITY:
        return waveform.clone()
    limits = {
        QSAFE_FAMILY_GENERAL: (3.0, 20.0, 28.0, 0.35),
        QSAFE_FAMILY_SPEECH_CONTENT: (2.0, 24.0, 32.0, 0.20),
        QSAFE_FAMILY_SPEAKER_PROSODY: (1.5, 28.0, 36.0, 0.10),
        QSAFE_FAMILY_TEMPORAL: (1.0, 30.0, 38.0, 0.05),
    }
    if family not in limits:
        raise ValueError(f"unknown Q-SAFE family {family}")
    gain_db, snr_low, snr_high, color_mix = limits[family]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    gain = (torch.rand((), generator=generator).item() * 2.0 - 1.0) * gain_db
    augmented = waveform.float() * float(10.0 ** (gain / 20.0))
    noise = torch.randn(waveform.shape, generator=generator, dtype=torch.float32)
    if color_mix > 0.0 and noise.shape[-1] >= 9:
        smooth = torch.nn.functional.avg_pool1d(
            noise.unsqueeze(0), kernel_size=9, stride=1, padding=4
        ).squeeze(0)
        noise = (1.0 - color_mix) * noise + color_mix * smooth
    signal_rms = augmented.square().mean().sqrt().clamp_min(1e-5)
    noise_rms = noise.square().mean().sqrt().clamp_min(1e-5)
    snr_db = snr_low + torch.rand((), generator=generator).item() * (snr_high - snr_low)
    noise_scale = signal_rms / (noise_rms * float(10.0 ** (snr_db / 20.0)))
    return (augmented + noise * noise_scale).clamp(-1.0, 1.0).to(waveform.dtype)


def resolve_naive_kd_target_scores(
    row, target_scores_key, mask_key="is_kd_sample"
):
    """Resolve one row's KD target without allowing collate-time imputation."""
    if (
        not isinstance(target_scores_key, str)
        or not target_scores_key.isidentifier()
        or not target_scores_key.endswith("_scores_abcd")
    ):
        raise ValueError(
            "target_scores_key must be an identifier ending in _scores_abcd, "
            f"got {target_scores_key!r}"
        )
    if not isinstance(mask_key, str) or not mask_key.isidentifier():
        raise ValueError(f"mask_key must be an identifier, got {mask_key!r}")
    is_kd_sample = bool(row.get(mask_key, False))
    scores = row.get(target_scores_key)
    if scores is None:
        if is_kd_sample:
            raise KeyError(
                f"KD sample is missing {target_scores_key}: "
                f"{row.get('sample_key') or row.get('id')}"
            )
        return [0.0, 0.0, 0.0, 0.0]
    if (
        not isinstance(scores, list)
        or len(scores) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in scores
        )
    ):
        raise ValueError(
            f"Invalid {target_scores_key} for "
            f"{row.get('sample_key') or row.get('id')}"
        )
    return scores


def ke_cot_output_control_region_ids(tokenizer, completion: str, max_len: int) -> torch.Tensor:
    """Mark padded/structure/reasoning/answer tokens as 0/1/2/3."""

    if not isinstance(completion, str) or not completion:
        raise ValueError("output-control completion must be non-empty")
    if not isinstance(max_len, int) or max_len <= 0:
        raise ValueError("output-control max_len must be positive")
    think_open = "<think>"
    think_close = "</think>"
    answer_open = "<answer>"
    answer_close = "</answer>"
    if not completion.startswith(think_open) or completion.count(think_open) != 1:
        raise ValueError("output-control completion has invalid think opening tag")
    if any(completion.count(tag) != 1 for tag in (think_close, answer_open, answer_close)):
        raise ValueError("output-control completion must contain one copy of every closing tag")
    think_start = len(think_open)
    think_stop = completion.index(think_close)
    answer_start = completion.index(answer_open, think_stop) + len(answer_open)
    answer_stop = completion.index(answer_close, answer_start)
    if think_start >= think_stop or answer_start >= answer_stop:
        raise ValueError("output-control reasoning and answer spans must be non-empty")

    serialized = completion + " <|endoftext|>"
    encoded = tokenizer.encode_plus(
        serialized,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    attention = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]
    if len(attention) != max_len or len(offsets) != max_len:
        raise ValueError("output-control tokenization did not preserve fixed length")
    regions = torch.zeros(max_len, dtype=torch.long)
    for position, (active, offset) in enumerate(zip(attention, offsets)):
        if not active:
            continue
        char_start, char_stop = offset
        if char_stop > char_start and think_start <= char_start and char_stop <= think_stop:
            regions[position] = 2
        elif char_stop > char_start and answer_start <= char_start and char_stop <= answer_stop:
            regions[position] = 3
        else:
            regions[position] = 1
    if not bool((regions == 2).any()) or not bool((regions == 3).any()):
        raise ValueError("output-control tokenization lost reasoning or answer span")
    return regions

class AudioTextDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """
    def __init__(
        self,
        data_path="",
        datafiles=[''],
        sampling_rate=44100, 
        max_clip_len=10,
        tokenizer_type="gpt2",
        ip_text_len=40,
        op_text_len=300,
        audio_mode="normal",
        audio_length_policy="fixed",
        ced_hidden_cache=None,
        beats_hidden_cache=None,
        alignkd_teacher_cache=None,
        sequence_retention_cache=None,
        naive_kd_target_scores_key="teacher_scores_abcd",
        output_control_regions=False,
        mixture_config=None,
        order_config=None,
        question_safe_augmentation=None,
    ):
        self.collect_data_jsons(data_path, datafiles)
        self.apply_mixture(mixture_config)
        self.apply_data_order(order_config)
        self.ip_text_len = ip_text_len
        self.op_text_len = op_text_len
        self.audio_mode = audio_mode
        self.audio_length_policy = str(audio_length_policy or "fixed").lower()
        self.variable_audio_length = self.audio_length_policy in {
            "variable",
            "raw",
            "no_crop",
            "no_crop_no_pad",
        }

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate
        self.tokenizer_type = tokenizer_type
        self.tokenizer = self._create_tokenizer(tokenizer_type)
        self.ced_hidden_cache = self._load_ced_hidden_cache(ced_hidden_cache)
        self.beats_hidden_cache = self._load_beats_hidden_cache(beats_hidden_cache)
        self.alignkd_teacher_cache = self._load_alignkd_teacher_cache(
            alignkd_teacher_cache
        )
        self.sequence_retention_cache = self._load_sequence_retention_cache(
            sequence_retention_cache
        )
        if (
            not isinstance(naive_kd_target_scores_key, str)
            or not naive_kd_target_scores_key.isidentifier()
            or not naive_kd_target_scores_key.endswith("_scores_abcd")
        ):
            raise ValueError(
                "naive_kd_target_scores_key must be an identifier ending in "
                f"_scores_abcd, got {naive_kd_target_scores_key!r}"
            )
        self.naive_kd_target_scores_key = naive_kd_target_scores_key
        self.output_control_regions = bool(output_control_regions)
        self.question_safe_augmentation = dict(question_safe_augmentation or {})
        self.qsafe_enabled = bool(self.question_safe_augmentation.get("enabled", False))
        self.qsafe_apply_transforms = bool(
            self.question_safe_augmentation.get("apply_transforms", False)
        )
        self.qsafe_seed = int(self.question_safe_augmentation.get("seed", 20260828))
        if self.qsafe_enabled and self.ced_hidden_cache is not None:
            raise ValueError(
                "Q-SAFE acoustic views require raw waveforms; remove data.ced_hidden_cache"
            )
        if self.ced_hidden_cache is not None and self.beats_hidden_cache is not None:
            raise ValueError("Use only one hidden cache: ced_hidden_cache or beats_hidden_cache")
    
    def _create_tokenizer(self, tokenizer_type):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        # if 'gpt' in tokenizer_type or:
        tokenizer.add_special_tokens({'pad_token': '!'})
        return tokenizer

    def _load_ced_hidden_cache(self, cache_config):
        if not cache_config:
            return None
        if isinstance(cache_config, str):
            config = {}
            roots = [Path(cache_config)]
        else:
            config = dict(cache_config)
            root_value = config.get("root")
            roots_value = config.get("roots")
            if root_value and roots_value:
                raise ValueError(
                    "data.ced_hidden_cache accepts either root or roots, not both"
                )
            if roots_value is not None:
                if (
                    not isinstance(roots_value, (list, tuple))
                    or not roots_value
                    or any(not str(value) for value in roots_value)
                ):
                    raise ValueError(
                        "data.ced_hidden_cache.roots must be a non-empty list"
                    )
                roots = [Path(str(value)) for value in roots_value]
            elif root_value:
                roots = [Path(str(root_value))]
            else:
                raise ValueError(
                    "data.ced_hidden_cache.root or .roots is required"
                )
        strict = bool(config.get("strict", False))
        expected_policy = str(config.get("policy", ""))
        expected_cache_point = str(config.get("cache_point", ""))
        if strict and len(set(map(str, roots))) != len(roots):
            raise ValueError("Duplicate data.ced_hidden_cache.roots entry")
        shard_roots = []
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"CED hidden cache root not found: {root}")
            discovered = (
                [root]
                if (root / "metadata.json").exists()
                else sorted(p for p in root.iterdir() if (p / "metadata.json").exists())
            )
            if not discovered:
                raise FileNotFoundError(
                    f"No CED hidden cache metadata found under {root}"
                )
            shard_roots.extend(discovered)
        if not shard_roots:
            raise FileNotFoundError("No CED hidden cache metadata found")
        shards = []
        key_to_index = {}
        reference_policy = None
        reference_cache_point = None
        reference_hidden_shape = None
        for shard_id, shard_root in enumerate(shard_roots):
            metadata = json.loads((shard_root / "metadata.json").read_text())
            shape = tuple(metadata["shape"])
            if strict:
                if len(shape) != 3 or any(int(value) <= 0 for value in shape):
                    raise ValueError(f"Invalid CED cache shape in {shard_root}: {shape}")
                if expected_policy and metadata.get("policy") != expected_policy:
                    raise ValueError(
                        f"CED cache policy mismatch in {shard_root}: "
                        f"{metadata.get('policy')} != {expected_policy}"
                    )
                if expected_cache_point and metadata.get("cache_point") != expected_cache_point:
                    raise ValueError(
                        f"CED cache point mismatch in {shard_root}: "
                        f"{metadata.get('cache_point')} != {expected_cache_point}"
                    )
                policy = metadata.get("policy")
                cache_point = metadata.get("cache_point")
                hidden_shape = shape[1:]
                if reference_policy is None:
                    reference_policy = policy
                    reference_cache_point = cache_point
                    reference_hidden_shape = hidden_shape
                elif (
                    policy != reference_policy
                    or cache_point != reference_cache_point
                    or hidden_shape != reference_hidden_shape
                ):
                    raise ValueError(
                        "Incompatible CED hidden cache roots: policy, cache point, "
                        f"or non-batch shape differs in {shard_root}"
                    )
                if int(metadata.get("num_cached_this_shard", -1)) != int(shape[0]):
                    raise ValueError(f"CED cache row-count mismatch in {shard_root}")
                hidden_path = shard_root / "hidden.f16.dat"
                expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
                if not hidden_path.is_file() or hidden_path.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"CED cache hidden byte-size mismatch in {shard_root}"
                    )
            valid_tokens = np.load(shard_root / "valid_tokens.npy")
            segment_lengths = np.load(shard_root / "segment_lengths.npy")
            if strict:
                if valid_tokens.shape != (shape[0],):
                    raise ValueError(f"CED valid_tokens shape mismatch in {shard_root}")
                if segment_lengths.ndim != 2 or segment_lengths.shape[0] != shape[0]:
                    raise ValueError(f"CED segment_lengths shape mismatch in {shard_root}")
                if bool(((valid_tokens <= 0) | (valid_tokens > shape[1])).any()):
                    raise ValueError(f"CED valid_tokens values are invalid in {shard_root}")
            shards.append({
                "root": shard_root,
                "metadata": metadata,
                "hidden": LazyMemmap(
                    shard_root / "hidden.f16.dat",
                    dtype=np.float16,
                    mode="r",
                    shape=shape,
                ),
                "valid_tokens": valid_tokens,
                "segment_lengths": segment_lengths,
            })
            shard_local_indices = set()
            shard_key_count = 0
            with open(shard_root / "keys.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    global_key = str(row["global_key"])
                    local_index = int(row["local_index"])
                    shard_key_count += 1
                    if strict and global_key in key_to_index:
                        raise ValueError(f"Duplicate CED cache key: {global_key}")
                    if strict and not 0 <= local_index < shape[0]:
                        raise ValueError(
                            f"CED cache local_index out of bounds in {shard_root}: {local_index}"
                        )
                    if strict:
                        if (
                            len(global_key) != 40
                            or any(char not in "0123456789abcdef" for char in global_key)
                        ):
                            raise ValueError(
                                f"Invalid CED cache global_key in {shard_root}: {global_key}"
                            )
                        if local_index in shard_local_indices:
                            raise ValueError(
                                f"Duplicate CED cache local_index in {shard_root}: {local_index}"
                            )
                        shard_local_indices.add(local_index)
                    key_to_index[global_key] = (shard_id, local_index)
            if strict and (
                shard_key_count != shape[0]
                or shard_local_indices != set(range(shape[0]))
            ):
                raise ValueError(
                    f"CED cache key/local-index coverage mismatch in {shard_root}: "
                    f"keys={shard_key_count}, shape_rows={shape[0]}"
                )
        return {
            "root": roots[0],
            "roots": roots,
            "metadata": shards[0]["metadata"],
            "shards": shards,
            "key_to_index": key_to_index,
        }

    def _load_beats_hidden_cache(self, cache_config):
        if not cache_config:
            return None
        if isinstance(cache_config, str):
            root = Path(cache_config)
        else:
            root = Path(cache_config.get("root", ""))
        if not root:
            return None
        shard_roots = [root] if (root / "metadata.json").exists() else sorted(
            p for p in root.iterdir() if (p / "metadata.json").exists()
        )
        if not shard_roots:
            raise FileNotFoundError(f"No BEATs hidden cache metadata found under {root}")
        shards = []
        key_to_index = {}
        for shard_id, shard_root in enumerate(shard_roots):
            metadata = json.loads((shard_root / "metadata.json").read_text())
            shape = tuple(metadata["shape"])
            shards.append({
                "root": shard_root,
                "metadata": metadata,
                "hidden": LazyMemmap(
                    shard_root / "hidden.f16.dat",
                    dtype=np.float16,
                    mode="r",
                    shape=shape,
                ),
                "valid_tokens": np.load(shard_root / "valid_tokens.npy"),
                "segment_lengths": np.load(shard_root / "segment_lengths.npy"),
            })
            with open(shard_root / "keys.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key_to_index[row["global_key"]] = (shard_id, int(row["local_index"]))
        return {
            "root": root,
            "metadata": shards[0]["metadata"],
            "shards": shards,
            "key_to_index": key_to_index,
        }

    def _load_alignkd_teacher_cache(self, cache_config):
        if not cache_config:
            return None
        if isinstance(cache_config, str):
            index_path = Path(cache_config)
            config = {}
        else:
            config = dict(cache_config)
            index_path = Path(config.get("index", ""))
        if not str(index_path):
            raise ValueError("data.alignkd_teacher_cache.index is required")
        if not index_path.is_file():
            raise FileNotFoundError(f"AlignKD cache index not found: {index_path}")

        expected_schema = str(
            config.get("schema_version", "audio_alignkd_teacher_v1")
        )
        requested_fields = set(config.get("fields", []) or [])
        allowed_fields = {
            "attention",
            "mass",
            "gram",
            "focus",
            "top16",
        }
        unknown = requested_fields - allowed_fields
        if unknown:
            raise ValueError(f"Unknown AlignKD cache fields: {sorted(unknown)}")
        if "attention" in requested_fields:
            requested_fields.add("mass")

        entries = {}
        shard_paths = set()
        with index_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                sample_key = str(item.get("sample_key") or "")
                if not sample_key:
                    raise ValueError(
                        f"Missing sample_key in AlignKD cache index line {line_no}"
                    )
                if sample_key in entries:
                    raise ValueError(f"Duplicate AlignKD cache sample_key: {sample_key}")
                if item.get("cache_schema_version") != expected_schema:
                    raise ValueError(
                        f"AlignKD cache schema mismatch for {sample_key}: "
                        f"{item.get('cache_schema_version')} != {expected_schema}"
                    )
                # v1 extractor keeps the numeric shard id in ``cache_shard``
                # and the durable absolute directory in ``cache_shard_dir``.
                # Older hand-built caches used a string ``cache_shard`` (or
                # ``teacher_cache_shard``), so retain those as read-only
                # compatibility fallbacks without ever interpreting an int as
                # a filesystem path.
                shard_value = item.get("cache_shard_dir")
                if not isinstance(shard_value, str) or not shard_value:
                    legacy_shard = item.get("cache_shard")
                    if isinstance(legacy_shard, str) and legacy_shard:
                        shard_value = legacy_shard
                    else:
                        shard_value = item.get("teacher_cache_shard")
                if not isinstance(shard_value, str) or not shard_value:
                    raise ValueError(f"Missing cache_shard_dir for {sample_key}")
                shard_path = Path(shard_value)
                if not shard_path.is_absolute():
                    shard_path = (index_path.parent / shard_path).resolve()
                item = dict(item)
                item["cache_shard"] = str(shard_path)
                item["cache_row"] = int(item.get("cache_row", -1))
                if item["cache_row"] < 0:
                    raise ValueError(f"Invalid cache_row for {sample_key}")
                entries[sample_key] = item
                shard_paths.add(shard_path)

        field_specs = {
            "attention": (
                "teacher_attn_qabcd_126",
                "attn_qabcd_126.f16.dat",
                np.float16,
                (5, 126),
            ),
            "mass": (
                "teacher_audio_mass_qabcd",
                "audio_mass_qabcd.f16.dat",
                np.float16,
                (5,),
            ),
            "gram": (
                "teacher_feature_gram_126",
                "feature_gram_126.f16.dat",
                np.float16,
                (126, 126),
            ),
            "focus": (
                "teacher_focus_126",
                "focus_126.f16.dat",
                np.float16,
                (126,),
            ),
            "top16": (
                "teacher_top16_indices",
                "top16_indices.i16.dat",
                np.int16,
                (16,),
            ),
        }
        shards = {}
        for shard_path in sorted(shard_paths):
            metadata_path = shard_path / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"AlignKD shard metadata not found: {metadata_path}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            shard_schema = metadata.get("cache_schema_version") or metadata.get(
                "schema_version"
            )
            if shard_schema != expected_schema:
                raise ValueError(f"AlignKD shard schema mismatch: {shard_path}")
            rows = int(metadata.get("rows", -1))
            if rows <= 0:
                raise ValueError(f"Invalid AlignKD shard row count: {shard_path}")
            required_metadata = {
                "fixed_teacher_audio_tokens": 500,
                "target_audio_tokens": 126,
                "audio_hash_algorithm": "sha256_f32le_mono_32000_exact20",
                "prompt_hash_algorithm": "audio_alignkd_semantic_prompt_v1_nfkc_casefold_compact_json",
                "time_resample_algorithm": "area_overlap_mass_preserving_500_to_126_v1",
                "feature_resample_algorithm": "area_overlap_average_500_to_126_v1",
                "teacher_attention_layer_index": 0,
                "semantic_group_order": ["question", "a", "b", "c", "d"],
                "attention_softmax_denominator": "all_causally_visible_nonpadding_keys",
                "attention_head_and_group_token_reduction": "mean",
                "attention_cache_value": "conditional_distribution_within_audio_keys",
                "audio_mass_cache_value": "unconditional_attention_mass_over_all_audio_keys",
                "feature_source": "pre_text_llm_audio_embeddings",
                "feature_normalization": "parameter_free_layernorm_then_l2",
                "feature_cache_value": "temporal_cosine_gram_126x126",
                "focus_temperature": 2.0,
                "top_k": 16,
                "top_k_tie_break": "descending_score_then_ascending_time_index",
            }
            for metadata_key, expected_value in required_metadata.items():
                if metadata.get(metadata_key) != expected_value:
                    raise ValueError(
                        f"AlignKD shard metadata mismatch in {shard_path}: "
                        f"{metadata_key}={metadata.get(metadata_key)!r} "
                        f"expected={expected_value!r}"
                    )
            arrays = {}
            metadata_arrays = metadata.get("arrays", {}) or {}
            for requested in sorted(requested_fields):
                logical_name, default_file, dtype, suffix = field_specs[requested]
                spec = metadata_arrays.get(logical_name, {}) or {}
                filename = str(spec.get("file", default_file))
                shape = tuple(spec.get("shape", [rows, *suffix]))
                expected_shape = (rows, *suffix)
                if shape != expected_shape:
                    raise ValueError(
                        f"{logical_name} shape mismatch in {shard_path}: "
                        f"{shape} != {expected_shape}"
                    )
                path = shard_path / filename
                if not path.is_file():
                    raise FileNotFoundError(f"AlignKD cache array not found: {path}")
                expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                if path.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"AlignKD cache byte-size mismatch for {path}: "
                        f"{path.stat().st_size} != {expected_bytes}"
                    )
                arrays[logical_name] = LazyMemmap(
                    path, dtype=dtype, mode="r", shape=shape
                )
            shards[str(shard_path)] = {
                "metadata": metadata,
                "arrays": arrays,
            }

        return {
            "index_path": index_path,
            "schema_version": expected_schema,
            "fields": requested_fields,
            "entries": entries,
            "shards": shards,
        }

    def _load_sequence_retention_cache(self, cache_config):
        """Open the write-once sparse REF next-token cache used by Round 6."""
        if not cache_config:
            return None
        config = {"root": cache_config} if isinstance(cache_config, str) else dict(cache_config)
        root = Path(str(config.get("root") or ""))
        if not str(root):
            raise ValueError("data.sequence_retention_cache.root is required")
        root = root.resolve()
        metadata_path = root / "metadata.json"
        index_path = root / "index.jsonl"
        if not metadata_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"Sequence-retention cache metadata/index not found under {root}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_schema = str(
            config.get("schema_version", "audio_alignkd_sequence_retention_v1")
        )
        if metadata.get("schema_version") != expected_schema:
            raise ValueError(
                "Sequence-retention cache schema mismatch: "
                f"{metadata.get('schema_version')} != {expected_schema}"
            )
        rows = int(metadata.get("rows", -1))
        active_tokens = int(metadata.get("active_tokens", -1))
        top_k = int(metadata.get("top_k", -1))
        op_text_len = int(metadata.get("op_text_len", -1))
        tau = float(metadata.get("tau", -1.0))
        dataset_role = str(metadata.get("dataset_role") or "")
        if rows <= 0 or active_tokens <= 0:
            raise ValueError("Sequence-retention cache must contain rows and active tokens")
        if top_k != int(config.get("top_k", 32)):
            raise ValueError(f"Sequence-retention top_k mismatch: {top_k}")
        if op_text_len != int(self.op_text_len):
            raise ValueError(
                f"Sequence-retention op_text_len mismatch: {op_text_len} != {self.op_text_len}"
            )
        if tau != float(config.get("tau", 2.0)) or tau <= 0.0:
            raise ValueError(f"Sequence-retention tau mismatch: {tau}")
        if dataset_role != str(config.get("dataset_role", "reasonaqa_ce_replay")):
            raise ValueError(f"Sequence-retention dataset_role mismatch: {dataset_role}")
        required_metadata = {
            "selection_rule": "answer_nonpad_and_ref_top1_equals_gold_v1",
            "target_ids_hash_algorithm": "sha256_i32le_fixed_answer_tokens_v1",
            "probability_dtype": "float32",
            "token_id_dtype": "uint16",
            "position_dtype": "uint16",
            "normalization": "per_active_token_then_per_active_sample",
            "top_k_order": "descending_probability_then_ascending_token_id",
        }
        if expected_schema == "audio_alignkd_sequence_retention_v2":
            required_metadata.update(
                {
                    "eligibility_rule": "deterministic_freeform_prompt_only_v1",
                    "prompt_ids_hash_algorithm": "sha256_i32le_fixed_prompt_tokens_v1",
                }
            )
        for key, expected in required_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Sequence-retention metadata mismatch: {key}="
                    f"{metadata.get(key)!r} expected={expected!r}"
                )
        for key in ("tokenizer_sha256", "ref_initializer_sha256"):
            expected = config.get(key)
            if expected and str(metadata.get(key)) != str(expected):
                raise ValueError(
                    f"Sequence-retention {key} mismatch: {metadata.get(key)} != {expected}"
                )

        array_specs = {
            "row_offsets": ("row_offsets.i64.dat", np.int64, (rows + 1,)),
            "token_positions": ("token_positions.u16.dat", np.uint16, (active_tokens,)),
            "topk_ids": ("topk_ids.u16.dat", np.uint16, (active_tokens, top_k)),
            "topk_probs": ("topk_probs.f32.dat", np.float32, (active_tokens, top_k)),
            "tail_probs": ("tail_probs.f32.dat", np.float32, (active_tokens,)),
            "gold_token_ids": ("gold_token_ids.u16.dat", np.uint16, (active_tokens,)),
        }
        arrays = {}
        metadata_arrays = metadata.get("arrays", {}) or {}
        for name, (default_file, dtype, expected_shape) in array_specs.items():
            spec = metadata_arrays.get(name, {}) or {}
            path = root / str(spec.get("file", default_file))
            shape = tuple(int(value) for value in spec.get("shape", expected_shape))
            if shape != expected_shape:
                raise ValueError(
                    f"Sequence-retention {name} shape mismatch: {shape} != {expected_shape}"
                )
            expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"Sequence-retention array byte-size mismatch: {path}")
            arrays[name] = LazyMemmap(path, dtype=dtype, mode="r", shape=shape)
        offsets = arrays["row_offsets"]
        if (
            int(offsets[0]) != 0
            or int(offsets[-1]) != active_tokens
            or bool(np.any(offsets[1:] < offsets[:-1]))
        ):
            raise ValueError("Sequence-retention row_offsets are invalid")

        entries = {}
        seen_rows = set()
        with index_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                sample_key = str(item.get("sample_key") or "")
                cache_row = int(item.get("cache_row", -1))
                if not sample_key or sample_key in entries:
                    raise ValueError(
                        f"Invalid/duplicate sequence-retention key at line {line_no}"
                    )
                if not 0 <= cache_row < rows or cache_row in seen_rows:
                    raise ValueError(
                        f"Invalid/duplicate sequence-retention row at line {line_no}"
                    )
                if str(item.get("dataset_role") or "") != dataset_role:
                    raise ValueError(f"Sequence-retention role mismatch for {sample_key}")
                entries[sample_key] = item
                seen_rows.add(cache_row)
        if len(entries) != rows or len(seen_rows) != rows:
            raise ValueError(
                f"Sequence-retention index coverage mismatch: {len(entries)} != {rows}"
            )
        vocab_size = len(self.tokenizer)
        if vocab_size > np.iinfo(np.uint16).max + 1:
            raise ValueError(f"Tokenizer vocabulary exceeds uint16 cache ids: {vocab_size}")
        return {
            "root": root,
            "metadata": metadata,
            "entries": entries,
            "arrays": arrays,
            "top_k": top_k,
            "dataset_role": dataset_role,
            "vocab_size": vocab_size,
        }

    @staticmethod
    def _fixed_answer_token_sha256(answer_token_ids):
        values = np.asarray(answer_token_ids, dtype="<i4")
        return hashlib.sha256(values.tobytes(order="C")).hexdigest()

    def _sequence_retention_sentinel(self):
        cache = self.sequence_retention_cache
        top_k = int(cache["top_k"])
        return {
            "sequence_retention_topk_ids": torch.zeros(
                self.op_text_len, top_k, dtype=torch.long
            ),
            "sequence_retention_topk_probs": torch.zeros(
                self.op_text_len, top_k, dtype=torch.float32
            ),
            "sequence_retention_tail_probs": torch.zeros(
                self.op_text_len, dtype=torch.float32
            ),
            "sequence_retention_active_mask": torch.zeros(
                self.op_text_len, dtype=torch.bool
            ),
        }

    def _read_sequence_retention_cache(
        self, row, tokenized_answer, tokenized_input=None
    ):
        cache = self.sequence_retention_cache
        values = self._sequence_retention_sentinel()
        if str(row.get("dataset_role") or "") != cache["dataset_role"]:
            return values, False
        sample_key = str(row.get("sample_key") or row.get("id") or "")
        item = cache["entries"].get(sample_key)
        if item is None:
            raise KeyError(f"Sequence-retention cache miss for {sample_key}")
        if str(item.get("audio_hash") or "") != str(row.get("audio_hash") or ""):
            raise ValueError(f"Sequence-retention audio_hash mismatch for {sample_key}")
        target_ids = tokenized_answer["input_ids"].reshape(-1).long()
        if target_ids.numel() != self.op_text_len:
            raise ValueError(f"Sequence-retention target length mismatch for {sample_key}")
        target_sha = self._fixed_answer_token_sha256(target_ids.cpu().numpy())
        if target_sha != str(item.get("target_ids_sha256") or ""):
            raise ValueError(f"Sequence-retention target ids mismatch for {sample_key}")

        cache_row = int(item["cache_row"])
        offsets = cache["arrays"]["row_offsets"]
        start, end = int(offsets[cache_row]), int(offsets[cache_row + 1])
        if int(item.get("active_tokens", end - start)) != end - start:
            raise ValueError(f"Sequence-retention active-count mismatch for {sample_key}")
        if (
            end > start
            and cache["metadata"]["schema_version"]
            == "audio_alignkd_sequence_retention_v2"
        ):
            if tokenized_input is None:
                raise ValueError(
                    f"Sequence-retention active row lacks tokenized prompt: {sample_key}"
                )
            prompt_ids = tokenized_input["input_ids"].reshape(-1).long()
            if prompt_ids.numel() != self.ip_text_len:
                raise ValueError(
                    f"Sequence-retention prompt length mismatch for {sample_key}"
                )
            prompt_sha = self._fixed_answer_token_sha256(
                prompt_ids.cpu().numpy()
            )
            if prompt_sha != str(item.get("prompt_ids_sha256") or ""):
                raise ValueError(
                    f"Sequence-retention prompt ids mismatch for {sample_key}"
                )
        positions = torch.from_numpy(
            np.array(
                cache["arrays"]["token_positions"][start:end],
                dtype=np.int64,
                copy=True,
            )
        )
        ids = torch.from_numpy(
            np.array(
                cache["arrays"]["topk_ids"][start:end],
                dtype=np.int64,
                copy=True,
            )
        )
        probabilities = torch.from_numpy(
            np.array(cache["arrays"]["topk_probs"][start:end], copy=True)
        ).float()
        tail = torch.from_numpy(
            np.array(cache["arrays"]["tail_probs"][start:end], copy=True)
        ).float()
        gold = torch.from_numpy(
            np.array(
                cache["arrays"]["gold_token_ids"][start:end],
                dtype=np.int64,
                copy=True,
            )
        )
        if positions.numel() == 0:
            return values, True
        if (
            bool(((positions < 0) | (positions >= self.op_text_len)).any())
            or positions.unique().numel() != positions.numel()
            or not bool(torch.all(positions[1:] > positions[:-1]))
        ):
            raise ValueError(f"Invalid sequence-retention positions for {sample_key}")
        if not torch.equal(gold, target_ids.index_select(0, positions)):
            raise ValueError(f"Sequence-retention gold-token mismatch for {sample_key}")
        answer_mask = tokenized_answer.get("attention_mask")
        if answer_mask is None or not bool(answer_mask.reshape(-1)[positions].bool().all()):
            raise ValueError(f"Sequence-retention active padding token for {sample_key}")
        if (
            bool(((ids < 0) | (ids >= cache["vocab_size"])).any())
            or bool((ids.sort(dim=-1).values.diff(dim=-1) == 0).any())
        ):
            raise ValueError(f"Invalid sequence-retention top-k ids for {sample_key}")
        if not bool(torch.isfinite(probabilities).all()) or not bool(torch.isfinite(tail).all()):
            raise FloatingPointError(f"Non-finite sequence-retention probabilities for {sample_key}")
        mass = probabilities.sum(dim=-1) + tail
        if (
            bool((probabilities < 0).any())
            or bool((tail < 0).any())
            or not bool(torch.all(probabilities[:, :-1] >= probabilities[:, 1:]))
            or not torch.equal(ids[:, 0], gold)
            or not torch.allclose(mass, torch.ones_like(mass), atol=1.0e-5, rtol=0.0)
        ):
            raise ValueError(f"Invalid sequence-retention probability mass for {sample_key}")
        values["sequence_retention_topk_ids"][positions] = ids
        values["sequence_retention_topk_probs"][positions] = probabilities
        values["sequence_retention_tail_probs"][positions] = tail
        values["sequence_retention_active_mask"][positions] = True
        return values, True

    def _read_alignkd_teacher_cache(self, row):
        cache = self.alignkd_teacher_cache
        sample_key = str(row.get("sample_key") or row.get("id") or "")
        is_kd_sample = bool(row.get("is_kd_sample", False))
        if not is_kd_sample:
            return self._alignkd_sentinel(), False
        item = cache["entries"].get(sample_key)
        if item is None:
            raise KeyError(f"AlignKD teacher cache miss for {sample_key}")
        if str(row.get("audio_hash") or "") != str(item.get("audio_hash") or ""):
            raise ValueError(f"AlignKD audio_hash mismatch for {sample_key}")
        if str(row.get("prompt_hash") or "") != str(item.get("prompt_hash") or ""):
            raise ValueError(f"AlignKD prompt_hash mismatch for {sample_key}")

        shard = cache["shards"][item["cache_shard"]]
        cache_row = int(item["cache_row"])
        rows = int(shard["metadata"]["rows"])
        if not 0 <= cache_row < rows:
            raise IndexError(f"AlignKD cache_row out of bounds for {sample_key}")
        values = self._alignkd_sentinel()
        for name, array in shard["arrays"].items():
            tensor = torch.from_numpy(np.array(array[cache_row], copy=True))
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"{name} is non-finite for {sample_key}")
            values[name] = tensor

        if "teacher_attn_qabcd_126" in shard["arrays"]:
            attention = values["teacher_attn_qabcd_126"].float()
            if bool((attention < 0).any()) or not torch.allclose(
                attention.sum(dim=-1), torch.ones(5), atol=3e-3, rtol=3e-3
            ):
                raise ValueError(f"Invalid teacher attention distribution for {sample_key}")
        if "teacher_audio_mass_qabcd" in shard["arrays"]:
            mass = values["teacher_audio_mass_qabcd"].float()
            if bool(((mass < 0) | (mass > 1)).any()):
                raise ValueError(f"Invalid teacher audio mass for {sample_key}")
        if "teacher_focus_126" in shard["arrays"]:
            focus = values["teacher_focus_126"].float()
            if bool((focus < 0).any()) or not torch.allclose(
                focus.sum(), torch.tensor(1.0), atol=3e-3, rtol=3e-3
            ):
                raise ValueError(f"Invalid teacher focus for {sample_key}")
        if "teacher_top16_indices" in shard["arrays"]:
            indices = values["teacher_top16_indices"].long()
            if (
                indices.unique().numel() != 16
                or bool(((indices < 0) | (indices >= 126)).any())
            ):
                raise ValueError(f"Invalid teacher top16 indices for {sample_key}")
        return values, True

    def _alignkd_sentinel(self):
        fields = self.alignkd_teacher_cache["fields"]
        values = {}
        if "attention" in fields:
            values["teacher_attn_qabcd_126"] = torch.zeros(5, 126, dtype=torch.float16)
        if "mass" in fields:
            values["teacher_audio_mass_qabcd"] = torch.zeros(5, dtype=torch.float16)
        if "gram" in fields:
            values["teacher_feature_gram_126"] = torch.zeros(126, 126, dtype=torch.float16)
        if "focus" in fields:
            values["teacher_focus_126"] = torch.zeros(126, dtype=torch.float16)
        if "top16" in fields:
            values["teacher_top16_indices"] = torch.arange(16, dtype=torch.int16)
        return values

    def _alignkd_query_group_mask(self, row, is_kd_sample):
        mask = torch.zeros(5, self.ip_text_len, dtype=torch.bool)
        if not is_kd_sample:
            return mask
        spans = row.get("student_query_spans_qabcd")
        if not isinstance(spans, list) or len(spans) != 5:
            raise ValueError(
                f"Missing student_query_spans_qabcd for {row.get('sample_key') or row.get('id')}"
            )
        previous_end = -1
        for group, span in enumerate(spans):
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError(f"Invalid AlignKD query span {span}")
            start, end = int(span[0]), int(span[1])
            if not (0 <= start < end <= self.ip_text_len):
                raise ValueError(f"Out-of-range AlignKD query span {span}")
            if start < previous_end:
                raise ValueError(f"Overlapping AlignKD query spans: {spans}")
            mask[group, start:end] = True
            previous_end = end
        return mask

    def _ced_cache_key(self, filepath1, filepath2):
        policy = self.ced_hidden_cache["metadata"]["policy"]
        raw = f"{policy}\n{filepath1}\n{filepath2}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _beats_cache_key(self, filepath1):
        policy = self.beats_hidden_cache["metadata"]["policy"]
        raw = f"{policy}\n{filepath1}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _tokenize_text(self, sentence, max_len):
        # if 'gpt' in self.tokenizer_type:
        sentence = sentence + ' <|endoftext|>'
        return self.tokenizer.encode_plus(
            text=sentence, 
            add_special_tokens=True,
            max_length=max_len, 
            padding='max_length',  # Use padding instead of deprecated pad_to_max_length
            truncation=True,  # Explicitly enable truncation
            return_tensors="pt"
        )

    def collect_data_jsons(self, data_path, datafiles):
        # collect per class data json
        all_data_json = []
        self.data_path = data_path
        for i, d in enumerate(datafiles):
            with open(d, 'r', encoding='utf-8') as fp:
                data_json = json.load(fp)
                # Only print from rank 0 to avoid duplicate messages
                import os
                local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
                if local_rank == 0:
                    print(f"Dataset: {d.split(os.path.sep)[-1]}, \t Examples: {str(len(data_json))}")
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json
        self._refresh_exists_filepaths1()

    def _refresh_exists_filepaths1(self):
        self.exists_filepaths1 = sorted(set([
            self.all_data_json[i]["filepath1"]
            for i in range(len(self.all_data_json))
            if self.all_data_json[i]["filepath1"] != ''
        ]))

    def _mixture_bucket(self, row):
        source = str(row.get("source", "")).lower()
        if source == "dcase2025":
            return "dcase2025"
        if source == "dcase2026":
            return "dcase2026"
        if source == "reasonaqa":
            filepath2 = str(row.get("filepath2", "") or "").strip()
            return "mellow_two_audio" if filepath2 else "mellow_single"
        filepath2 = str(row.get("filepath2", "") or "").strip()
        return "unknown_two_audio" if filepath2 else "unknown_single"

    def apply_mixture(self, mixture_config):
        self.mixture_summary = {
            "enabled": False,
            "id": "M0",
            "original_count": len(self.all_data_json),
            "original_bucket_counts": dict(Counter(self._mixture_bucket(r) for r in self.all_data_json)),
        }
        if not mixture_config or not mixture_config.get("enabled", False):
            self.mixture_summary["sampled_count"] = len(self.all_data_json)
            self.mixture_summary["sampled_bucket_counts"] = self.mixture_summary["original_bucket_counts"]
            return

        weights = dict(mixture_config.get("weights", {}) or {})
        target_size = int(mixture_config.get("target_size", 0) or len(self.all_data_json))
        seed = int(mixture_config.get("sample_seed", mixture_config.get("seed", 0)))
        replacement = bool(mixture_config.get("replacement", True))

        buckets = defaultdict(list)
        for row in self.all_data_json:
            buckets[self._mixture_bucket(row)].append(row)

        bucket_names = sorted(buckets)
        raw_weighted_counts = {
            name: len(buckets[name]) * float(weights.get(name, 1.0))
            for name in bucket_names
        }
        total_weighted = sum(raw_weighted_counts.values())
        if total_weighted <= 0:
            raise ValueError(f"Invalid data.mixture weights: {weights}")

        rng = random.Random(seed)
        sampled_rows = []
        requested_counts = {}
        for name in bucket_names:
            request = int(round(target_size * raw_weighted_counts[name] / total_weighted))
            requested_counts[name] = request
            rows = buckets[name]
            if request <= 0:
                continue
            if replacement:
                sampled_rows.extend(rng.choice(rows) for _ in range(request))
            else:
                sampled_rows.extend(rows[:request] if request <= len(rows) else rows)

        if not sampled_rows:
            raise ValueError(f"data.mixture produced no rows: {mixture_config}")
        rng.shuffle(sampled_rows)
        self.all_data_json = sampled_rows
        self._refresh_exists_filepaths1()
        self.mixture_summary.update({
            "enabled": True,
            "id": str(mixture_config.get("id", "custom")),
            "weights": weights,
            "target_size": target_size,
            "seed": seed,
            "sample_seed": seed,
            "replacement": replacement,
            "requested_bucket_counts": requested_counts,
            "sampled_count": len(self.all_data_json),
            "sampled_bucket_counts": dict(Counter(self._mixture_bucket(r) for r in self.all_data_json)),
        })

        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        if local_rank == 0:
            print(f"Mixture {self.mixture_summary['id']}: {json.dumps(self.mixture_summary, sort_keys=True)}")

    def apply_data_order(self, order_config):
        self.data_order_summary = {
            "enabled": False,
            "shuffle": False,
            "seed": None,
            "sampler_seed": None,
            "count": len(self.all_data_json),
            "bucket_counts": dict(Counter(self._mixture_bucket(r) for r in self.all_data_json)),
        }
        if not order_config or not order_config.get("shuffle", False):
            return

        seed = int(order_config.get("seed", 0))
        sampler_seed = order_config.get("sampler_seed", None)
        if sampler_seed is not None:
            sampler_seed = int(sampler_seed)

        rng = random.Random(seed)
        indexed_rows = list(enumerate(self.all_data_json))
        rng.shuffle(indexed_rows)
        order_head = [idx for idx, _row in indexed_rows[:32]]
        self.all_data_json = [row for _idx, row in indexed_rows]
        self._refresh_exists_filepaths1()

        self.data_order_summary.update({
            "enabled": True,
            "shuffle": True,
            "seed": seed,
            "sampler_seed": sampler_seed,
            "count": len(self.all_data_json),
            "bucket_counts": dict(Counter(self._mixture_bucket(r) for r in self.all_data_json)),
            "pre_shuffle_indices_head": order_head,
        })

        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        if local_rank == 0:
            print(f"Data order: {json.dumps(self.data_order_summary, sort_keys=True)}")

    def __len__(self):
        return len(self.all_data_json)

    def _cut_or_randomcrop(self, waveform):
        # waveform: [1, samples]
        if waveform.shape[0] > 1:
            waveform = (waveform[0] + waveform[1]) / 2
            waveform = waveform.unsqueeze(0)
        # random crop
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1)-self.max_length)
            waveform = waveform[:, random_idx:random_idx+self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform

    def _cut_or_centercrop(self, waveform, target_length=None):
        target_length = int(target_length or self.max_length)
        waveform = self._to_mono(waveform)
        if waveform.size(1) > target_length:
            start = (waveform.size(1) - target_length) // 2
            waveform = waveform[:, start : start + target_length]
        elif waveform.size(1) < target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_length - waveform.size(1))
            )
        if waveform.size(1) != target_length:
            raise ValueError("center-crop/pad length drift")
        return waveform

    def _read_qsafe_fixed20_audio(self, index):
        """Reproduce the frozen fixed20 cache policy before making view two."""

        row = self.all_data_json[index]
        raw1, raw2 = str(row["filepath1"]), str(row["filepath2"])
        if not raw1:
            raise ValueError("Q-SAFE does not permit an empty primary audio path")
        path1 = os.path.join(self.data_path, raw1).replace("/", os.path.sep).replace("\\", os.path.sep)
        wav1, rate1 = load_audio(path1, channels_first=True)
        wav1 = self._to_mono(wav1)
        if rate1 != self.sampling_rate:
            wav1 = torchaudio.functional.resample(wav1, rate1, self.sampling_rate)
        if raw2:
            path2 = os.path.join(self.data_path, raw2).replace("/", os.path.sep).replace("\\", os.path.sep)
            wav2, rate2 = load_audio(path2, channels_first=True)
            wav2 = self._to_mono(wav2)
            if rate2 != self.sampling_rate:
                wav2 = torchaudio.functional.resample(wav2, rate2, self.sampling_rate)
            audio1 = self._cut_or_centercrop(wav1, self.max_length)
            audio2 = self._cut_or_centercrop(wav2, self.max_length)
        else:
            path2 = ""
            fixed20 = self._cut_or_centercrop(wav1, 2 * self.max_length)
            audio1 = fixed20[:, : self.max_length].contiguous()
            audio2 = fixed20[:, self.max_length :].contiguous()
        return audio1, audio2, path1, path2

    def _to_mono(self, waveform):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform
        return waveform

    def _load_audio_clip(self, file_path):
        info = torchaudio.info(file_path)
        audio_rate = info.sample_rate
        if self.variable_audio_length:
            return load_audio(file_path, channels_first=True)
        max_source_frames = int(math.ceil(self.max_length * audio_rate / self.sampling_rate))
        if info.num_frames > max_source_frames > 0:
            frame_offset = random.randint(0, info.num_frames - max_source_frames)
            return load_audio(
                file_path,
                channels_first=True,
                frame_offset=frame_offset,
                num_frames=max_source_frames,
            )
        return load_audio(file_path, channels_first=True)

    def _read_audio(self, index):
        file_path1, file_path2 = self.all_data_json[index]["filepath1"], self.all_data_json[index]["filepath2"]
        if self.all_data_json[index]["filepath1"] == '':
            file_path1 = random.choice(self.exists_filepaths1)
        
        if self.all_data_json[index]["filepath2"] == '':
            file_path2 = random.choice(self.exists_filepaths1)
        
        file_path1 = os.path.join(self.data_path, file_path1)
        file_path1 = file_path1.replace("/",os.path.sep).replace("\\",os.path.sep)
        file_path2 = os.path.join(self.data_path, file_path2)
        file_path2 = file_path2.replace("/",os.path.sep).replace("\\",os.path.sep)
        try:
            audio_data1, audio_rate1 = self._load_audio_clip(file_path1)
            audio_data2, audio_rate2 = self._load_audio_clip(file_path2)

            # resample audio clip
            if audio_rate1 != self.sampling_rate:
                audio_data1 = torchaudio.functional.resample(audio_data1, orig_freq=audio_rate1, new_freq=self.sampling_rate)
            if audio_rate2 != self.sampling_rate:
                audio_data2 = torchaudio.functional.resample(audio_data2, orig_freq=audio_rate2, new_freq=self.sampling_rate)
            
            if self.variable_audio_length:
                audio_data1 = self._to_mono(audio_data1)
                if os.path.abspath(file_path1) == os.path.abspath(file_path2):
                    audio_data2 = audio_data1.new_zeros(1, 0)
                else:
                    audio_data2 = self._to_mono(audio_data2)
            else:
                audio_data1 = self._cut_or_randomcrop(audio_data1)
                audio_data2 = self._cut_or_randomcrop(audio_data2)

            if self.audio_mode in ("zero_audio", "question_only"):
                audio_data1 = torch.zeros_like(audio_data1)
                audio_data2 = torch.zeros_like(audio_data2)
            elif self.audio_mode != "normal":
                raise ValueError(f"Unknown data.audio_mode: {self.audio_mode}")

            return audio_data1, audio_data2, self.sampling_rate, self.sampling_rate, file_path1, file_path2
        
        except Exception as e:
            print(f'error: {e} occurs, when loading {file_path1} or {file_path2}')
            random_index = random.randint(0, len(self.all_data_json)-1)
            return self._read_audio(index=random_index)
        
    def _metadata_text(self, row, key):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        metadata = row.get("metadata", {})
        if isinstance(metadata, dict):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _answer_text(self, row):
        answer = row.get("answer", "")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        raise KeyError("Missing non-empty answer field")

    def _create_answer_input(self, index):
        row = self.all_data_json[index]
        input = row['input']
        if input == "caption both audios":
            input = random.choice(BOTH)
            caption1 = self._metadata_text(row, "caption1")
            caption2 = self._metadata_text(row, "caption2")
            if caption1 and caption2:
                answer = "The audio 1 is " + caption1.lower() + ". The audio 2 is " + caption2.lower() + "."
            else:
                answer = self._answer_text(row)
            answer = answer.replace("..",".")
        elif input == "caption first audio":
            input = random.choice(FIRST)
            caption1 = self._metadata_text(row, "caption1")
            if caption1:
                answer = "The audio 1 is " + caption1.lower() + "."
            else:
                answer = self._answer_text(row)
            answer = answer.replace("..",".")
        elif input == "caption second audio":
            input = random.choice(SECOND)
            caption2 = self._metadata_text(row, "caption2")
            if caption2:
                answer = "The audio 2 is " + caption2.lower() + "."
            else:
                answer = self._answer_text(row)
            answer = answer.replace("..",".")
        elif input == "explain the difference in few words":
            input = random.choice(WORDONLY)
            answer = row['answer']
        elif input == "explain the difference in a sentence":
            input = random.choice(LONGLINEONLY)
            answer = row['answer']
        elif input == "explain the difference in detail":
            input = random.choice(DETAILONLY)
            answer = row['answer']
        elif "emo_emo_emo" in input:
            input = input.replace("emo_emo_emo", random.choice(EMOTION))
            answer = row['answer']
        elif row.get("prompt_contract") == "ke_omni_r_cot_v1":
            input = row['input']
            answer = row['answer']
        else:
            input = row['input'].lower()
            answer = row['answer'].lower()
            
        return answer, input
    
    def _read_text(self, index):
        answer, input = self._create_answer_input(index)

        tok_answer = self._tokenize_text(answer, self.op_text_len)
        tok_input = self._tokenize_text(input, self.ip_text_len)
        return tok_answer, tok_input, answer

    def _read_cached_ced_hidden(self, index):
        row = self.all_data_json[index]
        file_path1 = row["filepath1"]
        file_path2 = row["filepath2"]
        key = self._ced_cache_key(file_path1, file_path2)
        cache_ref = self.ced_hidden_cache["key_to_index"].get(key)
        if cache_ref is None:
            raise KeyError(f"CED hidden cache miss for {file_path1} / {file_path2}")
        shard_id, cache_idx = cache_ref
        shard = self.ced_hidden_cache["shards"][shard_id]
        valid = int(shard["valid_tokens"][cache_idx])
        hidden_np = shard["hidden"][cache_idx, :valid, :]
        seg_np = shard["segment_lengths"][cache_idx]
        return (
            torch.from_numpy(np.array(hidden_np, copy=True)),
            torch.from_numpy(np.array(seg_np, copy=True)).long(),
            os.path.join(self.data_path, file_path1).replace("/", os.path.sep).replace("\\", os.path.sep),
            os.path.join(self.data_path, file_path2).replace("/", os.path.sep).replace("\\", os.path.sep),
        )

    def _read_cached_beats_hidden(self, index):
        row = self.all_data_json[index]
        file_path1 = row["filepath1"]
        file_path2 = row["filepath2"]
        key = self._beats_cache_key(file_path1)
        cache_ref = self.beats_hidden_cache["key_to_index"].get(key)
        if cache_ref is None:
            raise KeyError(f"BEATs hidden cache miss for {file_path1}")
        shard_id, cache_idx = cache_ref
        shard = self.beats_hidden_cache["shards"][shard_id]
        valid = int(shard["valid_tokens"][cache_idx])
        hidden_np = shard["hidden"][cache_idx, :valid, :]
        seg_np = shard["segment_lengths"][cache_idx]
        return (
            torch.from_numpy(np.array(hidden_np, copy=True)),
            torch.from_numpy(np.array(seg_np, copy=True)).long(),
            os.path.join(self.data_path, file_path1).replace("/", os.path.sep).replace("\\", os.path.sep),
            os.path.join(self.data_path, file_path2).replace("/", os.path.sep).replace("\\", os.path.sep),
        )

    def __getitem__(self, index):
        cached_hidden = None
        cached_segment_lengths = None
        cached_beats_hidden = None
        cached_beats_segment_lengths = None
        if self.qsafe_enabled:
            audio_data1, audio_data2, file_path1, file_path2 = self._read_qsafe_fixed20_audio(index)
        elif self.ced_hidden_cache is not None:
            cached_hidden, cached_segment_lengths, file_path1, file_path2 = self._read_cached_ced_hidden(index)
            audio_data1 = torch.zeros(1, 1)
            audio_data2 = torch.zeros(1, 1)
        elif self.beats_hidden_cache is not None:
            cached_beats_hidden, cached_beats_segment_lengths, file_path1, file_path2 = self._read_cached_beats_hidden(index)
            audio_data1 = torch.zeros(1, 1)
            audio_data2 = torch.zeros(1, 1)
        else:
            # create a audio tensor
            audio_data1, audio_data2, audio_rate1, audio_rate2, file_path1, file_path2 = self._read_audio(index)
        # audio_data1, audio_data2, audio_rate1, audio_rate2, file_path1, file_path2 = 0,0,0,0,0,0 #self._read_audio(index)
        tok_answer, tok_input, answer_text = self._read_text(index)
        
        # resample audio clip
        # if audio_rate1 != self.sampling_rate:
        #     audio_data1 = torchaudio.functional.resample(audio_data1, orig_freq=audio_rate1, new_freq=self.sampling_rate)
        # if audio_rate2 != self.sampling_rate:
        #     audio_data2 = torchaudio.functional.resample(audio_data2, orig_freq=audio_rate2, new_freq=self.sampling_rate)
        
        # # audio_data1 = audio_data1.unsqueeze(0)
        # # audio_data2 = audio_data2.unsqueeze(0)
        
        # audio_data1 = self._cut_or_randomcrop(audio_data1)
        # audio_data2 = self._cut_or_randomcrop(audio_data2)

        data_dict = {
            'waveform1': audio_data1,
            'waveform2': audio_data2,
            'answer': tok_answer,
            'answer_text': answer_text,
            'input': tok_input,
            'file_path1': file_path1,
            'file_path2': file_path2,
        }
        row = self.all_data_json[index]
        if self.qsafe_enabled:
            family = classify_question_safe_family(row)
            stable_identity = str(
                row.get("sample_key") or row.get("id") or f"row-{index}"
            )
            digest = hashlib.sha256(
                f"{self.qsafe_seed}\n{stable_identity}".encode("utf-8")
            ).digest()
            base_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
            if self.qsafe_apply_transforms:
                aug1 = question_safe_audio_augment(audio_data1, family, base_seed)
                aug2 = question_safe_audio_augment(audio_data2, family, base_seed + 1)
            else:
                aug1, aug2 = audio_data1.clone(), audio_data2.clone()
            data_dict["waveform1_aug"] = aug1
            data_dict["waveform2_aug"] = aug2
            data_dict["qsafe_family_id"] = family
        if self.output_control_regions:
            if row.get("prompt_contract") != "ke_omni_r_cot_v1":
                raise ValueError("output-control token regions require canonical Ke CoT rows")
            region_ids = ke_cot_output_control_region_ids(
                self.tokenizer, answer_text, self.op_text_len
            )
            answer_mask = tok_answer["attention_mask"].reshape(-1).bool()
            if not torch.equal(region_ids.gt(0), answer_mask):
                raise ValueError("output-control regions and answer attention mask drift")
            data_dict["output_control_region_ids"] = region_ids
        is_kd_sample = bool(row.get("is_kd_sample", False))
        if "is_kd_sample" in row:
            required_key = self.naive_kd_target_scores_key
            required_scores = resolve_naive_kd_target_scores(row, required_key)
            data_dict[required_key] = required_scores
            # Preserve the legacy field for diagnostics and old objectives,
            # but do not require it when a different target field is frozen.
            if required_key != "teacher_scores_abcd":
                teacher_scores = row.get(
                    "teacher_scores_abcd", [0.0, 0.0, 0.0, 0.0]
                )
                data_dict["teacher_scores_abcd"] = teacher_scores
        if "teacher_probs_abcd" in row:
            data_dict["teacher_probs_abcd"] = row["teacher_probs_abcd"]
        for key, value in row.items():
            if (
                (key.endswith("_scores_abcd") or key.endswith("_probs_abcd"))
                and isinstance(value, list)
                and len(value) == 4
            ):
                data_dict[key] = value
        if "gold_option_index" in row:
            data_dict["gold_option_index"] = row["gold_option_index"]
        if "is_kd_sample" in row:
            data_dict["is_kd_sample"] = row["is_kd_sample"]
        if "is_retention_kd_sample" in row:
            data_dict["is_retention_kd_sample"] = bool(
                row["is_retention_kd_sample"]
            )
            data_dict["ref_scores_abcd"] = resolve_naive_kd_target_scores(
                row,
                "ref_scores_abcd",
                mask_key="is_retention_kd_sample",
            )
        if "replay_group_id" in row:
            data_dict["replay_group_id"] = int(row["replay_group_id"])
        if "option_valid_mask_abcd" in row:
            option_mask = row["option_valid_mask_abcd"]
            if not isinstance(option_mask, list) or len(option_mask) != 4:
                raise ValueError("option_valid_mask_abcd must contain four entries")
            data_dict["option_valid_mask_abcd"] = [bool(value) for value in option_mask]
        if self.alignkd_teacher_cache is not None:
            cache_values, cache_valid = self._read_alignkd_teacher_cache(row)
            data_dict.update(cache_values)
            data_dict["teacher_cache_valid"] = cache_valid
            data_dict["alignkd_query_group_mask"] = self._alignkd_query_group_mask(
                row, is_kd_sample
            )
        if self.sequence_retention_cache is not None:
            sequence_values, sequence_cache_valid = self._read_sequence_retention_cache(
                row, tok_answer, tok_input
            )
            data_dict.update(sequence_values)
            data_dict["sequence_retention_cache_valid"] = sequence_cache_valid
        for scalar_key in (
            "teacher_drop_raw",
            "teacher_drop_positive",
            "teacher_drop_capped",
            "external_teacher_drop_raw",
            "external_teacher_drop_positive",
            "external_teacher_drop_capped",
        ):
            if scalar_key in row:
                data_dict[scalar_key] = row[scalar_key]
        for bool_key in (
            "real_teacher_correct",
            "masked_teacher_correct",
            "external_real_teacher_correct",
            "external_masked_teacher_correct",
        ):
            if bool_key in row:
                data_dict[bool_key] = row[bool_key]
        for metadata_key in (
            "id",
            "sample_key",
            "maskedkd_real_id",
            "source_index",
            "gold",
            "gold_letter",
            "maskedkd_sample_type",
        ):
            if metadata_key in row:
                data_dict[metadata_key] = row[metadata_key]
        if cached_hidden is not None:
            data_dict["ced_hidden"] = cached_hidden
            data_dict["ced_hidden_segment_lengths"] = cached_segment_lengths
        if cached_beats_hidden is not None:
            data_dict["beats_hidden"] = cached_beats_hidden
            data_dict["beats_hidden_segment_lengths"] = cached_beats_segment_lengths

        return data_dict
    

def collate_fn(list_data_dict):
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {
                'text': 'a sound of dog',
                'waveform': (1, samples),
                'modality': 'audio_text'
            }
            ...
            ]
    Returns:
        data_dict: e.g. 
            'audio_text': {
                'text': ['a sound of dog', ...]
                'waveform': (batch_size, 1, samples)
        }
    """
    at_data_dict = {}
    
    optional_metadata_keys = {
        "id",
        "sample_key",
        "maskedkd_real_id",
        "source_index",
        "gold",
        "gold_letter",
        "maskedkd_sample_type",
    }
    optional_float_keys = {
        "teacher_drop_raw",
        "teacher_drop_positive",
        "teacher_drop_capped",
        "external_teacher_drop_raw",
        "external_teacher_drop_positive",
        "external_teacher_drop_capped",
    }
    optional_bool_keys = {
        "real_teacher_correct",
        "masked_teacher_correct",
        "external_real_teacher_correct",
        "external_masked_teacher_correct",
        "teacher_cache_valid",
        "sequence_retention_cache_valid",
        "is_retention_kd_sample",
    }
    optional_long_keys = {
        "replay_group_id",
        "qsafe_family_id",
    }
    optional_abcd_bool_keys = {
        "option_valid_mask_abcd",
    }
    optional_abcd_float_keys = {
        key
        for item in list_data_dict
        for key, value in item.items()
        if (
            (key.endswith("_scores_abcd") or key.endswith("_probs_abcd"))
            and isinstance(value, list)
            and len(value) == 4
        )
    }

    if len(list_data_dict) > 0:
        keys = list(list_data_dict[0].keys())
        for optional_key in (
            optional_metadata_keys
            | optional_float_keys
            | optional_bool_keys
            | optional_long_keys
            | optional_abcd_float_keys
            | optional_abcd_bool_keys
        ):
            if optional_key not in keys and any(optional_key in item for item in list_data_dict):
                keys.append(optional_key)

        for key in keys:
            if key in optional_metadata_keys:
                at_data_dict[key] = [item.get(key) for item in list_data_dict]
            elif key in optional_abcd_float_keys:
                at_data_dict[key] = [
                    item.get(key, [0.0, 0.0, 0.0, 0.0])
                    for item in list_data_dict
                ]
            elif key in optional_abcd_bool_keys:
                at_data_dict[key] = [
                    item.get(key, [False, False, False, False])
                    for item in list_data_dict
                ]
            elif key in optional_float_keys:
                at_data_dict[key] = [item.get(key, 0.0) for item in list_data_dict]
            elif key in optional_bool_keys:
                at_data_dict[key] = [item.get(key, False) for item in list_data_dict]
            elif key in optional_long_keys:
                at_data_dict[key] = [item.get(key, -1) for item in list_data_dict]
            else:
                at_data_dict[key] = [at_data_dict[key] for at_data_dict in list_data_dict]
            if key in {"waveform1", "waveform2", "waveform1_aug", "waveform2_aug"}:
                lengths = torch.tensor([x.shape[-1] for x in at_data_dict[key]], dtype=torch.long)
                if len(set(lengths.tolist())) == 1:
                    at_data_dict[key] = torch.stack(at_data_dict[key]).squeeze(1)
                else:
                    max_len = int(lengths.max().item())
                    padded = []
                    for waveform in at_data_dict[key]:
                        if waveform.shape[-1] < max_len:
                            pad = torch.zeros(
                                waveform.shape[0],
                                max_len - waveform.shape[-1],
                                dtype=waveform.dtype,
                            )
                            waveform = torch.cat((waveform, pad), dim=-1)
                        padded.append(waveform)
                    at_data_dict[key] = torch.stack(padded).squeeze(1)
                    at_data_dict[f"{key}_lengths"] = lengths
            elif key == "ced_hidden":
                lengths = torch.tensor([x.shape[0] for x in at_data_dict[key]], dtype=torch.long)
                max_len = int(lengths.max().item())
                hidden_dim = int(at_data_dict[key][0].shape[-1])
                padded = []
                for hidden in at_data_dict[key]:
                    if hidden.shape[0] < max_len:
                        pad = torch.zeros(max_len - hidden.shape[0], hidden_dim, dtype=hidden.dtype)
                        hidden = torch.cat((hidden, pad), dim=0)
                    padded.append(hidden)
                at_data_dict[key] = torch.stack(padded)
                at_data_dict["ced_hidden_lengths"] = lengths
            elif key == "ced_hidden_segment_lengths":
                max_len = max(x.shape[0] for x in at_data_dict[key])
                padded = []
                for lengths in at_data_dict[key]:
                    if lengths.shape[0] < max_len:
                        lengths = torch.cat(
                            (lengths, torch.zeros(max_len - lengths.shape[0], dtype=lengths.dtype)),
                            dim=0,
                        )
                    padded.append(lengths)
                at_data_dict[key] = torch.stack(padded)
            elif key == "beats_hidden":
                lengths = torch.tensor([x.shape[0] for x in at_data_dict[key]], dtype=torch.long)
                max_len = int(lengths.max().item())
                hidden_dim = int(at_data_dict[key][0].shape[-1])
                padded = []
                for hidden in at_data_dict[key]:
                    if hidden.shape[0] < max_len:
                        pad = torch.zeros(max_len - hidden.shape[0], hidden_dim, dtype=hidden.dtype)
                        hidden = torch.cat((hidden, pad), dim=0)
                    padded.append(hidden)
                at_data_dict[key] = torch.stack(padded)
                at_data_dict["beats_hidden_lengths"] = lengths
            elif key == "beats_hidden_segment_lengths":
                max_len = max(x.shape[0] for x in at_data_dict[key])
                padded = []
                for lengths in at_data_dict[key]:
                    if lengths.shape[0] < max_len:
                        lengths = torch.cat(
                            (lengths, torch.zeros(max_len - lengths.shape[0], dtype=lengths.dtype)),
                            dim=0,
                        )
                    padded.append(lengths)
                at_data_dict[key] = torch.stack(padded)
            elif key == 'answer' or key == "input":
                stack = {k:[] for k in at_data_dict[key][0].keys()}
                for entry in at_data_dict[key]:
                    for k in entry.keys():
                        stack[k].append(entry[k])
                at_data_dict[key] = {k:torch.stack(stack[k]).squeeze(1) for k in stack.keys()}                       
            elif key == "output_control_region_ids":
                at_data_dict[key] = torch.stack(at_data_dict[key])
            elif key in optional_abcd_float_keys:
                at_data_dict[key] = torch.tensor(at_data_dict[key], dtype=torch.float32)
            elif key in optional_abcd_bool_keys:
                at_data_dict[key] = torch.tensor(at_data_dict[key], dtype=torch.bool)
            elif key == "gold_option_index":
                at_data_dict[key] = torch.tensor(at_data_dict[key], dtype=torch.long)
            elif key == "is_kd_sample":
                at_data_dict[key] = torch.tensor(at_data_dict[key], dtype=torch.bool)
            elif key in {
                "teacher_attn_qabcd_126",
                "teacher_audio_mass_qabcd",
                "teacher_focus_126",
                "teacher_feature_gram_126",
                "teacher_top16_indices",
                "alignkd_query_group_mask",
                "sequence_retention_topk_ids",
                "sequence_retention_topk_probs",
                "sequence_retention_tail_probs",
                "sequence_retention_active_mask",
            }:
                at_data_dict[key] = torch.stack(at_data_dict[key])
            elif key in optional_float_keys:
                at_data_dict[key] = torch.tensor(at_data_dict[key], dtype=torch.float32)
            elif key in optional_bool_keys:
                at_data_dict[key] = torch.tensor(at_data_dict[key], dtype=torch.bool)
            elif key in optional_long_keys:
                at_data_dict[key] = torch.tensor(at_data_dict[key], dtype=torch.long)
            elif key in {
                "file_path1",
                "file_path2",
                "answer_text",
                "id",
                "sample_key",
                "maskedkd_real_id",
                "source_index",
                "gold",
                "gold_letter",
                "maskedkd_sample_type",
            }:
                at_data_dict[key] = [text for text in at_data_dict[key]]
    
    return at_data_dict
