# Training and reproducibility

Run the commands below from the repository root. See the [README](../README.md) for installation and inference.

## Three-stage training

### 1. Prepare paths and data

```bash
conda activate mizar-train
python pipeline/download_assets.py --training-data
python pipeline/unpack_data.py
python pipeline/configure_local.py
```

This creates `configs/assets.local.yaml` with paths to bundled assets and
unpacked manifests. Configure raw-audio root translations and obtain or
regenerate the S1/S2 CED caches as described in [docs/DATA.md](DATA.md).
Mapping paths at audio I/O keeps original manifest hashes and cache keys intact.

```bash
export MIZAR_AUDIO_PATH_MAP="$PWD/configs/audio_paths.local.json"
```

Skip that environment variable when the paths in your manifests already exist.
Check the relevant assets before each stage; `--quick` checks only presence and
sizes and is not a replacement for checksum verification.

### 2. Stage 1 — broad audio–language alignment

The historical S1 mixture contains 1,477,976 rows: 756,968 ReasonAQA,
571,008 AudioMCQ direct-answer rows and 150,000 historical rationale views.
The frozen CED encoder supplies features; the mapper and full language decoder
are trained for three epochs. This configuration documents the executed
historical lineage. It does not change S2/S3 into rationale training.

```bash
python pipeline/validate_assets.py --assets configs/assets.local.yaml --stage s1
python pipeline/render_config.py s1 \
  --assets configs/assets.local.yaml --output resolved_configs/s1.yaml
python pipeline/launch_stage.py s1 \
  --config resolved_configs/s1.yaml --output runs/s1 --execute
```

Reference geometry: **4 GPUs, global batch 32, 138,558 updates**. The decoder LR
is 3e-4 and mapper LR is 6e-4. Adopt the epoch-3 / update-138,558 checkpoint.
For a complete new replay, set `s1_checkpoint` in the local asset file to that
new checkpoint before rendering S2. The public release does not include an S1 checkpoint.

### 3. Stage 2 — audio-dependent fine-tuning

S2 uses 256,077 strongAC rows plus 33,875 AVQA rows (sound and both subsets,
without video). It starts from S1 with a **fresh optimizer** and gold-answer CE.

```bash
python pipeline/validate_assets.py --assets configs/assets.local.yaml --stage s2
python pipeline/render_config.py s2 \
  --assets configs/assets.local.yaml --output resolved_configs/s2.yaml
python pipeline/launch_stage.py s2 \
  --config resolved_configs/s2.yaml --output runs/s2 --execute
```

Reference geometry: **2 GPUs, global batch 64, 4,530-update LR horizon**.
The adopted checkpoint is **update 906**, corresponding to 57,984 example
exposures. Preserve the original LR horizon when reproducing that checkpoint.
Set `s2_checkpoint` to your new update-906 checkpoint before rendering S3, after S2 training. The public release does not include an S2 checkpoint.

### 4. Stage 3 — joint strongAC + AVQA Q4 refinement

Teacher and fixed S2 student correctness comes from free generation under four
cyclic option orders. Consistent correctness means all four generated answers
pass the official content matcher. Q4 samples the joint training pool with
teacher-correct/student-wrong, both-correct, teacher-wrong/student-correct, and
both-wrong proportions of **50/35/10/5%**. Correct-answer positions are balanced.
The training objective remains ordinary gold-answer CE.

The five complete 38,400-row manifests are available through `--training-data`. Optionally verify
their exact reconstruction from the released candidate file and labels:

```bash
for seed in 20260905 20260906 20260907 20260908 20260909; do
  python pipeline/materialize_q4.py \
    --candidates data/prepared/s3_candidates.json \
    --labels data/q4/labels_merged_gen0_gen4.executed.json.gz \
    --seed "$seed" --output "data/reconstructed/q4_s${seed}.json"
done
```

The materializer checks the executed per-seed SHA-256. Candidate rows must refer
to original recordings; S2 AVQA chunk-pair input is not interchangeable.

```bash
python pipeline/validate_assets.py --assets configs/assets.local.yaml --stage s3-q4
for seed in 20260905 20260906 20260907 20260908 20260909; do
  python pipeline/render_config.py s3-q4 --seed "$seed" \
    --assets configs/assets.local.yaml \
    --output "resolved_configs/s3_q4_${seed}.yaml"
  python pipeline/launch_stage.py s3-q4 \
    --config "resolved_configs/s3_q4_${seed}.yaml" \
    --output "runs/s3/q4_s${seed}" --execute
done
```

Each run uses **2 GPUs, global batch 64, 600 updates**, saving 100/200/400/600.
The primary released model is **update 200, seed 20260905**, after 12,800 exposures. Do not shorten the
cosine schedule to 200. The loop runs seeds sequentially; independent scheduler
allocations may run them concurrently. Omit `--execute` to inspect the command.

## Evaluation and reproducibility scope

MMAU-mini (1,000) is validation only. The test protocol uses MMAU full9k,
MMAR (1,000) and the fixed ADQA-clean (1,577) IDs. Full9k requires an official
MMAU service receipt; its public placeholder answers are not ground truth.
See [Evaluation](EVALUATION.md) for generation and official scoring commands.

The joint-pool extension inherits the earlier validation-selected update 200.
It was developed after test exposure and is not a renewed blind evaluation.
The inherited S1 training list has known canonical audio matches to four MMAU
full9k questions (one recording) and 14 ADQA-clean questions. This release does
not claim exhaustive absence of contamination. The frozen 1,577-ID primary
ADQA mask must not change per model.

S1/S2 historical cache preprocessing and S3 raw first-20-second preprocessing
are preserved separately. Changing all three to the newest loader would change
the historical recipe. Architecture/optimizer source is retained, with the
small portability-only edits documented in `provenance/release_patches.json`.
New inference defaults to vLLM where supported. Forced-choice scoring, option
likelihood ranking and constrained A/B/C/D decoding are excluded.
