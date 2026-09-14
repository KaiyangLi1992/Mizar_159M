# Mizar: A 159M-Parameter Audio-Language Model for Audio Understanding

Mizar connects a frozen **CED-Small** audio encoder to **SmolLM2-135M** through
an audio–language mapper. This repository contains the three-stage training code, environment definitions
and data preparation tools. The released **Stage-3 Q4 model (seed 20260905,
update 200)** was trained on the joint **strongAC + AVQA** pool.

**[Download the model and training data on Hugging Face](https://huggingface.co/KaiyangLi/Mizar-159M).**

Start with [checkpoint inference](#use-a-checkpoint) to try the model, or
[three-stage training](#three-stage-training) to inspect and reproduce the recipe.
All commands below run from this directory. No cluster-specific scheduler is
required; training commands should run inside your own GPU allocation.

## Included assets

| Directory | Contents |
| --- | --- |
| `runtime/stage1`, `stage2`, `stage3` | Stage-specific model, loader and training code |
| `environment/` | Separate training and vLLM Conda specifications |
| `configs/` | Training templates and local path configuration |
| `pipeline/` | Setup, training, data reconstruction, inference and integrity tools |
| `data/manifests/` | Eight complete training/candidate files, downloaded from HF |
| `data/q4/` | Executed free-generation labels for Q4 reconstruction |
| `data/evaluation/` | Evaluation metadata, official evaluator snapshots and fixed ADQA-clean IDs |
| `checkpoints/final/` | Primary Stage-3 checkpoint, downloaded from Hugging Face |
| `assets/` | Pinned CED-Small and SmolLM2 base assets and tokenizer |
| `provenance/` | Source identities, checksums and documented release changes |

Raw dataset audio and large CED hidden-feature caches are **not bundled**.
The included JSON files are complete metadata and supervision, not synthetic
examples. Obtain the corresponding audio from its upstream source; see
[Data preparation](docs/DATA.md). The downloaded primary checkpoint and base assets are sufficient for inference
on your own audio.

## Environment

```bash
conda env create -f environment/train.yaml
conda activate mizar-train
python -m unittest discover -s tests -v
python pipeline/download_assets.py
python pipeline/verify_bundle.py
```

The training reference is Python 3.10, PyTorch/Torchaudio 2.5.1 with CUDA 12.4,
and Transformers 4.46.3. Keep vLLM in a separate environment:

```bash
conda env create -f environment/vllm.yaml
```

See [Environment notes](docs/ENVIRONMENT.md) for the historical precision
behavior and the scope of the local release checks.

## Use a checkpoint

All `.ckpt` files contain a plain PyTorch tensor state dictionary, including
encoder, mapper and decoder weights. Optimizer and RNG state have been removed;
every exported tensor is checked against its original checkpoint. These are
full model weights, not adapters. They are not an `AutoModel` Hugging Face
architecture and should be loaded with the supplied code.

| File | Role |
| --- | --- |
| `checkpoints/final/stage3_q4_seed20260905.ckpt` | Released joint strongAC + AVQA Q4 model, update 200 |

The public model repository contains one final checkpoint. S1/S2 intermediate
weights are not uploaded; to reproduce training, produce them with the commands
below and set their paths in your local asset configuration.

Create an input JSON file like this:

```json
[
  {
    "id": "my-recording",
    "audio": "/absolute/path/to/recording.wav",
    "question": "What can be heard in this recording?"
  }
]
```

An optional `choices` array supplies the four answer texts for a multiple-choice
question. It changes the prompt only; generation remains unrestricted.

```bash
conda activate mizar-train
python pipeline/prepare_inference.py \
  --checkpoint checkpoints/final/stage3_q4_seed20260905.ckpt \
  --input my_input.json --output inference/my_recording --device cpu

conda activate mizar-vllm
python pipeline/generate_vllm.py \
  --bundle inference/my_recording \
  --output inference/my_recording/answers.json
```

The preparation command computes continuous audio/text prefixes and exports the
**trained** decoder. vLLM generates full answers in FP32, greedily, with at most
300 new tokens. Raw text, generated token IDs, finish reasons, input token IDs,
waveform hashes, checkpoint identity and engine version are retained. Existing
output directories are not overwritten. Audio uses mono, 32 kHz, first 20 seconds
and zero padding. Questions/options are lowercased, text is limited to 256 tokens,
and no extra question-side end-of-text token is added.

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
regenerate the S1/S2 CED caches as described in [docs/DATA.md](docs/DATA.md).
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
See [Evaluation](docs/EVALUATION.md) for generation and official scoring commands.

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

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for code attribution. Included
pretrained model assets and dataset metadata retain their upstream licenses;
this repository's code license does not relicense those assets. Raw audio is
obtained separately from the original dataset providers.
