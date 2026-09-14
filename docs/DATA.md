# Data preparation

The code repository includes frozen Q4 labels and evaluation metadata.
Download complete supervision/candidate manifests from the linked HF model
repository using `python pipeline/download_assets.py --training-data`. It does not bundle source audio or the large
historical CED hidden-feature caches. Dataset/model assets remain governed by
their upstream licenses; the code license does not grant new dataset rights.

## Included files

| File in `data/manifests/` | Rows | Use |
| --- | ---: | --- |
| `s1.json.gz` | 1,477,976 | Historical ReasonAQA + AudioMCQ mixture |
| `s2.json.gz` | 289,952 | 256,077 strongAC + 33,875 AVQA direct-answer rows |
| `s3_candidates.json.gz` | 289,952 | Same identities, original recording paths for S3 |
| `s3_q4_seed20260905.json.gz` through `...20260909.json.gz` | 38,400 each | Five exact joint-pool Q4 training files |

`python pipeline/unpack_data.py` verifies both the compressed SHA and the
uncompressed executed SHA before exposing a manifest for training. Original
row order, sample identity, text and fields are preserved. The S1 rationale
views are disclosed historical supervision, not newly generated targets.

Common loader fields are `filepath1`, `filepath2`, `input`, and `answer`.
S2/S3 also preserve `sample_key`, `source`, `question`, `choices`, and gold answer
identity. Empty `filepath2` means one recording; nonempty S2 AVQA pair paths
must retain their original chunk pairing. Do not replace a pair with a guessed
full-recording crop.

## Obtain and locate the audio

The sources are [ReasonAQA (Mellow)](https://github.com/soham97/mellow),
[AudioMCQ](https://huggingface.co/datasets/inclusionAI/AudioMCQ) (including its
strongAC subset), and [AVQA](https://github.com/AlyssaYoung/AVQA). Follow each provider's access and license terms. Use the filenames
and source/sample identifiers in the included manifests to recover the exact
recordings. AVQA uses sound and both categories without video; visual-only
questions are excluded.

The manifests retain the executed path strings because these are part of CED
cache identities. To use a new location, create a JSON object mapping old roots
to your local roots:

```json
{
  "/original/dataset/audio": "/datasets/downloaded/audio"
}
```

```bash
export MIZAR_AUDIO_PATH_MAP="$PWD/configs/audio_paths.local.json"
```

The longest matching path-component prefix wins. Only the file-open path is
translated; original manifest IDs, text, order and cache keys remain unchanged.
The same resolver is used by training, cache construction and prefix inference.
Do not hash-rewrite a manifest just to relocate audio.

## S1/S2 CED caches

For exact artifact replay, obtain the original versioned cache directories and
set `s1_ced_cache` / `s2_ced_cache_roots` in `configs/assets.local.yaml`.
Each cache shard contains `metadata.json`, `keys.jsonl`, `hidden.f16.dat`,
`valid_tokens.npy` and `segment_lengths.npy`. The encoded features precede the
trainable mapper; they do not contain teacher answers.

Alternatively, regenerate them with the included historical feature extractor.
This is feature preprocessing, not answer generation. It preserves the original
`fixed20` center-crop/pair policy, final hidden features and canonical path keys.
Generated caches may differ numerically from the historical CUDA environment;
regeneration is not a claim of bitwise cache replay.

```bash
conda activate mizar-train
python pipeline/build_ced_cache.py \
  --train-json data/prepared/s1.json --out-root cache/s1 \
  --policy fixed20 --cache-point final_hidden \
  --ced-model-name "$PWD/assets/ced-small" --device cuda

for shard in 0 1 2 3; do
  python pipeline/build_ced_cache.py \
    --train-json data/prepared/s2.json --out-root "cache/s2/shard${shard}" \
    --policy fixed20 --cache-point final_hidden \
    --ced-model-name "$PWD/assets/ced-small" --device cuda \
    --num-shards 4 --shard-index "$shard"
done
```

Run feature extraction inside a GPU allocation and provide enough disk for
unique-audio hidden features. S1 can also be sharded with `--num-shards` and
`--shard-index` into subdirectories below `cache/s1`. The loader discovers the
shards. S3 uses original raw audio and does not consume these caches.

## Joint-pool Q4 identity

The train universe is 261,171 rows (230,599 strongAC and 30,572 AVQA); the probe
universe is 28,781. Source proportions in the selected Q4 set are sampling
outcomes, not a fixed AVQA quota. The committed executed labels come from free
generations, with teacher/student correctness for four cyclic orderings.
The deterministic materializer verifies the exact five training-file hashes.
Historical strongAC-only labels remain in provenance for identification only;
they are not inputs to this release's active recipe.

Fresh screening requires the fixed S2 student and AudioMCQ's tuned teacher,
`inclusionAI/AudioMCQ-Mixed-To-Strong`, revision
`1405063d68760b17c705b38a8fc6633a47bf68df`. It requires unrestricted generation,
retention of raw answers and official content matching. Exact screening inputs, label hashes and teacher revision are recorded in
provenance. Replay uses the provided labels, so downloading the large teacher
is unnecessary.

## Evaluation metadata

Only ADQA-clean 1,577 metadata and gold rows are included for active evaluation.
Its frozen ID file is `data/evaluation/ADQA_DEV_CLEAN_V1.json`. Historical
unfiltered ADQA files are not active evaluation inputs. The inherited S1
canonical-hash audit identified four full9k questions sharing one recording
and 14 ADQA-clean matches; this is not an exhaustive near-duplicate audit.
