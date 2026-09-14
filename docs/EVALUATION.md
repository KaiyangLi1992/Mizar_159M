# Evaluation

Evaluation always generates a full, unrestricted answer and retains raw output
before official parsing. There is no option scoring, candidate ranking or
constrained letter-only decoding. Mizar uses FP32 greedy vLLM inference, an
EOS stop token of 0 and a maximum of 300 new tokens. Audio is resampled to
32 kHz, converted to mono, truncated to its first 20 seconds and zero-padded.
The prefix is 126 audio tokens, one separator, and 256 padded/truncated text
tokens. Question/option text is lowercase without an extra question EOT.

## Generate predictions

After locating benchmark audio and setting `MIZAR_AUDIO_PATH_MAP` if necessary:

```bash
conda activate mizar-train
python pipeline/prepare_inference.py \
  --checkpoint checkpoints/final/stage3_q4_seed20260905.ckpt \
  --input data/evaluation/mmar_model_input.json \
  --output inference/mmar_seed20260905 --device cuda

conda activate mizar-vllm
python pipeline/generate_vllm.py \
  --bundle inference/mmar_seed20260905 \
  --output inference/mmar_seed20260905/answers.json
```

Use `mmau_mini_model_input.json`, `mmau_full9k_model_input.json` or
`adqa_clean.json` as the input for the other datasets, with a fresh output
directory for each checkpoint/dataset. Only the fixed ADQA-clean metadata is
used. Generation records include raw text, token IDs, finish reasons, input
IDs, prefix/audio hashes and the engine version. Interrupted runs leave a
partial JSONL journal; an incomplete journal is not a scored submission.

## MMAU-mini: validation only

```bash
python pipeline/score_mmau.py \
  --references data/evaluation/mmau_mini_official.json \
  --predictions inference/mini_seed20260905/answers.json \
  --output inference/mini_seed20260905/official_score.json
```

The implementation uses the designated MMAU token-set content-matching rule.
Prediction/reference IDs must match exactly. Mini is never included in the
three-test-set mean. The joint-pool extension inherits the previously selected
step 200 and does not reselect a checkpoint using the new test results.

## MMAU full9k

```bash
python pipeline/format_predictions.py --benchmark mmau-full \
  --references data/evaluation/mmau_full9k_official_input.json \
  --predictions inference/full9k_seed20260905/answers.json \
  --output inference/full9k_seed20260905/submission.json
```

Submit this complete 9,000-question prediction file through the official MMAU
benchmark service and preserve its receipt. The public answer fields are
placeholders and cannot be used to calculate a local test accuracy. This
repository does not upload predictions automatically.

## MMAR

```bash
python pipeline/format_predictions.py --benchmark mmar \
  --references data/evaluation/mmar_model_input.json \
  --predictions inference/mmar_seed20260905/answers.json \
  --output inference/mmar_seed20260905/official_input.json
python data/evaluation/mmar_official_evaluator.py \
  --input inference/mmar_seed20260905/official_input.json
```

The formatter validates all 1,000 unique IDs before calling the unchanged
upstream evaluator, which would otherwise silently skip missing predictions.

## ADQA-clean

```bash
python pipeline/score_adqa.py \
  --predictions inference/adqa_seed20260905/answers.json \
  --output inference/adqa_seed20260905/clean_score.json
```

The strict wrapper checks the frozen official evaluator's SHA, enforces exactly
1,577 unique clean IDs and calls its unchanged parser/evaluation function.
Empty answers remain incorrect entries rather than disappearing from the
denominator. It preserves raw answers and per-item judgments. Matching a freely
generated answer to official choices is post-generation parsing, not option
likelihood scoring. The primary ADQA mask is identical for every model.
