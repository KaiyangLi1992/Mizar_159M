# Environment and runtime notes

Use `environment/train.yaml` for historical training code and audio-prefix
preparation, and `environment/vllm.yaml` for decoder inference. Do not install
vLLM into the training environment: it requires a different PyTorch build.

The training reference is Linux x86-64, Python 3.10.14, PyTorch/Torchaudio
2.5.1+cu124, Transformers 4.46.3 and Tokenizers 0.20.3. S1 used four RTX A6000s;
S2/S3 use two GPUs per run. Inference uses vLLM 0.11.0, PyTorch 2.8.0 and FP32.
The YAML specifications pin the principal packages; they are not a complete
OS/driver/container lock or a claim of bitwise reproducibility on every GPU.

Historical mixed-precision configuration labels should not be interpreted as
a guarantee that every operation executed in that dtype. Stage-specific
training source is preserved for replay rather than replacing it with later
AMP implementations from unrelated experiments.

The release removes an unused `pysepm` import from each runtime. This avoids
installing unrelated audio-quality metric dependencies from mutable Git
branches. Audio-path relocation is applied only when `MIZAR_AUDIO_PATH_MAP` is
set. Exact original/release hashes and the reasons for each change are recorded
in `provenance/release_patches.json`.

`pipeline/verify_bundle.py` validates the actual local model/data assets.
`python -m unittest discover -s tests -v` exercises CPU-only release contracts.
Model loading and launcher smoke checks are recorded separately in `provenance/release_checks.json`. A full three-stage retraining is not part of release packaging.
