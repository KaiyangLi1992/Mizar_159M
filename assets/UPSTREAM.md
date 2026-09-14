# Bundled base assets

The bundle includes local snapshots needed to instantiate the released model
without downloading pretrained weights at inference time. The trained Mizar
state dictionary subsequently replaces the model's initialized weights.

| Component | Upstream | Snapshot revision | Upstream license |
| --- | --- | --- | --- |
| CED-Small | [mispeech/ced-small](https://huggingface.co/mispeech/ced-small) | `06bb40c5ec089e96867ebc5246be02441f4a71e4` | Apache-2.0 |
| SmolLM2-135M | [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/README.md) | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | Apache-2.0 |

The CED source headers and model card are preserved. See the included
`LICENSE-APACHE-2.0`; Mizar's code license does not replace these terms.
File-level hashes are in `provenance/local_bundle.json`.
